"""
Wind Turbine Blade Panorama Stitching Pipeline v2

Pipeline:
  1. Load metadata, images, masks
     - PS/SS: convex hull post-processing on SAM masks
     - LE/TE: raw SAM segmentation masks
  2. Brightness alignment + SAM segmentation
  3. LoFTR keypoint matching at high resolution (2160x1440)
  4. DBSCAN cluster filtering on match vectors (min_samples=4)
  5. Grid-based spatial sampling (5x5 grid, 25 samples) for transform estimation
  6. Coarse transforms (DCM) + fine transforms (scale + rotation + translation)
     - PS/SS: conditional scale (apply if ratio in [0.9,0.96] or [1.06,1.10])
     - LE/TE: always use LiDAR scale, no rotation
     - PS/SS: conditional rotation (apply if |rot| in [3, 10] degrees)
  7. Fallback: if proj outside [0.3, 2.0], use coarse
  8. Match-skip: skip images with 0 matches or step < 0.1
  9. Cut-skip: remove redundant images whose cut is covered by next image
  10. Stitch panorama (rotation-aware affine warp)
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import cv2
import torch
from pathlib import Path
from collections import defaultdict
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR))

from modules.segmentation import load_sam, segment_image
from modules.matching import load_loftr, match_loftr, filter_by_mask
from modules.brightness import align_brightness
from modules.coarse import compute_coarse_transforms, clamp_lidar_distances
from modules.stitching import stitch_trans_scale

LOFTR_W, LOFTR_H = 2160, 1440
WORK_LONG_EDGE = 720
PROJ_MIN, PROJ_MAX = 0.2, 2.0
ROT_MIN, ROT_MAX = 3.0, 10.0
STEP_MIN = 0.1
GRID_N = 5
GRID_SAMPLES = 25


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def get_disconnection_edges(img1_rect, img2_rect):
    """Return overlap bbox and disconnection edges (overlap bbox edges on img2's boundary)."""
    ax1, ay1, ax2, ay2 = img1_rect
    bx1, by1, bx2, by2 = img2_rect
    ox1, oy1 = max(ax1, bx1), max(ay1, by1)
    ox2, oy2 = min(ax2, bx2), min(ay2, by2)
    if ox1 >= ox2 or oy1 >= oy2:
        return None, []
    edges = []
    if abs(ox1 - bx1) < 1:
        edges.append({'edge': 'left', 'segment': (ox1, oy1, ox1, oy2)})
    if abs(ox2 - bx2) < 1:
        edges.append({'edge': 'right', 'segment': (ox2, oy1, ox2, oy2)})
    if abs(oy1 - by1) < 1:
        edges.append({'edge': 'top', 'segment': (ox1, oy1, ox2, oy1)})
    if abs(oy2 - by2) < 1:
        edges.append({'edge': 'bottom', 'segment': (ox1, oy2, ox2, oy2)})
    return (ox1, oy1, ox2, oy2), edges


def get_cuts(disc_edges, mask1_region):
    """Find contiguous segments of disconnection edges where mask1 > 0."""
    cuts = []
    for e in disc_edges:
        sx1, sy1, sx2, sy2 = [int(v) for v in e['segment']]
        if sx1 == sx2:  # vertical edge
            x = max(0, min(sx1, mask1_region.shape[1] - 1))
            col = mask1_region[min(sy1, sy2):max(sy1, sy2), x]
            in_mask, start = False, None
            for j, val in enumerate(col):
                y = min(sy1, sy2) + j
                if val > 0 and not in_mask:
                    in_mask, start = True, y
                elif val == 0 and in_mask:
                    in_mask = False
                    cuts.append((x, start, x, y - 1))
            if in_mask:
                cuts.append((x, start, x, min(sy1, sy2) + len(col) - 1))
        else:  # horizontal edge
            y = max(0, min(sy1, mask1_region.shape[0] - 1))
            row = mask1_region[y, min(sx1, sx2):max(sx1, sx2)]
            in_mask, start = False, None
            for j, val in enumerate(row):
                x = min(sx1, sx2) + j
                if val > 0 and not in_mask:
                    in_mask, start = True, x
                elif val == 0 and in_mask:
                    in_mask = False
                    cuts.append((start, y, x - 1, y))
            if in_mask:
                cuts.append((start, y, min(sx1, sx2) + len(row) - 1, y))
    return cuts


def get_pair_cuts(mask1, mask2, transform):
    """Get cuts (disconnection edge segments on mask1) in img1's coordinate system."""
    h1, w1 = mask1.shape[:2]
    h2, w2 = mask2.shape[:2]
    tx, ty, scale = transform['tx'], transform['ty'], transform['scale']
    new_w2, new_h2 = int(w2 * scale), int(h2 * scale)
    if new_w2 <= 0 or new_h2 <= 0:
        return []
    img1_rect = (0, 0, w1, h1)
    img2_rect = (int(tx), int(ty), int(tx) + new_w2, int(ty) + new_h2)
    _, disc_edges = get_disconnection_edges(img1_rect, img2_rect)
    if not disc_edges:
        return []
    return get_cuts(disc_edges, mask1)


def _to_matrix(t):
    """Convert transform dict to 3x3 homogeneous matrix."""
    s = t['scale']
    rad = np.radians(t.get('rotation', 0.0))
    c, sn = np.cos(rad), np.sin(rad)
    return np.array([[s * c, -s * sn, t['tx']],
                     [s * sn, s * c, t['ty']],
                     [0, 0, 1]])


def _from_matrix(M):
    """Extract transform dict from 3x3 homogeneous matrix."""
    scale = np.sqrt(M[0, 0]**2 + M[1, 0]**2)
    rotation = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
    return {'tx': M[0, 2], 'ty': M[1, 2], 'scale': scale, 'rotation': rotation}


def chain_transforms(t1, t2):
    """Chain t1 (base->mid) and t2 (mid->end) into base->end."""
    return _from_matrix(_to_matrix(t1) @ _to_matrix(t2))


def check_cut_covered(cuts, transform, img2_shape):
    """Check if either endpoint of every cut segment is inside img2's region."""
    if not cuts:
        return False
    h2, w2 = img2_shape[:2]
    tx, ty, scale = transform['tx'], transform['ty'], transform['scale']
    new_w2, new_h2 = int(w2 * scale), int(h2 * scale)
    if new_w2 <= 0 or new_h2 <= 0:
        return False
    x2_min, y2_min = tx, ty
    x2_max, y2_max = tx + new_w2, ty + new_h2
    for cx1, cy1, cx2, cy2 in cuts:
        ep1_inside = x2_min <= cx1 < x2_max and y2_min <= cy1 < y2_max
        ep2_inside = x2_min <= cx2 < x2_max and y2_min <= cy2 < y2_max
        if not (ep1_inside or ep2_inside):
            return False
    return True


# ---------------------------------------------------------------------------
# Transform estimation helpers
# ---------------------------------------------------------------------------

def estimate_scale(pts1, pts2):
    """Estimate scale from keypoint pairwise distance ratios."""
    if len(pts1) < 3:
        return None
    d1 = pdist(pts1)
    d2 = pdist(pts2)
    valid = d2 > 1e-6
    if valid.sum() < 1:
        return None
    return np.median(d1[valid] / d2[valid])


def estimate_rotation(pts1, pts2):
    """Estimate rotation angle (degrees) from keypoint matches.
    Uses median of pairwise angle differences."""
    if len(pts1) < 2:
        return 0.0
    n = len(pts1)
    angles = []
    for ii in range(n):
        for jj in range(ii + 1, min(n, ii + 50)):
            d1 = pts1[jj] - pts1[ii]
            d2 = pts2[jj] - pts2[ii]
            if np.linalg.norm(d1) < 1e-6 or np.linalg.norm(d2) < 1e-6:
                continue
            a1 = np.arctan2(d1[1], d1[0])
            a2 = np.arctan2(d2[1], d2[0])
            diff = a1 - a2
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            angles.append(diff)
    if not angles:
        return 0.0
    return np.degrees(np.median(angles))


def rotation_matrix(deg):
    """2x2 rotation matrix for angle in degrees."""
    rad = np.radians(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s], [s, c]])


def grid_sample(pts, n_samples=GRID_SAMPLES, grid_n=GRID_N):
    """Sample up to n_samples points spatially distributed across a grid_n x grid_n grid."""
    n = len(pts)
    if n <= n_samples:
        return np.arange(n)
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    range_xy = max_xy - min_xy
    range_xy[range_xy == 0] = 1
    cell_indices = np.floor((pts - min_xy) / range_xy * grid_n).astype(int)
    cell_indices = np.clip(cell_indices, 0, grid_n - 1)
    cell_keys = cell_indices[:, 0] * grid_n + cell_indices[:, 1]
    cells = defaultdict(list)
    for idx, key in enumerate(cell_keys):
        cells[key].append(idx)
    for cl in cells.values():
        np.random.shuffle(cl)
    selected = []
    cell_lists = list(cells.values())
    while len(selected) < n_samples:
        added = False
        for cl in cell_lists:
            if cl and len(selected) < n_samples:
                selected.append(cl.pop(0))
                added = True
        if not added:
            break
    return np.array(selected)


def compute_fine_transform_full(pts1, pts2, scale_prior, section_side):
    """Compute fine transform: decide scale, rotation, then compute translation.

    Returns (transform_dict, est_scale_ratio, est_rotation).
    """
    if len(pts1) < 1:
        return None, 1.0, 0.0

    # Grid-based spatial sampling
    sample_idx = grid_sample(pts1, n_samples=GRID_SAMPLES, grid_n=GRID_N)
    pts1_s = pts1[sample_idx]
    pts2_s = pts2[sample_idx]

    # 1. Decide scale
    est_scale = estimate_scale(pts1_s, pts2_s)
    if est_scale is not None:
        scale_ratio = est_scale / scale_prior
        if section_side in ('PS', 'SS'):
            apply_scale = (0.9 < scale_ratio < 0.96) or (1.06 < scale_ratio < 1.10)
        else:  # LE/TE: always trust LiDAR scale
            apply_scale = False
        chosen_scale = est_scale if apply_scale else scale_prior
    else:
        scale_ratio = 1.0
        chosen_scale = scale_prior

    # 2. Compute translation with decided scale (no rotation)
    trans = np.median(pts1_s - pts2_s * chosen_scale, axis=0)

    return ({'tx': trans[0], 'ty': trans[1], 'scale': chosen_scale, 'rotation': 0.0},
            scale_ratio, 0.0)


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------

def dbscan_filter_matches(pts1_thumb, pts2_thumb, coarse_t, thumb_w, thumb_h):
    """Filter keypoint matches using DBSCAN clustering on match vectors.

    Selects the cluster whose average projected magnitude along the
    center-crossing axis is closest to zero.
    """
    tx, ty, sc = coarse_t['tx'], coarse_t['ty'], coarse_t['scale']
    new_w2, new_h2 = int(thumb_w * sc), int(thumb_h * sc)

    min_x, min_y = min(0, tx), min(0, ty)
    x1_off, y1_off = int(-min_x), int(-min_y)
    x2_off, y2_off = int(tx - min_x), int(ty - min_y)

    pts1_c = pts1_thumb + np.array([x1_off, y1_off])
    pts2_c = pts2_thumb * sc + np.array([x2_off, y2_off])

    c1 = np.array([x1_off + thumb_w / 2, y1_off + thumb_h / 2])
    c2 = np.array([x2_off + new_w2 / 2, y2_off + new_h2 / 2])
    axis_vec = c2 - c1
    axis_mag = np.linalg.norm(axis_vec)
    axis_unit = axis_vec / axis_mag if axis_mag > 0 else np.array([1.0, 0.0])

    match_vecs = pts2_c - pts1_c

    if len(match_vecs) >= 3:
        mags = np.linalg.norm(match_vecs, axis=1)
        eps = max(np.median(mags) * 0.15, 5.0)
        clustering = DBSCAN(eps=eps, min_samples=4).fit(match_vecs)
        labels = clustering.labels_

        proj1 = (pts1_c - c1) @ axis_unit
        proj2 = (pts2_c - c1) @ axis_unit
        proj_mags = proj2 - proj1

        cluster_ids = [l for l in set(labels) if l >= 0]
        if cluster_ids:
            best = max(cluster_ids, key=lambda l: (labels == l).sum())
            return labels == best
    return np.ones(len(pts1_thumb), dtype=bool)


def match_pair_loftr(img1_seg, img2_seg, mask1_hr, mask2_hr, scale_xy):
    """Run LoFTR matching between two high-res segmented images, return thumbnail-scale points."""
    pts1, pts2, conf = match_loftr(img1_seg, img2_seg)
    pts1, pts2, conf = filter_by_mask(pts1, pts2, mask1_hr, mask2_hr, conf=conf)
    pts1_thumb = pts1 * scale_xy
    pts2_thumb = pts2 * scale_xy
    return pts1_thumb, pts2_thumb, conf


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def make_convex_mask(mask):
    """Fill the convex hull of the mask region."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    all_pts = np.concatenate(contours)
    hull = cv2.convexHull(all_pts)
    convex = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillConvexPoly(convex, hull, 1)
    return convex


def load_draft_data(draft_dir):
    """Load metadata and return photos_by_id and sections dict.

    Section keys include mission UUID: 'A-PS-<uuid>'.
    """
    with open(os.path.join(draft_dir, 'metadata.json')) as f:
        metadata = json.load(f)
    photos_by_id = {p['id']: p for p in metadata['photos']}

    sections = defaultdict(list)
    for p in metadata['photos']:
        blade = p.get('blade_tag')
        side = p.get('blade_side_tag')
        if not blade or not side:
            continue
        mission = p['metadata'].get('mission_uuid') or p['metadata'].get('missionUuid') or 'unknown'
        sections[f"{blade}-{side}-{mission}"].append(p['id'])

    for key in sections:
        sections[key].sort(key=lambda pid: photos_by_id[pid]['metadata'].get('r', 0))

    return photos_by_id, dict(sections)


def resize_long_edge(img, long_edge=WORK_LONG_EDGE):
    """Resize image so that the longer edge matches long_edge, keeping aspect ratio."""
    h, w = img.shape[:2]
    scale = long_edge / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h))


def load_images_and_masks(draft_dir, photos_by_id, photo_ids, section_side):
    """Load original images, resize to working resolution, align brightness, segment masks."""
    images, loaded_pids = [], []
    for pid in photo_ids:
        photo = photos_by_id[pid]
        img_path = os.path.join(draft_dir, photo['original_path'])
        if not os.path.exists(img_path):
            img_path = os.path.join(draft_dir, photo['local_path'])
        img = cv2.imread(img_path)
        if img is not None:
            images.append(resize_long_edge(img))
            loaded_pids.append(pid)
    # Clamp distances with running default (matching create_stitching)
    distances = clamp_lidar_distances(photos_by_id, loaded_pids)
    print(f"Loaded {len(images)} images")

    images = align_brightness(images)
    use_convex = section_side in ('PS', 'SS')
    raw_masks = [segment_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in images]
    masks = [make_convex_mask(m) if use_convex else m.astype(np.uint8) for m in raw_masks]
    print(f"Generated {len(masks)} masks (convex={use_convex}, side={section_side})")

    return images, masks, distances


def load_hires_segmented(draft_dir, photos_by_id, photo_ids, masks):
    """Load high-res images, apply masks, return segmented images + hires masks."""
    images_hires_seg = []
    masks_hires = []
    for pid, mask in zip(photo_ids, masks):
        photo = photos_by_id[pid]
        orig_path = os.path.join(draft_dir, photo['original_path'])
        img = cv2.resize(cv2.imread(orig_path), (LOFTR_W, LOFTR_H))
        mask_hr = cv2.resize(mask.astype(np.uint8), (LOFTR_W, LOFTR_H))
        img[mask_hr == 0] = 0
        images_hires_seg.append(img)
        masks_hires.append(mask_hr)
    print(f"Loaded {len(images_hires_seg)} high-res segmented images")
    return images_hires_seg, masks_hires


def _match_and_fine(idx_i, idx_j, images_hires_seg, masks_hires, scale_xy,
                    photos_by_id, photo_ids, distances, thumb_w, thumb_h,
                    compute_proj_fn, section_side):
    """Match pair (idx_i, idx_j), filter, compute fine transform + fallback.

    Returns (fine_t, fine_t_original, is_fallback, n_matches, est_rot, est_scale_ratio).
    """
    torch.cuda.empty_cache()
    pts1, pts2, _ = match_pair_loftr(
        images_hires_seg[idx_i], images_hires_seg[idx_j],
        masks_hires[idx_i], masks_hires[idx_j], scale_xy,
    )
    coarse_t = compute_coarse_transforms(
        photos_by_id, [photo_ids[idx_i], photo_ids[idx_j]],
        img_width=thumb_w, img_height=thumb_h,
    )[0]
    inlier = dbscan_filter_matches(pts1, pts2, coarse_t, thumb_w, thumb_h)
    pts1_f, pts2_f = pts1[inlier], pts2[inlier]
    n_matches = len(pts1_f)

    scale = distances[idx_j] / distances[idx_i]
    fine_t, est_scale_ratio, est_rot = compute_fine_transform_full(pts1_f, pts2_f, scale, section_side)

    if fine_t is None:
        return None, None, True, 0, 0.0, 1.0

    # Proj fallback check using coarse at working resolution
    coarse_t_fb = compute_coarse_transforms(
        photos_by_id, [photo_ids[idx_i], photo_ids[idx_j]],
        img_width=thumb_w, img_height=thumb_h,
    )[0]
    coarse_vec = np.array([coarse_t_fb['tx'], coarse_t_fb['ty']])
    fine_vec = np.array([fine_t['tx'], fine_t['ty']])
    proj = compute_proj_fn(coarse_vec, fine_vec)

    if proj < PROJ_MIN or proj > PROJ_MAX:
        coarse_t_fb['rotation'] = 0.0
        return coarse_t_fb, fine_t, True, n_matches, est_rot, est_scale_ratio

    return fine_t, fine_t, False, n_matches, est_rot, est_scale_ratio


def compute_all_transforms(
    images, masks, distances, photo_ids, photos_by_id,
    images_hires_seg, masks_hires, scale_xy, thumb_w, thumb_h,
    section_side,
):
    """Compute coarse and fine transforms, skipping images with 0 matches or small step.

    Returns:
        selected_idx, coarse_transforms, fine_transforms, fine_original,
        fallback_flags, rotation_angles, est_scale_ratios,
        global_axis_unit, compute_proj_fn
    """
    # Global stitching axis from all coarse transforms
    all_coarse = compute_coarse_transforms(
        photos_by_id, photo_ids, img_width=thumb_w, img_height=thumb_h,
    )
    global_pos = sum(
        (np.array([ct['tx'], ct['ty']]) for ct in all_coarse),
        np.array([0.0, 0.0]),
    )
    global_norm = np.linalg.norm(global_pos)
    global_axis_unit = global_pos / global_norm if global_norm > 0 else np.array([1.0, 0.0])
    print(f"Global stitching axis: ({global_pos[0]:.1f}, {global_pos[1]:.1f})")

    def compute_proj(coarse_vec, fine_vec):
        proj_c = np.dot(coarse_vec, global_axis_unit)
        proj_f = np.dot(fine_vec, global_axis_unit)
        return proj_f / proj_c if abs(proj_c) > 1e-6 else 0.0

    # Compute step threshold (diagonal slice length along global axis)
    ux, uy = global_axis_unit
    h_img, w_img = images[0].shape[:2]
    diag_slice = min(w_img / abs(ux) if abs(ux) > 1e-6 else float('inf'),
                     h_img / abs(uy) if abs(uy) > 1e-6 else float('inf'))

    # Match-skip: skip images with 0 matches or step < STEP_MIN
    print(f"Match-skip (0 matches or step < {STEP_MIN})...")
    n = len(images)
    selected_idx = [0]
    fine_transforms, fine_original, fallback_flags = [], [], []
    rotation_angles, est_scale_ratios = [], []

    i = 0
    while i < n - 1:
        matched = False
        for j in range(i + 1, n):
            # Check step first
            coarse_t_ij = compute_coarse_transforms(
                photos_by_id, [photo_ids[i], photo_ids[j]],
                img_width=thumb_w, img_height=thumb_h,
            )[0]
            step = np.dot(np.array([coarse_t_ij['tx'], coarse_t_ij['ty']]), global_axis_unit) / diag_slice
            if abs(step) < STEP_MIN and j < n - 1:
                print(f"  Skip image {j}: step={step:.3f} < {STEP_MIN}")
                continue

            ft, ft_orig, fb, n_matches, rot, est_sr = _match_and_fine(
                i, j, images_hires_seg, masks_hires, scale_xy,
                photos_by_id, photo_ids, distances, thumb_w, thumb_h,
                compute_proj, section_side,
            )
            if n_matches == 0 and j < n - 1:
                print(f"  Skip image {j}: 0 matches with image {i}")
                continue

            if j > i + 1:
                skipped = list(range(i + 1, j))
                print(f"  Pair ({i},{j}): skipped {skipped}, {n_matches} matches")
            else:
                label = "FALLBACK" if fb else f"FINE d={np.linalg.norm([ft['tx'], ft['ty']]):.1f}"
                print(f"  Pair {i}: {n_matches} matches, {label}")

            selected_idx.append(j)
            if ft is None:
                # No matches at all, use coarse
                coarse_t_ij['rotation'] = 0.0
                fine_transforms.append(coarse_t_ij)
                fine_original.append(coarse_t_ij)
                fallback_flags.append(True)
                rotation_angles.append(0.0)
                est_scale_ratios.append(1.0)
            else:
                fine_transforms.append(ft)
                fine_original.append(ft_orig)
                fallback_flags.append(fb)
                rotation_angles.append(rot)
                est_scale_ratios.append(est_sr)
            i = j
            matched = True
            break

        if not matched:
            print(f"  Pair {i}: no valid match found, fallback for ({i},{i+1})")
            coarse_t_fb = compute_coarse_transforms(
                photos_by_id, [photo_ids[i], photo_ids[i + 1]],
                img_width=thumb_w, img_height=thumb_h,
            )[0]
            coarse_t_fb['rotation'] = 0.0
            selected_idx.append(i + 1)
            fine_transforms.append(coarse_t_fb)
            fine_original.append(coarse_t_fb)
            fallback_flags.append(True)
            rotation_angles.append(0.0)
            est_scale_ratios.append(1.0)
            i = i + 1

    match_skipped = sorted(set(range(n)) - set(selected_idx))
    if match_skipped:
        print(f"Match-skipped {len(match_skipped)} images: {match_skipped}")
    print(f"After match-skip: {len(selected_idx)}/{n} images")

    # Recompute coarse for selected images
    selected_photo_ids = [photo_ids[k] for k in selected_idx]
    coarse_transforms = compute_coarse_transforms(
        photos_by_id, selected_photo_ids, img_width=thumb_w, img_height=thumb_h,
    )

    return (selected_idx, coarse_transforms, fine_transforms, fine_original,
            fallback_flags, rotation_angles, est_scale_ratios,
            global_axis_unit, compute_proj)


def match_and_compute_fine_pair(
    idx_i, idx_j,
    images_hires_seg, masks_hires, photos_by_id, photo_ids,
    distances, scale_xy, thumb_w, thumb_h, compute_proj_fn, section_side,
):
    """Recompute LoFTR matching + fine transform for an arbitrary pair.

    Used when an intermediate image is skipped in cut-skip.
    Returns (fine_t, fine_t_original, is_fallback, est_rot, est_scale_ratio).
    """
    ft, ft_orig, fb, n_matches, est_rot, est_sr = _match_and_fine(
        idx_i, idx_j, images_hires_seg, masks_hires, scale_xy,
        photos_by_id, photo_ids, distances, thumb_w, thumb_h,
        compute_proj_fn, section_side,
    )
    if ft is not None:
        d_f = np.linalg.norm([ft['tx'], ft['ty']])
        label = "fallback" if fb else f"FINE d={d_f:.1f}"
    else:
        label = "fallback (no matches)"
    print(f"    Recomputed ({idx_i},{idx_j}): {n_matches} matches, {label}")
    return ft, ft_orig, fb, est_rot, est_sr


def filter_redundant_images(
    images, masks, distances, photo_ids, photos_by_id,
    fine_transforms, fine_original, fallback_flags,
    rotation_angles, est_scale_ratios,
    images_hires_seg, masks_hires, scale_xy, thumb_w, thumb_h,
    compute_proj_fn, section_side,
):
    """Remove images whose cut is fully covered by the next image."""
    print("Filtering redundant images...")

    selected_idx = [0, 1]
    sel_fine_t = [fine_transforms[0]]
    sel_fine_t_orig = [fine_original[0]]
    sel_fallback = [fallback_flags[0]]
    sel_rot = [rotation_angles[0]]
    sel_sr = [est_scale_ratios[0]]
    prev_cuts = get_pair_cuts(masks[0], masks[1], fine_transforms[0])
    print(f"  KEEP image 0, 1: {len(prev_cuts)} cut(s)")

    for j in range(2, len(images)):
        base = selected_idx[-2]
        last = selected_idx[-1]

        t_last_j = fine_transforms[last]
        t_base_j = chain_transforms(sel_fine_t[-1], t_last_j)

        covered = prev_cuts and check_cut_covered(
            prev_cuts, t_base_j, masks[j].shape,
        )

        if covered:
            print(f"  SKIP image {last}: image {j} covers pair ({base},{last}) cut")
            selected_idx.pop()
            sel_fine_t.pop()
            sel_fine_t_orig.pop()
            sel_fallback.pop()
            sel_rot.pop()
            sel_sr.pop()
            selected_idx.append(j)

            new_ft, new_ft_orig, new_fb, new_rot, new_sr = match_and_compute_fine_pair(
                base, j,
                images_hires_seg, masks_hires, photos_by_id, photo_ids,
                distances, scale_xy, thumb_w, thumb_h, compute_proj_fn, section_side,
            )
            sel_fine_t.append(new_ft)
            sel_fine_t_orig.append(new_ft_orig)
            sel_fallback.append(new_fb)
            sel_rot.append(new_rot)
            sel_sr.append(new_sr)
            prev_cuts = get_pair_cuts(masks[base], masks[j], new_ft)
        else:
            selected_idx.append(j)
            sel_fine_t.append(t_last_j)
            sel_fine_t_orig.append(fine_original[last])
            sel_fallback.append(fallback_flags[last])
            sel_rot.append(rotation_angles[last])
            sel_sr.append(est_scale_ratios[last])
            prev_cuts = get_pair_cuts(masks[last], masks[j], t_last_j)
            print(f"  KEEP image {j}: {len(prev_cuts)} cut(s)")

    skipped = sorted(set(range(len(images))) - set(selected_idx))
    print(f"Result: {len(selected_idx)}/{len(images)} kept, skipped: {skipped}")

    return selected_idx, sel_fine_t, sel_fine_t_orig, sel_fallback


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_section(section_name, photo_ids, photos_by_id, draft_dir, output_dir):
    """Run the full pipeline for one section."""
    torch.cuda.empty_cache()
    t_section_start = time.time()

    print(f"\n{'='*60}")
    print(f"Processing: {section_name}")
    print(f"{'='*60}")

    # Extract section side for conditional scale/rotation/mask
    section_side = section_name.split('-')[1]  # key format: blade-side-mission

    t0 = time.time()
    images, masks, distances = load_images_and_masks(draft_dir, photos_by_id, photo_ids, section_side)
    if len(images) < 2:
        print(f"Skipping {section_name}: need at least 2 images")
        return
    t_load_seg = time.time() - t0

    thumb_w, thumb_h = images[0].shape[1], images[0].shape[0]
    scale_xy = np.array([thumb_w / LOFTR_W, thumb_h / LOFTR_H])

    t0 = time.time()
    images_hires_seg, masks_hires = load_hires_segmented(
        draft_dir, photos_by_id, photo_ids, masks,
    )
    t_hires = time.time() - t0

    # Compute transforms (with match-skip)
    t0 = time.time()
    (match_sel_idx, coarse_transforms, fine_transforms, fine_original, fallback_flags,
     rotation_angles, est_scale_ratios,
     global_axis_unit, compute_proj_fn) = compute_all_transforms(
        images, masks, distances, photo_ids, photos_by_id,
        images_hires_seg, masks_hires, scale_xy, thumb_w, thumb_h,
        section_side,
    )
    t_match_skip = time.time() - t0

    # Remap to match-skip-selected arrays
    ms_images = [images[i] for i in match_sel_idx]
    ms_masks = [masks[i] for i in match_sel_idx]
    ms_distances = [distances[i] for i in match_sel_idx]
    ms_photo_ids = [photo_ids[i] for i in match_sel_idx]
    ms_hires_seg = [images_hires_seg[i] for i in match_sel_idx]
    ms_masks_hires = [masks_hires[i] for i in match_sel_idx]

    # Cut-skip filter
    t0 = time.time()
    cut_sel_idx, sel_fine_t, sel_fine_t_orig, sel_fallback = filter_redundant_images(
        ms_images, ms_masks, ms_distances, ms_photo_ids, photos_by_id,
        fine_transforms, fine_original, fallback_flags,
        rotation_angles, est_scale_ratios,
        ms_hires_seg, ms_masks_hires, scale_xy, thumb_w, thumb_h,
        compute_proj_fn, section_side,
    )
    t_cut_skip = time.time() - t0

    # Build final selected data
    images_sel = [ms_images[i] for i in cut_sel_idx]
    photo_ids_sel = [ms_photo_ids[i] for i in cut_sel_idx]

    # Recompute coarse for selected images
    sel_coarse_t = compute_coarse_transforms(
        photos_by_id, photo_ids_sel, img_width=thumb_w, img_height=thumb_h,
    )

    # Stitch panoramas
    t0 = time.time()
    panorama_coarse = stitch_trans_scale(images_sel, sel_coarse_t)
    panorama_fine = stitch_trans_scale(images_sel, sel_fine_t)
    t_stitch = time.time() - t0

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    blade, side, mission_uuid = section_name.split('-', 2)
    section_dir = os.path.join(output_dir, blade, side, mission_uuid)
    os.makedirs(section_dir, exist_ok=True)

    coarse_path = os.path.join(section_dir, 'panorama_coarse.jpg')
    fine_path = os.path.join(section_dir, 'panorama_fine.jpg')
    cv2.imwrite(coarse_path, panorama_coarse)
    cv2.imwrite(fine_path, panorama_fine)

    # Save image positions for both panoramas
    def compute_cumulative_positions(transforms, images_list, photo_ids_list):
        """Compute cumulative position, scale, rotation for each image."""
        cum = [{'tx': 0.0, 'ty': 0.0, 'scale': 1.0, 'rotation': 0.0}]
        for t in transforms:
            M_prev = _to_matrix(cum[-1])
            M_t = _to_matrix(t)
            M_cum = M_prev @ M_t
            cum.append(_from_matrix(M_cum))
        positions = []
        for i, (pos, pid) in enumerate(zip(cum, photo_ids_list)):
            h, w = images_list[i].shape[:2]
            positions.append({
                'photo_id': pid,
                'tx': float(pos['tx']),
                'ty': float(pos['ty']),
                'scale': float(pos['scale']),
                'rotation': float(pos.get('rotation', 0.0)),
                'width': w,
                'height': h,
            })
        return positions

    coarse_positions = compute_cumulative_positions(sel_coarse_t, images_sel, photo_ids_sel)
    fine_positions = compute_cumulative_positions(sel_fine_t, images_sel, photo_ids_sel)

    with open(os.path.join(section_dir, 'panorama_coarse.json'), 'w') as f:
        json.dump({'images': coarse_positions}, f, indent=2)
    with open(os.path.join(section_dir, 'panorama_fine.json'), 'w') as f:
        json.dump({'images': fine_positions}, f, indent=2)

    t_section_total = time.time() - t_section_start

    n_total = len(images)
    n_match_skip = n_total - len(match_sel_idx)
    n_cut_skip = len(match_sel_idx) - len(images_sel)
    print(f"\n  Panorama coarse: {panorama_coarse.shape} -> {coarse_path}")
    print(f"  Panorama fine:   {panorama_fine.shape} -> {fine_path}")
    print(f"  Total: {n_total}, Match-skipped: {n_match_skip}, Cut-skipped: {n_cut_skip}, "
          f"Final: {len(images_sel)}, Fallback: {sum(sel_fallback)}/{len(sel_fallback)}")
    print(f"  Timing: load+SAM={t_load_seg:.1f}s, hires={t_hires:.1f}s, "
          f"match-skip={t_match_skip:.1f}s, cut-skip={t_cut_skip:.1f}s, "
          f"stitch={t_stitch:.1f}s, total={t_section_total:.1f}s")


def find_draft_dirs(data_dir):
    """Find all subdirectories containing metadata.json."""
    draft_dirs = []
    for entry in sorted(os.listdir(data_dir)):
        d = os.path.join(data_dir, entry)
        if os.path.isdir(d) and os.path.exists(os.path.join(d, 'metadata.json')):
            draft_dirs.append((entry, d))
    return draft_dirs


def main():
    parser = argparse.ArgumentParser(description='Blade stitching pipeline v2')
    parser.add_argument('--data-dir', default=str(REPO_DIR / 'data'), help='Data directory')
    parser.add_argument('--output-dir', default=None, help='Output directory (default: blade_stitching/output)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--diu-id', nargs='*', default=None, help='DIU ID(s) to process (default: all)')
    args = parser.parse_args()

    output_dir = args.output_dir or str(REPO_DIR / 'output')
    device = torch.device(args.device)
    print(f"Device: {device}")

    # Load models
    weights_dir = REPO_DIR / 'weights'
    load_sam(
        base_checkpoint=str(weights_dir / 'sam_vit_b_01ec64.pth'),
        finetune_checkpoint=str(weights_dir / 'best_model.pth'),
        device=device,
    )
    load_loftr(device=device)

    if args.diu_id:
        draft_dirs = [(d, os.path.join(args.data_dir, d)) for d in args.diu_id]
    else:
        draft_dirs = find_draft_dirs(args.data_dir)
    print(f"Found {len(draft_dirs)} draft IDs")

    for draft_id, draft_dir in draft_dirs:
        print(f"\n{'#'*60}")
        print(f"Draft ID: {draft_id}")
        print(f"{'#'*60}")

        draft_output_dir = os.path.join(output_dir, draft_id)

        try:
            photos_by_id, sections = load_draft_data(draft_dir)
            print(f"Sections: {list(sections.keys())}")

            for section_name, photo_ids in sections.items():
                try:
                    process_section(section_name, photo_ids, photos_by_id, draft_dir, draft_output_dir)
                except Exception as e:
                    print(f"Error processing {section_name}: {e}")
                    import traceback
                    traceback.print_exc()
        except Exception as e:
            print(f"Error loading {draft_id}: {e}")

    print("\nDone.")


if __name__ == '__main__':
    main()
