"""
Wind Turbine Blade Panorama Stitching Pipeline v3

Pipeline:
  1. Load original images, resize to working resolution (longer edge = 720)
  2. Brightness alignment + SAM segmentation (convex hull for PS/SS)
  3. LoFTR keypoint matching at high resolution (2160x1440)
  4. DBSCAN cluster filtering on match vectors
  5. Grid-based spatial sampling (5x5 grid, 25 samples) for transform estimation
  6. Fine transforms (scale + translation) with LiDAR scale prior
     - PS/SS: conditional scale override
     - LE/TE: always use LiDAR scale
  7. Fallback: if fine/coarse projection ratio outside [0.2, 2.0], use coarse
  8. Match-skip: skip images with 0 matches or step < 0.1
  9. Cut-skip: remove redundant images whose cut is covered by next image
  10. Stitch panorama (affine warp)
"""

import os
import sys
import json
import argparse
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

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

from modules.segmentation import load_sam, free_sam, segment_image, segment_images_batch
from modules.matching import load_loftr, match_loftr, filter_by_mask
from modules.brightness import align_brightness
from modules.coarse import CoarseStitcher, clamp_lidar_distances
from modules.stitching import stitch_trans_scale

LOFTR_W, LOFTR_H = 2160, 1440
WORK_LONG_EDGE = 720
PROJ_MIN, PROJ_MAX = 0.2, 2.0
STEP_MIN = 0.1
GRID_N = 5
GRID_SAMPLES = 25


# ---------------------------------------------------------------------------
# Section context — bundles all per-section data
# ---------------------------------------------------------------------------

@dataclass
class SectionCtx:
    """All data for processing one section, avoiding long parameter lists."""
    section_name: str
    section_side: str  # PS, SS, LE, TE
    photo_ids: List[int]
    photos_by_id: Dict
    distances: List[float]
    images: List[np.ndarray]           # working resolution (long edge = 720)
    masks: List[np.ndarray]            # binary masks at working resolution
    hires_seg: List[np.ndarray] = field(default_factory=list)
    masks_hires: List[np.ndarray] = field(default_factory=list)
    thumb_w: int = 0
    thumb_h: int = 0
    scale_xy: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0]))
    stitcher: Optional[CoarseStitcher] = None

    def coarse_transform(self, idx_i: int, idx_j: int) -> dict:
        """Get coarse transform from image idx_i to idx_j using cached positions."""
        pos_i = self._positions[self.photo_ids[idx_i]]
        pos_j = self._positions[self.photo_ids[idx_j]]
        tx = pos_j[0] - pos_i[0]
        ty = pos_j[1] - pos_i[1]
        scale = self.distances[idx_j] / self.distances[idx_i]
        return {'tx': tx, 'ty': ty, 'scale': scale}

    def coarse_transforms_for(self, indices: List[int]) -> List[dict]:
        """Get consecutive coarse transforms for a list of image indices."""
        return [self.coarse_transform(indices[i], indices[i + 1])
                for i in range(len(indices) - 1)]

    def init_coarse(self):
        """Build the CoarseStitcher and cache all positions. Call once after construction."""
        self.stitcher = CoarseStitcher(
            self.photos_by_id, self.photo_ids,
            img_width=self.thumb_w, img_height=self.thumb_h,
        )
        self._positions = self.stitcher.compute_all_positions()


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


def compute_cumulative_positions(transforms, images_list, photo_ids_list):
    """Compute cumulative position, scale, rotation for each image."""
    cum = [{'tx': 0.0, 'ty': 0.0, 'scale': 1.0, 'rotation': 0.0}]
    for t in transforms:
        M_cum = _to_matrix(cum[-1]) @ _to_matrix(t)
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


def compute_fine_transform(pts1, pts2, scale_prior, section_side):
    """Compute fine transform: grid-sample, decide scale, compute translation.

    Returns transform dict or None.
    """
    if len(pts1) < 1:
        return None

    sample_idx = grid_sample(pts1, n_samples=GRID_SAMPLES, grid_n=GRID_N)
    pts1_s = pts1[sample_idx]
    pts2_s = pts2[sample_idx]

    est_scale = estimate_scale(pts1_s, pts2_s)
    if est_scale is not None and section_side in ('PS', 'SS'):
        ratio = est_scale / scale_prior
        chosen_scale = est_scale if (0.9 < ratio < 0.96) or (1.06 < ratio < 1.10) else scale_prior
    else:
        chosen_scale = scale_prior

    trans = np.median(pts1_s - pts2_s * chosen_scale, axis=0)
    return {'tx': trans[0], 'ty': trans[1], 'scale': chosen_scale, 'rotation': 0.0}


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------

def dbscan_filter_matches(pts1_thumb, pts2_thumb, coarse_t, thumb_w, thumb_h):
    """Filter keypoint matches using DBSCAN clustering on match vectors.
    Selects the largest cluster."""
    tx, ty, sc = coarse_t['tx'], coarse_t['ty'], coarse_t['scale']
    new_w2, new_h2 = int(thumb_w * sc), int(thumb_h * sc)

    min_x, min_y = min(0, tx), min(0, ty)
    x1_off, y1_off = int(-min_x), int(-min_y)
    x2_off, y2_off = int(tx - min_x), int(ty - min_y)

    pts1_c = pts1_thumb + np.array([x1_off, y1_off])
    pts2_c = pts2_thumb * sc + np.array([x2_off, y2_off])
    match_vecs = pts2_c - pts1_c

    if len(match_vecs) >= 3:
        mags = np.linalg.norm(match_vecs, axis=1)
        eps = max(np.median(mags) * 0.15, 5.0)
        labels = DBSCAN(eps=eps, min_samples=4).fit(match_vecs).labels_
        cluster_ids = [l for l in set(labels) if l >= 0]
        if cluster_ids:
            best = max(cluster_ids, key=lambda l: (labels == l).sum())
            return labels == best
    return np.ones(len(pts1_thumb), dtype=bool)


def match_pair_loftr(img1_seg, img2_seg, mask1_hr, mask2_hr, scale_xy):
    """Run LoFTR matching between two high-res segmented images, return working-resolution points."""
    pts1, pts2, conf = match_loftr(img1_seg, img2_seg)
    pts1, pts2, conf = filter_by_mask(pts1, pts2, mask1_hr, mask2_hr, conf=conf)
    return pts1 * scale_xy, pts2 * scale_xy, conf


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


def resize_long_edge(img, long_edge=WORK_LONG_EDGE):
    """Resize image so that the longer edge matches long_edge, keeping aspect ratio."""
    h, w = img.shape[:2]
    scale = long_edge / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)))


def load_draft_data(draft_dir):
    """Load metadata and return photos_by_id and sections dict.

    Section keys: 'blade-side-missionUUID'.
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


def build_section_ctx(section_name, photo_ids, photos_by_id, draft_dir) -> Optional[SectionCtx]:
    """Load all data for one section, returning a SectionCtx or None if insufficient images."""
    section_side = section_name.split('-')[1]

    # Load originals once, produce both working and LoFTR resolutions
    images, hires_seg = [], []
    loaded_pids = []
    for pid in photo_ids:
        photo = photos_by_id[pid]
        img_path = os.path.join(draft_dir, photo['original_path'])
        img_orig = cv2.imread(img_path)
        if img_orig is None:
            continue
        images.append(resize_long_edge(img_orig))
        hires_seg.append(cv2.resize(img_orig, (LOFTR_W, LOFTR_H)))
        loaded_pids.append(pid)

    if len(images) < 2:
        return None

    distances = clamp_lidar_distances(photos_by_id, loaded_pids)
    print(f"Loaded {len(images)} images")

    # Brightness alignment
    images = align_brightness(images)

    # SAM segmentation (batched encoder)
    use_convex = section_side in ('PS', 'SS')
    raw_masks = segment_images_batch([cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images])
    masks = [make_convex_mask(m) if use_convex else m.astype(np.uint8) for m in raw_masks]
    print(f"Generated {len(masks)} masks (convex={use_convex}, side={section_side})")
    free_sam()

    # Apply masks to hires images
    masks_hires = [cv2.resize(m.astype(np.uint8), (LOFTR_W, LOFTR_H)) for m in masks]
    for img_hr, mask_hr in zip(hires_seg, masks_hires):
        img_hr[mask_hr == 0] = 0

    thumb_w, thumb_h = images[0].shape[1], images[0].shape[0]

    ctx = SectionCtx(
        section_name=section_name,
        section_side=section_side,
        photo_ids=loaded_pids,
        photos_by_id=photos_by_id,
        distances=distances,
        images=images,
        masks=masks,
        hires_seg=hires_seg,
        masks_hires=masks_hires,
        thumb_w=thumb_w,
        thumb_h=thumb_h,
        scale_xy=np.array([thumb_w / LOFTR_W, thumb_h / LOFTR_H]),
    )
    ctx.init_coarse()
    return ctx


# ---------------------------------------------------------------------------
# Match and fine transform for a pair
# ---------------------------------------------------------------------------

def _match_and_fine(ctx: SectionCtx, idx_i: int, idx_j: int, compute_proj_fn):
    """Match pair (idx_i, idx_j), filter, compute fine transform + fallback.

    Returns (fine_t, is_fallback, n_matches).
    """
    pts1, pts2, _ = match_pair_loftr(
        ctx.hires_seg[idx_i], ctx.hires_seg[idx_j],
        ctx.masks_hires[idx_i], ctx.masks_hires[idx_j], ctx.scale_xy,
    )
    coarse_t = ctx.coarse_transform(idx_i, idx_j)
    inlier = dbscan_filter_matches(pts1, pts2, coarse_t, ctx.thumb_w, ctx.thumb_h)
    pts1_f, pts2_f = pts1[inlier], pts2[inlier]
    n_matches = len(pts1_f)

    scale = ctx.distances[idx_j] / ctx.distances[idx_i]
    fine_t = compute_fine_transform(pts1_f, pts2_f, scale, ctx.section_side)

    if fine_t is None:
        return None, True, 0

    # Projection fallback check
    coarse_vec = np.array([coarse_t['tx'], coarse_t['ty']])
    fine_vec = np.array([fine_t['tx'], fine_t['ty']])
    proj = compute_proj_fn(coarse_vec, fine_vec)

    if proj < PROJ_MIN or proj > PROJ_MAX:
        coarse_t['rotation'] = 0.0
        return coarse_t, True, n_matches

    return fine_t, False, n_matches


# ---------------------------------------------------------------------------
# Match-skip: skip images with small step or 0 matches
# ---------------------------------------------------------------------------

def match_skip(ctx: SectionCtx):
    """Walk through images, skipping those too close or with 0 matches.

    Returns (selected_indices, fine_transforms, fallback_flags, global_axis_unit, compute_proj_fn).
    The match-skip loop logic is identical to v2:
      - For each anchor i, try j = i+1, i+2, ... until a valid pair is found.
      - Skip j if coarse step < STEP_MIN (unless j is the last image).
      - Skip j if 0 LoFTR matches (unless j is the last image).
      - If no valid j found, force coarse fallback for (i, i+1).
    """
    n = len(ctx.images)

    # Global stitching axis from consecutive coarse transforms
    all_coarse = ctx.coarse_transforms_for(list(range(n)))
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

    # Diagonal slice length for step normalization
    ux, uy = global_axis_unit
    h_img, w_img = ctx.images[0].shape[:2]
    diag_slice = min(w_img / abs(ux) if abs(ux) > 1e-6 else float('inf'),
                     h_img / abs(uy) if abs(uy) > 1e-6 else float('inf'))

    print(f"Match-skip (0 matches or step < {STEP_MIN})...")
    torch.cuda.empty_cache()

    selected_idx = [0]
    fine_transforms = []
    fallback_flags = []

    i = 0
    while i < n - 1:
        matched = False
        for j in range(i + 1, n):
            # Step check using cached coarse
            coarse_t_ij = ctx.coarse_transform(i, j)
            step = np.dot(np.array([coarse_t_ij['tx'], coarse_t_ij['ty']]), global_axis_unit) / diag_slice
            if abs(step) < STEP_MIN and j < n - 1:
                print(f"  Skip image {j}: step={step:.3f} < {STEP_MIN}")
                continue

            ft, fb, n_matches = _match_and_fine(ctx, i, j, compute_proj)
            if n_matches == 0 and j < n - 1:
                print(f"  Skip image {j}: 0 matches with image {i}")
                continue

            if j > i + 1:
                print(f"  Pair ({i},{j}): skipped {list(range(i + 1, j))}, {n_matches} matches")
            else:
                label = "FALLBACK" if fb else f"FINE d={np.linalg.norm([ft['tx'], ft['ty']]):.1f}"
                print(f"  Pair {i}: {n_matches} matches, {label}")

            selected_idx.append(j)
            if ft is None:
                coarse_t_ij['rotation'] = 0.0
                fine_transforms.append(coarse_t_ij)
                fallback_flags.append(True)
            else:
                fine_transforms.append(ft)
                fallback_flags.append(fb)
            i = j
            matched = True
            break

        if not matched:
            print(f"  Pair {i}: no valid match found, fallback for ({i},{i+1})")
            coarse_fb = ctx.coarse_transform(i, i + 1)
            coarse_fb['rotation'] = 0.0
            selected_idx.append(i + 1)
            fine_transforms.append(coarse_fb)
            fallback_flags.append(True)
            i = i + 1

    match_skipped = sorted(set(range(n)) - set(selected_idx))
    if match_skipped:
        print(f"Match-skipped {len(match_skipped)} images: {match_skipped}")
    print(f"After match-skip: {len(selected_idx)}/{n} images")

    return selected_idx, fine_transforms, fallback_flags, global_axis_unit, compute_proj


# ---------------------------------------------------------------------------
# Cut-skip: remove redundant images whose cut is covered by next image
# ---------------------------------------------------------------------------

def cut_skip(ctx: SectionCtx, fine_transforms, fallback_flags, compute_proj_fn):
    """Remove images whose disconnection-edge cut is fully covered by the next image.

    The cut-skip loop logic is identical to v2:
      - For triplet (base, last, j), check if j covers the cut of (base, last).
      - If covered: pop last, re-match (base, j), update transform.
      - If not covered: keep last, advance.
    ctx.images/masks/etc must already be remapped to match-skip-selected arrays.

    Returns (selected_indices, fine_transforms, fallback_flags).
    """
    n = len(ctx.images)
    print("Filtering redundant images...")

    selected_idx = [0, 1]
    sel_fine_t = [fine_transforms[0]]
    sel_fallback = [fallback_flags[0]]
    prev_cuts = get_pair_cuts(ctx.masks[0], ctx.masks[1], fine_transforms[0])
    print(f"  KEEP image 0, 1: {len(prev_cuts)} cut(s)")

    for j in range(2, n):
        base = selected_idx[-2]
        last = selected_idx[-1]

        t_last_j = fine_transforms[last]
        t_base_j = chain_transforms(sel_fine_t[-1], t_last_j)

        covered = prev_cuts and check_cut_covered(
            prev_cuts, t_base_j, ctx.masks[j].shape,
        )

        if covered:
            print(f"  SKIP image {last}: image {j} covers pair ({base},{last}) cut")
            selected_idx.pop()
            sel_fine_t.pop()
            sel_fallback.pop()
            selected_idx.append(j)

            # Re-match the new pair (base, j)
            ft, fb, n_matches = _match_and_fine(ctx, base, j, compute_proj_fn)
            if ft is None:
                ft = ctx.coarse_transform(base, j)
                ft['rotation'] = 0.0
                fb = True
            label = "fallback" if fb else f"FINE d={np.linalg.norm([ft['tx'], ft['ty']]):.1f}"
            print(f"    Recomputed ({base},{j}): {n_matches} matches, {label}")
            sel_fine_t.append(ft)
            sel_fallback.append(fb)
            prev_cuts = get_pair_cuts(ctx.masks[base], ctx.masks[j], ft)
        else:
            selected_idx.append(j)
            sel_fine_t.append(t_last_j)
            sel_fallback.append(fallback_flags[last])
            prev_cuts = get_pair_cuts(ctx.masks[last], ctx.masks[j], t_last_j)
            print(f"  KEEP image {j}: {len(prev_cuts)} cut(s)")

    skipped = sorted(set(range(n)) - set(selected_idx))
    print(f"Result: {len(selected_idx)}/{n} kept, skipped: {skipped}")

    return selected_idx, sel_fine_t, sel_fallback


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def is_disconnected(positions):
    """Check if the stitched photos form more than one connected component.

    Two photos are connected if their bounding boxes overlap. Connectivity
    is transitive (union-find). Returns True if there are 2+ components.
    """
    n = len(positions)
    if n <= 1:
        return False

    # Compute bounding boxes: (x1, y1, x2, y2) for each photo
    bboxes = []
    for p in positions:
        x1 = p['tx']
        y1 = p['ty']
        x2 = x1 + p['width'] * p['scale']
        y2 = y1 + p['height'] * p['scale']
        bboxes.append((x1, y1, x2, y2))

    # Union-Find
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[a] = b

    for i in range(n):
        ax1, ay1, ax2, ay2 = bboxes[i]
        for j in range(i + 1, n):
            bx1, by1, bx2, by2 = bboxes[j]
            if ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1:
                union(i, j)

    num_components = len(set(find(i) for i in range(n)))
    return num_components > 1


def process_section(section_name, photo_ids, photos_by_id, draft_dir, output_dir):
    """Run the full pipeline for one section, check for disconnected photos."""
    torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"Processing: {section_name}")
    print(f"{'='*60}")

    if len(photo_ids) <= 10:
        print(f"  Disconnected: True (only {len(photo_ids)} images)")
        return

    ctx = build_section_ctx(section_name, photo_ids, photos_by_id, draft_dir)
    if ctx is None:
        print(f"Skipping {section_name}: need at least 2 images")
        return

    # Load LoFTR after SAM is freed
    load_loftr()

    # Phase 1: Match-skip
    match_sel_idx, fine_transforms, fallback_flags, global_axis_unit, compute_proj_fn = match_skip(ctx)

    # Remap ctx to match-skip-selected images
    ms_ctx = SectionCtx(
        section_name=ctx.section_name,
        section_side=ctx.section_side,
        photo_ids=[ctx.photo_ids[i] for i in match_sel_idx],
        photos_by_id=ctx.photos_by_id,
        distances=[ctx.distances[i] for i in match_sel_idx],
        images=[ctx.images[i] for i in match_sel_idx],
        masks=[ctx.masks[i] for i in match_sel_idx],
        hires_seg=[ctx.hires_seg[i] for i in match_sel_idx],
        masks_hires=[ctx.masks_hires[i] for i in match_sel_idx],
        thumb_w=ctx.thumb_w,
        thumb_h=ctx.thumb_h,
        scale_xy=ctx.scale_xy,
    )
    ms_ctx.init_coarse()

    # Phase 2: Cut-skip
    cut_sel_idx, sel_fine_t, sel_fallback = cut_skip(
        ms_ctx, fine_transforms, fallback_flags, compute_proj_fn,
    )

    # Build final data
    images_sel = [ms_ctx.images[i] for i in cut_sel_idx]
    photo_ids_sel = [ms_ctx.photo_ids[i] for i in cut_sel_idx]

    # Check disconnection using fine transforms
    fine_positions = compute_cumulative_positions(sel_fine_t, images_sel, photo_ids_sel)
    disconnected = is_disconnected(fine_positions)
    print(f"  Disconnected: {disconnected}")

    # Save fine panorama
    panorama_fine = stitch_trans_scale(images_sel, sel_fine_t)
    blade, side, mission_uuid = section_name.split('-', 2)
    section_dir = os.path.join(output_dir, blade, side, mission_uuid)
    os.makedirs(section_dir, exist_ok=True)
    fine_path = os.path.join(section_dir, 'panorama_fine.jpg')
    cv2.imwrite(fine_path, panorama_fine)
    print(f"  Panorama: {panorama_fine.shape} -> {fine_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Check disconnected photos in stitching',
        usage='%(prog)s --diu-id ID --section BLADE/SIDE [options]',
    )
    parser.add_argument('--data-dir', default=str(REPO_DIR / 'data'), help='Data directory')
    parser.add_argument('--output-dir', '-o', default=None, help='Output directory (default: blade_stitching/output_disconnect)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--diu-id', type=str, required=True, help='DIU ID to process')
    parser.add_argument('--section', type=str, required=True, help='Section to process, e.g. A/LE')
    args = parser.parse_args()

    output_dir = args.output_dir or str(REPO_DIR / 'output_disconnect')
    draft_dir = os.path.join(args.data_dir, args.diu_id)
    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"DIU: {args.diu_id}, Section: {args.section}")

    photos_by_id, sections = load_draft_data(draft_dir)

    blade_f, side_f = args.section.split('/')
    sections = {k: v for k, v in sections.items()
                if k.startswith(f"{blade_f}-{side_f}-")}
    if not sections:
        print(f"No sections matching {args.section}")
        return

    # Load SAM first, LoFTR loaded later (after SAM is freed)
    weights_dir = REPO_DIR / 'weights'
    load_sam(
        finetune_checkpoint=str(weights_dir / 'best_model.pth'),
        device=device,
    )

    for section_name, photo_ids in sections.items():
        draft_output_dir = os.path.join(output_dir, args.diu_id)
        process_section(section_name, photo_ids, photos_by_id, draft_dir, draft_output_dir)

    print("\nDone.")


if __name__ == '__main__':
    main()
