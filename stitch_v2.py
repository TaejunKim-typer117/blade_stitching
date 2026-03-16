"""
Wind Turbine Blade Panorama Stitching Pipeline v2

Pipeline:
  1. Load metadata, images, masks
  2. Brightness alignment + SAM segmentation
  3. LoFTR keypoint matching at high resolution
  4. DBSCAN cluster filtering on match vectors
  5. Coarse transforms (DCM) + fine transforms (keypoint median)
  6. Fallback: if proj outside [0.5, 2.0], use coarse
  7. Cut-skip: remove redundant images whose cut is covered by next image
  8. Stitch panorama
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
import torch
from pathlib import Path
from collections import defaultdict
from sklearn.cluster import DBSCAN


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR))

from modules.segmentation import load_sam, segment_image
from modules.matching import load_loftr, match_loftr, filter_by_mask
from modules.brightness import align_brightness
from modules.coarse import compute_coarse_transforms, compute_fine_transform
from modules.stitching import stitch_trans_scale

LOFTR_W, LOFTR_H = 2160, 1440
PROJ_MIN, PROJ_MAX = 0.5, 2.0


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

    overlap = (ox1, oy1, ox2, oy2)
    edges = []
    tol = 1e-6

    if abs(ox1 - bx1) < tol:
        edges.append({'side': 'left', 'segment': (ox1, oy1, ox1, oy2)})
    if abs(ox2 - bx2) < tol:
        edges.append({'side': 'right', 'segment': (ox2, oy1, ox2, oy2)})
    if abs(oy1 - by1) < tol:
        edges.append({'side': 'top', 'segment': (ox1, oy1, ox2, oy1)})
    if abs(oy2 - by2) < tol:
        edges.append({'side': 'bottom', 'segment': (ox1, oy2, ox2, oy2)})

    return overlap, edges


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


def chain_transforms(t1, t2):
    """Chain t1 (base->mid) and t2 (mid->end) into base->end."""
    return {
        'tx': t1['tx'] + t2['tx'] * t1['scale'],
        'ty': t1['ty'] + t2['ty'] * t1['scale'],
        'scale': t1['scale'] * t2['scale'],
    }


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
# Matching helpers
# ---------------------------------------------------------------------------

def dbscan_filter_matches(pts1_thumb, pts2_thumb, coarse_t, thumb_w, thumb_h):
    """Filter keypoint matches using DBSCAN clustering on match vectors.

    Selects the cluster whose average projected magnitude along the
    center-crossing axis is closest to zero (smallest residual motion).
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
        clustering = DBSCAN(eps=eps, min_samples=3).fit(match_vecs)
        labels = clustering.labels_

        proj1 = (pts1_c - c1) @ axis_unit
        proj2 = (pts2_c - c1) @ axis_unit
        proj_mags = proj2 - proj1

        cluster_ids = [l for l in set(labels) if l >= 0]
        if cluster_ids:
            best = min(cluster_ids, key=lambda l: abs(np.mean(proj_mags[labels == l])))
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


def load_images_and_masks(draft_dir, photos_by_id, photo_ids):
    """Load images, align brightness, segment masks."""
    images, distances = [], []
    for pid in photo_ids:
        photo = photos_by_id[pid]
        img_path = os.path.join(draft_dir, photo['local_path'])
        if not os.path.exists(img_path):
            img_path = os.path.join(
                draft_dir, photo['blade_tag'], photo['blade_side_tag'],
                os.path.basename(photo['local_path']),
            )
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
            distances.append(photo['metadata']['measured_distance_to_blade'])
    print(f"Loaded {len(images)} images")

    images = align_brightness(images)
    masks = [segment_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in images]
    print(f"Generated {len(masks)} masks")

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
                    photos_by_id, photo_ids, distances, thumb_w, thumb_h, compute_proj_fn):
    """Match pair (idx_i, idx_j), filter, compute fine transform + fallback.

    Returns (fine_t, fine_t_original, is_fallback, n_matches).
    n_matches=0 means LoFTR found nothing after filtering.
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
    fine_t = compute_fine_transform(pts1_f, pts2_f, scale)

    if fine_t is None:
        return coarse_t, coarse_t, True, 0

    coarse_vec = np.array([coarse_t['tx'], coarse_t['ty']])
    fine_vec = np.array([fine_t['tx'], fine_t['ty']])
    proj = compute_proj_fn(coarse_vec, fine_vec)

    if proj < PROJ_MIN or proj > PROJ_MAX:
        return coarse_t, fine_t, True, n_matches

    return fine_t, fine_t, False, n_matches


def compute_all_transforms(
    images, masks, distances, photo_ids, photos_by_id,
    images_hires_seg, masks_hires, scale_xy, thumb_w, thumb_h,
):
    """Compute coarse and fine transforms, skipping images with 0 keypoint matches.

    If LoFTR returns 0 matches between image i and image i+1, image i+1 is
    removed and matching is retried with image i+2, i+3, etc.

    Returns:
        selected_idx: indices of images that survived match-skip
        coarse_transforms: coarse transforms for selected pairs
        fine_transforms: fine transforms (fallback applied) for selected pairs
        fine_transforms_original: original fine transforms for selected pairs
        fallback_flags: list of bools for selected pairs
        global_axis_unit: unit vector of global stitching direction
        compute_proj_fn: function(coarse_vec, fine_vec) -> proj value
    """
    # Global stitching axis from all consecutive coarse transforms
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

    # Build selected image list, skipping images with 0 matches
    n = len(images)
    selected_idx = [0]
    fine_transforms, fine_original, fallback_flags = [], [], []

    i = 0
    while i < n - 1:
        # Try matching image i with i+1, i+2, ... until we find matches
        matched = False
        for j in range(i + 1, n):
            ft, ft_orig, fb, n_matches = _match_and_fine(
                i, j, images_hires_seg, masks_hires, scale_xy,
                photos_by_id, photo_ids, distances, thumb_w, thumb_h, compute_proj,
            )
            if n_matches > 0:
                if j > i + 1:
                    skipped = list(range(i + 1, j))
                    print(f"  Pair ({i},{j}): {n_matches} matches "
                          f"(skipped images {skipped} due to 0 matches)")
                else:
                    d_f = np.linalg.norm([ft['tx'], ft['ty']])
                    proj = compute_proj(
                        np.array([ft_orig['tx'], ft_orig['ty']]) if not fb else np.array([ft['tx'], ft['ty']]),
                        np.array([ft_orig['tx'], ft_orig['ty']]),
                    )
                    label = "FALLBACK" if fb else f"FINE d={d_f:.1f}, proj={proj:.2f}"
                    print(f"  Pair {i}: {n_matches} matches, {label}")
                selected_idx.append(j)
                fine_transforms.append(ft)
                fine_original.append(ft_orig)
                fallback_flags.append(fb)
                i = j
                matched = True
                break

        if not matched:
            # No matches found with any subsequent image — keep i+1 with coarse fallback
            print(f"  Pair {i}: no matches with any image -> fallback for ({i},{i+1})")
            coarse_t = compute_coarse_transforms(
                photos_by_id, [photo_ids[i], photo_ids[i + 1]],
                img_width=thumb_w, img_height=thumb_h,
            )[0]
            selected_idx.append(i + 1)
            fine_transforms.append(coarse_t)
            fine_original.append(coarse_t)
            fallback_flags.append(True)
            i = i + 1

    match_skipped = sorted(set(range(n)) - set(selected_idx))
    if match_skipped:
        print(f"Match-skip: removed {len(match_skipped)} images: {match_skipped}")
    print(f"After match-skip: {len(selected_idx)}/{n} images")

    # Recompute coarse transforms for selected images
    selected_photo_ids = [photo_ids[k] for k in selected_idx]
    coarse_transforms = compute_coarse_transforms(
        photos_by_id, selected_photo_ids, img_width=thumb_w, img_height=thumb_h,
    )

    return (selected_idx, coarse_transforms, fine_transforms, fine_original,
            fallback_flags, global_axis_unit, compute_proj)


def match_and_compute_fine_pair(
    idx_i, idx_j,
    images_hires_seg, masks_hires, photos_by_id, photo_ids,
    distances, scale_xy, thumb_w, thumb_h, compute_proj_fn,
):
    """Recompute LoFTR matching + fine transform for an arbitrary (non-consecutive) pair.

    Used when an intermediate image is skipped and we need a fresh transform.
    Returns (fine_t, fine_t_original, is_fallback).
    """
    ft, ft_orig, fb, n_matches = _match_and_fine(
        idx_i, idx_j, images_hires_seg, masks_hires, scale_xy,
        photos_by_id, photo_ids, distances, thumb_w, thumb_h, compute_proj_fn,
    )
    d_f = np.linalg.norm([ft['tx'], ft['ty']])
    label = "fallback" if fb else f"FINE d={d_f:.1f}"
    print(f"    Recomputed ({idx_i},{idx_j}): {n_matches} matches, {label}")
    return ft, ft_orig, fb


def filter_redundant_images(
    images, masks, distances, photo_ids, photos_by_id,
    fine_transforms, fine_original, fallback_flags,
    images_hires_seg, masks_hires, scale_xy, thumb_w, thumb_h,
    compute_proj_fn,
):
    """Remove images whose cut is fully covered by the next image.

    Returns selected indices and their transforms.
    """
    print("Filtering redundant images...")

    selected_idx = [0, 1]
    sel_fine_t = [fine_transforms[0]]
    sel_fine_t_orig = [fine_original[0]]
    sel_fallback = [fallback_flags[0]]
    prev_cuts = get_pair_cuts(masks[0], masks[1], fine_transforms[0])
    print(f"  KEEP image 0, 1: {len(prev_cuts)} cut(s)")

    for j in range(2, len(images)):
        base = selected_idx[-2]
        last = selected_idx[-1]

        # Chain for coverage check only
        t_last_j = fine_transforms[last]  # pair (last, last+1) = (j-1, j)
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
            selected_idx.append(j)

            new_ft, new_ft_orig, new_fb = match_and_compute_fine_pair(
                base, j,
                images_hires_seg, masks_hires, photos_by_id, photo_ids,
                distances, scale_xy, thumb_w, thumb_h, compute_proj_fn,
            )
            sel_fine_t.append(new_ft)
            sel_fine_t_orig.append(new_ft_orig)
            sel_fallback.append(new_fb)
            prev_cuts = get_pair_cuts(masks[base], masks[j], new_ft)
        else:
            selected_idx.append(j)
            sel_fine_t.append(t_last_j)
            sel_fine_t_orig.append(fine_original[last])
            sel_fallback.append(fallback_flags[last])
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
    # Free GPU memory from previous section
    torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"Processing: {section_name}")
    print(f"{'='*60}")

    images, masks, distances = load_images_and_masks(draft_dir, photos_by_id, photo_ids)
    if len(images) < 2:
        print(f"Skipping {section_name}: need at least 2 images")
        return

    thumb_w, thumb_h = images[0].shape[1], images[0].shape[0]
    scale_xy = np.array([thumb_w / LOFTR_W, thumb_h / LOFTR_H])

    images_hires_seg, masks_hires = load_hires_segmented(
        draft_dir, photos_by_id, photo_ids, masks,
    )

    # Compute transforms (with match-skip: removes images with 0 keypoint matches)
    (match_sel_idx, coarse_transforms, fine_transforms, fine_original, fallback_flags,
     global_axis_unit, compute_proj_fn) = compute_all_transforms(
        images, masks, distances, photo_ids, photos_by_id,
        images_hires_seg, masks_hires, scale_xy, thumb_w, thumb_h,
    )

    # Remap to match-skip-selected arrays
    ms_images = [images[i] for i in match_sel_idx]
    ms_masks = [masks[i] for i in match_sel_idx]
    ms_distances = [distances[i] for i in match_sel_idx]
    ms_photo_ids = [photo_ids[i] for i in match_sel_idx]
    ms_hires_seg = [images_hires_seg[i] for i in match_sel_idx]
    ms_masks_hires = [masks_hires[i] for i in match_sel_idx]

    # Cut-skip filter (operates on match-skip-selected images)
    cut_sel_idx, sel_fine_t, sel_fine_t_orig, sel_fallback = filter_redundant_images(
        ms_images, ms_masks, ms_distances, ms_photo_ids, photos_by_id,
        fine_transforms, fine_original, fallback_flags,
        ms_hires_seg, ms_masks_hires, scale_xy, thumb_w, thumb_h,
        compute_proj_fn,
    )

    # Build final selected data
    images_sel = [ms_images[i] for i in cut_sel_idx]
    photo_ids_sel = [ms_photo_ids[i] for i in cut_sel_idx]

    # Recompute coarse for selected images
    sel_coarse_t = compute_coarse_transforms(
        photos_by_id, photo_ids_sel, img_width=thumb_w, img_height=thumb_h,
    )

    # Stitch panoramas
    panorama_coarse = stitch_trans_scale(images_sel, sel_coarse_t)
    panorama_fine = stitch_trans_scale(images_sel, sel_fine_t)

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    blade, side, mission_uuid = section_name.split('-', 2)
    section_dir = os.path.join(output_dir, blade, side, mission_uuid)
    os.makedirs(section_dir, exist_ok=True)

    coarse_path = os.path.join(section_dir, 'panorama_coarse.jpg')
    fine_path = os.path.join(section_dir, 'panorama_fine.jpg')
    cv2.imwrite(coarse_path, panorama_coarse)
    cv2.imwrite(fine_path, panorama_fine)

    n_total = len(images)
    n_match_skip = n_total - len(match_sel_idx)
    n_cut_skip = len(match_sel_idx) - len(images_sel)
    print(f"\n  Panorama coarse: {panorama_coarse.shape} -> {coarse_path}")
    print(f"  Panorama fine:   {panorama_fine.shape} -> {fine_path}")
    print(f"  Total: {n_total}, Match-skipped: {n_match_skip}, Cut-skipped: {n_cut_skip}, "
          f"Final: {len(images_sel)}, Fallback: {sum(sel_fallback)}/{len(sel_fallback)}")


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
