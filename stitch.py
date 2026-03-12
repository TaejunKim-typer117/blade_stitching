#!/usr/bin/env python3
"""
Wind turbine blade panorama stitching.

Usage:
    python stitch.py --data-dir /path/to/dataset
    python stitch.py --data-dir /path/to/dataset --output-dir /path/to/output

data-dir should contain draft ID folders, each with metadata.json and images:
    dataset/
    ├── 40012/
    │   ├── metadata.json
    │   └── A/LE/photo_*.jpg
    ├── 40013/
    │   └── ...
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from modules import (
    load_sam, segment_image,
    align_brightness,
    load_loftr, match_loftr, filter_by_mask, ransac_filter,
    compute_coarse_transforms, compute_transforms,
    compute_edge_aligned_transforms,
    stitch_trans_scale,
)

REPO_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = REPO_DIR / 'weights'
DEFAULT_OUTPUT_DIR = REPO_DIR / 'output'


def load_models(device):
    load_sam(
        base_checkpoint=str(WEIGHTS_DIR / 'sam_vit_b_01ec64.pth'),
        finetune_checkpoint=str(WEIGHTS_DIR / 'best_model.pth'),
        device=device,
    )
    load_loftr(device=device)


def load_data(draft_dir):
    metadata_path = os.path.join(draft_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    photos_by_id = {p['id']: p for p in metadata['photos']}

    sections = defaultdict(list)
    for p in metadata['photos']:
        key = f"{p['blade_tag']}-{p['blade_side_tag']}"
        sections[key].append(p['id'])

    return metadata, photos_by_id, dict(sections)


def process_section(section_name, photo_ids, photos_by_id, draft_dir, output_dir):
    print(f"\n{'='*60}")
    print(f"Processing: {section_name}")
    print('='*60)

    # Load images and distances
    images = []
    distances = []
    for pid in photo_ids:
        photo = photos_by_id[pid]
        local_path = photo['local_path']
        img_path = local_path if os.path.isabs(local_path) else os.path.join(draft_dir, local_path)
        if not os.path.exists(img_path):
            img_path = os.path.join(draft_dir, photo['blade_tag'], photo['blade_side_tag'], os.path.basename(local_path))
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
            distances.append(photo['metadata']['measured_distance_to_blade'])
    print(f"Loaded {len(images)} images")

    if len(images) < 2:
        print(f"Skipping {section_name}: need at least 2 images")
        return

    # Brightness alignment
    images = align_brightness(images)
    print("Brightness aligned")

    # Generate masks
    masks = [segment_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in images]
    print(f"Generated {len(masks)} masks")

    # Create segmented images for matching
    images_segmented = [img.copy() for img in images]
    for img, mask in zip(images_segmented, masks):
        img[mask == 0] = 0

    # Match keypoints with LoFTR
    match_results = []
    for i in range(len(images_segmented) - 1):
        pts1, pts2 = match_loftr(images_segmented[i], images_segmented[i+1])
        pts1, pts2 = filter_by_mask(pts1, pts2, masks[i], masks[i+1])
        match_results.append({'pts1': pts1, 'pts2': pts2})
        print(f"Pair {i}: {len(pts1)} matches")

    # Apply RANSAC outlier filtering
    filtered_results = []
    for r in match_results:
        pts1, pts2 = r['pts1'], r['pts2']
        mask = ransac_filter(pts1, pts2) if len(pts1) >= 4 else np.ones(len(pts1), dtype=bool)
        filtered_results.append({'pts1': pts1[mask], 'pts2': pts2[mask]})
        print(f"After RANSAC: {mask.sum()}/{len(pts1)} inliers")

    # Compute coarse transforms using DCM
    coarse_transforms = compute_coarse_transforms(photos_by_id, photo_ids)

    # Compute fallback transforms, then apply edge alignment
    print("\nComputing fallback transforms...")
    fallback_transforms, _ = compute_transforms(filtered_results, coarse_transforms, distances, masks=masks, mode='fallback')
    print("Applying edge alignment (two-step with IoU validation)...")
    transforms, _ = compute_edge_aligned_transforms(images, masks, fallback_transforms)

    # Stitch and save panorama
    panorama = stitch_trans_scale(images, transforms)
    print(f"Panorama shape: {panorama.shape}")

    blade, side = section_name.split('-')
    output_folder = os.path.join(output_dir, blade, side)
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, 'panorama.jpg')
    cv2.imwrite(output_path, panorama, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Saved: {output_path}")


def find_draft_dirs(data_dir):
    """Find all subdirectories containing metadata.json."""
    draft_dirs = []
    for entry in sorted(os.listdir(data_dir)):
        d = os.path.join(data_dir, entry)
        if os.path.isdir(d) and os.path.exists(os.path.join(d, 'metadata.json')):
            draft_dirs.append((entry, d))
    return draft_dirs


def main():
    parser = argparse.ArgumentParser(description='Wind turbine blade panorama stitching')
    parser.add_argument('--data-dir', default=str(REPO_DIR / 'data'), help='Directory containing draft ID folders with metadata.json and images')
    parser.add_argument('--output-dir', default=None, help='Output directory (default: blade_stitching/output)')
    args = parser.parse_args()

    output_dir = args.output_dir or str(DEFAULT_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    load_models(device)

    draft_dirs = find_draft_dirs(args.data_dir)
    print(f"Found {len(draft_dirs)} draft IDs")

    for draft_id, draft_dir in draft_dirs:
        print(f"\n{'#'*60}")
        print(f"Draft ID: {draft_id}")
        print(f"{'#'*60}")

        draft_output_dir = os.path.join(output_dir, draft_id)
        os.makedirs(draft_output_dir, exist_ok=True)

        try:
            metadata, photos_by_id, sections = load_data(draft_dir)
            print(f"Sections: {list(sections.keys())}")

            for section_name, photo_ids in sections.items():
                process_section(section_name, photo_ids, photos_by_id, draft_dir, draft_output_dir)
        except Exception as e:
            print(f"Error processing {draft_id}: {e}")
            continue

    print("\nDone.")


if __name__ == '__main__':
    main()
