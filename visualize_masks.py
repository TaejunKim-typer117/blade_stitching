#!/usr/bin/env python3
"""
Visualize SAM segmentation masks for a specific blade section.

Loads thumbnail images, runs finetuned SAM with aspect-ratio-preserving
resize (longer edge → 1024, black padding to 1024x1024), and saves
mask-overlaid images.

Usage:
    python visualize_masks.py --diu-id 64923 --section A/LE
    python visualize_masks.py --diu-id 64923 --section A/LE --output-dir output/masks
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR))

import modules.segmentation as seg_module
from modules.segmentation import load_sam, postprocess_mask, MEAN, STD


def segment_image_ar(img_rgb, target_size=1024):
    """Segment with aspect-ratio-preserving resize + black padding."""
    h, w = img_rgb.shape[:2]

    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img_rgb, (new_w, new_h))

    pad_top = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2
    img_padded = np.zeros((target_size, target_size, 3), dtype=img_resized.dtype)
    img_padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = img_resized

    img_norm = (img_padded / 255.0 - MEAN) / STD
    img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0).to(seg_module._device)
    with torch.no_grad():
        pred_mask, _ = seg_module._sam_model(img_tensor)
        mask = torch.sigmoid(pred_mask).cpu().numpy()[0, 0]

    mask = mask[pad_top:pad_top + new_h, pad_left:pad_left + new_w]
    mask = cv2.resize(mask, (w, h))
    return postprocess_mask(mask)


def overlay_mask(img_bgr, mask, color=(0, 255, 0), alpha=0.4):
    """Overlay a binary mask on an image with transparency."""
    overlay = img_bgr.copy()
    overlay[mask > 0] = (
        (1 - alpha) * overlay[mask > 0] + alpha * np.array(color)
    ).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 2)
    return overlay


def main():
    parser = argparse.ArgumentParser(description='Visualize SAM masks for a blade section')
    parser.add_argument('--diu-id', type=str, required=True, help='DIU ID')
    parser.add_argument('--section', type=str, required=True, help='Section as blade/side, e.g. A/LE')
    parser.add_argument('--data-dir', type=str, default=str(REPO_DIR / 'data'))
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    blade, side = args.section.split('/')
    section_dir = Path(args.data_dir) / args.diu_id / 'thumbnail' / blade / side

    if not section_dir.exists():
        print(f"Section directory not found: {section_dir}")
        sys.exit(1)

    # Collect all images across mission UUIDs
    image_paths = sorted(section_dir.rglob('*.jpg'))
    if not image_paths:
        print(f"No images found in {section_dir}")
        sys.exit(1)
    print(f"Found {len(image_paths)} images in {section_dir}")

    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else REPO_DIR / 'mask_viz' / args.diu_id / blade / side
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load SAM
    device = torch.device(args.device)
    weights_dir = REPO_DIR / 'weights'
    load_sam(
        finetune_checkpoint=str(weights_dir / 'best_model.pth'),
        device=device,
    )

    for img_path in image_paths:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  Skip (unreadable): {img_path.name}")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask = segment_image_ar(img_rgb)
        overlay = overlay_mask(img_bgr, mask)

        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), overlay)
        print(f"  {img_path.name} -> {out_path}")

    print(f"\nDone. {len(image_paths)} images saved to {output_dir}")


if __name__ == '__main__':
    main()
