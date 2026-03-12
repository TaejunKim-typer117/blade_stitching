#!/usr/bin/env python3
"""
Download wind turbine blade images and metadata from Zoomable API.

Uses the Worker API which returns quality-checked (included) photos only.
Output format is compatible with stitch.py.

Usage:
    python download.py --diu-id 40012
    python download.py --diu-id 40012 40013 40014
    python download.py --diu-id 40012 -o ./data
    python download.py --diu-id 40012 --workers 8

Output structure:
    {output_dir}/{diu_id}/
    ├── metadata.json
    └── {blade}/{side}/photo_{id}.jpg
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import boto3
from botocore.config import Config

REPO_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = REPO_DIR / 'data'


# ── Authentication ───────────────────────────────────────────────────────────

def get_cognito_token(
    region='ap-northeast-2',
    client_id='2mt6ifvu9rksdb9ucltri00d0i',
    username='dolfin.demo',
    password='Dolfin0201!',
) -> str:
    client = boto3.client(
        'cognito-idp',
        region_name=region,
        config=Config(signature_version='v4'),
    )
    response = client.initiate_auth(
        ClientId=client_id,
        AuthFlow='USER_PASSWORD_AUTH',
        AuthParameters={'USERNAME': username, 'PASSWORD': password},
    )
    return response['AuthenticationResult']['AccessToken']


# ── API ──────────────────────────────────────────────────────────────────────

BASE_URL = 'https://zoomable.nearthlab.com/worker/ai-quality-check'


def fetch_photos(diu_id, token):
    """Fetch photo list with metadata from Worker API (included photos only)."""
    url = f'{BASE_URL}/draft-inspection-unit/{diu_id}/draft-photos'
    headers = {'Authorization': token}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_photo_tags(diu_id, token):
    """Fetch blade and blade-side tag lists."""
    url = f'{BASE_URL}/draft-inspection-unit/{diu_id}/photo-tags'
    headers = {'Authorization': token}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    data = response.json()

    blade_tags = []
    blade_side_tags = []
    for group in data:
        if group.get('type') == 'Blade':
            blade_tags = [tag['slug'] for tag in group.get('photoTags', [])]
        elif group.get('type') == 'BladeSide':
            blade_side_tags = [tag['slug'] for tag in group.get('photoTags', [])]
    return blade_tags, blade_side_tags


# ── Metadata normalization ───────────────────────────────────────────────────

def normalize_metadata(api_meta):
    """Convert API metadata (camelCase) to snake_case format for stitch.py."""
    if not api_meta:
        api_meta = {}
    return {
        'n': float(api_meta.get('n', 0)),
        'e': float(api_meta.get('e', 0)),
        'alt': float(api_meta.get('alt', 0)),
        'body_yaw': float(api_meta.get('bodyYaw', 0)),
        'gimbal_roll': float(api_meta.get('gimbalRoll', 0)),
        'gimbal_pitch': float(api_meta.get('gimbalPitch', 0)),
        'gimbal_yaw': float(api_meta.get('gimbalYaw', 0)),
        'focal_length': float(api_meta['focalLength']) if api_meta.get('focalLength') is not None else None,
        'measured_distance_to_blade': float(api_meta.get('measuredDistanceToBlade', 7)),
        'blade_side': api_meta.get('bladeSide', ''),
        'blade_position': api_meta.get('bladePosition'),
        'direction': api_meta.get('direction', ''),
        'drone': api_meta.get('drone', ''),
        'app_version': api_meta.get('appVersion', ''),
        'meta_version': str(api_meta.get('metaVersion', '')),
        'width': int(api_meta.get('width', 1280)),
        'height': int(api_meta.get('height', 853)),
    }


# ── Image download ───────────────────────────────────────────────────────────

def download_one(photo, diu_dir):
    """Download a single photo. Returns (photo_id, status, local_path)."""
    photo_id = photo['id']
    image_url = photo.get('thumbnailImage')
    if not image_url:
        return photo_id, 'failed', None

    blade = photo.get('bladeTag', {}).get('slug') if photo.get('bladeTag') else None
    side = photo.get('bladeSideTag', {}).get('slug') if photo.get('bladeSideTag') else None

    if blade and side:
        subdir = diu_dir / blade / side
        filename = f'photo_{photo_id}.jpg'
    else:
        subdir = diu_dir / 'untagged'
        filename = f'photo_{photo_id}.jpg'

    dest = subdir / filename
    local_path = str(dest.relative_to(diu_dir))

    if dest.exists():
        return photo_id, 'skipped', local_path

    try:
        resp = requests.get(image_url, timeout=60)
        resp.raise_for_status()
        subdir.mkdir(parents=True, exist_ok=True)
        with open(dest, 'wb') as f:
            f.write(resp.content)
        return photo_id, 'downloaded', local_path
    except Exception:
        return photo_id, 'failed', None


# ── Process one DIU ──────────────────────────────────────────────────────────

def process_diu(diu_id, token, output_dir, workers):
    diu_dir = output_dir / str(diu_id)
    metadata_path = diu_dir / 'metadata.json'

    # Skip if already done
    if metadata_path.exists():
        print(f'  Skipped (metadata.json exists)')
        return

    # Fetch from API
    photos = fetch_photos(diu_id, token)
    blade_tags, blade_side_tags = fetch_photo_tags(diu_id, token)

    valid_photos = [p for p in photos if p.get('thumbnailImage') and p.get('metadata')]
    print(f'  {len(valid_photos)} photos')

    if not valid_photos:
        print(f'  No photos to download')
        return

    # Download images
    local_paths = {}
    stats = {'downloaded': 0, 'skipped': 0, 'failed': 0}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_one, p, diu_dir): p for p in valid_photos}
        for future in as_completed(futures):
            photo_id, status, local_path = future.result()
            stats[status] += 1
            if local_path:
                local_paths[photo_id] = local_path

    # Build metadata JSON
    metadata_output = {
        'draft_id': diu_id,
        'blade_tags': blade_tags,
        'blade_side_tags': blade_side_tags,
        'total_photos': len(valid_photos),
        'photos': [
            {
                'id': p['id'],
                'blade_tag': p.get('bladeTag', {}).get('slug') if p.get('bladeTag') else None,
                'blade_side_tag': p.get('bladeSideTag', {}).get('slug') if p.get('bladeSideTag') else None,
                'local_path': local_paths.get(p['id']),
                'thumbnail_url': p.get('thumbnailImage'),
                'metadata': normalize_metadata(p.get('metadata')),
            }
            for p in valid_photos
        ],
    }

    diu_dir.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(metadata_output, f, indent=2)

    print(f'  Done (down={stats["downloaded"]}, skip={stats["skipped"]}, fail={stats["failed"]})')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Download blade images from Zoomable API')
    parser.add_argument('--diu-id', type=int, nargs='+', required=True, help='Draft inspection unit ID(s)')
    parser.add_argument('--output-dir', '-o', type=str, default=None, help='Output directory (default: blade_stitching/data)')
    parser.add_argument('--workers', '-w', type=int, default=5, help='Concurrent downloads (default: 5)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print('Authenticating...')
    token = get_cognito_token()

    for i, diu_id in enumerate(args.diu_id, 1):
        print(f'[{i}/{len(args.diu_id)}] DIU {diu_id}')
        try:
            process_diu(diu_id, token, output_dir, args.workers)
        except Exception as e:
            print(f'  Error: {e}')

    print('\nDone.')


if __name__ == '__main__':
    main()
