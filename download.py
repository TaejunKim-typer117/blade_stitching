#!/usr/bin/env python3
"""
Download wind turbine blade images and metadata from Zoomable API.

Uses the paginated API which returns all photos (included + excluded).
Downloads both thumbnail and original resolution images.
Output format is compatible with stitch.py.

Usage:
    python download.py --diu-id 40012
    python download.py --diu-id 40012 40013 40014
    python download.py --diu-id 40012 -o ./data
    python download.py --diu-id 40012 --workers 8

Output structure:
    {output_dir}/{diu_id}/
    ├── metadata.json
    ├── thumbnail/{blade}/{side}/{missionUuid}/photo_{id}.jpg
    └── original/{blade}/{side}/{missionUuid}/photo_{id}.jpg
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

BASE_URL = 'https://zoomable.nearthlab.com'


def fetch_photos(diu_id, token):
    """Fetch all photos (included + excluded) from paginated API."""
    url = f'{BASE_URL}/draft-inspection-units/{diu_id}/draft-photos?size=10000'
    headers = {'Authorization': token, 'Referer': 'https://worker.zoomable.io/'}
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()
    return response.json()['result']


def fetch_tag_map(diu_id, token):
    """Fetch tag ID -> slug mapping."""
    url = f'{BASE_URL}/worker/ai-quality-check/draft-inspection-unit/{diu_id}/photo-tags'
    headers = {'Authorization': token}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    tag_map = {}
    for group in response.json():
        for tag in group.get('photoTags', []):
            tag_map[tag['id']] = tag['slug']
    return tag_map


def resolve_tags(photo, tag_map):
    """Resolve blade and blade-side tags from photo data."""
    blade = None
    side = None
    if photo.get('bladeTag'):
        blade = photo['bladeTag'].get('slug')
    elif photo.get('bladeTagId'):
        blade = tag_map.get(photo['bladeTagId'])
    if photo.get('bladeSideTag'):
        side = photo['bladeSideTag'].get('slug')
    elif photo.get('bladeSideTagId'):
        side = tag_map.get(photo['bladeSideTagId'])
    return blade, side


# ── Metadata normalization ───────────────────────────────────────────────────

def normalize_metadata(api_meta):
    """Convert API metadata (camelCase) to snake_case format for stitch.py."""
    if not api_meta:
        api_meta = {}
    return {
        'r': float(api_meta.get('r', 0)),
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
        'mission_uuid': api_meta.get('missionUuid'),
    }


# ── Image download ───────────────────────────────────────────────────────────

def get_subpath(photo, tag_map):
    """Return the sub-path components (subdir, filename) for a photo."""
    blade, side = resolve_tags(photo, tag_map)
    mission_uuid = photo.get('metadata', {}).get('missionUuid', 'unknown')
    filename = f'photo_{photo["id"]}.jpg'

    if blade and side:
        subdir = Path(blade) / side / mission_uuid
    else:
        subdir = Path('untagged') / mission_uuid

    return subdir, filename


def download_one(photo, diu_dir, tag_map):
    """Download thumbnail and original for a single photo.
    Returns (photo_id, thumb_status, orig_status, thumb_path, orig_path)."""
    photo_id = photo['id']
    thumb_url = photo.get('thumbnailImage')
    orig_url = photo.get('originalImage')

    subdir, filename = get_subpath(photo, tag_map)

    results = {}
    for kind, url in [('thumbnail', thumb_url), ('original', orig_url)]:
        if not url:
            results[kind] = ('failed', None)
            continue

        dest = diu_dir / kind / subdir / filename
        local_path = str(dest.relative_to(diu_dir))

        if dest.exists():
            results[kind] = ('skipped', local_path)
            continue

        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, 'wb') as f:
                f.write(resp.content)
            results[kind] = ('downloaded', local_path)
        except Exception:
            results[kind] = ('failed', None)

    return photo_id, results


# ── Process one DIU ──────────────────────────────────────────────────────────

def process_diu(diu_id, token, output_dir, workers):
    diu_dir = output_dir / str(diu_id)
    metadata_path = diu_dir / 'metadata.json'

    # Skip if already done
    if metadata_path.exists():
        print(f'  Skipped (metadata.json exists)')
        return

    # Fetch from API
    tag_map = fetch_tag_map(diu_id, token)
    photos = fetch_photos(diu_id, token)

    valid_photos = [p for p in photos if p.get('thumbnailImage')]
    print(f'  {len(valid_photos)} photos (all)')

    if not valid_photos:
        print(f'  No photos to download')
        return

    # Download images
    local_paths = {}  # photo_id -> {'thumbnail': path, 'original': path}
    stats = {
        'thumbnail': {'downloaded': 0, 'skipped': 0, 'failed': 0},
        'original': {'downloaded': 0, 'skipped': 0, 'failed': 0},
    }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_one, p, diu_dir, tag_map): p for p in valid_photos}
        for future in as_completed(futures):
            photo_id, results = future.result()
            paths = {}
            for kind in ['thumbnail', 'original']:
                status, path = results[kind]
                stats[kind][status] += 1
                if path:
                    paths[kind] = path
            local_paths[photo_id] = paths

    # Build metadata JSON
    blade_tags = sorted({resolve_tags(p, tag_map)[0] for p in valid_photos if resolve_tags(p, tag_map)[0]})
    blade_side_tags = sorted({resolve_tags(p, tag_map)[1] for p in valid_photos if resolve_tags(p, tag_map)[1]})

    metadata_output = {
        'draft_id': diu_id,
        'blade_tags': blade_tags,
        'blade_side_tags': blade_side_tags,
        'total_photos': len(valid_photos),
        'photos': [
            {
                'id': p['id'],
                'blade_tag': resolve_tags(p, tag_map)[0],
                'blade_side_tag': resolve_tags(p, tag_map)[1],
                'quality_checked': bool(p.get('qualityChecked', True)),
                'local_path': local_paths.get(p['id'], {}).get('thumbnail'),
                'original_path': local_paths.get(p['id'], {}).get('original'),
                'thumbnail_url': p.get('thumbnailImage'),
                'original_url': p.get('originalImage'),
                'metadata': normalize_metadata(p.get('metadata', {})),
            }
            for p in valid_photos
        ],
    }

    diu_dir.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(metadata_output, f, indent=2)

    for kind in ['thumbnail', 'original']:
        s = stats[kind]
        print(f'  {kind}: down={s["downloaded"]}, skip={s["skipped"]}, fail={s["failed"]}')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Download all blade images (included + excluded)')
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
