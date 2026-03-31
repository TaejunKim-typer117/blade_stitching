#!/usr/bin/env python3
"""
Download images for 4 random DIU IDs in [60000, 69999], then run stitch_v2.py on them.
Skips DIU IDs that return no images or fail to download.

Usage:
    nohup python run_pipeline.py > log/pipeline.log 2>&1 &
"""

import random
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
TARGET_COUNT = 4
DIU_RANGE = (60000, 69999)


def try_download(diu_id):
    """Run download.py for a single DIU ID. Returns True if successful."""
    result = subprocess.run(
        [sys.executable, str(REPO_DIR / 'download.py'), '--diu-id', str(diu_id)],
        capture_output=True, text=True, timeout=300,
    )
    print(result.stdout, end='', flush=True)
    if result.stderr:
        print(result.stderr, end='', flush=True)

    if result.returncode != 0:
        return False
    if 'No photos' in result.stdout or 'Error' in result.stdout:
        return False
    if 'down=' in result.stdout or 'skip=' in result.stdout:
        return True
    return False


def main():
    completed = []
    tried = set()

    print(f'=== Phase 1: Download {TARGET_COUNT} DIU IDs from range {DIU_RANGE} ===', flush=True)

    while len(completed) < TARGET_COUNT:
        diu_id = random.randint(*DIU_RANGE)
        if diu_id in tried:
            continue
        tried.add(diu_id)

        print(f'\n[{len(completed)}/{TARGET_COUNT}] Trying DIU {diu_id} (tried {len(tried)} so far)', flush=True)
        try:
            success = try_download(diu_id)
        except Exception as e:
            print(f'  -> SKIP (exception: {e})', flush=True)
            continue

        if success:
            completed.append(diu_id)
            print(f'  -> SUCCESS ({len(completed)}/{TARGET_COUNT})', flush=True)
        else:
            print(f'  -> SKIP', flush=True)

    print(f'\n=== Phase 2: Stitching DIU IDs: {completed} ===', flush=True)

    diu_args = ['--diu-id'] + [str(d) for d in completed]

    result = subprocess.run(
        [sys.executable, str(REPO_DIR / 'stitch_v2.py')] + diu_args,
        timeout=3600,
    )

    if result.returncode != 0:
        print(f'\nStitching exited with code {result.returncode}', flush=True)
    else:
        print(f'\nAll done. Processed DIU IDs: {completed}', flush=True)


if __name__ == '__main__':
    main()
