#!/usr/bin/env python3

"""Ensure 'shots' subdirectory exists under each story directory in ViStoryBench outputs, and move shot images into it.

This fixes outputs where images like 'shot_00.png' or '00.png' are directly under the story_id directory:
    data/outputs/{method}/{mode}/{language}/{timestamp}/{story_id}/shot_00.png
to the canonical structure:
    data/outputs/{method}/{mode}/{language}/{timestamp}/{story_id}/shots/shot_00.png

The script scans all timestamp directories under outputs_root, then iterates story_id subdirectories.
It detects shot-like images and moves them into a 'shots' subdirectory, creating it if missing.
It supports both directory layouts (mode-language order variants).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path
from typing import List, Tuple

# Allowed image extensions
IMG_EXTS = (".png", ".jpg", ".jpeg")

# Timestamp like 20250819_091354 or 20250819-091354
TIMESTAMP_RE = re.compile(r"^\d{8}[_-]\d{6}$")

# Story id like '01', '1', '002'
STORY_ID_RE = re.compile(r"^\d{1,3}$")

# Patterns for shot-like filenames (without extension)
SHOT_BASE_RES = [
    re.compile(r"^shot[-_]\d{1,3}$"),   # shot_00, shot-12
    re.compile(r"^shot\d{1,3}$"),       # shot00
    re.compile(r"^\d{1,3}$"),           # 00, 1, 012
]

def is_timestamp_dir(name: str) -> bool:
    """Return True if name matches timestamp format."""
    return bool(TIMESTAMP_RE.match(name))

def is_story_id_dir(name: str) -> bool:
    """Return True if name looks like a story id (digits)."""
    return bool(STORY_ID_RE.match(name))

def is_shot_like_file(filename: str) -> bool:
    """Heuristically determine if filename represents a shot image."""
    base, ext = os.path.splitext(filename)
    ext = ext.lower()
    if ext not in IMG_EXTS:
       return False
    base = base.lower()
    return any(rx.match(base) for rx in SHOT_BASE_RES)

def ensure_shots_subdir(story_dir: Path, dry_run: bool = False, verbose: bool = True) -> Tuple[int, int]:
    """
    Ensure 'shots' subdirectory exists in a given story_dir and move shot-like images into it.

    Returns a tuple: (moved_count, skipped_conflicts)
    """
    moved = 0
    skipped = 0
    shots_dir = story_dir / "shots"

    # Collect top-level files directly under story_dir
    try:
        entries = list(story_dir.iterdir())
    except FileNotFoundError:
        return (0, 0)

    top_level_files = [p for p in entries if p.is_file()]
    shot_candidates = [p for p in top_level_files if is_shot_like_file(p.name)]

    if not shot_candidates and not shots_dir.exists():
        # Nothing to do
        return (0, 0)

    if not shots_dir.exists():
        if verbose:
            print(f"[CREATE] {shots_dir}")
        if not dry_run:
            shots_dir.mkdir(parents=True, exist_ok=True)

    # Move candidates into shots/
    for src in shot_candidates:
        dst = shots_dir / src.name
        if dst.exists():
            # Conflict: destination file already exists
            skipped += 1
            if verbose:
                print(f"[SKIP] {src} -> {dst} (exists)")
            continue
        if verbose:
            print(f"[MOVE] {src} -> {dst}")
        if not dry_run:
            try:
                shutil.move(str(src), str(dst))
                moved += 1
            except Exception as e:
                skipped += 1
                print(f"[ERROR] Failed to move {src} -> {dst}: {e}")

    return (moved, skipped)

def scan_outputs(outputs_root: Path, dry_run: bool = False, verbose: bool = True) -> Tuple[int, int, int]:
    """
    Scan outputs_root for timestamp directories and process each story subdir.

    Returns totals: (processed_story_dirs, moved_files, skipped_conflicts)
    """
    processed = 0
    moved_total = 0
    skipped_total = 0

    root_str = str(outputs_root)
    for dirpath, dirnames, filenames in os.walk(root_str):
        base = os.path.basename(dirpath)
        if not is_timestamp_dir(base):
            continue
        ts_dir = Path(dirpath)

        # Immediate subdirectories of timestamp are expected story ids
        for d in sorted(os.listdir(ts_dir)):
            story_dir = ts_dir / d
            if not story_dir.is_dir():
                continue
            if not is_story_id_dir(d):
                # not a story id; ignore (may be artifacts)
                continue
            processed += 1
            moved, skipped = ensure_shots_subdir(story_dir, dry_run=dry_run, verbose=verbose)
            moved_total += moved
            skipped_total += skipped

    return (processed, moved_total, skipped_total)

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Normalize ViStoryBench outputs by ensuring 'shots' subdirectory exists under each story directory "
            "and moving shot images into it. Supports both outputs layouts."
        )
    )
    parser.add_argument("--outputs-root", type=str, default="data/outputs_avif",
                        help="Root of outputs directory (default: data/outputs)")
    parser.add_argument("--dry-run", action="store_true",default=False, help="Do not modify files, only print actions")
    parser.add_argument("--quiet", action="store_true", help="Reduce log output")
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    if not outputs_root.exists():
        raise SystemExit(f"Outputs root not found: {outputs_root}")

    verbose = not args.quiet
    processed, moved, skipped = scan_outputs(outputs_root, dry_run=args.dry_run, verbose=verbose)
    print(f"[SUMMARY] story_dirs_processed={processed}, files_moved={moved}, conflicts_skipped={skipped}, dry_run={args.dry_run}")

if __name__ == "__main__":
    main()