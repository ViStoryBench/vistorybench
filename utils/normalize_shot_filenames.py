#!/usr/bin/env python3
from __future__ import annotations

"""
Normalize shot filenames under ViStoryBench outputs:
- If a 'shots' directory contains any shot_00.<ext>, shift all matching shot_<NN>.<ext> by +1 so that numbering starts at 01.
- Only files strictly matching ^shot_(\\d+)\\.<ext>$ are processed.
- Safe renaming using descending index order to avoid collisions.

Default root: data/outputs
Default extension: png
"""

import argparse
import os
import re
from pathlib import Path
from typing import List, Tuple, Optional

SHOT_RE_TEMPLATE = r"^shot_(\d+)\.{ext}$"

def compile_shot_re(ext: str) -> re.Pattern[str]:
    ext = ext.lower().lstrip(".")
    return re.compile(SHOT_RE_TEMPLATE.format(ext=re.escape(ext)))

def find_shots_dirs(outputs_root: Path) -> List[Path]:
    dirs: List[Path] = []
    root_str = str(outputs_root)
    for dirpath, dirnames, filenames in os.walk(root_str):
        if os.path.basename(dirpath) == "shots":
            dirs.append(Path(dirpath))
    return dirs

def parse_index(filename: str, rx: re.Pattern[str]) -> Optional[Tuple[int, int]]:
    """
    Return (index_value, width) if filename matches, else None.
    width = length of the digit group in the filename (zero-padded width).
    """
    m = rx.match(filename)
    if not m:
        return None
    digits = m.group(1)
    try:
        return int(digits), len(digits)
    except ValueError:
        return None

def plan_shift_one(shots_dir: Path, rx: re.Pattern[str]) -> List[Tuple[Path, Path]]:
    """
    If the directory contains any index == 0 file, plan renames for all matching files: idx -> idx+1.
    Returns list of (src, dst) paths in descending src index order.
    """
    candidates: List[Tuple[Path, int, int]] = []  # (path, idx, width)
    has_zero = False
    try:
        entries = list(shots_dir.iterdir())
    except FileNotFoundError:
        return []
    for p in entries:
        if not p.is_file():
            continue
        parsed = parse_index(p.name, rx)
        if not parsed:
            continue
        idx, width = parsed
        candidates.append((p, idx, width))
        if idx == 0:
            has_zero = True
    if not has_zero or not candidates:
        return []
    # Sort by index descending to avoid collisions
    candidates.sort(key=lambda t: t[1], reverse=True)
    renames: List[Tuple[Path, Path]] = []
    for p, idx, width in candidates:
        new_idx = idx + 1
        # Preserve original zero-padding width; if new index has more digits, keep the expanded width
        new_name = f"shot_{str(new_idx).zfill(width)}{p.suffix}"
        dst = p.with_name(new_name)
        if dst == p:
            continue
        renames.append((p, dst))
    return renames

def apply_renames(renames: List[Tuple[Path, Path]], dry_run: bool, verbose: bool) -> Tuple[int, int]:
    """
    Apply renames in given order. Returns (done, skipped_due_to_exists).
    """
    done = 0
    skipped = 0
    for src, dst in renames:
        if dst.exists():
            skipped += 1
            if verbose:
                print(f"[SKIP] {src} -> {dst} (destination exists)")
            continue
        if verbose:
            print(f"[RENAME] {src} -> {dst}")
        if not dry_run:
            try:
                src.rename(dst)
                done += 1
            except Exception as e:
                skipped += 1
                print(f"[ERROR] Failed to rename {src} -> {dst}: {e}")
    return done, skipped

def scan_and_fix(outputs_root: Path, ext: str, dry_run: bool, verbose: bool) -> Tuple[int, int, int, int]:
    """
    Scan outputs_root for 'shots' directories and shift numbering where needed.
    Returns totals: (shots_dirs_seen, shots_dirs_changed, files_renamed, conflicts_skipped)
    """
    rx = compile_shot_re(ext)
    shots_dirs = find_shots_dirs(outputs_root)
    seen = len(shots_dirs)
    changed = 0
    renamed_total = 0
    skipped_total = 0
    for sd in sorted(shots_dirs):
        plans = plan_shift_one(sd, rx)
        if not plans:
            continue
        changed += 1
        done, skipped = apply_renames(plans, dry_run=dry_run, verbose=verbose)
        print(f"[INFO] Processed shots dir: {sd}, planned renames: {len(plans)}, done: {done}, skipped: {skipped}")
        renamed_total += done
        skipped_total += skipped
    return seen, changed, renamed_total, skipped_total

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Normalize shot filenames: if a shots/ dir contains shot_00.<ext>, "
            "shift all shot_<NN>.<ext> by +1 so numbering starts at 01."
        )
    )
    parser.add_argument("--outputs-root", type=str, default="data/outputs",
                        help="Root of outputs directory (default: data/outputs)")
    parser.add_argument("--ext", type=str, default="png",
                        help="Image extension to process (default: png)")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Do not modify files, only print planned actions")
    parser.add_argument("--quiet", action="store_true", default=True,
                        help="Reduce log output")
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    if not outputs_root.exists():
        raise SystemExit(f"Outputs root not found: {outputs_root}")

    verbose = not args.quiet
    seen, changed, renamed, skipped = scan_and_fix(outputs_root, ext=args.ext, dry_run=args.dry_run, verbose=verbose)
    print(
        "[SUMMARY] shots_dirs_seen={seen}, shots_dirs_changed={chg}, files_renamed={ren}, conflicts_skipped={skp}, dry_run={dr}".format(
            seen=seen, chg=changed, ren=renamed, skp=skipped, dr=args.dry_run
        )
    )

if __name__ == "__main__":
    main()