#!/usr/bin/env python3
from __future__ import annotations

"""
Fix nested timestamp directories under story directories for GPT4o and Gemini outputs.

This script converts:
  data/outputs/{method}/{mode}/{language}/{T1}/{story_id}/{T2}/shot_01.png
into:
  data/outputs/{method}/{mode}/{language}/{T1}/{story_id}/shots/shot_01.png

Rules:
- Within each story directory, if there are one or more inner timestamp directories (T2),
  pick the one with the largest number of image files (png/jpg/jpeg) as the source.
- Move those images into a 'shots' subdirectory under the story directory.
- Delete all inner timestamp directories (including the chosen one) afterwards.
- Also move any shot-like images directly under the story directory into 'shots'.

Defaults:
- Only process methods: GPT4o and Gemini (configurable via --methods).
- Outputs root: data/outputs (configurable via --outputs-root).
"""

import argparse
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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

DEFAULT_METHOD_WHITELIST: Set[str] = {"GPT4o", "Gemini"}

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

def list_immediate_subdirs(p: Path) -> List[Path]:
    try:
        return [x for x in p.iterdir() if x.is_dir()]
    except FileNotFoundError:
        return []

def count_images_in_dir(d: Path) -> int:
    count = 0
    try:
        for p in d.iterdir():
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                count += 1
    except FileNotFoundError:
        return 0
    return count

def ensure_dir(path: Path, dry_run: bool, verbose: bool) -> None:
    if not path.exists():
        if verbose:
            print(f"[CREATE] {path}")
        if not dry_run:
            path.mkdir(parents=True, exist_ok=True)

def move_images(src: Path, dst: Path, dry_run: bool, verbose: bool) -> Tuple[int, int]:
    """
    Move all image files (by extension) from src to dst. Returns (moved, skipped_conflicts).
    """
    moved = 0
    skipped = 0
    try:
        entries = sorted(src.iterdir())
    except FileNotFoundError:
        return (0, 0)
    for p in entries:
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMG_EXTS:
            continue
        dst_path = dst / p.name
        if dst_path.exists():
            skipped += 1
            if verbose:
                print(f"[SKIP] {p} -> {dst_path} (exists)")
            continue
        if verbose:
            print(f"[MOVE] {p} -> {dst_path}")
        if not dry_run:
            try:
                shutil.move(str(p), str(dst_path))
                moved += 1
            except Exception as e:
                skipped += 1
                print(f"[ERROR] Failed to move {p} -> {dst_path}: {e}")
    return (moved, skipped)

def move_top_level_shot_like(story_dir: Path, shots_dir: Path, dry_run: bool, verbose: bool) -> Tuple[int, int]:
    """Move shot-like image files directly under story_dir into shots_dir."""
    moved = 0
    skipped = 0
    try:
        entries = sorted(story_dir.iterdir())
    except FileNotFoundError:
        return (0, 0)
    for p in entries:
        if not p.is_file():
            continue
        if not is_shot_like_file(p.name):
            continue
        dst_path = shots_dir / p.name
        if dst_path.exists():
            skipped += 1
            if verbose:
                print(f"[SKIP] {p} -> {dst_path} (exists)")
            continue
        if verbose:
            print(f"[MOVE] {p} -> {dst_path}")
        if not dry_run:
            try:
                shutil.move(str(p), str(dst_path))
                moved += 1
            except Exception as e:
                skipped += 1
                print(f"[ERROR] Failed to move {p} -> {dst_path}: {e}")
    return (moved, skipped)

def remove_tree(p: Path, dry_run: bool, verbose: bool) -> bool:
    if not p.exists():
        return False
    if verbose:
        print(f"[DELETE] {p}")
    if not dry_run:
        try:
            shutil.rmtree(str(p))
        except Exception as e:
            print(f"[ERROR] Failed to delete {p}: {e}")
            return False
    return True

def choose_winner(inner_ts_dirs: List[Path]) -> Optional[Tuple[Path, int, Dict[str, int]]]:
    """
    Choose the inner timestamp dir with the largest number of images.
    Tie-breaker: lexicographically latest directory name.
    Returns (winner_path, winner_count, counts_by_name) or None if list empty.
    """
    if not inner_ts_dirs:
        return None
    counts: Dict[str, int] = {}
    for d in inner_ts_dirs:
        counts[d.name] = count_images_in_dir(d)
    max_count = max(counts.values()) if counts else 0
    candidates = [d for d in inner_ts_dirs if counts.get(d.name, 0) == max_count]
    winner = sorted(candidates, key=lambda x: x.name)[-1] if candidates else None
    if winner is None:
        return None
    return (winner, max_count, counts)

def process_story_dir(story_dir: Path, dry_run: bool, verbose: bool) -> Dict[str, int]:
    """
    For a given story_dir:
      - pick best inner timestamp dir (if any), move its images to shots/
      - optionally move top-level shot-like images to shots/
      - delete all inner timestamp dirs
    Returns stats dict.
    """
    stats = {
        "inner_ts_found": 0,
        "winner_images": 0,
        "moved_from_winner": 0,
        "skipped_from_winner": 0,
        "inner_ts_deleted": 0,
        "top_level_moved": 0,
        "top_level_skipped": 0,
    }

    inner_ts_dirs = [d for d in list_immediate_subdirs(story_dir) if is_timestamp_dir(d.name)]
    stats["inner_ts_found"] = len(inner_ts_dirs)

    shots_dir = story_dir / "shots"
    # We'll create shots dir only if we actually move something

    # Decide winner among inner timestamp dirs
    winner_info = choose_winner(inner_ts_dirs) if inner_ts_dirs else None
    if winner_info:
        winner_dir, winner_count, counts_by_name = winner_info
        stats["winner_images"] = winner_count
        if verbose:
            counts_repr = ", ".join(f"{k}:{v}" for k, v in sorted(counts_by_name.items()))
            print(f"[FOUND] inner_ts in {story_dir}: {counts_repr}; winner={winner_dir.name}")
        if winner_count > 0:
            ensure_dir(shots_dir, dry_run=dry_run, verbose=verbose)
            moved, skipped = move_images(winner_dir, shots_dir, dry_run=dry_run, verbose=verbose)
            stats["moved_from_winner"] += moved
            stats["skipped_from_winner"] += skipped

    # Move any top-level shot-like images
    # Do not redundantly create shots dir if nothing to move
    # First check whether there are candidates
    has_top_level_candidates = any(
        p.is_file() and is_shot_like_file(p.name)
        for p in story_dir.iterdir()
    ) if story_dir.exists() else False
    if has_top_level_candidates:
        ensure_dir(shots_dir, dry_run=dry_run, verbose=verbose)
        moved, skipped = move_top_level_shot_like(story_dir, shots_dir, dry_run=dry_run, verbose=verbose)
        stats["top_level_moved"] += moved
        stats["top_level_skipped"] += skipped

    # Remove all inner timestamp dirs (even if empty or no images moved),
    # as they are considered redundant under the corrected structure.
    for d in inner_ts_dirs:
        if remove_tree(d, dry_run=dry_run, verbose=verbose):
            stats["inner_ts_deleted"] += 1

    return stats

def scan_and_fix(outputs_root: Path, methods: Set[str], dry_run: bool, verbose: bool) -> Dict[str, int]:
    """
    Traverse outputs_root for the given methods and fix nested timestamp structures.
    Structure assumed:
      outputs_root/{method}/{mode}/{language}/{T1}/{story_id}/
    """
    totals = {
        "story_dirs_processed": 0,
        "files_moved": 0,
        "conflicts_skipped": 0,
        "inner_ts_deleted": 0,
    }
    for method in sorted(os.listdir(outputs_root)):
        method_dir = outputs_root / method
        if not method_dir.is_dir():
            continue
        if methods and method not in methods:
            continue
        if verbose:
            print(f"[METHOD] {method_dir}")
        # mode
        for mode in sorted(os.listdir(method_dir)):
            mode_dir = method_dir / mode
            if not mode_dir.is_dir():
                continue
            # language
            for lang in sorted(os.listdir(mode_dir)):
                lang_dir = mode_dir / lang
                if not lang_dir.is_dir():
                    continue
                # top-level timestamp T1
                for t1 in sorted(os.listdir(lang_dir)):
                    t1_dir = lang_dir / t1
                    if not t1_dir.is_dir() or not is_timestamp_dir(t1):
                        continue
                    # story_id
                    for story in sorted(os.listdir(t1_dir)):
                        story_dir = t1_dir / story
                        if not story_dir.is_dir():
                            continue
                        if not is_story_id_dir(story):
                            continue
                        totals["story_dirs_processed"] += 1
                        stats = process_story_dir(story_dir, dry_run=dry_run, verbose=verbose)
                        totals["files_moved"] += stats["moved_from_winner"] + stats["top_level_moved"]
                        totals["conflicts_skipped"] += stats["skipped_from_winner"] + stats["top_level_skipped"]
                        totals["inner_ts_deleted"] += stats["inner_ts_deleted"]
    return totals

def parse_methods_arg(arg: str, outputs_root: Path) -> Set[str]:
    """
    Parse the --methods argument.
    - "all" to process all method directories under outputs_root.
    - Comma-separated list otherwise.
    """
    arg = (arg or "").strip()
    if arg.lower() == "all":
        try:
            return {d for d in os.listdir(outputs_root) if (outputs_root / d).is_dir()}
        except FileNotFoundError:
            return set()
    methods = {m.strip() for m in arg.split(",") if m.strip()}
    return methods

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Repair nested timestamp directories under story outputs by selecting the inner timestamp "
            "with the most images as the canonical 'shots' and deleting the rest. "
            "Defaults to methods: GPT4o, Gemini."
        )
    )
    parser.add_argument("--outputs-root", type=str, default="data/outputs",
                        help="Root of outputs directory (default: data/outputs)")
    parser.add_argument("--methods", type=str, default="GPT4o,Gemini",
                        help='Comma-separated method names to process (default: "GPT4o,Gemini"). Use "all" to process all.')
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Do not modify files, only print actions")
    parser.add_argument("--quiet", action="store_true", default=False,
                        help="Reduce log output")
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    if not outputs_root.exists():
        raise SystemExit(f"Outputs root not found: {outputs_root}")

    methods = parse_methods_arg(args.methods, outputs_root)
    if not methods:
        # default whitelist if none parsed
        methods = set(DEFAULT_METHOD_WHITELIST)

    verbose = not args.quiet
    totals = scan_and_fix(outputs_root, methods=methods, dry_run=args.dry_run, verbose=verbose)
    print(
        "[SUMMARY] story_dirs_processed={sd}, files_moved={fm}, conflicts_skipped={cs}, inner_ts_deleted={itd}, dry_run={dr}".format(
            sd=totals["story_dirs_processed"],
            fm=totals["files_moved"],
            cs=totals["conflicts_skipped"],
            itd=totals["inner_ts_deleted"],
            dr=args.dry_run,
        )
    )

if __name__ == "__main__":
    main()