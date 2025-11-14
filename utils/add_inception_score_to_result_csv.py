#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add or recompute Inception Score column for ViStoryBench result CSVs.

This script updates CSVs under data/result (lite/full aggregated CSV) by adding
a column 'metrics.diversity.metrics.inception_score'. It recomputes the IS
using vistorybench.bench.diversity.inception_score.calculate_inception_score
over the allowed story subset for the given split (lite or full).

- Default behavior: runs both lite and full splits when --split is omitted.
- Default targets:
  * split=lite -> data/result/bench_results_lite.csv
  * split=full -> data/result/bench_results_full.csv
- Row keys are interpreted as:
  method=row['method'], mode=row['model'], language=row['mode'], timestamp=row['timestamp']
- Only fills empty cells by default; use --overwrite to replace existing values.
- Writes UTF-8 BOM by default; creates a .bak backup before writing.

Usage:
  # Run both lite and full (default)
  python utils/add_inception_score_to_result_csv.py

  # Run a single split
  python utils/add_inception_score_to_result_csv.py --split lite
  python utils/add_inception_score_to_result_csv.py --split full

  # Example with options for a single split
  python utils/add_inception_score_to_result_csv.py --split full --overwrite --device cpu --batch-size 16 --splits 1 --no-bom --no-backup

Note:
  When --split=both, do not pass --csv; the script will update the default lite/full CSVs.
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import shutil

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore

# Ensure project root on sys.path (this file is under utils/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Imports from project
from vistorybench.bench.diversity.inception_score import calculate_inception_score
from vistorybench.dataset_loader.dataset_load import StoryDataset  # type: ignore
from vistorybench.dataset_loader.read_outputs import load_outputs  # type: ignore

METRIC_COL = "metrics.diversity.metrics.inception_score"
FRONT_COLS = ["method", "model", "mode", "timestamp"]
IMG_EXTS = (".png", ".jpg", ".jpeg")

def load_yaml_config(config_path: Path = Path("config.yaml")) -> Dict[str, Any]:
    """Load config.yaml if present; return {} if unavailable or parse fails."""
    if yaml is None:
        return {}
    try:
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except Exception:
        return {}
    return {}

def resolve_paths_from_config(cfg: Dict[str, Any]) -> Tuple[Path, Path]:
    """
    Resolve dataset_root and outputs_root from config with fallbacks.
    dataset_root preference: core.paths.dataset -> dataset_path -> data/dataset
    outputs_root preference: core.paths.outputs -> outputs_path -> data/outputs
    """
    core = (cfg or {}).get("core") or {}
    paths_map = core.get("paths") or {}
    dataset_root = paths_map.get("dataset") or (cfg or {}).get("dataset_path") or "data/dataset"
    outputs_root = paths_map.get("outputs") or (cfg or {}).get("outputs_path") or "data/outputs"
    return Path(str(dataset_root)), Path(str(outputs_root))

def detect_device(opt: str) -> str:
    """
    Map device option to 'cuda' or 'cpu'.
    - 'auto' => 'cuda' if available else 'cpu'
    """
    if opt in ("cuda", "cpu"):
        return opt
    try:
        import torch  # local import to avoid mandatory dep at import time
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

def try_get_allowed_story_ids(dataset_root: Path, split: str) -> Optional[Set[str]]:
    """
    Try to load story id list from dataset_root/ViStory using StoryDataset.
    Return a set of ids (as strings) or None if dataset not accessible.
    """
    try:
        vis_dir = dataset_root / "ViStory"
        if not vis_dir.is_dir():
            return None
        ds = StoryDataset(str(vis_dir))
        ids = ds.get_story_name_list(split=split)
        return set(map(str, ids))
    except Exception:
        return None

def load_story_outputs_for_run(
    outputs_root: Path,
    method: str,
    mode: str,
    language: str,
    timestamp: str,
) -> Dict[str, Any]:
    """
    Load outputs for a specific (method, mode, language, timestamp) combination.
    Returns a mapping: story_id -> { 'shots': [image_paths], ... }
    """
    try:
        out = load_outputs(
            outputs_root=str(outputs_root),
            methods=[method],
            languages=[language],
            modes=[mode],
            return_latest=False,
        )
        # load_outputs returns a nested map: {story_id: {...}} for the given filters
        # Ensure keys are strings for consistency
        return {str(k): v for k, v in out.items()} if isinstance(out, dict) else {}
    except Exception:
        return {}

def collect_images_from_outputs(
    stories_outputs: Dict[str, Any],
    allowed_ids: Set[str],
    must_exist: bool = True,
) -> List[str]:
    """
    Collect shot image paths from stories_outputs for the allowed story ids.
    """
    image_paths: List[str] = []
    for sid, data in stories_outputs.items():
        if allowed_ids and str(sid) not in allowed_ids:
            continue
        shots: Optional[List[str]] = None
        if isinstance(data, dict):
            v = data.get("shots").values()
            shots = [p for p in v if isinstance(p, str)]
        # Fallback: if not standard schema, try to extract any string paths from values
        if shots is None and isinstance(data, dict):
            cand: List[str] = []
            for val in data.values():
                if isinstance(val, list):
                    cand.extend([p for p in val if isinstance(p, str)])
            shots = cand if cand else []
        if not shots:
            continue
        for p in shots:
            if not isinstance(p, str):
                continue
            if not p.lower().endswith(IMG_EXTS):
                continue
            if must_exist and not os.path.isfile(p):
                continue
            image_paths.append(p)
    image_paths.sort()
    return image_paths

def update_csv(
    csv_path: Path,
    split: str,
    dataset_root: Path,
    outputs_root: Path,
    batch_size: int,
    splits: int,
    device_opt: str,
    overwrite: bool,
    no_bom: bool,
    no_backup: bool,
    dry_run: bool,
) -> None:
    if not csv_path.exists():
        print(f"Error: CSV not found: {csv_path}")
        return

    encoding_read = "utf-8-sig"
    try:
        with csv_path.open("r", newline="", encoding=encoding_read) as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                print(f"Error: CSV has no header: {csv_path}")
                return
            fieldnames = list(reader.fieldnames)
            rows = [row for row in reader]
    except UnicodeDecodeError:
        # fallback to plain utf-8
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                print(f"Error: CSV has no header: {csv_path}")
                return
            fieldnames = list(reader.fieldnames)
            rows = [row for row in reader]

    if METRIC_COL not in fieldnames:
        fieldnames.append(METRIC_COL)

    device_selected = detect_device(device_opt)
    print(f"Device selected: {device_selected}")

    allowed_ids = try_get_allowed_story_ids(dataset_root, split=split)
    if allowed_ids is not None:
        print(f"Loaded {len(allowed_ids)} allowed story IDs from dataset for split='{split}'.")
    else:
        print(f"Dataset not available or failed to load; will derive allowed IDs from each run outputs.")

    # Cache to avoid recomputation: key -> is_mean
    cache: Dict[Tuple[str, str, str, str, str], float] = {}

    updated = 0
    skipped = 0
    failed = 0

    for idx, row in enumerate(rows):
        try:
            method = (row.get("method") or "").strip()
            mode = (row.get("model") or "").strip()      # 'model' column stores mode
            language = (row.get("mode") or "").strip()   # 'mode' column stores language
            timestamp = (row.get("timestamp") or "").strip()
        except Exception:
            failed += 1
            continue

        if not (method and mode and language and timestamp):
            skipped += 1
            continue

        current_val = row.get(METRIC_COL, "")
        if (current_val not in (None, "", "NaN")) and (not overwrite):
            skipped += 1
            continue

        cache_key = (method, mode, language, timestamp, split)
        if cache_key in cache:
            row[METRIC_COL] = cache[cache_key]
            updated += 1
            continue

        # Load story outputs for this run
        stories_outputs = load_story_outputs_for_run(outputs_root, method, mode, language, timestamp)
        if not stories_outputs:
            # No outputs found
            skipped += 1
            continue

        # Derive allowed set if dataset not available
        run_ids = set(map(str, stories_outputs.keys()))
        allowed = allowed_ids if (allowed_ids is not None) else run_ids

        # Collect images
        images = collect_images_from_outputs(stories_outputs, allowed_ids=allowed, must_exist=True)
        if not images:
            skipped += 1
            continue

        if dry_run:
            # Just simulate update
            row[METRIC_COL] = ""
            skipped += 1
            continue

        try:
            is_mean, is_std = calculate_inception_score(
                images, batch_size=batch_size, splits=splits, device=device_selected
            )
            # Store only mean in CSV
            row[METRIC_COL] = is_mean
            cache[cache_key] = is_mean
            updated += 1
        except Exception as e:
            print(f"Warning: IS calculation failed for run {method}/{mode}/{language}/{timestamp}: {e}")
            failed += 1
            continue

    # If dry-run, do not write back
    if dry_run:
        print(f"[Dry Run] Planned updates: updated={updated}, skipped={skipped}, failed={failed}")
        return

    # Backup before write
    if not no_backup:
        try:
            shutil.copy2(str(csv_path), str(csv_path) + ".bak")
            print(f"Backup created: {str(csv_path)}.bak")
        except Exception as e:
            print(f"Warning: Failed to create backup for {csv_path}: {e}")

    # Write back with BOM by default
    encoding_write = "utf-8" if no_bom else "utf-8-sig"
    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    try:
        with tmp_path.open("w", newline="", encoding=encoding_write) as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        os.replace(str(tmp_path), str(csv_path))
    except Exception as e:
        # Cleanup tmp on failure
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        print(f"Error: Failed to write CSV {csv_path}: {e}")
        return

    print(f"Done. CSV updated: {csv_path}")
    print(f"Summary: updated={updated}, skipped={skipped}, failed={failed}")

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Add or recompute Inception Score column for ViStoryBench result CSVs (lite/full)."
    )
    parser.add_argument("--split", required=False, choices=["lite", "full", "both"], default="both", help="Target dataset split to compute (default: both)")
    parser.add_argument("--csv", default=None, help="CSV file to update. Default depends on split.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing values (default: only fill empty)")
    parser.add_argument("--no-bom", action="store_true", help="Write without UTF-8 BOM (default: with BOM)")
    parser.add_argument("--no-backup", action="store_true", help="Do not create .bak backup (default: create)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for IS calculation")
    parser.add_argument("--splits", type=int, default=1, help="Number of splits for IS calculation")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device to use")
    parser.add_argument("--dry-run", action="store_true", help="Preview updates without computing or writing")
    parser.add_argument("--dataset-path", type=str, default=None, help="Override dataset root path (optional)")
    parser.add_argument("--outputs-path", type=str, default=None, help="Override outputs root path (optional)")
    args = parser.parse_args(argv)

    cfg = load_yaml_config(Path("config.yaml"))
    dataset_root, outputs_root = resolve_paths_from_config(cfg)

    # CLI overrides
    if args.dataset_path:
        dataset_root = Path(args.dataset_path)
    if args.outputs_path:
        outputs_root = Path(args.outputs_path)

    if args.split == "both":
        if args.csv:
            print("Error: --csv cannot be used when --split=both. Omit --csv or run twice with --split and --csv.", file=sys.stderr)
            return 2
        # Run both splits with default CSV paths
        for s, default_csv in (("lite", "data/result/bench_results_lite.csv"), ("full", "data/result/bench_results_full.csv")):
            update_csv(
                csv_path=Path(default_csv),
                split=s,
                dataset_root=dataset_root,
                outputs_root=outputs_root,
                batch_size=args.batch_size,
                splits=args.splits,
                device_opt=args.device,
                overwrite=bool(args.overwrite),
                no_bom=bool(args.no_bom),
                no_backup=bool(args.no_backup),
                dry_run=bool(args.dry_run),
            )
        return 0
    else:
        if args.csv:
            csv_path = Path(args.csv)
        else:
            csv_path = Path("data/result/bench_results_lite.csv" if args.split == "lite" else "data/result/bench_results_full.csv")

        update_csv(
            csv_path=csv_path,
            split=args.split,
            dataset_root=dataset_root,
            outputs_root=outputs_root,
            batch_size=args.batch_size,
            splits=args.splits,
            device_opt=args.device,
            overwrite=bool(args.overwrite),
            no_bom=bool(args.no_bom),
            no_backup=bool(args.no_backup),
            dry_run=bool(args.dry_run),
        )
        return 0

if __name__ == "__main__":
    raise SystemExit(main())