"""Export all results under data/bench_results to CSV, supporting:
- Flatten original summary.json aggregation (default output: data/result/bench_results.csv)
- Aggregated exports based on story subsets:
  * lite: aggregate only the lite story list (output: data/result/bench_results_lite.csv)
  * full: aggregate only the full story list (output: data/result/bench_results_full.csv)

Implementation notes:
- Recursively find summary.json as run-root anchors;
- Parse the four front columns method, model, mode, timestamp from path (consistent with prior logic);
- "ALL" export: continue reading and dot-flatten summary.json;
- Subset export: do not use summary.json overall values; instead read run/metrics/*/story_results.json,
  compute numeric means only for stories within allowed_ids (semantics aligned with ResultManager._aggregate_story_metrics_mean),
  and output column names like metrics.<metric>.metrics.<key>;
- lite/full story lists:
  * Prefer dataset root inferred from config.yaml core.paths.dataset or dataset_path;
  * Load [StoryDataset.get_story_name_list(split)] from data/dataset/ViStory to obtain story sets;
  * If dataset unavailable: lite falls back to built-in LITE constants; full falls back to union of run story_results;
- Default UTF-8 BOM (--no-bom switches to no BOM);
- Files that fail are warned and skipped; if none found, header-only CSV is still generated.

Usage:
  python utils/export_bench_results_csv.py
  python utils/export_bench_results_csv.py --root data/bench_results
  python utils/export_bench_results_csv.py --out data/result/bench_results.csv
  python utils/export_bench_results_csv.py --no-bom
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import sys
import os
import yaml  # type: ignore
# Add the project root to the Python module search path.
# __file__ is the path of the current file.
# os.path.abspath(__file__) gets the absolute path.
# os.path.dirname() gets the directory name, which we use twice to get to the project root.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

FRONT_COLS = ["method", "model", "mode", "timestamp"]

LITE_METHODs=['Sora2']
FULL_METHODs=[]
# LITE_METHODs=['Seedream4','CharaConsist','QwenImageEdit2509','GPT4o','Gemini','NaiveBaseline','OmniGen2','NanoBanana','StoryDiffusion', 'StoryAdapter', 'StoryGen', 'UNO', 'TheaterGen', 'SeedStory', 'Vlogger', 'MovieAgent', 'AnimDirector', 'MMStoryAgent', 'MOKI', 'MorphicStudio', 'AIbrm', 'ShenBi', 'TypeMovie', 'DouBao']
# FULL_METHODs=['CharaConsist','QwenImageEdit2509','NaiveBaseline','OmniGen2','StoryDiffusion', 'StoryAdapter', 'StoryGen', 'UNO', 'TheaterGen', 'SeedStory', 'Vlogger', 'MovieAgent', 'AnimDirector', 'MMStoryAgent']
def iter_summary_files(root: Path) -> List[Path]:
    """Return all summary.json files under root (recursive), sorted by path string."""
    if not root.exists():
        return []
    return sorted(root.rglob("summary.json"), key=lambda p: str(p))


def parse_front_fields(root: Path, summary_path: Path) -> Dict[str, str]:
    """Parse method, model, mode, timestamp from the path relative to root."""
    rel = summary_path.relative_to(root)
    parent = rel.parent
    parts = list(parent.parts)
    timestamp = parts[-1] if parts else ""
    prefix = parts[:-1] if parts else []
    method = prefix[0] if len(prefix) >= 1 else ""
    model = prefix[1] if len(prefix) >= 2 else ""
    mode_parts = prefix[2:] if len(prefix) >= 3 else []
    mode = "/".join(mode_parts) if mode_parts else ""
    return {"method": method, "model": model, "mode": mode, "timestamp": timestamp}


def flatten_dict(d: Dict[str, Any], sep: str = ".") -> Dict[str, Any]:
    """Flatten a nested dict using dot notation. Lists are not flattened."""
    out: Dict[str, Any] = {}

    def _rec(obj: Any, prefix: str = "") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = f"{prefix}{sep}{k}" if prefix else str(k)
                _rec(v, key)
        else:
            out[prefix] = obj

    _rec(d, "")
    return out


def normalize_value(val: Any) -> Any:
    """Normalize values for CSV writing."""
    if val is None:
        return ""
    if isinstance(val, (str, int, float, bool)):
        return val
    if isinstance(val, (list, dict)):
        try:
            return json.dumps(val, ensure_ascii=False)
        except Exception:
            return str(val)
    try:
        return json.dumps(val, ensure_ascii=False)
    except Exception:
        return str(val)


def read_json_file(path: Path) -> Dict[str, Any]:
    """Read JSON file into dict; strip potential BOM; ensure top-level is dict."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8-sig")
    if text and text[0] == "\ufeff":
        text = text.lstrip("\ufeff")
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Top-level JSON must be an object (dict)")
    return data


def process_summary_file(root: Path, path: Path) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Return (front_fields, metrics_dict) for a summary.json file."""
    front = parse_front_fields(root, path)
    data = read_json_file(path)
    flat = flatten_dict(data, sep=".")
    metrics: Dict[str, Any] = {}
    for k, v in flat.items():
        if k in FRONT_COLS:
            continue
        metrics[k] = normalize_value(v)
    return front, metrics


def write_csv(
    out_path: Path,
    rows: List[Tuple[Dict[str, str], Dict[str, Any]]],
    metrics_keys: Set[str],
    encoding: str,
) -> None:
    """Write consolidated CSV with fixed front columns + sorted metric columns."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = FRONT_COLS + sorted(metrics_keys)
    with out_path.open("w", newline="", encoding=encoding) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for front, metrics in rows:
            row: Dict[str, Any] = dict(front)
            for k in metrics_keys:
                row[k] = metrics.get(k, "")
            writer.writerow(row)


# ----------------------------
# Subset Aggregation Utilities
# ----------------------------

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


def resolve_dataset_root_from_config(cfg: Dict[str, Any]) -> Path:
    """
    Resolve dataset root from config:
    - Prefer cfg['core']['paths']['dataset']
    - Else prefer cfg['dataset_path']
    - Else default to data/dataset
    """
    ds = (
        (((cfg or {}).get("core") or {}).get("paths") or {}).get("dataset")
        or (cfg or {}).get("dataset_path")
        or "data/dataset"
    )
    return Path(ds)


def try_get_story_ids_from_dataset(dataset_root: Path, split: str) -> Optional[Set[str]]:
    """
    Try to load story id list from data/dataset/ViStory using StoryDataset.
    Return a set of ids or None if dataset not accessible.
    """
    dataset_dir = (dataset_root / "ViStory")
    if not dataset_dir.is_dir():
        return None
    # try:
    # Late import to avoid hard dependency if module path not resolvable in some environments
    from vistorybench.dataset_loader.dataset_load import StoryDataset  # type: ignore
    ds = StoryDataset(str(dataset_dir))
    ids = ds.get_story_name_list(split=split)
    return set(map(str, ids))
    # except Exception:
        # return None


def derive_story_ids_from_run_story_results(run_dir: Path) -> Set[str]:
    """
    Derive union of story ids from all metrics/*/story_results.json under a run directory.
    """
    metrics_root = run_dir / "metrics"
    ids: Set[str] = set()
    if not metrics_root.is_dir():
        return ids
    try:
        for child in metrics_root.iterdir():
            if not child.is_dir():
                continue
            story_file = child / "story_results.json"
            if not story_file.exists():
                continue
            try:
                data = read_json_file(story_file)
                # Keys are story ids
                for sid in data.keys():
                    ids.add(str(sid))
            except Exception:
                # Skip corrupted metric files silently
                continue
    except Exception:
        pass
    return ids


def compute_mean_numeric_metrics_for_story_results(
    story_results: Dict[str, Any], allowed_ids: Set[str]
) -> Dict[str, float]:
    """
    Given metrics/*/story_results.json loaded as dict and an allowed id set,
    compute mean for numeric values under each story's 'metrics' (or top-level dict).
    Align semantics with ResultManager._aggregate_story_metrics_mean, but filtered by allowed_ids.
    """
    if not allowed_ids:
        return {}
    numeric: Dict[str, List[float]] = {}
    for sid, obj in story_results.items():
        if allowed_ids and str(sid) not in allowed_ids:
            continue
        metrics = None
        if isinstance(obj, dict):
            m = obj.get("metrics") if isinstance(obj.get("metrics"), dict) else None
            metrics = m if m is not None else (obj if isinstance(obj, dict) else None)
        if not isinstance(metrics, dict):
            continue
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                numeric.setdefault(k, []).append(float(v))
    return {k: (sum(vals) / len(vals) if vals else 0.0) for k, vals in numeric.items()}


def aggregate_metrics_for_run_subset(run_dir: Path, allowed_ids: Set[str]) -> Dict[str, Any]:
    """
    For a run directory (parent of summary.json), aggregate metrics for the given story subset.
    Return a flattened metrics map with keys like: metrics.<metric>.metrics.<key>.
    """
    metrics_root = run_dir / "metrics"
    flat: Dict[str, Any] = {}
    if not metrics_root.is_dir():
        return flat
    try:
        for metric_dir in sorted([p for p in metrics_root.iterdir() if p.is_dir()], key=lambda p: p.name):
            metric_name = metric_dir.name
            story_file = metric_dir / "story_results.json"
            if not story_file.exists():
                # Cannot aggregate by subset, skip this metric
                continue
            try:
                story_results = read_json_file(story_file)
                if not isinstance(story_results, dict):
                    continue
                mean_map = compute_mean_numeric_metrics_for_story_results(story_results, allowed_ids)
                for k, v in mean_map.items():
                    flat[f"metrics.{metric_name}.metrics.{k}"] = v
            except Exception:
                # Single metric aggregation failed, does not affect other metrics
                continue
    except Exception:
        return flat
    return flat


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export ViStoryBench bench_results to consolidated CSV (all, lite-subset, full-subset)."
    )
    parser.add_argument(
        "--root",
        default="data/bench_results",
        help="Root directory to search recursively for summary.json (default: data/bench_results)",
    )
    parser.add_argument(
        "--out",
        default="data/result/bench_results.csv",
        help="Output CSV path for ALL (flatten summary.json, default: data/result/bench_results.csv)",
    )
    parser.add_argument(
        "--out-lite",
        default="data/result/bench_results_lite.csv",
        help="Output CSV path for LITE subset (default: data/result/bench_results_lite.csv)",
    )
    parser.add_argument(
        "--out-full",
        default="data/result/bench_results_full.csv",
        help="Output CSV path for FULL subset (default: data/result/bench_results_full.csv)",
    )
    parser.add_argument(
        "--no-bom",
        action="store_true",
        help="Use UTF-8 without BOM (default uses UTF-8 BOM for Excel)",
    )
    args = parser.parse_args(argv)

    root = Path(args.root)
    out_all = Path(args.out)
    out_lite = Path(args.out_lite)
    out_full = Path(args.out_full)
    env_no_bom = (os.environ.get("NO_BOM") or "").lower() in ("1", "true", "yes")
    no_bom_effective = bool(args.no_bom or env_no_bom)
    encoding = "utf-8" if no_bom_effective else "utf-8-sig"

    files = iter_summary_files(root)

    # Prepare rows/keys for three outputs
    rows_all: List[Tuple[Dict[str, str], Dict[str, Any]]] = []
    keys_all: Set[str] = set()
    rows_lite: List[Tuple[Dict[str, str], Dict[str, Any]]] = []
    keys_lite: Set[str] = set()
    rows_full: List[Tuple[Dict[str, str], Dict[str, Any]]] = []
    keys_full: Set[str] = set()

    succ_all = fail_all = 0
    succ_lite = fail_lite = 0
    succ_full = fail_full = 0

    if not root.exists():
        print(f"Note: Root directory does not exist: {root}. Will still generate header-only CSV.", file=sys.stdout)

    # Resolve dataset root to build global story id lists
    cfg = load_yaml_config(Path("config.yaml"))
    dataset_root = resolve_dataset_root_from_config(cfg)

    # Try derive global sets from dataset folder
    global_lite_ids: Optional[Set[str]] = try_get_story_ids_from_dataset(dataset_root, "lite")
    global_full_ids: Optional[Set[str]] = try_get_story_ids_from_dataset(dataset_root, "full")

    for fp in files:
        # Front fields once
        try:
            front, metrics_all = process_summary_file(root, fp)
            rows_all.append((front, metrics_all))
            keys_all.update(metrics_all.keys())
            succ_all += 1
        except Exception as e:
            fail_all += 1
            print(f"Warning: Failed to parse summary; skipped: {fp}; reason: {e}", file=sys.stderr)
            # Even if it fails, still try to aggregate the lite/full run (story_results might be normal)
            try:
                front = parse_front_fields(root, fp)
            except Exception:
                # If front cannot be parsed, skip lite/full as well
                continue

        # Subset aggregation
        run_dir = fp.parent

        # Parse the story id set of the run (for checking and warning only, does not block output)
        try:
            run_ids = derive_story_ids_from_run_story_results(run_dir)
        except Exception as e:
            print(f"Warning: Failed to parse run's story id set (continuing): {run_dir}; reason: {e}", file=sys.stderr)
            run_ids = set()

        # Allowed sets: lite prioritizes global_lite_ids, otherwise falls back to run_ids; full prioritizes global_full_ids, otherwise falls back to run_ids
        lite_allowed: Set[str] = global_lite_ids if global_lite_ids is not None else run_ids
        full_allowed: Set[str] = global_full_ids if global_full_ids is not None else run_ids

        method_name = front.get("method", "")

        # LITE output: output if the method is in the list; only warn if the story ID set is inconsistent with the LITE expectation
        if method_name in LITE_METHODs:
            if len(global_lite_ids - run_ids)>0:
                missing = sorted(global_lite_ids - run_ids)
                extra = sorted(run_ids - global_lite_ids)
                print(
                    f"Warning: LITE story ID set differs from expectation; still outputting: {run_dir}/summary.json\n"
                    f"  Missing {len(missing)}: {missing[:10]}{' ...' if len(missing) > 10 else ''}\n"
                    f"  Extra {len(extra)}: {extra[:10]}{' ...' if len(extra) > 10 else ''}",
                    file=sys.stderr,
                )
            try:
                metrics_lite = aggregate_metrics_for_run_subset(run_dir, lite_allowed)
                rows_lite.append((front, metrics_lite))
                keys_lite.update(metrics_lite.keys())
                succ_lite += 1
            except Exception as e:
                fail_lite += 1
                print(f"Warning: LITE aggregation failed; skipped: {run_dir}; reason: {e}", file=sys.stderr)

        # FULL output: output if the method is in the list; only warn if the story ID set is inconsistent with the FULL expectation
        if method_name in FULL_METHODs:
            if len(global_full_ids - run_ids) > 0:
                missing_f = sorted(global_full_ids - run_ids)
                extra_f = sorted(run_ids - global_full_ids)
                print(
                    f"Warning: FULL story ID set differs from expectation; still outputting: {run_dir}/summary.json\n"
                    f"  Missing {len(missing_f)}: {missing_f[:10]}{' ...' if len(missing_f) > 10 else ''}\n"
                    f"  Extra {len(extra_f)}: {extra_f[:10]}{' ...' if len(extra_f) > 10 else ''}",
                    file=sys.stderr,
                )
            try:
                metrics_full = aggregate_metrics_for_run_subset(run_dir, full_allowed)
                rows_full.append((front, metrics_full))
                keys_full.update(metrics_full.keys())
                succ_full += 1
            except Exception as e:
                fail_full += 1
                print(f"Warning: FULL aggregation failed; skipped: {run_dir}; reason: {e}", file=sys.stderr)

    # LITE
    try:
        write_csv(out_lite, rows_lite, keys_lite, encoding=encoding)
    except Exception as e:
        print(f"Error: failed to write CSV (LITE): {out_lite}; reason: {e}", file=sys.stderr)

    # FULL
    try:
        write_csv(out_full, rows_full, keys_full, encoding=encoding)
    except Exception as e:
        print(f"Error: failed to write CSV (FULL): {out_full}; reason: {e}", file=sys.stderr)

    # Summary logs
    if not files:
        print(
            f"Done: No summary.json found under {root}."
            f"Created header-only CSVs:\n"
            f"  ALL: {out_all}\n  LITE: {out_lite}\n  FULL: {out_full}",
            file=sys.stdout,
        )
    else:
        print(
            "Done:\n"
            f"  LITE: success {succ_lite}, failed {fail_lite}. Output: {out_lite}\n"
            f"  FULL: success {succ_full}, failed {fail_full}. Output: {out_full}",
            file=sys.stdout,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())