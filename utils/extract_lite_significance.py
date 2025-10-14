#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
On the ViStory-lite subset, automatically discover all available metrics and perform pairwise significance tests on all methods,
outputting a significance matrix for each metric, as well as a comprehensive significance rate across all metrics (average significance coverage),
to demonstrate the overall effectiveness of the evaluation system in a rebuttal.

Major enhancements (compared to previous versions):
1) Automatic metric discovery: No longer limited to 3-4 fixed metrics. The script iterates through the selected run's metrics/*/story_results.json for each method,
   automatically extracting all numerical sub-metrics (e.g., prompt_align.scene, prompt_align.camera, cids.single_character_action, etc.).
2) Full metric test output: For each discovered metric, generate three types of matrices: p-value, significance (0/1), and effect size (Cohen's d),
   and add a new significance matrix corrected by FDR (BH).
3) Global comprehensive metrics: Across all metrics and all method pairs, calculate a "significance coverage matrix" (pairwise rate),
   as well as a global overall significance rate (raw/FDR), and output a JSON/CSV summary for citation in papers/defenses.
4) Still maintains per-story mean output (means_lite.csv), but this file is for reference only.

CLI Example:
  - Automatically discover all metrics and output full significance tests and comprehensive significance:
    python utils/extract_lite_significance.py
  - Specify methods and language (the rest remain auto-discovered):
    python utils/extract_lite_significance.py --methods uno storyadapter storydiffusion --language en
  - Adjust test threshold and number of permutations (when SciPy is not available):
    python utils/extract_lite_significance.py --alpha 0.01 --permutations 5000

Output directory structure:
  data/result/significance/lite_{timestamp}/
    ├── selected_runs.json
    ├── means_lite.csv
    ├── pvalues_{metric}.csv
    ├── signif_{metric}.csv
    ├── signif_fdr_{metric}.csv
    ├── effect_{metric}.csv
    ├── global_significance_rate_pairwise_raw.csv    # Significance coverage matrix across all metrics (original alpha)
    ├── global_significance_rate_pairwise_fdr.csv    # Significance coverage matrix across all metrics (FDR-BH)
    ├── metric_significance_coverage_raw.csv         # Significant proportion of each metric across all method pairs (original)
    ├── metric_significance_coverage_fdr.csv         # Significant proportion of each metric across all method pairs (FDR)
    └── global_significance_overall.json             # Global overall significance rate (raw/FDR) and statistical summary

Note:
- Welch's t-test (equal_var=False) is preferred when SciPy is available; falls back to a two-sided permutation test otherwise.
- Significance tests only use story IDs where both methods coexist; if the number of common stories is < 2, the method pair is not included in the statistics for that metric.
- FDR uses the Benjamini-Hochberg (BH) procedure, correcting based on the set of p-values for all method pairs within a metric.

"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# Optional dependency: SciPy, fallback to permutation test if unavailable
try:
    from scipy import stats as _scipy_stats  # type: ignore
except Exception:
    _scipy_stats = None

# YAML (for reading dataset path from config.yaml)
try:
    import yaml  # type: ignore
except Exception:
    yaml = None

# Inject project root path for easy import of vistorybench.dataset_loader
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# ----------------------------
# Constants and Data Structures
# ----------------------------

DEFAULT_LITE_METHODS: List[str] = [
    "animdirector", "doubao", "gpt4o", "moki", "movieagent", "seedstory",
    "storyadapter", "storygen", "uno", "xunfeihuiying", "bairimeng_ai", "gemini",
    "mmstoryagent", "morphic_studio", "naive_baseline", "shenbimaliang",
    "storydiffusion", "theatergen", "vlogger"
]

FRONT_COLS = ["method", "mode", "language", "timestamp"]


@dataclass(frozen=True)
class RunInfo:
    method: str
    mode: str
    language: str
    timestamp: str
    run_dir: Path


# ----------------------------
# Basic Utility Functions
# ----------------------------

def read_json_file(path: Path) -> Dict[str, Any]:
    """Read a JSON file, compatible with BOM. The top level must be a dict."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8-sig")
    if text and text[0] == "\ufeff":
        text = text.lstrip("\ufeff")
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Top-level JSON must be an object (dict): {path}")
    return data


def load_yaml_config(config_path: Path = Path("config.yaml")) -> Dict[str, Any]:
    """Load YAML configuration; return an empty dict if unavailable or parsing fails."""
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
    Infer dataset root path from config:
    - Prioritize cfg['core']['paths']['dataset']
    - Then cfg['dataset_path']
    - Fallback to 'data/dataset'
    """
    ds = (
        (((cfg or {}).get("core") or {}).get("paths") or {}).get("dataset")
        or (cfg or {}).get("dataset_path")
        or "data/dataset"
    )
    return Path(ds)


def try_get_story_ids_from_dataset(dataset_root: Path, split: str) -> Optional[Set[str]]:
    """
    Try to load the story ID set for the specified split from data/dataset/ViStory, standardized to a two-digit string (zfill(2)).
    """
    dataset_dir = (dataset_root / "ViStory")
    if not dataset_dir.is_dir():
        return None
    try:
        from vistorybench.dataset_loader.dataset_load import StoryDataset  # type: ignore
    except Exception:
        return None
    try:
        ds = StoryDataset(str(dataset_dir))
        ids = ds.get_story_name_list(split=split)
        return set(map(lambda s: str(s).zfill(2), ids))
    except Exception:
        return None


def list_modes(results_root: Path, method: str) -> List[str]:
    mdir = results_root / method
    if not mdir.is_dir():
        return []
    return sorted([d.name for d in mdir.iterdir() if d.is_dir()])


def list_timestamps(results_root: Path, method: str, mode: str, language: str) -> List[str]:
    run_root = results_root / method / mode / language
    if not run_root.is_dir():
        return []
    return sorted([d.name for d in run_root.iterdir() if d.is_dir()], reverse=True)


def derive_story_ids_from_run_story_results(run_dir: Path) -> Set[str]:
    """
    Infer the story set of the run from the union of keys in run_dir/metrics/*/story_results.json (two-digit string).
    """
    metrics_root = run_dir / "metrics"
    ids: Set[str] = set()
    if not metrics_root.is_dir():
        return ids
    for child in metrics_root.iterdir():
        if not child.is_dir():
            continue
        story_file = child / "story_results.json"
        if not story_file.exists():
            continue
        try:
            data = read_json_file(story_file)
            for sid in data.keys():
                ids.add(str(sid).zfill(2))
        except Exception:
            continue
    return ids


def get_allowed_story_ids_for_run(
    dataset_lite_ids: Optional[Set[str]],
    run_dir: Path,
) -> Set[str]:
    """
    Get the allowed lite story set for this run:
    - Prioritize using dataset_lite_ids
    - If not available, fall back to the union of stories from metrics/*/story_results.json under this run
    """
    if dataset_lite_ids and len(dataset_lite_ids) > 0:
        return set(sorted(dataset_lite_ids))
    return derive_story_ids_from_run_story_results(run_dir)


def run_has_any_metric_with_story_results(run_dir: Path) -> bool:
    """
    Check if the run has at least one metrics/*/story_results.json.
    """
    metrics_root = run_dir / "metrics"
    if not metrics_root.is_dir():
        return False
    for child in metrics_root.iterdir():
        if child.is_dir() and (child / "story_results.json").exists():
            return True
    return False


def select_latest_valid_run_for_method(
    results_root: Path,
    method: str,
    preferred_language: Optional[str],
    dataset_lite_ids: Optional[Set[str]],
) -> Optional[RunInfo]:
    """
    Select the "latest valid run" for a method:
    - Mode: Iterate through all modes under the method
    - Language: If preferred_language is given, prioritize it, otherwise prioritize 'en', then 'ch'
    - Timestamp: Select the latest run that has at least one metrics/*/story_results.json
    """
    modes = list_modes(results_root, method)
    if not modes:
        return None
    lang_order: List[str] = [preferred_language] if preferred_language else ["en", "ch"]
    for lang in lang_order:
        for mode in modes:
            for ts in list_timestamps(results_root, method, mode, lang):
                run_dir = results_root / method / mode / lang / ts
                allowed_ids = get_allowed_story_ids_for_run(dataset_lite_ids, run_dir)
                if not allowed_ids:
                    continue
                if run_has_any_metric_with_story_results(run_dir):
                    return RunInfo(method=method, mode=mode, language=lang, timestamp=ts, run_dir=run_dir)
    return None


# ----------------------------
# Metric Data Reading/Discovery
# ----------------------------

def load_story_value_map(
    run_dir: Path,
    metric_name: str,
    key: str,
    allowed_ids: Set[str],
) -> Dict[str, float]:
    """
    Load per-story numerical values for a specified key from run_dir/metrics/{metric_name}/story_results.json (only for allowed_ids).
    Returns a sid -> float map. Ignores non-numeric or missing values.
    """
    out: Dict[str, float] = {}
    story_file = run_dir / "metrics" / metric_name / "story_results.json"
    if not story_file.exists():
        return out
    try:
        data = read_json_file(story_file)
    except Exception:
        return out
    if not isinstance(data, dict):
        return out
    for sid in sorted(allowed_ids):
        obj = data.get(sid) or data.get(int(sid))
        if isinstance(obj, dict):
            metrics = obj.get("metrics") if isinstance(obj.get("metrics"), dict) else obj
            v = metrics.get(key) if isinstance(metrics, dict) else None
            if isinstance(v, (int, float)):
                out[sid] = float(v)
    return out


def discover_metric_specs_across_selected(
    selected: Dict[str, RunInfo],
    dataset_lite_ids: Optional[Set[str]],
) -> List[Tuple[str, str]]:
    """
    Iterate through all selected runs, automatically discover all numerical sub-metrics, and return a (metric_name, key) list (deduplicated/sorted).
    """
    found: Dict[str, Set[str]] = {}
    for _, ri in selected.items():
        run_dir = ri.run_dir
        allowed_ids = get_allowed_story_ids_for_run(dataset_lite_ids, run_dir)
        metrics_root = run_dir / "metrics"
        if not metrics_root.is_dir():
            continue
        for metric_dir in sorted([p for p in metrics_root.iterdir() if p.is_dir()], key=lambda p: p.name):
            metric_name = metric_dir.name
            story_file = metric_dir / "story_results.json"
            if not story_file.exists():
                continue
            try:
                sr = read_json_file(story_file)
            except Exception:
                continue
            if not isinstance(sr, dict):
                continue
            keys_set = found.setdefault(metric_name, set())
            # Collect numeric keys that exist in allowed_ids
            for sid, obj in sr.items():
                sid2 = str(sid).zfill(2)
                if allowed_ids and sid2 not in allowed_ids:
                    continue
                if not isinstance(obj, dict):
                    continue
                metrics = obj.get("metrics") if isinstance(obj.get("metrics"), dict) else obj
                if not isinstance(metrics, dict):
                    continue
                for k, v in metrics.items():
                    if metric_name=='prompt_align' and k=='single_character_action':
                        # prompt_align.single_character_action is a string, skip it
                        continue
                    if isinstance(v, (int, float)):
                        keys_set.add(str(k))
    specs: List[Tuple[str, str]] = []
    for m, ks in found.items():
        for k in sorted(ks):
            specs.append((m, k))
    # Sort: by metric_name then key
    specs.sort(key=lambda t: (t[0], t[1]))
    return specs


# ----------------------------
# Statistical Calculations
# ----------------------------

def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def sample_std(xs: List[float]) -> float:
    n = len(xs)
    if n <= 1:
        return 0.0
    mu = mean(xs)
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    return math.sqrt(max(var, 0.0))


def cohen_d(a: List[float], b: List[float]) -> float:
    """
    Cohen's d effect size: (mean_a - mean_b) / s_pooled
    s_pooled = sqrt(((n_a-1)*s_a^2 + (n_b-1)*s_b^2) / (n_a + n_b - 2))
    Returns NaN if the sample is insufficient.
    """
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    ma, mb = mean(a), mean(b)
    sa, sb = sample_std(a), sample_std(b)
    denom = ((na - 1) * (sa ** 2) + (nb - 1) * (sb ** 2))
    denom_n = na + nb - 2
    if denom_n <= 0 or denom <= 0:
        return float("nan")
    s_pooled = math.sqrt(denom / denom_n)
    if s_pooled == 0.0:
        return float("nan")
    return (ma - mb) / s_pooled


def welch_t_pvalue(a: List[float], b: List[float]) -> float:
    """If SciPy is available, use Welch's t-test two-sided p-value; otherwise, return NaN."""
    if _scipy_stats is None:
        return float("nan")
    try:
        tval, pval = _scipy_stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        return float(pval)
    except Exception:
        return float("nan")


def permutation_test_pvalue(
    a: List[float],
    b: List[float],
    permutations: int = 2000,
    seed: int = 42,
) -> float:
    """
    Permutation test two-sided p-value:
    - The statistic is the mean difference |mean(a) - mean(b)|
    - Randomly shuffle the labels on the combined sample, assign them to two groups, and count the proportion of extreme cases
    """
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return float("nan")
    obs = abs(mean(a) - mean(b))
    combined = list(a) + list(b)
    n = len(combined)
    rng = random.Random(seed)
    count = 0
    denom = permutations
    for _ in range(permutations):
        rng.shuffle(combined)
        a_perm = combined[:na]
        b_perm = combined[na:n]
        stat = abs(mean(a_perm) - mean(b_perm))
        if stat >= obs - 1e-12:
            count += 1
    return (count + 1) / (denom + 1)


def pairwise_significance(
    methods: List[str],
    method_to_valmap: Dict[str, Dict[str, float]],
    alpha: float,
    permutations: int,
    prefer_welch: bool = True,
) -> Tuple[List[List[float]], List[List[int]], List[List[float]]]:
    """
    Perform pairwise tests on a given set of methods for a certain metric:
    Returns (pvals_matrix, signif_matrix(0/1), effect_matrix)
    Note: Each pair of methods only uses the values of common story IDs; if the number of common stories is < 2, then p=NaN, signif=0, effect=NaN.
    """
    m = len(methods)
    pvals = [[float("nan")] * m for _ in range(m)]
    signif = [[0] * m for _ in range(m)]
    effects = [[float("nan")] * m for _ in range(m)]
    for i in range(m):
        mi = methods[i]
        for j in range(m):
            mj = methods[j]
            if i == j:
                pvals[i][j] = 1.0
                signif[i][j] = 0
                effects[i][j] = 0.0
                continue
            vi = method_to_valmap.get(mi, {})
            vj = method_to_valmap.get(mj, {})
            common = sorted(set(vi.keys()) & set(vj.keys()))
            if len(common) < 2:
                pvals[i][j] = float("nan")
                signif[i][j] = 0
                effects[i][j] = float("nan")
                continue
            a = [vi[sid] for sid in common]
            b = [vj[sid] for sid in common]
            p = welch_t_pvalue(a, b) if prefer_welch else float("nan")
            if math.isnan(p):
                p = permutation_test_pvalue(a, b, permutations=permutations, seed=42)
            e = cohen_d(a, b)
            pvals[i][j] = p
            signif[i][j] = 1 if (not math.isnan(p) and p < alpha) else 0
            effects[i][j] = e
    return pvals, signif, effects


def bh_significance_from_pvals(
    pvals: List[List[float]],
    alpha: float,
) -> List[List[int]]:
    """
    Apply Benjamini-Hochberg (BH) FDR correction to a p-value matrix for one metric, and output a significance (0/1) matrix.
    Only considers the upper triangular elements where i < j; NaN is skipped.
    """
    msize = len(pvals)
    pairs: List[Tuple[float, Tuple[int, int]]] = []
    for i in range(msize):
        for j in range(i + 1, msize):
            p = pvals[i][j]
            if p is None or math.isnan(p):
                continue
            pairs.append((float(p), (i, j)))
    M = len(pairs)
    signif = [[0] * msize for _ in range(msize)]
    if M == 0:
        return signif
    pairs.sort(key=lambda t: t[0])  # Sort by p-value in ascending order
    # Find the largest k such that p_(k) <= alpha * k / M
    k_star = 0
    for idx, (p, _) in enumerate(pairs, start=1):
        thresh = alpha * idx / M
        if p <= thresh:
            k_star = idx
    # Mark the first k_star as significant
    for idx in range(k_star):
        _, (i, j) = pairs[idx]
        signif[i][j] = 1
        signif[j][i] = 1
    # Keep the diagonal as 0
    return signif


# ----------------------------
# I/O
# ----------------------------

def write_matrix_csv(
    out_path: Path,
    methods: List[str],
    matrix: List[List[Any]],
) -> None:
    """Write a matrix CSV, with method names in the first row and column."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([""] + methods)
        for i, row in enumerate(matrix):
            writer.writerow([methods[i]] + row)


def write_means_csv(
    out_path: Path,
    means_rows: List[Dict[str, Any]],
) -> None:
    """Write a summary CSV of the lite mean values for each method."""
    import csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["method", "mode", "language", "timestamp", "metric", "mean"]
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in means_rows:
            writer.writerow(r)


def write_selected_runs_json(
    out_path: Path,
    selected: Dict[str, RunInfo],
) -> None:
    """Write the selected run information to a JSON file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        m: {
            "method": ri.method,
            "mode": ri.mode,
            "language": ri.language,
            "timestamp": ri.timestamp,
            "run_dir": str(ri.run_dir),
        }
        for m, ri in selected.items()
    }
    out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_global_overall_json(
    out_path: Path,
    summary: Dict[str, Any],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


# ----------------------------
# Comprehensive Significance Rate (Across All Metrics)
# ----------------------------

def compute_pairwise_significance_rate(
    methods: List[str],
    per_metric_pvals: Dict[str, List[List[float]]],
    per_metric_signif: Dict[str, List[List[int]]],
) -> List[List[float]]:
    """
    For all metrics, calculate the "significance coverage rate" for method pair (i,j): number of significant results / number of participations.
    Number of participations = number of non-NaN p-values for (i,j) on this metric; number of significant results = number of corresponding signif==1.
    """
    m = len(methods)
    rate = [[float("nan")] * m for _ in range(m)]
    for i in range(m):
        for j in range(m):
            if i == j:
                rate[i][j] = 0.0
                continue
            total = 0
            hit = 0
            for mk, pmat in per_metric_pvals.items():
                smat = per_metric_signif.get(mk)
                if smat is None:
                    continue
                p = pmat[i][j]
                if not (p is None or math.isnan(p)):
                    total += 1
                    if smat[i][j] == 1:
                        hit += 1
            rate[i][j] = (hit / total) if total > 0 else float("nan")
    return rate


def coverage_per_metric(
    methods: List[str],
    signif: List[List[int]],
    pvals: List[List[float]],
) -> float:
    """
    Calculate the proportion of significant pairs for a given metric across all method pairs (number of significant pairs / number of participating pairs).
    """
    m = len(methods)
    total = 0
    hit = 0
    for i in range(m):
        for j in range(i + 1, m):
            p = pvals[i][j]
            if not (p is None or math.isnan(p)):
                total += 1
                if signif[i][j] == 1:
                    hit += 1
    return (hit / total) if total > 0 else float("nan")


def write_metric_coverage_csv(
    out_path: Path,
    rows: List[Tuple[str, float]],
) -> None:
    """Write a CSV of the significance coverage proportion for each metric: metric_id, coverage"""
    import csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["metric_id", "significance_coverage"])
        for mk, cov in rows:
            w.writerow([mk, cov])


# ----------------------------
# Main Process
# ----------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auto-discover all metrics on ViStory-lite and compute cross-method significance with global summaries.")
    p.add_argument("--results_root", default="data/bench_results", help="Root directory for evaluation results")
    p.add_argument("--dataset_root", default="", help="Dataset root directory (default: inferred from config.yaml or fallback to data/dataset)")
    p.add_argument("--split", default="lite", choices=["lite"], help="Currently only supports the lite subset")
    p.add_argument("--methods", nargs="*", default=DEFAULT_LITE_METHODS, help="List of methods (default: predefined LITE method set)")
    p.add_argument("--language", default="", help="Preferred language (default: prioritize en then ch)")
    p.add_argument("--metrics", nargs="*", default=[], help="Optional: explicitly specify a list of metrics (in the form of metric_name.key); if empty, automatically discover all available metrics")
    p.add_argument("--alpha", type=float, default=0.05, help="Significance test threshold alpha")
    p.add_argument("--permutations", type=int, default=2000, help="Number of iterations for permutation test (enabled when SciPy is not available)")
    p.add_argument("--out_dir", default="", help="Output directory (default: data/result/significance/lite_{timestamp})")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    results_root = Path(args.results_root)
    if not results_root.exists():
        print(f"Error: Results root directory does not exist: {results_root}", file=sys.stderr)
        return 2

    # Output directory
    ts_label = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir or f"data/result/significance/lite_{ts_label}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dataset lite list
    cfg = load_yaml_config(Path("config.yaml"))
    dataset_root = Path(args.dataset_root) if args.dataset_root else resolve_dataset_root_from_config(cfg)
    dataset_lite_ids = try_get_story_ids_from_dataset(dataset_root, args.split)
    if dataset_lite_ids is None:
        print(f"Warning: Could not load {args.split} list from dataset, will fall back to the union of story_results from the run", file=sys.stderr)

    # Method selection
    methods = args.methods or []
    if not methods:
        print("Error: No method list specified", file=sys.stderr)
        return 2

    # Select a run for each method
    selected: Dict[str, RunInfo] = {}
    for method in methods:
        ri = select_latest_valid_run_for_method(
            results_root=results_root,
            method=method,
            preferred_language=(args.language or None),
            dataset_lite_ids=dataset_lite_ids,
        )
        if ri is None:
            print(f"Warning: No available run found for method '{method}', skipped", file=sys.stderr)
            continue
        selected[method] = ri
    if not selected:
        print("Error: No methods were selected", file=sys.stderr)
        return 2

    # Write selected run information
    write_selected_runs_json(out_dir / "selected_runs.json", selected)

    # Automatically discover metrics (if not explicitly specified)
    if args.metrics:
        metric_specs: List[Tuple[str, str]] = []
        for mk in (args.metrics or []):
            mk = str(mk).strip()
            if "." not in mk:
                print(f"Warning: Metric '{mk}' is not in 'metric.key' format, skipped", file=sys.stderr)
                continue
            m, k = mk.split(".", 1)
            metric_specs.append((m.strip(), k.strip()))
    else:
        metric_specs = discover_metric_specs_across_selected(selected, dataset_lite_ids)
        if not metric_specs:
            print("Error: No numerical metrics were automatically discovered", file=sys.stderr)
            return 2
        print(f"Info: Automatically discovered {len(metric_specs)} metrics, for example, the first 10: {[f'{m}.{k}' for (m,k) in metric_specs[:10]]}")

    # Collect mean values (mean of per-story values)
    means_rows: List[Dict[str, Any]] = []

    # Matrix accumulation per metric (for global summary)
    methods_list = sorted(selected.keys())
    per_metric_pvals_raw: Dict[str, List[List[float]]] = {}
    per_metric_signif_raw: Dict[str, List[List[int]]] = {}
    per_metric_signif_fdr: Dict[str, List[List[int]]] = {}

    # Process each metric
    for (metric_name, key) in metric_specs:
        mk_id = f"{metric_name}.{key}"

        # Collect method -> sid->value
        method_valmaps: Dict[str, Dict[str, float]] = {}
        for method in methods_list:
            ri = selected[method]
            allowed_ids = get_allowed_story_ids_for_run(dataset_lite_ids, ri.run_dir)
            valmap = load_story_value_map(ri.run_dir, metric_name, key, allowed_ids)
            method_valmaps[method] = valmap
            vals = list(valmap.values())
            mu = mean(vals) if vals else float("nan")
            means_rows.append({
                "method": method,
                "mode": ri.mode,
                "language": ri.language,
                "timestamp": ri.timestamp,
                "metric": mk_id,
                "mean": mu,
            })

        # Significance test (original alpha)
        pvals, signif_raw, effects = pairwise_significance(
            methods=methods_list,
            method_to_valmap=method_valmaps,
            alpha=float(args.alpha),
            permutations=int(args.permutations),
            prefer_welch=True,
        )
        # FDR (BH) corrected significance matrix
        signif_fdr = bh_significance_from_pvals(pvals, alpha=float(args.alpha))

        # Write matrices
        write_matrix_csv(out_dir / f"pvalues_{mk_id.replace('.', '_')}.csv", methods_list, pvals)
        write_matrix_csv(out_dir / f"signif_{mk_id.replace('.', '_')}.csv", methods_list, signif_raw)
        write_matrix_csv(out_dir / f"signif_fdr_{mk_id.replace('.', '_')}.csv", methods_list, signif_fdr)
        write_matrix_csv(out_dir / f"effect_{mk_id.replace('.', '_')}.csv", methods_list, effects)

        # Accumulate to global
        per_metric_pvals_raw[mk_id] = pvals
        per_metric_signif_raw[mk_id] = signif_raw
        per_metric_signif_fdr[mk_id] = signif_fdr

    # Write mean summary
    write_means_csv(out_dir / "means_lite.csv", means_rows)

    # Calculate the significance coverage rate of method pairs "across all metrics"
    rate_pairwise_raw = compute_pairwise_significance_rate(methods_list, per_metric_pvals_raw, per_metric_signif_raw)
    rate_pairwise_fdr = compute_pairwise_significance_rate(methods_list, per_metric_pvals_raw, per_metric_signif_fdr)

    write_matrix_csv(out_dir / "global_significance_rate_pairwise_raw.csv", methods_list, rate_pairwise_raw)
    write_matrix_csv(out_dir / "global_significance_rate_pairwise_fdr.csv", methods_list, rate_pairwise_fdr)

    # Calculate the significance coverage proportion for each metric (raw / fdr)
    metric_cov_rows_raw: List[Tuple[str, float]] = []
    metric_cov_rows_fdr: List[Tuple[str, float]] = []
    for mk_id, pmat in per_metric_pvals_raw.items():
        cov_raw = coverage_per_metric(methods_list, per_metric_signif_raw[mk_id], pmat)
        cov_fdr = coverage_per_metric(methods_list, per_metric_signif_fdr[mk_id], pmat)
        metric_cov_rows_raw.append((mk_id, cov_raw))
        metric_cov_rows_fdr.append((mk_id, cov_fdr))
    write_metric_coverage_csv(out_dir / "metric_significance_coverage_raw.csv", metric_cov_rows_raw)
    write_metric_coverage_csv(out_dir / "metric_significance_coverage_fdr.csv", metric_cov_rows_fdr)

    # Global overall significance rate (aggregated over all method pairs and all metrics)
    def overall_rate(rate_matrix: List[List[float]]) -> float:
        vals: List[float] = []
        m = len(rate_matrix)
        for i in range(m):
            for j in range(i + 1, m):
                r = rate_matrix[i][j]
                if not (r is None or math.isnan(r)):
                    vals.append(float(r))
        return (sum(vals) / len(vals)) if vals else float("nan")

    overall_raw = overall_rate(rate_pairwise_raw)
    overall_fdr = overall_rate(rate_pairwise_fdr)

    write_global_overall_json(out_dir / "global_significance_overall.json", {
        "methods": methods_list,
        "n_metrics": len(per_metric_pvals_raw),
        "alpha": float(args.alpha),
        "overall_significance_rate_raw": overall_raw,
        "overall_significance_rate_fdr": overall_fdr,
        "notes": "rates are averaged over all method pairs; per-metric coverage CSVs provide individual metric coverage proportions."
    })

    print("Done: Output saved to", out_dir)
    print("Overall conclusion (can be used directly for Rebuttal):")
    print(f"  - Number of metrics: {len(per_metric_pvals_raw)}")
    print(f"  - Average significance coverage rate of all method pairs (raw): {overall_raw:.4f}")
    print(f"  - Average significance coverage rate of all method pairs (FDR-BH): {overall_fdr:.4f}")
    print("  - See global_significance_rate_pairwise_raw.csv / global_significance_rate_pairwise_fdr.csv and per-metric coverage CSV for details")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())