#!/usr/bin/env python3
"""
Compute cross-model metric significance on ViStoryBench summaries with per-story list
control. Different story ID groups can request different statistical tests (paired
t-test, Wilcoxon signed-rank, bootstrap CI).

Example:
  python utils/story_range_significance.py \
      --metrics copy_paste_score \
      --methods StoryAdapter_TextOnly_Scale5 UNO_Base \
      --story-range 01,02,03,04:t_test --story-range 05-20:wilcoxon --story-range 21,22,30:bootstrap \
      --pairs StoryAdapter_TextOnly_Scale5,UNO_Base

Notes:
- Story selectors can be comma-separated ID lists or numeric ranges.
- Metrics listed in LOWER_IS_BETTER (currently copy_paste_score) will be flipped automatically so
  the reported statistics are still "method A better than method B".
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    from scipy import stats as _scipy_stats  # type: ignore
except Exception:  # pragma: no cover - SciPy is optional
    _scipy_stats = None


REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_ROOT = REPO_ROOT / "data" / "bench_results_summary"


LOWER_IS_BETTER = {"copy_paste_score", "copy_paste"}

DEFAULT_STORY_RANGE_EXPR = "01,08,09,15,17,19,24,27,28,29,32,41,52,53,55,57,60,64,68,79"
DEFAULT_TEST_SEQUENCE = ("paired_t", "wilcoxon", "bootstrap")
HEATMAP_VALUE_CHOICES = ("mean_diff", "mean_diff_raw", "p_value", "statistic", "ci_low", "ci_high", "ci_width")


@dataclass(frozen=True)
class StorySelector:
    story_ids: Optional[Set[str]]
    start: Optional[int]
    end: Optional[int]
    test: str
    label: str

    def contains(self, sid: str) -> bool:
        norm = normalize_story_id(sid)
        if self.story_ids is not None:
            return norm in self.story_ids
        val = safe_story_int(norm)
        if self.start is not None and val < self.start:
            return False
        if self.end is not None and val > self.end:
            return False
        return True


def metric_direction(metric: str) -> int:
    return -1 if metric.lower() in LOWER_IS_BETTER else 1


def preference_from_direction(direction: int) -> str:
    return "higher" if direction >= 0 else "lower"


@dataclass
class TestOutcome:
    method: str
    statistic: Optional[float]
    p_value: Optional[float]
    ci_low: Optional[float]
    ci_high: Optional[float]
    significant: Optional[bool]
    extra: Dict[str, float]


def safe_story_int(story_id: str) -> int:
    digits = "".join(ch for ch in str(story_id) if ch.isdigit())
    if not digits:
        raise ValueError(f"Story ID cannot be parsed as int: {story_id}")
    return int(digits)


def normalize_story_id(story_id: str) -> str:
    return str(safe_story_int(story_id)).zfill(2)


def discover_methods(results_root: Path) -> List[str]:
    return sorted([d.name for d in results_root.iterdir() if d.is_dir()])


def discover_metrics(results_root: Path, methods: Sequence[str]) -> List[str]:
    metrics: set[str] = set()
    for method in methods:
        mdir = results_root / method
        if not mdir.is_dir():
            continue
        for f in mdir.glob("*.json"):
            metrics.add(f.stem)
    return sorted(metrics)


def load_metric_scores(results_root: Path, methods: Sequence[str], metric: str) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for method in methods:
        path = results_root / method / f"{metric}.json"
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Failed reading {path}: {exc}") from exc
        scores: Dict[str, float] = {}
        for sid, value in data.items():
            try:
                scores[str(sid).zfill(2)] = float(value)
            except Exception as exc:
                raise ValueError(f"Metric {metric} of {method} has invalid value for story {sid}: {value}") from exc
        out[method] = scores
    return out


def normalize_test_name(name: str) -> str:
    mapping = {
        "t": "paired_t",
        "ttest": "paired_t",
        "t_test": "paired_t",
        "paired_t": "paired_t",
        "wilcoxon": "wilcoxon",
        "wilcoxon_signed_rank": "wilcoxon",
        "bootstrap": "bootstrap",
        "boot": "bootstrap",
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unknown test type '{name}'. Expected one of {sorted(set(mapping.values()))}.")
    return mapping[key]


def parse_story_selector(expr: str, default_label_idx: int) -> StorySelector:
    # Accepts formats:
    #   id1,id2,id3
    #   start-end
    #   id1,id2,id3:test[:label]
    #   start-end:test[:label]
    parts = [part.strip() for part in expr.split(":")]
    if not parts or not parts[0]:
        raise ValueError(f"Story selector '{expr}' is empty.")
    raw_span = parts[0]
    test_part = parts[1] if len(parts) >= 2 else ""
    label_part = parts[2] if len(parts) >= 3 else ""
    test = normalize_test_name(test_part) if test_part else "paired_t"
    label = label_part if label_part else f"range_{default_label_idx}"
    span = raw_span.replace(" ", "")
    tokens = [tok for tok in span.split(",") if tok]
    story_ids: Optional[Set[str]] = None
    start: Optional[int] = None
    end: Optional[int] = None
    if len(tokens) == 1 and "-" in tokens[0]:
        start_str, end_str = tokens[0].split("-", 1)
        start = int(start_str) if start_str else None
        end = int(end_str) if end_str else None
    else:
        if not tokens:
            raise ValueError(f"Story selector '{expr}' is empty.")
        story_ids = {normalize_story_id(token) for token in tokens}
    if start is not None and end is not None and end < start:
        raise ValueError(f"Invalid story range '{expr}' with end < start.")
    if label == f"range_{default_label_idx}":
        label = build_selector_label(story_ids, start, end)
    return StorySelector(story_ids=story_ids, start=start, end=end, test=test, label=label)


def build_selector_label(story_ids: Optional[Set[str]], start: Optional[int], end: Optional[int]) -> str:
    if story_ids:
        ordered = sorted(story_ids)
        if len(ordered) <= 4:
            joined = ",".join(ordered)
        else:
            joined = ",".join(ordered[:3]) + ",...," + ordered[-1]
        return f"ids[{joined}]"
    if start is None and end is None:
        return "all"
    if start is None:
        return f"<= {end}"
    if end is None:
        return f">= {start}"
    if start == end:
        return f"{start}"
    return f"{start}-{end}"


def normalize_test_sequence(test_names: Optional[Sequence[str]]) -> List[str]:
    base = list(test_names) if test_names else list(DEFAULT_TEST_SEQUENCE)
    normalized: List[str] = []
    for name in base:
        canonical = normalize_test_name(name)
        if canonical not in normalized:
            normalized.append(canonical)
    return normalized or [normalize_test_name("paired_t")]


def expand_story_ranges(range_exprs: Sequence[str], tests: Sequence[str]) -> List[StorySelector]:
    selectors: List[StorySelector] = []
    for expr in range_exprs:
        expr_clean = expr.strip()
        if not expr_clean:
            continue
        if ":" in expr_clean:
            selectors.append(parse_story_selector(expr_clean, len(selectors)))
            continue
        for test in tests:
            selectors.append(parse_story_selector(f"{expr_clean}:{test}", len(selectors)))
    return selectors


def paired_t_test(scores_a: Sequence[float], scores_b: Sequence[float], alpha: float) -> TestOutcome:
    if len(scores_a) != len(scores_b):
        raise ValueError("Paired samples must have the same length")
    if len(scores_a) < 2:
        return TestOutcome("paired_t", None, None, None, None, None, {})
    if _scipy_stats is not None:
        stat, p_two_sided = _scipy_stats.ttest_rel(scores_a, scores_b, nan_policy="omit")
        if math.isnan(stat):
            p_value = None
        else:
            if stat > 0:
                p_value = max(p_two_sided / 2.0, 0.0)
            else:
                p_value = min(1.0, 1.0 - p_two_sided / 2.0)
        significant = p_value is not None and p_value <= alpha
        return TestOutcome("paired_t", float(stat) if stat is not None else None, p_value, None, None, significant, {})
    # Manual fallback with normal approximation.
    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    mean_diff = sum(diffs) / len(diffs)
    variance = sum((d - mean_diff) ** 2 for d in diffs) / (len(diffs) - 1)
    if variance == 0:
        stat = math.inf if mean_diff > 0 else -math.inf
        p_value = 0.0 if mean_diff > 0 else 1.0
    else:
        se = math.sqrt(variance / len(diffs))
        stat = mean_diff / se
        # Approximate using normal distribution when SciPy is unavailable.
        norm = NormalDist()
        p_value = 1.0 - norm.cdf(stat)
    significant = p_value is not None and p_value <= alpha
    return TestOutcome("paired_t", stat, p_value, None, None, significant, {})


def wilcoxon_test(scores_a: Sequence[float], scores_b: Sequence[float], alpha: float) -> TestOutcome:
    if len(scores_a) != len(scores_b):
        raise ValueError("Paired samples must have the same length")
    if len(scores_a) == 0:
        return TestOutcome("wilcoxon", None, None, None, None, None, {})
    if _scipy_stats is not None:
        stat, p_value = _scipy_stats.wilcoxon(scores_a, scores_b, zero_method="wilcox", correction=True, alternative="greater")
        significant = p_value is not None and p_value <= alpha
        return TestOutcome("wilcoxon", float(stat) if stat is not None else None, float(p_value) if p_value is not None else None, None, None, significant, {})
    diffs = [a - b for a, b in zip(scores_a, scores_b) if not math.isclose(a, b)]
    n = len(diffs)
    if n == 0:
        return TestOutcome("wilcoxon", None, None, None, None, None, {})
    abs_diffs = [abs(d) for d in diffs]
    ranks = rankdata(abs_diffs)
    w_plus = sum(rank for rank, diff in zip(ranks, diffs) if diff > 0)
    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0
    # Normal approximation with continuity correction.
    z = (w_plus - mean_w - 0.5) / math.sqrt(var_w) if var_w > 0 else 0.0
    norm = NormalDist()
    p_value = 1.0 - norm.cdf(z)
    significant = p_value is not None and p_value <= alpha
    return TestOutcome("wilcoxon", w_plus, p_value, None, None, significant, {})


def rankdata(values: Sequence[float]) -> List[float]:
    sorted_pairs = sorted((val, idx) for idx, val in enumerate(values))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(sorted_pairs):
        j = i
        total = 0.0
        while j < len(sorted_pairs) and math.isclose(sorted_pairs[j][0], sorted_pairs[i][0], rel_tol=1e-12, abs_tol=1e-12):
            total += j + 1
            j += 1
        avg_rank = total / (j - i)
        for k in range(i, j):
            _, idx = sorted_pairs[k]
            ranks[idx] = avg_rank
        i = j
    return ranks


def bootstrap_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    iterations: int,
    confidence: float,
    rng: random.Random,
) -> TestOutcome:
    if len(scores_a) != len(scores_b):
        raise ValueError("Paired samples must have the same length")
    n = len(scores_a)
    if n == 0:
        return TestOutcome("bootstrap", None, None, None, None, None, {})
    pairs = list(zip(scores_a, scores_b))
    observed = sum(a - b for a, b in pairs) / n
    if n == 1:
        return TestOutcome("bootstrap", observed, None, observed, observed, observed > 0, {})
    estimates: List[float] = []
    for _ in range(iterations):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        diff = sum(a - b for a, b in sample) / n
        estimates.append(diff)
    estimates.sort()
    lower_q = (1.0 - confidence) / 2.0
    upper_q = 1.0 - lower_q
    lower_idx = max(0, int(math.floor(lower_q * (iterations - 1))))
    upper_idx = min(iterations - 1, int(math.ceil(upper_q * (iterations - 1))))
    ci_low = estimates[lower_idx]
    ci_high = estimates[upper_idx]
    p_value = sum(1 for est in estimates if est <= 0.0) / iterations
    significant = ci_low > 0.0
    return TestOutcome("bootstrap", observed, p_value, ci_low, ci_high, significant, {"ci_confidence": confidence})


def collect_pairs(methods: Sequence[str], pair_exprs: Optional[Sequence[str]]) -> List[Tuple[str, str]]:
    if not pair_exprs:
        return [(a, b) for a in methods for b in methods if a != b]
    pairs: List[Tuple[str, str]] = []
    method_set = set(methods)
    for expr in pair_exprs:
        token = expr.replace(">", ",").replace("|", ",")
        parts = [p.strip() for p in token.split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError(f"Pair expression '{expr}' is invalid. Use A,B or A>B format.")
        a, b = parts
        if a not in method_set or b not in method_set:
            raise ValueError(f"Methods in pair '{expr}' must be included in --methods list.")
        if a == b:
            continue
        pairs.append((a, b))
    return pairs


def run_test(range_conf: StorySelector, scores_a: Sequence[float], scores_b: Sequence[float], args: argparse.Namespace, rng: random.Random) -> TestOutcome:
    if range_conf.test == "paired_t":
        return paired_t_test(scores_a, scores_b, args.alpha)
    if range_conf.test == "wilcoxon":
        return wilcoxon_test(scores_a, scores_b, args.alpha)
    if range_conf.test == "bootstrap":
        return bootstrap_test(scores_a, scores_b, args.bootstrap_iterations, args.bootstrap_confidence, rng)
    raise ValueError(f"Unsupported test '{range_conf.test}'")


def summarize_results(results: List[Dict[str, object]]) -> None:
    if not results:
        print("No overlapping story IDs for the requested configuration.")
        return
    grouped: Dict[Tuple[str, str, str], List[Dict[str, object]]] = {}
    for row in results:
        key = (str(row["metric"]), str(row["range_label"]), str(row["test"]))
        grouped.setdefault(key, []).append(row)
    for (metric, range_label, test_name), subset in sorted(grouped.items()):
        preference = subset[0].get("preference", "higher")
        pref_txt = "higher is better" if preference == "higher" else "lower is better"
        print(f"\nMetric: {metric} ({pref_txt}) | Story group: {range_label} | Test: {test_name}")
        for row in sorted(subset, key=lambda r: (str(r["method_a"]), str(r["method_b"]))):
            sig = "YES" if row["significant"] else "no"
            raw_diff = row.get("mean_diff_raw")
            raw_txt = f" raw(A-B)={raw_diff:.4f}" if isinstance(raw_diff, float) else ""
            base = (
                f"  {row['method_a']} > {row['method_b']}: "
                f"n={row['n']} mean_diff_dir={row['mean_diff']:.4f}{raw_txt}"
            )
            stat = f" statistic={row['statistic']:.4f}" if isinstance(row["statistic"], float) else ""
            if row["test"] == "bootstrap":
                ci_info = ""
                if row["ci_low"] is not None and row["ci_high"] is not None:
                    ci_info = f" ci=[{row['ci_low']:.4f}, {row['ci_high']:.4f}]"
                p_txt = f" p={row['p_value']:.4g}" if isinstance(row["p_value"], float) else " p=NA"
                print(f"{base}{stat}{ci_info}{p_txt} significant={sig}")
            else:
                pval = row["p_value"]
                p_txt = f" p={pval:.4g}" if isinstance(pval, float) else " p=NA"
                print(f"{base}{stat}{p_txt} significant={sig}")


def generate_heatmaps(
    results: Sequence[Dict[str, object]],
    methods: Sequence[str],
    output_dir: Optional[Path],
    value_fields: Sequence[str],
    annotate: bool,
    image_format: str,
    split_by_test: bool,
) -> None:
    if output_dir is None:
        return
    if not results:
        print("No results available for heatmap generation.")
        return
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"Skipping heatmap generation because matplotlib is unavailable: {exc}")
        return

    canonical_fields = []
    for field in value_fields:
        if field in HEATMAP_VALUE_CHOICES and field not in canonical_fields:
            canonical_fields.append(field)
    if not canonical_fields:
        canonical_fields = ["mean_diff"]

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = image_format.lstrip(".") or "png"
    method_indices = {method: idx for idx, method in enumerate(methods)}
    grouped: Dict[Tuple[str, str, str], List[Dict[str, object]]] = {}
    for row in results:
        metric = str(row["metric"])
        test = str(row["test"])
        label = str(row["range_label"])
        grouped.setdefault((metric, test, label), []).append(row)

    for value_field in canonical_fields:
        value_dir = output_dir / value_field if len(canonical_fields) > 1 else output_dir
        value_dir.mkdir(parents=True, exist_ok=True)
        for (metric, test, label), rows in sorted(grouped.items()):
            matrix = np.full((len(methods), len(methods)), np.nan, dtype=float)
            annotations: List[List[str]] = [["" for _ in methods] for _ in methods]
            for row in rows:
                method_a = str(row["method_a"])
                method_b = str(row["method_b"])
                if method_a not in method_indices or method_b not in method_indices:
                    continue
                value = extract_heatmap_value(row, value_field)
                if not isinstance(value, (int, float)):
                    continue
                value = float(value)
                if math.isnan(value):
                    continue
                i = method_indices[method_a]
                j = method_indices[method_b]
                matrix[i, j] = value
                if annotate:
                    txt = f"{value:.3f}"
                    if bool(row.get("significant")):
                        txt += "*"
                    annotations[i][j] = txt
            if np.all(np.isnan(matrix)):
                continue
            for diag in range(len(methods)):
                matrix[diag, diag] = 0.0
                if annotate:
                    annotations[diag][diag] = "-"
            finite_mask = np.isfinite(matrix)
            if not finite_mask.any():
                continue
            abs_max = np.nanmax(np.abs(matrix[finite_mask]))
            if not math.isfinite(abs_max) or abs_max == 0.0:
                abs_max = 1.0
            fig, ax = plt.subplots(figsize=(max(8.0, len(methods) * 1.4), max(7, len(methods) * 1.4)))
            cmap = plt.get_cmap("coolwarm")
            im = ax.imshow(matrix, cmap=cmap, vmin=-abs_max, vmax=abs_max)
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels(methods, rotation=45, ha="right")
            ax.set_yticks(range(len(methods)))
            ax.set_yticklabels(methods)
            ax.set_title(f"{metric} | {test} | {label} | {value_field}")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(value_field)
            if annotate:
                for i in range(len(methods)):
                    for j in range(len(methods)):
                        text = annotations[i][j]
                        if not text:
                            continue
                        ax.text(j, i, text, ha="center", va="center", fontsize=8, color="black")
            fig.tight_layout()
            raw_name = f"{metric}_{test}_{label}.{suffix}"
            safe_name = "".join(ch if ch.isalnum() or ch in "-_. " else "_" for ch in raw_name).replace(" ", "_")
            target_dir = value_dir / test if split_by_test else value_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(target_dir / safe_name, dpi=200)
            plt.close(fig)
            print(f"Saved heatmap to {target_dir / safe_name}")


def extract_heatmap_value(row: Dict[str, object], field: str) -> Optional[float]:
    if field == "ci_width":
        low = row.get("ci_low")
        high = row.get("ci_high")
        if isinstance(low, (int, float)) and isinstance(high, (int, float)):
            return float(high) - float(low)
        return None
    value = row.get(field)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute per-story-range significance over ViStoryBench summary metrics.")
    parser.add_argument("--results-root", type=Path, default=SUMMARY_ROOT, help="Directory with per-method metric JSON (default: data/bench_results_summary)")
    parser.add_argument("--methods", nargs="*", default=['Sora2_ALL_ImgRef','DouBao_Base','MovieAgent_SD3','UNO_Base','NaiveBaseline_Base'], help="Method directories under results-root. Defaults to all detected methods.")
    parser.add_argument("--metrics", nargs="*", default=None, help="Metric file stems (e.g., copy_paste_score). Defaults to all metrics present in --methods.")
    parser.add_argument(
        "--story-range",
        action="append",
        dest="story_ranges",
        default=None,
        help="Story list (id1,id2,...) or start-end range with :test[:label]. Repeatable. Defaults to a curated ID list.",
    )
    parser.add_argument(
        "--story-tests",
        nargs="*",
        default=None,
        help="When --story-range entries omit :test, duplicate them across these tests (default: paired_t, wilcoxon, bootstrap).",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level for t-test/Wilcoxon (default: 0.05)")
    parser.add_argument("--bootstrap-iterations", type=int, default=10000, help="Bootstrap iterations for CI (default: 10000)")
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95, help="Bootstrap confidence interval (default: 0.95)")
    parser.add_argument("--pairs", nargs="*", help="Optional ordered method pairs (A,B) to evaluate. Defaults to all ordered pairs.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for bootstrap sampling (default: 13)")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON file to store detailed results.")
    parser.add_argument("--min-samples", type=int, default=2, help="Minimum overlapping stories per pair+range to run a test (default: 2)")
    parser.add_argument("--heatmap-dir", type=Path, default=None, help="Directory to store metric/test heatmap matrices.")
    parser.add_argument(
        "--heatmap-value",
        choices=HEATMAP_VALUE_CHOICES,
        default="mean_diff",
        help="Which statistic to visualize in heatmaps (options: mean_diff, mean_diff_raw, p_value, statistic, ci_low, ci_high, ci_width).",
    )
    parser.add_argument(
        "--heatmap-values",
        nargs="+",
        choices=HEATMAP_VALUE_CHOICES,
        default=None,
        help="Render multiple statistics at once (each gets its own subdirectory).",
    )
    parser.add_argument("--heatmap-format", type=str, default="png", help="Image format for heatmaps (default: png).")
    parser.add_argument("--heatmap-annotate", action="store_true", help="Annotate heatmap cells with numeric values.")
    parser.add_argument(
        "--heatmap-split-tests",
        dest="heatmap_split_tests",
        action="store_true",
        help="Store heatmaps into per-test subdirectories under --heatmap-dir (default).",
    )
    parser.add_argument(
        "--heatmap-flat",
        dest="heatmap_split_tests",
        action="store_false",
        help="Save all heatmaps directly under --heatmap-dir (disables per-test subdirectories).",
    )
    parser.set_defaults(heatmap_split_tests=True)
    args = parser.parse_args()

    results_root = args.results_root
    if not results_root.is_dir():
        raise SystemExit(f"{results_root} does not exist or is not a directory.")

    methods = args.methods or discover_methods(results_root)
    if not methods:
        raise SystemExit(f"No methods discovered under {results_root}.")

    metrics = args.metrics or discover_metrics(results_root, methods)
    if not metrics:
        raise SystemExit(f"No metrics discovered for methods {methods}.")

    default_tests = normalize_test_sequence(args.story_tests)
    raw_range_exprs = args.story_ranges if args.story_ranges else [DEFAULT_STORY_RANGE_EXPR]
    range_exprs = [expr for expr in raw_range_exprs if expr and expr.strip()]
    if not range_exprs:
        story_configs = [
            StorySelector(story_ids=None, start=None, end=None, test=test, label="all")
            for test in default_tests
        ]
    else:
        story_configs = expand_story_ranges(range_exprs, default_tests)

    pairs = collect_pairs(methods, args.pairs)
    rng = random.Random(args.seed)

    all_results: List[Dict[str, object]] = []
    for metric in metrics:
        metric_scores = load_metric_scores(results_root, methods, metric)
        direction = metric_direction(metric)
        preference = preference_from_direction(direction)
        for method_a, method_b in pairs:
            scores_a = metric_scores.get(method_a)
            scores_b = metric_scores.get(method_b)
            if not scores_a or not scores_b:
                continue
            common_ids = sorted(set(scores_a.keys()) & set(scores_b.keys()), key=lambda sid: safe_story_int(sid))
            if not common_ids:
                continue
            for selector in story_configs:
                selected_ids = [sid for sid in common_ids if selector.contains(sid)]
                if len(selected_ids) < args.min_samples:
                    continue
                raw_a = [scores_a[sid] for sid in selected_ids]
                raw_b = [scores_b[sid] for sid in selected_ids]
                oriented_a = [direction * val for val in raw_a]
                oriented_b = [direction * val for val in raw_b]
                outcome = run_test(selector, oriented_a, oriented_b, args, rng)
                mean_diff_oriented = sum(a - b for a, b in zip(oriented_a, oriented_b)) / len(oriented_a)
                mean_diff_raw = sum(a - b for a, b in zip(raw_a, raw_b)) / len(raw_a)
                record = {
                    "metric": metric,
                    "range_label": selector.label,
                    "range_start": selector.start,
                    "range_end": selector.end,
                    "story_ids": sorted(selector.story_ids) if selector.story_ids else None,
                    "test": selector.test,
                    "method_a": method_a,
                    "method_b": method_b,
                    "n": len(selected_ids),
                    "mean_a": sum(raw_a) / len(raw_a),
                    "mean_b": sum(raw_b) / len(raw_b),
                    "mean_diff": mean_diff_oriented,
                    "mean_diff_raw": mean_diff_raw,
                    "preference": preference,
                    "statistic": outcome.statistic,
                    "p_value": outcome.p_value,
                    "ci_low": outcome.ci_low,
                    "ci_high": outcome.ci_high,
                    "significant": outcome.significant,
                    "extra": outcome.extra,
                }
                all_results.append(record)

    summarize_results(all_results)

    if args.heatmap_dir:
        heatmap_fields = args.heatmap_values or ([args.heatmap_value] if args.heatmap_value else ["mean_diff"])
        generate_heatmaps(
            all_results,
            methods,
            args.heatmap_dir,
            heatmap_fields,
            args.heatmap_annotate,
            args.heatmap_format,
            args.heatmap_split_tests,
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        print(f"\nSaved {len(all_results)} rows to {args.output}")


if __name__ == "__main__":
    main()
    # 019aa470-eb82-7a00-9721-2802c6ce9a2b
