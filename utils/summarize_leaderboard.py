#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize leaderboard rankings from ViStoryBench CSVs for lite, full and combined.

This script:
- Reads data/result/bench_results_lite.csv and data/result/bench_results_full.csv
- Aggregates each split independently by (method, model)
- Optionally merges the two splits (combined = average of common metrics)
- Computes category ranks using averaged submetric ranks within each category
- Selects the best mode per method by minimal average rank across the five categories
- Outputs compact leaderboard tables with ranks per category and overall rank

Outputs:
- Prints one or more leaderboards to stdout (controlled by --which)
- Optionally writes CSVs via --out (combined), --out-lite, --out-full

Usage:
  python utils/summarize_leaderboard.py
  python utils/summarize_leaderboard.py --lite data/result/bench_results_lite.csv --full data/result/bench_results_full.csv --which all --out data/result/leaderboard_summary_combined.csv --out-lite data/result/leaderboard_summary_lite.csv --out-full data/result/leaderboard_summary_full.csv
"""

from __future__ import annotations
import argparse
import csv
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Set

# Category -> list of submetric CSV column keys
CATEGORY_SUBMETRICS: Dict[str, List[str]] = {
    "char": [
        "metrics.cids.metrics.cids_self_mean",
        "metrics.cids.metrics.cids_cross_mean",
    ],
    "style": [
        "metrics.csd.metrics.self_csd",
        "metrics.csd.metrics.cross_csd",
    ],
    "prompt": [
        "metrics.prompt_align.metrics.camera",
        "metrics.prompt_align.metrics.character_action",
        "metrics.prompt_align.metrics.scene",
        "metrics.cids.metrics.single_character_action",
    ],
    "quality": [
        "metrics.aesthetic.metrics.aesthetic_score",
    ],
    "diversity": [
        "metrics.diversity.metrics.inception_score",
    ],
}

# All metrics are "higher is better" in ViStoryBench
HIGHER_IS_BETTER: Dict[str, bool] = defaultdict(lambda: True)  # default True

# Included methods grouped by display method category
from typing import Dict, List

INCLUDED_GROUPS: Dict[str, List[str]] = {
    "Story image method": [
        "CharaConsist",
        "OmniGen2",
        "QwenImageEdit2509",
        "SeedStory",
        "StoryAdapter",
        "StoryDiffusion",
        "StoryGen",
        "TheaterGen",
        "UNO",
    ],
    "Story video method": ["AnimDirector", "MMStoryAgent", "MovieAgent", "Vlogger"],
    "Commercial platform": [
        "AIbrm",
        "DouBao",
        "MOKI",
        "MorphicStudio",
        "ShenBi",
        "TypeMovie",
    ],
    "MLLM model": ["GPT4o", "Gemini","NanoBanana"],
    "Naive baseline": ["NaiveBaseline"],
}

FRONT_COLS = ["method", "model", "mode", "timestamp"]

@dataclass(frozen=True)
class RunKey:
    method: str
    model: str

def safe_float(s: Any) -> Optional[float]:
    try:
        if s is None:
            return None
        if isinstance(s, (int, float)):
            val = float(s)
        else:
            s2 = str(s).strip()
            if s2 == "" or s2.lower() == "nan":
                return None
            val = float(s2)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    except Exception:
        return None

def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    encodings = ["utf-8-sig", "utf-8"]
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            with path.open("r", newline="", encoding=enc) as f:
                reader = csv.DictReader(f)
                return [row for row in reader]
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to read CSV: {path} ({last_err})")

def _group_rows_by_run(rows: List[Dict[str, Any]]) -> Dict[RunKey, List[Dict[str, Any]]]:
    g: Dict[RunKey, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = RunKey(method=(r.get("method") or "").strip(), model=(r.get("model") or "").strip())
        if key.method and key.model:
            g[key].append(r)
    return g

def _avg_rows(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    acc: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        for k, v in r.items():
            if k in FRONT_COLS:
                continue
            fv = safe_float(v)
            if fv is not None:
                acc[k].append(fv)
    return {k: (sum(vals) / len(vals)) for k, vals in acc.items() if vals}

def aggregate_split_rows(rows: List[Dict[str, Any]]) -> Dict[RunKey, Dict[str, float]]:
    """
    Aggregate a single split (lite OR full) by (method, model).
    When multiple timestamps/languages exist, average them.
    Returns: map RunKey -> {metric_key: avg_value}
    """
    grouped = _group_rows_by_run(rows)
    return {k: _avg_rows(v) for k, v in grouped.items()}

def merge_splits(lite_rows: List[Dict[str, Any]], full_rows: List[Dict[str, Any]]) -> Dict[RunKey, Dict[str, float]]:
    """
    Merge lite and full rows by (method, model) averaging numeric metric columns.
    Returns: map RunKey -> merged_metrics
    """
    lite_map_by_key = aggregate_split_rows(lite_rows)
    full_map_by_key = aggregate_split_rows(full_rows)

    all_keys: Set[RunKey] = set(lite_map_by_key.keys()) | set(full_map_by_key.keys())
    merged: Dict[RunKey, Dict[str, float]] = {}

    for key in all_keys:
        lite_map = lite_map_by_key.get(key, {})
        full_map = full_map_by_key.get(key, {})
        merged_keys = set(lite_map.keys()) | set(full_map.keys())
        out: Dict[str, float] = {}
        for m in merged_keys:
            lv = lite_map.get(m)
            fv = full_map.get(m)
            if lv is not None and fv is not None:
                out[m] = (lv + fv) / 2.0
            elif lv is not None:
                out[m] = lv
            elif fv is not None:
                out[m] = fv
        merged[key] = out
    return merged

def rank_values(id_to_value: Dict[RunKey, Optional[float]], higher_is_better: bool) -> Dict[RunKey, float]:
    """
    Compute average-rank for ties. Returns rank starting from 1.0 (lower is better).
    Non-numeric (None) values get worst rank (len items).
    """
    items = list(id_to_value.items())
    n = len(items)
    # Prepare list with fallback for None
    prepared: List[Tuple[RunKey, float, bool]] = []
    for k, v in items:
        if v is None:
            prepared.append((k, float("-inf") if higher_is_better else float("inf"), True))
        else:
            prepared.append((k, float(v), False))
    # Sort by value desc if higher is better, else asc
    prepared.sort(key=lambda t: t[1], reverse=higher_is_better)
    # Assign ranks with averaging ties
    ranks: Dict[RunKey, float] = {}
    i = 0
    while i < n:
        j = i
        val = prepared[i][1]
        while j + 1 < n and prepared[j + 1][1] == val:
            j += 1
        # positions i..j inclusive -> ordinal positions 1-based are (i+1)..(j+1)
        avg_rank = ((i + 1) + (j + 1)) / 2.0
        for k2, _, _ in prepared[i:j+1]:
            ranks[k2] = avg_rank
        i = j + 1
    # Put worst rank for Nones explicitly (already handled by -inf/+inf, but ensure worst)
    worst = float(n)
    for k, v in id_to_value.items():
        if v is None:
            ranks[k] = worst
    return ranks

def compute_category_rank(
    merged: Dict[RunKey, Dict[str, float]],
    submetrics: List[str],
    higher_is_better: bool,
    scope: Optional[Set[RunKey]] = None,
) -> Tuple[Dict[RunKey, float], Dict[RunKey, float]]:
    """
    For a set of runs, compute:
      - submetric ranks per run (averaged across submetrics) -> category_score (lower better)
      - category ordinal rank among runs based on category_score
    Returns: (category_score_map, category_rank_map)
    """
    # Limit scope
    run_keys = [k for k in merged.keys() if (scope is None or k in scope)]
    # For each submetric, compute rank map
    per_sub_ranks: List[Dict[RunKey, float]] = []
    for m in submetrics:
        id_to_val: Dict[RunKey, Optional[float]] = {k: merged[k].get(m) for k in run_keys}
        per_sub_ranks.append(rank_values(id_to_val, higher_is_better))
    # Average submetric ranks
    category_scores: Dict[RunKey, float] = {}
    for k in run_keys:
        vals = [r[k] for r in per_sub_ranks if k in r]
        category_scores[k] = sum(vals) / len(vals) if vals else float("inf")
    # Ordinal rank among runs based on category_scores (lower better)
    # Convert to values for rank_values (lower better -> invert)
    inv = {k: -v for k, v in category_scores.items()}
    category_ranks = rank_values(inv, higher_is_better=True)
    return category_scores, category_ranks

def average(lst: List[float]) -> float:
    return sum(lst) / len(lst) if lst else float("inf")

def pick_best_mode_per_method(
    merged: Dict[RunKey, Dict[str, float]],
    allowed_methods: Set[str],
) -> Dict[str, RunKey]:
    """
    For each method in allowed_methods, pick the (method, model) run with minimal
    average category rank across all five categories.
    Returns: method_id -> chosen RunKey
    """
    # Compute category ranks across all runs for scope of allowed methods only
    scope = {k for k in merged.keys() if k.method in allowed_methods}
    cat_ranks: Dict[str, Dict[RunKey, float]] = {}
    for cat, subs in CATEGORY_SUBMETRICS.items():
        _, ranks = compute_category_rank(merged, subs, HIGHER_IS_BETTER[subs[0]], scope=scope)
        cat_ranks[cat] = ranks
    chosen: Dict[str, RunKey] = {}
    for method in sorted(allowed_methods):
        candidates = [k for k in scope if k.method == method]
        if not candidates:
            continue
        # Compute mean of category ranks for each candidate
        best_key = None
        best_mean = float("inf")
        for k in candidates:
            ranks = [cat_ranks[c][k] for c in CATEGORY_SUBMETRICS.keys()]
            mean_rank = average(ranks)
            if mean_rank < best_mean:
                best_mean = mean_rank
                best_key = k
        if best_key:
            chosen[method] = best_key
    return chosen

def finalize_ranks_for_selected(
    merged: Dict[RunKey, Dict[str, float]],
    selected: Dict[str, RunKey],
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
    """
    For selected runs (one per method), compute integer category ranks and overall ranks.
    Returns:
      cat_rank_ints: {category -> {method_id -> rank_int}}
      overall_ranks: {method_id -> rank_int}
    """
    scope = set(selected.values())
    # Category ordinal ranks
    cat_rank_ints: Dict[str, Dict[str, int]] = {}
    for cat, subs in CATEGORY_SUBMETRICS.items():
        _, ranks = compute_category_rank(merged, subs, HIGHER_IS_BETTER[subs[0]], scope=scope)
        # Convert to integer ranks, preserving ties
        items = list(ranks.items())
        items.sort(key=lambda kv: kv[1])
        rank_int_map: Dict[RunKey, int] = {}
        current_rank = 0
        prev_val: Optional[float] = None
        for k, v in items:
            if prev_val is None or v > prev_val:
                current_rank += 1
                prev_val = v
            rank_int_map[k] = current_rank
        # Map to method id
        cat_rank_ints[cat] = {k.method: rank_int_map[k] for k in scope}
    # Overall: average category ranks then ordinal
    overall_scores: Dict[str, float] = {}
    for method, rk in selected.items():
        vals = [cat_rank_ints[c][method] for c in CATEGORY_SUBMETRICS.keys()]
        overall_scores[method] = sum(vals) / len(vals)
    # Ordinal
    items2 = sorted(overall_scores.items(), key=lambda kv: kv[1])
    overall_ints: Dict[str, int] = {}
    current = 0
    prev: Optional[float] = None
    for m, v in items2:
        if prev is None or v > prev:
            current += 1
            prev = v
        overall_ints[m] = current
    return cat_rank_ints, overall_ints

def build_output_rows(
    selected: Dict[str, RunKey],
    cat_ranks: Dict[str, Dict[str, int]],
    overall_ranks: Dict[str, int],
) -> List[Dict[str, Any]]:
    # Build helper for group label
    method_to_group: Dict[str, str] = {}
    for g, methods in INCLUDED_GROUPS.items():
        for m in methods:
            method_to_group[m] = g
    rows: List[Dict[str, Any]] = []
    # Sorting order: by overall rank ascending
    sorted_methods = sorted(selected.keys(), key=lambda m: overall_ranks.get(m, 9999))
    for method in sorted_methods:
        rk = selected[method]
        group = method_to_group.get(method, "")
        model_disp = rk.model
        method_disp = method
        row = {
            "Method": group,
            "model": method_disp,
            "mode": model_disp,
            "Character Consistency(CRef)_rank": cat_ranks["char"][method],
            "Style Consistency(SRef)_rank": cat_ranks["style"][method],
            "Prompt Alignment_rank": cat_ranks["prompt"][method],
            "Generative Quality_rank": cat_ranks["quality"][method],
            "Diversity_rank": cat_ranks["diversity"][method],
            "overall_rank": overall_ranks[method],
            "model_key": f"{group} - {method_disp}",
        }
        rows.append(row)
    return rows

def write_output_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "Method",
        "model",
        "mode",
        "Character Consistency(CRef)_rank",
        "Style Consistency(SRef)_rank",
        "Prompt Alignment_rank",
        "Generative Quality_rank",
        "Diversity_rank",
        "overall_rank",
        "model_key",
    ]
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

def print_rows(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("No rows to display.")
        return
    headers = [
        "Method",
        "model",
        "mode",
        "Character Consistency(CRef)_rank",
        "Style Consistency(SRef)_rank",
        "Prompt Alignment_rank",
        "Generative Quality_rank",
        "Diversity_rank",
        "overall_rank",
        "model_key",
    ]
    print(",".join(headers))
    for r in rows:
        vals = [str(r[h]) for h in headers]
        print(",".join(vals))

def print_section(title: str, rows: List[Dict[str, Any]]) -> None:
    print(f"# {title}")
    print_rows(rows)
    print()

def compute_leaderboard_rows(
    merged: Dict[RunKey, Dict[str, float]],
    allowed_methods: Set[str],
) -> List[Dict[str, Any]]:
    chosen = pick_best_mode_per_method(merged, allowed_methods)
    cat_ranks, overall_ranks = finalize_ranks_for_selected(merged, chosen)
    rows = build_output_rows(chosen, cat_ranks, overall_ranks)
    return rows


def _method_to_group_map() -> Dict[str, str]:
    d: Dict[str, str] = {}
    for g, methods in INCLUDED_GROUPS.items():
        for m in methods:
            d[m] = g
    return d


def _pick_existence_key(metrics: Dict[str, float]) -> Optional[str]:
    for k in (
        "metrics.prompt_align.metrics.character_existence",
        "metrics.prompt_align.metrics.character_existence_number",
        "metrics.prompt_align.metrics.character_number",
        "metrics.prompt_align.metrics.character_count",
    ):
        if k in metrics:
            return k
    return None


def _compute_pa_avg_from_row(row: Dict[str, Any]) -> Optional[float]:
    keys = [
        "metrics.prompt_align.metrics.scene",
        "metrics.prompt_align.metrics.camera",
        "metrics.prompt_align.metrics.character_action",
        "metrics.cids.metrics.single_character_action",
    ]
    ex_k: Optional[str] = None
    for k in (
        "metrics.prompt_align.metrics.character_existence",
        "metrics.prompt_align.metrics.character_existence_number",
        "metrics.prompt_align.metrics.character_number",
        "metrics.prompt_align.metrics.character_count",
    ):
        if row.get(k) is not None:
            ex_k = k
            break
    vals: List[float] = []
    for k in keys:
        v = safe_float(row.get(k))
        if v is not None:
            vals.append(v)
    if ex_k:
        v = safe_float(row.get(ex_k))
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    return sum(vals) / len(vals)


def build_raw_metric_rows(
    agg: Dict[RunKey, Dict[str, float]],
    allowed_methods: Set[str],
) -> List[Dict[str, Any]]:
    """
    Build raw per-run rows (no ranking) for TSV export:
    - One row per (method, model) run contained in agg
    - Includes Method group, model display, mode display, and raw metric values
    """
    method_to_group = _method_to_group_map()
    sortable_rows: List[Tuple[str, str, str, Dict[str, Any]]] = []

    for rk, metrics in agg.items():
        if rk.method not in allowed_methods:
            continue
        group = method_to_group.get(rk.method, "")
        method_disp = rk.method
        mode_disp = rk.model

        row: Dict[str, Any] = {
            "Method": group,
            "model": method_disp,
            "mode": mode_disp,
            # Style
            "metrics.csd.metrics.cross_csd": metrics.get("metrics.csd.metrics.cross_csd"),
            "metrics.csd.metrics.self_csd": metrics.get("metrics.csd.metrics.self_csd"),
            # Char
            "metrics.cids.metrics.cids_cross_mean": metrics.get("metrics.cids.metrics.cids_cross_mean"),
            "metrics.cids.metrics.cids_self_mean": metrics.get("metrics.cids.metrics.cids_self_mean"),
            "metrics.cids.metrics.occm": metrics.get("metrics.cids.metrics.occm"),
            # Quality
            "metrics.aesthetic.metrics.aesthetic_score": metrics.get("metrics.aesthetic.metrics.aesthetic_score"),
            # Diversity
            "metrics.diversity.metrics.inception_score": metrics.get("metrics.diversity.metrics.inception_score"),
            # Prompt Align
            "metrics.prompt_align.metrics.scene": metrics.get("metrics.prompt_align.metrics.scene"),
            "metrics.prompt_align.metrics.camera": metrics.get("metrics.prompt_align.metrics.camera"),
            "metrics.prompt_align.metrics.character_action": metrics.get("metrics.prompt_align.metrics.character_action"),
            "metrics.cids.metrics.single_character_action": metrics.get("metrics.cids.metrics.single_character_action"),
        }
        ex_k = _pick_existence_key(metrics)
        if ex_k:
            val = metrics.get(ex_k)
            row[ex_k] = val
            # Normalize existence/number metric into a standard key for output columns
            row["metrics.prompt_align.metrics.character_existence"] = val

        sortable_rows.append((group, method_disp, mode_disp, row))

    # Sort: by group (INCLUDED_GROUPS order), then model (display), then mode (display)
    group_order = {g: i for i, g in enumerate(INCLUDED_GROUPS.keys())}
    sortable_rows.sort(key=lambda t: (group_order.get(t[0], 999), t[1].lower(), t[2].lower()))
    return [t[3] for t in sortable_rows]


def print_tsv_js_const(
    const_name: str,
    rows: List[Dict[str, Any]],
    columns: List[Tuple[str, Optional[str]]],
) -> None:
    """
    Print a JS constant with TSV content:
      const <name> = `
      col1\tcol2\t...
      v1\tv2\t...
      `.trim();
    columns: list of (header_text, key)
      - If key is "Method"/"model"/"mode" they are taken from row fields
      - If key is None -> compute Prompt Align Avg over available PA submetrics
      - Other keys are direct metric dict keys
    """
    text=f"const {const_name} = `"
    header = "\t".join([h for h, _ in columns])
    text += f"{header}\n"
    for r in rows:
        cells: List[str] = []
        for h, key in columns:
            if key is None:
                v = _compute_pa_avg_from_row(r)
                cells.append("" if v is None else f"{v:.3f}")
            else:
                v = r.get(key)
                cells.append("" if v is None else str(v))
        text += f'\t'.join(cells) + "\n"
    text += "`.trim();"
    return text

def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Summarize leaderboard from lite/full CSVs and compute ranks.")
    p.add_argument("--lite", default="data/result/bench_results_lite.csv", help="Path to lite CSV")
    p.add_argument("--full", default="data/result/bench_results_full.csv", help="Path to full CSV")
    p.add_argument("--which", default="all", choices=["all", "combined", "lite", "full"], help="Which leaderboard(s) to print")
    p.add_argument("--out", default="", help="Optional output CSV path for combined leaderboard")
    p.add_argument("--out-lite", dest="out_lite", default="", help="Optional output CSV path for lite leaderboard")
    p.add_argument("--out-full", dest="out_full", default="", help="Optional output CSV path for full leaderboard")
    args = p.parse_args(argv)

    lite_path = Path(args.lite)
    full_path = Path(args.full)
    lite_rows = read_csv_rows(lite_path)
    full_rows = read_csv_rows(full_path)

    # Build three datasets: combined, lite-only, full-only
    # merged_combined = merge_splits(lite_rows, full_rows)
    lite_agg = aggregate_split_rows(lite_rows)
    full_agg = aggregate_split_rows(full_rows)

    # Restrict to included methods
    allowed_methods: Set[str] = set(sum(INCLUDED_GROUPS.values(), []))

    # Compute rows for each dataset
    # rows_combined = compute_leaderboard_rows(merged_combined, allowed_methods)
    rows_lite = compute_leaderboard_rows(lite_agg, allowed_methods)
    rows_full = compute_leaderboard_rows(full_agg, allowed_methods)

    # Print selections
    # if args.which in ("all", "combined"):
    #     print_section("Leaderboard (Combined lite+full)", rows_combined)
    if args.which in ("all", "lite"):
        print_section("Leaderboard (Lite only)", rows_lite)
    if args.which in ("all", "full"):
        print_section("Leaderboard (Full only)", rows_full)
    text = ""
    # Additional JS TSV outputs for raw metric tables
    if args.which in ("all", "full"):
        raw_full_rows = build_raw_metric_rows(full_agg, allowed_methods)
        columns_full: List[Tuple[str, Optional[str]]] = [
            ("Method", "Method"),
            ("model", "model"),
            ("mode", "mode"),
            ("Cross CSD Score (Ref-Gen)", "metrics.csd.metrics.cross_csd"),
            ("Self CSD Score (Gen-Gen)", "metrics.csd.metrics.self_csd"),
            ("Cross Cref Score (Ref-Gen)", "metrics.cids.metrics.cids_cross_mean"),
            ("Self Cref Score (Gen-Gen)", "metrics.cids.metrics.cids_self_mean"),
            ("Aesthetics Score", "metrics.aesthetic.metrics.aesthetic_score"),
            ("Inception Score", "metrics.diversity.metrics.inception_score"),
            ("Prompt Align (Scene)", "metrics.prompt_align.metrics.scene"),
            ("Prompt Align (Camera)", "metrics.prompt_align.metrics.camera"),
            ("Prompt Align (Character Existence|Number) OCCM", "metrics.cids.metrics.occm"),
            ("Prompt Align (Global Character Action|Script)", "metrics.prompt_align.metrics.character_action"),
            ("Prompt Align (Local/Single Character Action|Script)", "metrics.cids.metrics.single_character_action"),
            ("Prompt Align Avg", None),
        ]
        text += print_tsv_js_const("fullDatasetTSV", raw_full_rows, columns_full)

    if args.which in ("all", "lite"):
        raw_lite_rows = build_raw_metric_rows(lite_agg, allowed_methods)
        columns_lite: List[Tuple[str, Optional[str]]] = [
            ("Method", "Method"),
            ("model", "model"),
            ("mode", "mode"),
            ("Cross CSD Score (Ref-Gen)", "metrics.csd.metrics.cross_csd"),
            ("Self CSD Score (Gen-Gen)", "metrics.csd.metrics.self_csd"),
            ("Cross Cref Score (Ref-Gen)", "metrics.cids.metrics.cids_cross_mean"),
            ("Self Cref Score (Gen-Gen)", "metrics.cids.metrics.cids_self_mean"),
            ("Aesthetics Score", "metrics.aesthetic.metrics.aesthetic_score"),
            ("Inception Score", "metrics.diversity.metrics.inception_score"),
            ("Prompt Align (Scene)", "metrics.prompt_align.metrics.scene"),
            ("Prompt Align (Camera)", "metrics.prompt_align.metrics.camera"),
            ("Prompt Align (Character Existence|Number) OCCM", "metrics.cids.metrics.occm"),
            ("Prompt Align (Global Character Action|Script)", "metrics.prompt_align.metrics.character_action"),
            ("Prompt Align (Local/Single Character Action|Script)", "metrics.cids.metrics.single_character_action"),
            ("Prompt Align Avg（Alignment score）", None),
        ]
        text += print_tsv_js_const("liteDatasetTSV", raw_lite_rows, columns_lite)

    # Optional write
    # if args.out:
    #     write_output_csv(rows_combined, Path(args.out))
    if args.out_lite:
        write_output_csv(rows_lite, Path(args.out_lite))
    if args.out_full:
        write_output_csv(rows_full, Path(args.out_full))
    open("data/result/leaderboard_data.js", "w", encoding="utf-8").write(text + "\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())