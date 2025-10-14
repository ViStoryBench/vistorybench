#!/usr/bin/env python3
import os
import json
import argparse
from typing import List, Dict, Any, Optional
from datetime import datetime
from statistics import stdev
import math
import csv


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze stability for CIDS and PromptAlign metrics across multiple runs")
    parser.add_argument("--methods", type=str, nargs="+", default=["uno", "storyadapter", "storydiffusion"], help="Methods to analyze")
    parser.add_argument("--stories", type=str, default="01,09,27,41,53", help="Comma-separated story IDs (zero-padded, e.g., 01,09,27,41,53)")
    parser.add_argument("--method_modes", type=str, default="uno=base,storyadapter=img_ref_results_xl,storydiffusion=original", help="Per-method mode mapping, e.g., 'uno=base,storyadapter=sdxl,storydiffusion=sdxl'")
    parser.add_argument("--language", type=str, default=None, choices=["en", "ch"], help="Language; if omitted, auto-discover from bench_results")
    parser.add_argument("--results_root", type=str, default="data/bench_results", help="Bench results root")
    parser.add_argument("--use_latest_n", type=int, default=5, help="Pick latest N valid runs per method/mode/language")
    parser.add_argument("--errorbar", type=str, default="sem", choices=["sem", "std", "ci95"], help="Type of error bar to report")
    parser.add_argument("--output_csv", type=str, default=None, help="Output CSV path; default to data/result/stability/stability_cids_promptalign_{ts}.csv")
    parser.add_argument("--plot", action="store_true", help="Also save bar plots with error bars")
    return parser.parse_args()


def parse_method_modes(method_modes_str: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not method_modes_str:
        return mapping
    for part in method_modes_str.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        m, mode = part.split("=", 1)
        m = m.strip()
        mode = mode.strip()
        if m and mode:
            mapping[m] = mode
    return mapping


def read_json(path: str) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def discover_result_languages(results_root: str, method: str, mode: str) -> List[str]:
    langs_dir = os.path.join(results_root, method, mode)
    if not os.path.isdir(langs_dir):
        return []
    return [name for name in os.listdir(langs_dir) if os.path.isdir(os.path.join(langs_dir, name))]


def list_result_timestamps(results_root: str, method: str, mode: str, language: str) -> List[str]:
    run_root = os.path.join(results_root, method, mode, language)
    if not os.path.isdir(run_root):
        return []
    # Newest first
    return sorted([d for d in os.listdir(run_root) if os.path.isdir(os.path.join(run_root, d))], reverse=True)


def run_has_all_stories(results_root: str, method: str, mode: str, language: str, ts: str, story_ids: List[str]) -> bool:
    run_dir = os.path.join(results_root, method, mode, language, ts)

    def has_for_metric(metric: str) -> bool:
        story_file = os.path.join(run_dir, "metrics", metric, "story_results.json")
        data = read_json(story_file)
        if not isinstance(data, dict):
            return False
        keys = {str(k).zfill(2) for k in data.keys()}
        return all(sid in keys for sid in story_ids)

    # Ensure both metrics have all requested stories
    return has_for_metric("prompt_align") and has_for_metric("cids")


def extract_metric_from_run(results_root: str, method: str, mode: str, language: str, ts: str, metric: str, key: str, story_ids: List[str]) -> Optional[float]:
    """
    Prefer averaging metric[key] across the provided story_ids from story_results.json.
    If missing, fallback to summary.json value.
    """
    run_dir = os.path.join(results_root, method, mode, language, ts)
    # Primary: story_results.json average over selected stories
    story_file = os.path.join(run_dir, "metrics", metric, "story_results.json")
    data = read_json(story_file)
    if isinstance(data, dict):
        vals: List[float] = []
        for sid in story_ids:
            obj = data.get(str(sid)) or data.get(int(sid))
            if isinstance(obj, dict):
                m = obj.get("metrics") if isinstance(obj.get("metrics"), dict) else obj
                v = m.get(key) if isinstance(m, dict) else None
                if isinstance(v, (int, float)):
                    vals.append(float(v))
        if len(vals) == len(story_ids):
            return sum(vals) / len(vals)
        if vals:
            # Partial fallback: average what we have
            return sum(vals) / len(vals)

    # Fallback: summary.json (dataset-level average, may include extra stories)
    summary = read_json(os.path.join(run_dir, "summary.json"))
    try:
        if isinstance(summary, dict):
            v = summary.get("metrics", {}).get(metric, {}).get("metrics", {}).get(key)
            if isinstance(v, (int, float)):
                return float(v)
    except Exception:
        pass
    return None


def compute_stats(values: List[float]) -> Dict[str, float]:
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "sem": float("nan"), "ci95": float("nan")}
    mu = sum(values) / n
    if n <= 1:
        s = 0.0
    else:
        try:
            s = float(stdev(values))  # sample std
        except Exception:
            var = sum((x - mu) ** 2 for x in values) / (n - 1)
            s = math.sqrt(max(var, 0.0))
    sem = s / math.sqrt(n) if n > 0 else float("nan")
    ci95 = 1.96 * sem if n > 0 else float("nan")
    return {"n": n, "mean": mu, "std": s, "sem": sem, "ci95": ci95}


def choose_errorbar(stats: Dict[str, float], typ: str) -> float:
    if typ == "std":
        return stats["std"]
    if typ == "ci95":
        return stats["ci95"]
    return stats["sem"]


def main():
    args = parse_args()
    methods = args.methods or []
    stories = [sid.strip().zfill(2) for sid in args.stories.split(",") if sid.strip()]
    if not methods:
        raise SystemExit("No methods specified.")
    if not stories:
        raise SystemExit("No story IDs specified.")

    # target metrics
    metric_specs = [
        ("prompt_align", "scene"),
        ("prompt_align", "character_action"),
        ("prompt_align", "camera"),
        ("cids", "single_character_action"),
    ]
    method_modes = parse_method_modes(args.method_modes)

    # Output paths
    out_dir = os.path.join("data", "result", "stability")
    os.makedirs(out_dir, exist_ok=True)
    ts_label = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = args.output_csv or os.path.join(out_dir, f"stability_cids_promptalign_{ts_label}.csv")

    rows: List[Dict[str, Any]] = []
    per_metric_method_stats: Dict[str, Dict[str, Dict[str, float]]] = {}

    for method in methods:
        mode = method_modes.get(method)
        if not mode:
            print(f"Warning: missing mode for method '{method}', skip.")
            continue

        # Language selection
        langs = [args.language] if args.language else discover_result_languages(args.results_root, method, mode)
        if not langs:
            langs = ["en"]

        for lang in langs:
            all_ts = list_result_timestamps(args.results_root, method, mode, lang)
            if not all_ts:
                print(f"Warning: no runs at {args.results_root}/{method}/{mode}/{lang}")
                continue

            # Pick latest N valid runs that contain all target stories for both metrics
            selected_ts: List[str] = []
            for ts in all_ts:
                if run_has_all_stories(args.results_root, method, mode, lang, ts, stories):
                    selected_ts.append(ts)
                if len(selected_ts) >= int(args.use_latest_n):
                    break

            if not selected_ts:
                print(f"Warning: no valid runs with all stories {stories} for {method}/{mode}/{lang}")
                continue

            # Collect per-run values
            metric_values: Dict[str, List[float]] = {f"{m}.{k}": [] for (m, k) in metric_specs}
            for ts in selected_ts:
                for (m, k) in metric_specs:
                    val = extract_metric_from_run(args.results_root, method, mode, lang, ts, m, k, stories)
                    if isinstance(val, (int, float)):
                        metric_values[f"{m}.{k}"].append(float(val))

            # Compute stats and fill rows
            print(f"== {method}/{mode}/{lang} using {len(selected_ts)} runs: {', '.join(selected_ts)} ==")
            for mk, vals in metric_values.items():
                stats = compute_stats(vals)
                err = choose_errorbar(stats, args.errorbar)
                print(f"{mk}: mean={stats['mean']:.4f} std={stats['std']:.4f} sem={stats['sem']:.4f} ci95={stats['ci95']:.4f} errorbar({args.errorbar})={err:.4f} (n={stats['n']})")
                row = {
                    "method": method,
                    "mode": mode,
                    "language": lang,
                    "stories": ",".join(stories),
                    "metric": mk,
                    "n_runs": stats["n"],
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "sem": stats["sem"],
                    "ci95": stats["ci95"],
                    "errorbar_type": args.errorbar,
                    "errorbar": err,
                    "runs_used": "|".join(selected_ts),
                }
                rows.append(row)
                per_metric_method_stats.setdefault(mk, {})[f"{method}/{mode}/{lang}"] = {"mean": stats["mean"], "error": err, "n": stats["n"]}

    # Write CSV
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["method", "mode", "language", "stories", "metric", "n_runs", "mean", "std", "sem", "ci95", "errorbar_type", "errorbar", "runs_used"],
            )
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"[Stability] CSV saved: {csv_path}")
    except Exception as e:
        print(f"Error writing CSV: {e}")

    # Optional plots
    if args.plot:
        try:
            import matplotlib.pyplot as plt  # optional
            for mk, meth_stats in per_metric_method_stats.items():
                labels = list(meth_stats.keys())
                means = [meth_stats[l]["mean"] for l in labels]
                errs = [meth_stats[l]["error"] for l in labels]
                plt.figure(figsize=(7, 4))
                plt.bar(labels, means, yerr=errs, capsize=5)
                plt.title(f"{mk} stability ({args.errorbar})")
                plt.xticks(rotation=20, ha="right")
                plt.tight_layout()
                out_png = os.path.join(out_dir, f"stability_{mk.replace('.', '_')}_{ts_label}.png")
                plt.savefig(out_png, dpi=150)
                plt.close()
                print(f"[Stability] Plot saved: {out_png}")
        except Exception as e:
            print(f"Warning: plotting skipped due to error: {e}")


if __name__ == "__main__":
    main()