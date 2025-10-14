#!/usr/bin/env python3
import os
import argparse
from typing import List
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
# Reuse discovery and config helpers from bench_run
from bench_run import discover_languages, discover_modes, discover_timestamps, merge_config_with_args, load_config

from vistorybench.dataset_loader.dataset_load import StoryDataset
from vistorybench.result_management.manager import ResultManager
from vistorybench.bench.content.cids_evaluator import CIDSEvaluator
from vistorybench.bench.prompt_align.prompt_align_evaluator import PromptAlignEvaluator

def parse_args():
    parser = argparse.ArgumentParser(description="Batch evaluate CIDS and PromptAlign for specific methods/stories over multiple runs")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    parser.add_argument("--methods", type=str, nargs="+", default=["storyadapter"], help="Methods to evaluate")
    parser.add_argument("--stories", type=str, default="01,09,27,41,53", help="Comma-sstorydiffusioneparated story IDs to evaluate (e.g., 01,09,27,41,53)")
    parser.add_argument("--runs", type=int, default=5, help="Number of times to repeat evaluation for each combination")
    parser.add_argument("--language", type=str, default=None, choices=["en", "ch"], help="Language to evaluate; if omitted, auto-discover")
    parser.add_argument("--mode", type=str, default=None, help="Mode to evaluate; if omitted, auto-discover")
    parser.add_argument("--method_modes", type=str, default='uno=base,storyadapter=img_ref_results_xl,storydiffusion=original', help="Per-method mode mapping, e.g., 'uno=base,storyadapter=sdxl,storydiffusion=sdxl'")
    parser.add_argument("--timestamp", type=str, default=None, help="Outputs timestamp to evaluate; if omitted, pick the latest")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False, help="Align result timestamp to outputs timestamp (True) or create new timestamp per run (False)")
    parser.add_argument("--dataset_path", type=str, default=None, help="Override dataset root path")
    parser.add_argument("--outputs_path", type=str, default=None, help="Override outputs root path")
    parser.add_argument("--pretrain_path", type=str, default=None, help="Override pretrain root path")
    parser.add_argument("--result_path", type=str, default=None, help="Override bench results root path")
    parser.add_argument("--api_key", type=str, default=os.environ.get("VISTORYBENCH_API_KEY"), help="API key for PromptAlign GPT-V; defaults to env VISTORYBENCH_API_KEY")
    parser.add_argument("--base_url", type=str, default=None, help="Base URL for GPT-V service")
    parser.add_argument("--model_id", type=str, default=None, help="Model ID for GPT-V")
    return parser.parse_args()

def load_dataset_subset(dataset_root: str, language: str, story_ids: List[str]):
    ds = StoryDataset(os.path.join(dataset_root, "ViStory"))
    # ensure IDs are zero-padded strings
    story_ids = [sid.zfill(2) for sid in story_ids]
    data = ds.load_stories(story_ids, language)
    return ds, data

def parse_method_modes(method_modes_str: str) -> dict[str, str]:
    """
    Parse per-method mode mapping from CLI string, e.g.:
      'uno=base,storyadapter=sdxl,storydiffusion=sdxl'
    """
    mapping: dict[str, str] = {}
    if not method_modes_str:
        return mapping
    for part in method_modes_str.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        method, mode = part.split("=", 1)
        method = method.strip()
        mode = mode.strip()
        if method and mode:
            mapping[method] = mode
    return mapping

def main():
    args = parse_args()
    # Load YAML and merge CLI
    config = load_config(args.config)
    merged = merge_config_with_args(config, args)
    cli = merged.get("cli_args", {}) if isinstance(merged, dict) else {}
    # Resolve core paths with CLI precedence
    dataset_root = cli.get("dataset_path") or config.get("core", {}).get("paths", {}).get("dataset", "data/dataset")
    outputs_root = cli.get("outputs_path") or config.get("core", {}).get("paths", {}).get("outputs", "data/outputs")
    results_root = cli.get("result_path") or config.get("core", {}).get("paths", {}).get("results", "data/bench_results")

    methods = args.methods
    stories = [sid.strip() for sid in args.stories.split(",") if sid.strip()]

    if not methods:
        raise SystemExit("No methods specified")
    if not stories:
        raise SystemExit("No story IDs specified")

    for method_name in methods:
        # Languages
        langs = [args.language] if args.language else discover_languages(outputs_root, method_name)
        if not langs:
            print(f"Warning: No languages found for method '{method_name}' under {outputs_root}. Skipping.")
            continue
        for lang in langs:
            # Modes (per-method override > global --mode > auto-discover)
            discovered_modes = discover_modes(outputs_root, method_name, lang)
            selected_mode = None
            if args.method_modes:
                method_modes_map = parse_method_modes(args.method_modes)
                selected_mode = method_modes_map.get(method_name)
            if selected_mode:
                if selected_mode in discovered_modes:
                    modes = [selected_mode]
                else:
                    print(f"Warning: Selected mode '{selected_mode}' not found for {method_name}/{lang}. Available: {discovered_modes}. Skipping.")
                    continue
            else:
                modes = [args.mode] if args.mode else discovered_modes
            if not modes:
                print(f"Warning: No modes found for method '{method_name}' language '{lang}'. Skipping.")
                continue
            for mode in modes:
                # Timestamp
                if args.timestamp:
                    ts_list = [args.timestamp]
                else:
                    found_ts = discover_timestamps(outputs_root, method_name, lang, mode)
                    ts_list = found_ts[-1:] if found_ts else []
                if not ts_list:
                    print(f"Warning: No timestamps found for {method_name}/{lang}/{mode}. Skipping.")
                    continue
                for ts_out in ts_list:
                    for run_idx in range(1, args.runs + 1):
                        # Result timestamp policy
                        results_ts = ts_out if args.resume else ResultManager.create_timestamp()
                        print(f"=== Run {run_idx}/{args.runs} :: method={method_name} language={lang} mode={mode} ts_out={ts_out} -> result_ts={results_ts} ===")
                        # Initialize ResultManager
                        rm = ResultManager(method_name=method_name, mode=mode, language=lang, timestamp=results_ts, base_path=results_root, outputs_timestamp=ts_out)
                        merged["bench_result_run_dir"] = rm.result_path
                        # Evaluators
                        evaluators = {
                            "cids": CIDSEvaluator(config=merged, timestamp=rm.timestamp, mode=rm.mode, language=rm.language, outputs_timestamp=ts_out),
                            "prompt_align": PromptAlignEvaluator(config=merged, timestamp=rm.timestamp, mode=rm.mode, language=rm.language, outputs_timestamp=ts_out),
                        }
                        # Provide CLI overrides for GPT-V explicitly via merged.cli_args
                        merged["cli_args"]["api_key"] = args.api_key
                        merged["cli_args"]["base_url"] = args.base_url or merged["cli_args"].get("base_url")
                        merged["cli_args"]["model_id"] = args.model_id or merged["cli_args"].get("model_id")
                        # Dataset subset
                        ds, stories_data = load_dataset_subset(dataset_root, lang, stories)
                        # Evaluate each requested story
                        for story_id, story_data in stories_data.items():
                            story_id = str(story_id).zfill(2)
                            try:
                                print(f"--- Evaluating Story: {story_id} for {method_name}/{lang}/{mode}/{ts_out} ---")
                                for metric_name, evaluator in evaluators.items():
                                    result = evaluator.evaluate(method=method_name, story_id=story_id)
                                    if result:
                                        rm.save_story_result(metric_name, story_id, result)
                                        try:
                                            items = evaluator.build_item_records(method=method_name, story_id=story_id, story_result=result)
                                            rm.append_items(metric_name, items)
                                        except Exception as _e:
                                            print(f"Warning: failed to append item-level records for {metric_name}, story {story_id}: {_e}")
                                print(f"Story {story_id} complete.")
                            except Exception as e:
                                print(f"Error during evaluation for {method_name}/{lang}/{mode}/{ts_out} story {story_id}: {e}")
                        # Save summary
                        rm.compute_and_save_summary()
                        print(f"Run {run_idx} complete. Results at: {rm.result_path}")

if __name__ == "__main__":
    main()