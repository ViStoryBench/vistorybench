import os
import argparse
import yaml
from pathlib import Path
import torch
import sys
import time
from vistorybench.dataset_loader.read_outputs import load_outputs
from vistorybench.result_management.manager import ResultManager
from vistorybench.dataset_loader.dataset_load import StoryDataset

# Import all concrete evaluator classes
from vistorybench.bench.content.cids_evaluator import CIDSEvaluator
from vistorybench.bench.style.csd_evaluator import CSDEvaluator
from vistorybench.bench.diversity.diversity_evaluator import DiversityEvaluator
from vistorybench.bench.quality.aesthetic_evaluator import AestheticEvaluator
from vistorybench.bench.prompt_align.prompt_align_evaluator import PromptAlignEvaluator
from dotenv import load_dotenv

load_dotenv()


# Evaluator registry
EVALUATOR_REGISTRY = {
    'cids': CIDSEvaluator,
    'csd': CSDEvaluator,
    'diversity': DiversityEvaluator,
    'aesthetic': AestheticEvaluator,
    'prompt_align': PromptAlignEvaluator,
}

def blue_print(text, bright=True):
    color_code = "\033[94m" if bright else "\033[34m"
    print(f"{color_code}{text}\033[0m")

def yellow_print(text):
    print(f"\033[93m{text}\033[0m")

def green_print(text):
    print(f"\033[92m{text}\033[0m")

def load_dataset(_dataset_path, dataset_name, language, split='full'):
    dataset_path = f"{_dataset_path}/{dataset_name}"
    dataset = StoryDataset(dataset_path)
    story_name_list = dataset.get_story_name_list(split=split)
    print(f'\nStory name list: {story_name_list}')
    stories_data = dataset.load_stories(story_name_list, language)
    return stories_data

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def merge_config_with_args(config, args):
    """Merge configuration file with command line arguments, args kept in a non-overlapping namespace (cli_args)."""
    merged_config = config.copy() if isinstance(config, dict) else {}

    # Ensure core structure exists
    if not merged_config.get('core'):
        merged_config['core'] = {}
    if not merged_config['core'].get('paths'):
        merged_config['core']['paths'] = {}

    # Do not override YAML core.paths with CLI here.
    # Keep YAML as source-of-truth; CLI paths live under merged_config['cli_args'] and are resolved where needed.

    # Ensure runtime device default exists (do not read from CLI here)
    if not merged_config['core'].get('runtime'):
        merged_config['core']['runtime'] = {}
    if not merged_config['core']['runtime'].get('device'):
        merged_config['core']['runtime']['device'] = 'cuda'

    # Attach all CLI args under a dedicated namespace to avoid key overlap with YAML
    merged_config['cli_args'] = {
        'dataset_path': getattr(args, 'dataset_path', None),
        'outputs_path': getattr(args, 'outputs_path', None),
        'pretrain_path': getattr(args, 'pretrain_path', None),
        'result_path': getattr(args, 'result_path', None),
        'api_key': getattr(args, 'api_key', None),
        'base_url': getattr(args, 'base_url', None),
        'model_id': getattr(args, 'model_id', None),
        'method': getattr(args, 'method', None),
        'metrics': getattr(args, 'metrics', None),
        'language': getattr(args, 'language', None),
        'timestamp': getattr(args, 'timestamp', None),
        'mode': getattr(args, 'mode', None),
        'resume': getattr(args, 'resume', None),
        # Fast CIDS CLI overrides: only set when explicitly provided to avoid overriding YAML defaults
        'fast_cids': True if '--fast_cids' in sys.argv else None,
        'cids_batch_size': getattr(args, 'cids_batch_size', None) if '--cids_batch_size' in sys.argv else None,
        'cids_block_size': getattr(args, 'cids_block_size', None) if '--cids_block_size' in sys.argv else None,
        'cids_fast_only_copypaste': True if '--cids_fast_only_copypaste' in sys.argv else None,
        'cids_fast_parallel_shots': getattr(args, 'cids_fast_parallel_shots', None) if '--cids_fast_parallel_shots' in sys.argv else None,
        'cids_fast_num_workers': getattr(args, 'cids_fast_num_workers', None) if '--cids_fast_num_workers' in sys.argv else None,
    }

    return merged_config

# --- Discovery helpers (outputs directory introspection) ---
def list_subdirs(path: str):
    try:
        return sorted(
            [
                name
                for name in os.listdir(path)
                if not name.startswith(".") and os.path.isdir(os.path.join(path, name))
            ]
        )
    except Exception:
        return []


def discover_languages(outputs_root: str, method: str):
    """
    Discover languages under outputs for a given method.
    Supports both layouts:
      - {outputs_root}/{method}/{language}/{mode}/{timestamp}/...
      - {outputs_root}/{method}/{mode}/{language}/{timestamp}/...
    """
    method_dir = os.path.join(outputs_root, method)
    # Prefer direct language directories under method
    direct = [d for d in list_subdirs(method_dir) if d in ("en", "ch")]
    if direct:
        return direct

    # Fallback: search under possible mode dirs for language names
    found = set()
    for sub in list_subdirs(method_dir):
        for d in list_subdirs(os.path.join(method_dir, sub)):
            if d in ("en", "ch"):
                found.add(d)
    return sorted(found)


def discover_modes(outputs_root: str, method: str, language: str):
    """
    Discover modes for a given (method, language).
    Supports both layouts:
      - {outputs_root}/{method}/{language}/{mode}/...
      - {outputs_root}/{method}/{mode}/{language}/...
    """
    # Prefer language-first layout: {method}/{language}/<mode>/
    lang_dir = os.path.join(outputs_root, method, language)
    if os.path.isdir(lang_dir):
        return list_subdirs(lang_dir)

    # Fallback: mode-first layout: {method}/{mode}/{language}/
    method_dir = os.path.join(outputs_root, method)
    found = []
    for m in list_subdirs(method_dir):
        if os.path.isdir(os.path.join(method_dir, m, language)):
            found.append(m)
    return sorted(found)


def discover_timestamps(outputs_root: str, method: str, language: str, mode: str):
    """
    Discover timestamps for a given (method, language, mode).
    Supports both layouts.
    """
    ts = set()
    p1 = os.path.join(outputs_root, method, language, mode)
    p2 = os.path.join(outputs_root, method, mode, language)
    for p in (p1, p2):
        for d in list_subdirs(p):
            ts.add(d)
    return sorted(ts)
def main():
    base_parser = argparse.ArgumentParser(description='Application path configuration', add_help=False)
    base_parser.add_argument('--config', type=str, default=f'config.yaml', help='Path to configuration file')
    base_args, _ = base_parser.parse_known_args()
    config = load_config(base_args.config)

    parser = argparse.ArgumentParser(
        description='ViStoryBench Evaluation Tool',
        parents=[base_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Path and API configurations with fallbacks from config
    parser.add_argument('--dataset_path', type=str, default=(config.get('core', {}).get('paths', {}).get('dataset', 'data/dataset')))
    parser.add_argument('--outputs_path', type=str, default=(config.get('core', {}).get('paths', {}).get('outputs', 'data/outputs')))
    parser.add_argument('--pretrain_path', type=str, default=(config.get('core', {}).get('paths', {}).get('pretrain', 'data/pretrain')))
    parser.add_argument('--result_path', type=str, default=(config.get('core', {}).get('paths', {}).get('results', 'data/bench_results')))
    parser.add_argument('--api_key', type=str, default=os.environ.get('API_KEY'), help='API key for external services')
    parser.add_argument('--base_url', type=str, default=os.environ.get('BASE_URL'), help='Base URL for API services')
    parser.add_argument('--model_id', type=str, default=os.environ.get('MODEL_ID'), help='Model ID for evaluation')

    # Evaluation settings
    parser.add_argument('--method', type=str, nargs='+', default=None, help='Method name(s) to evaluate. Accept multiple values.')
    parser.add_argument('--metrics', type=str, nargs='+', choices=list(EVALUATOR_REGISTRY.keys()), default=None, help='List of metrics to run. Runs all if not specified.')
    parser.add_argument('--language', type=str, choices=['en', 'ch', 'all'], default=None, help='Language to evaluate. None/all => enumerate all available languages.')
    parser.add_argument('--split', type=str, choices=['full', 'lite'], default='full', help='Dataset split to use.')
    parser.add_argument('--timestamp', type=str, default=None, help='Specific timestamp to evaluate. If omitted, enumerate all available timestamps for each language/mode.')
    parser.add_argument('--mode', type=str, default=None, help='Mode for method. None => enumerate all available modes.')
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction, default=False, help='Only controls result timestamp alignment. True: align result timestamp to output timestamp. False: create a new result timestamp for each combination. Does not affect enumeration.')

    # Fast CIDS options (defaults preserve current behavior; CLI overrides are read via cli_args)
    parser.add_argument('--fast_cids', action='store_true', help='Enable fast CIDS copy-paste path (CLIP single-model only).')
    parser.add_argument('--cids_batch_size', type=int, default=32, help='Batch size for fast CIDS embedding.')
    parser.add_argument('--cids_block_size', type=int, default=4096, help='Block size for fast similarity matmul.')
    # Extended fast options (default off / 0)
    parser.add_argument('--cids_fast_only_copypaste', action='store_true', help='Fast mode: compute only Copy-Paste score and skip other CIDS stats.')
    parser.add_argument('--cids_fast_parallel_shots', type=int, default=0, help='Parallel shots for DINO detect+crop (0 to disable).')
    parser.add_argument('--cids_fast_num_workers', type=int, default=0, help='Num workers for CPU-side CLIP preprocessing (DataLoader).')
    
    args = parser.parse_args()
    
    # Merge config with args, args take precedence
    merged_config = merge_config_with_args(config, args)
    if args.method is None:
        args.method = os.listdir(args.outputs_path)

    # --- Main Evaluation Logic ---
    # Use unified config paths for results, preferring CLI overrides without mutating YAML
    _cli = merged_config.get('cli_args', {}) if isinstance(merged_config, dict) else {}
    results_root = _cli.get('result_path') or merged_config.get('core', {}).get('paths', {}).get('results', 'data/bench_results')

    # Resolve dataset/outputs roots
    _cli = merged_config.get('cli_args', {}) if isinstance(merged_config, dict) else {}
    dataset_root = _cli.get('dataset_path') or merged_config.get('core', {}).get('paths', {}).get('dataset', 'data/dataset')
    outputs_root = _cli.get('outputs_path') or merged_config.get('core', {}).get('paths', {}).get('outputs', 'data/outputs')

    # Determine metrics
    requested_metrics = args.metrics or list(EVALUATOR_REGISTRY.keys())

    # Normalize methods to list
    methods_list = args.method if isinstance(args.method, (list, tuple)) else [args.method]

    for method_name in methods_list:
        # Languages to run
        if args.language in (None, 'all'):
            langs_to_run = discover_languages(outputs_root, method_name)
        else:
            langs_to_run = [args.language]

        if not langs_to_run:
            yellow_print(f"Warning: No languages found under outputs for method '{method_name}'. Skipping.")
            continue

        for lang in langs_to_run:
            # Load dataset per-language
            stories_data = load_dataset(dataset_root, 'ViStory', lang, args.split)

            # Modes to run
            modes_to_run = discover_modes(outputs_root, method_name, lang) if args.mode is None else [args.mode]
            if not modes_to_run:
                yellow_print(f"Warning: No modes found for method '{method_name}', language '{lang}'. Skipping.")
                continue

            for mode in modes_to_run:
                # Timestamps to run
                if args.timestamp:
                    timestamps_to_run = [args.timestamp]
                else:
                    timestamps_to_run = discover_timestamps(outputs_root, method_name, lang, mode)

                if not timestamps_to_run:
                    yellow_print(f"Warning: No timestamps found for method '{method_name}', language '{lang}', mode '{mode}'. Skipping.")
                    continue

                for ts_out in timestamps_to_run:
                    # Result timestamp policy (Scheme B)
                    results_ts = ts_out if args.resume else ResultManager.create_timestamp()

                    blue_print(f"=== Evaluating: method={method_name} language={lang} mode={mode} ts={ts_out} -> result_ts={results_ts} ===")

                    # Initialize ResultManager per combination
                    result_manager = ResultManager(
                        method_name=method_name,
                        mode=mode,
                        language=lang,
                        timestamp=False if args.resume else results_ts,
                        base_path=results_root,
                        outputs_timestamp=ts_out
                    )

                    # Load outputs for this combination
                    stories_outputs = load_outputs(
                        outputs_root=outputs_root,
                        methods=[method_name],
                        languages=[lang],
                        modes=[mode],
                        return_latest=False,
                        timestamps=[ts_out]
                    )

                    # Add result path from manager into config for evaluators
                    merged_config['bench_result_run_dir'] = result_manager.result_path

                    # Initialize evaluators for this combination
                    evaluators = {}
                    for metric_name in requested_metrics:
                        if metric_name in EVALUATOR_REGISTRY:
                            evaluators[metric_name] = EVALUATOR_REGISTRY[metric_name](
                                config=merged_config,
                                timestamp=result_manager.timestamp,
                                mode=result_manager.mode,
                                language=result_manager.language,
                                outputs_timestamp=ts_out
                            )

                    # Story-level metrics
                    for story_id, story_data in stories_data.items():
                        story_id = str(story_id)
                        if story_id not in stories_outputs:
                            yellow_print(f"Warning: Story '{story_id}' not found in outputs for {method_name}/{lang}/{mode}/{ts_out}. Skipping.")
                            continue

                        blue_print(f"--- Evaluating Story: {story_id} for {method_name}/{lang}/{mode}/{ts_out} ---")

                        for metric_name, evaluator in evaluators.items():
                            if metric_name == 'diversity':
                                continue  # Skip method-level evaluators here
                            finish_data=result_manager.load_metric_result(metric_name, 'story')
                            if finish_data and story_id in finish_data and finish_data[story_id].get('status', '')=='complete':
                                green_print(f"Skipping {metric_name} for story {story_id}, already completed.")
                                continue
                            try:
                                green_print(f"Running {metric_name} evaluation...")
                                result = evaluator.evaluate(method=method_name, story_id=story_id)
                                if result:
                                    # Save story-level result (wrapped by ResultManager)
                                    result_manager.save_story_result(metric_name, story_id, result)

                                    # Append item-level records when available (delegated to evaluator)
                                    try:
                                        items = evaluator.build_item_records(
                                            method=method_name,
                                            story_id=story_id,
                                            story_result=result,
                                        ) 
                                        result_manager.append_items(metric_name, items)
                                    except Exception as _e:
                                        yellow_print(f"Warning: failed to append item-level records for {metric_name}, story {story_id}: {_e}")

                                green_print(f"{metric_name} evaluation complete.")
                            except Exception as e:
                                yellow_print(f"Error during {metric_name} evaluation for story {story_id}: {e}")

                    # Handle method-level evaluators like diversity
                    if 'diversity' in evaluators:
                        green_print("Running diversity evaluation for this method/mode/language/timestamp combination...")
                        try:
                            diversity_evaluator = evaluators['diversity']
                            result = diversity_evaluator.evaluate(
                                method=method_name,
                                mode=mode,
                                language=lang,
                                timestamp=ts_out,
                                stories_outputs=stories_outputs,
                            )
                            # Save dataset-level metric for diversity
                            ds_record = {
                                "metric": {"name": "diversity"},
                                "scope": {"level": "dataset"},
                                "metrics": result
                            }
                            result_manager.save_dataset_metric("diversity", ds_record)
                            green_print("Diversity evaluation complete.")
                        except Exception as e:
                            yellow_print(f"Error during diversity evaluation: {e}")

                    # Finalize and save summary (cross-metric dataset-level)
                    result_manager.compute_and_save_summary()
                    green_print(f"Evaluation complete for {method_name}/{lang}/{mode}/{results_ts}. Results at: {result_manager.result_path}")

if __name__ == "__main__":
    main()
