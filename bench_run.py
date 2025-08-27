import os
import argparse
import yaml
from pathlib import Path
import torch
import sys

from vistorybench.data_process.outputs_read.read_outputs import load_outputs
from vistorybench.result_management.manager import ResultManager
from vistorybench.data_process.dataset_process.dataset_load import StoryDataset

# Import all concrete evaluator classes
from vistorybench.bench.content.cids_evaluator import CIDSEvaluator
from vistorybench.bench.style.csd_evaluator import CSDEvaluator
from vistorybench.bench.diversity.diversity_evaluator import DiversityEvaluator
from vistorybench.bench.quality.aesthetic_evaluator import AestheticEvaluator
from vistorybench.bench.prompt_align.prompt_align_evaluator import PromptAlignEvaluator



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

def load_dataset(_dataset_path, dataset_name, language):
    dataset_path = f"{_dataset_path}/{dataset_name}"
    dataset = StoryDataset(dataset_path)
    story_name_list = dataset.get_story_name_list()
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
    }

    return merged_config

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
    parser.add_argument('--api_key', type=str, default=None, help='API key for external services')
    parser.add_argument('--base_url', type=str, default=None, help='Base URL for API services')
    parser.add_argument('--model_id', type=str, default=None, help='Model ID for evaluation')

    # Evaluation settings
    parser.add_argument('--method', type=str, required=True, help='Method name to evaluate.')
    parser.add_argument('--metrics', type=str, nargs='+', choices=list(EVALUATOR_REGISTRY.keys()), default=None, help='List of metrics to run. Runs all if not specified.')
    parser.add_argument('--language', type=str, choices=['en', 'ch'], default='en', help='Language of the dataset.')
    parser.add_argument('--timestamp', type=str, default=None, help='Specific timestamp to evaluate. If omitted and --resume is True, the latest outputs will be used; if --resume is False, a new timestamp will be created.')
    parser.add_argument('--mode', type=str, default=None, help='Mode for method, if applicable.')
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction, default=False, help='Resume mode: True to evaluate into an existing run (use specified/latest timestamp); False to create a new timestamp and re-evaluate.')

    args = parser.parse_args()
    
    # Merge config with args, args take precedence
    merged_config = merge_config_with_args(config, args)

    # --- Main Evaluation Logic ---
    # Determine run timestamp according to simplified resume semantics:
    # - if --resume False: always create a new timestamp
    # - else if --timestamp provided: use it
    # - else: let manager select the latest automatically
    mode_val = args.mode or "base"
    if args.resume is False:
        timestamp_to_use = ResultManager.create_timestamp()
    else:
        timestamp_to_use = args.timestamp if args.timestamp else None

    # Use unified config paths for results, preferring CLI overrides without mutating YAML
    _cli = merged_config.get('cli_args', {}) if isinstance(merged_config, dict) else {}
    results_root = _cli.get('result_path') or merged_config.get('core', {}).get('paths', {}).get('results', 'data/bench_results')
    result_manager = ResultManager(
        method_name=args.method,
        mode=mode_val,
        language=args.language,
        timestamp=timestamp_to_use,
        base_path=results_root
    )

    _cli = merged_config.get('cli_args', {}) if isinstance(merged_config, dict) else {}
    dataset_root = _cli.get('dataset_path') or merged_config.get('core', {}).get('paths', {}).get('dataset', 'data/dataset')
    outputs_root = _cli.get('outputs_path') or merged_config.get('core', {}).get('paths', {}).get('outputs', 'data/outputs')
    stories_data = load_dataset(dataset_root, 'ViStory', args.language)
    stories_outputs = load_outputs(
        outputs_root=outputs_root,
        methods=args.method,
        languages=[args.language],
        modes=[args.mode] if args.mode else None,
        return_latest=not args.timestamp,
        timestamps=[args.timestamp] if args.timestamp else None
    )

    requested_metrics = args.metrics or list(EVALUATOR_REGISTRY.keys())

    # Handle API key fallback
    # API key/base_url resolution is delegated to evaluators via BaseEvaluator; no top-level mutation here.

    # Add result path from manager
    merged_config['bench_result_run_dir'] = result_manager.result_path

    # Initialize all evaluators once with the merged config
    evaluators = {}
    for metric_name in requested_metrics:
        if metric_name in EVALUATOR_REGISTRY:
            evaluators[metric_name] = EVALUATOR_REGISTRY[metric_name](config=merged_config, timestamp=result_manager.timestamp,mode=result_manager.mode,language=result_manager.language)

    for story_id, story_data in stories_data.items():
        story_id = str(story_id) 
        if story_id not in stories_outputs:
            yellow_print(f"Warning: Story '{story_id}' not found in outputs for method '{args.method}'. Skipping.")
            continue
        
        blue_print(f"--- Evaluating Story: {story_id} for Method: {args.method} ---")

        for metric_name, evaluator in evaluators.items():
            if metric_name == 'diversity': continue # Skip method-level evaluators here

            try:
                green_print(f"Running {metric_name} evaluation...")
                result = evaluator.evaluate(method=args.method, story_id=story_id)
                if result:
                    # Save story-level result (wrapped by ResultManager)
                    result_manager.save_story_result(metric_name, story_id, result)

                    # Append item-level records when available (delegated to evaluator)
                    try:
                        run_info = {
                            "method": args.method,
                            "mode": mode_val,
                            "language": args.language,
                            "dataset": "ViStory",
                            "timestamp": result_manager.timestamp,
                        }
                        items = []
                        if hasattr(evaluator, "build_item_records"):
                            items = evaluator.build_item_records(
                                method=args.method,
                                story_id=story_id,
                                story_result=result,
                                run_info=run_info,
                            ) or []
                        if items:
                            result_manager.append_items(metric_name, items)
                    except Exception as _e:
                        yellow_print(f"Warning: failed to append item-level records for {metric_name}, story {story_id}: {_e}")

                green_print(f"{metric_name} evaluation complete.")
            except Exception as e:
                yellow_print(f"Error during {metric_name} evaluation for story {story_id}: {e}")

    # Handle method-level evaluators like diversity
    if 'diversity' in evaluators:
        green_print("Running diversity evaluation for the whole method...")
        try:
            diversity_evaluator = evaluators['diversity']
            result = diversity_evaluator.evaluate(method=args.method)
            # Save dataset-level metric for diversity
            ds_record = {
                "run": {
                    "method": args.method,
                    "mode": mode_val,
                    "language": args.language,
                    "dataset": "ViStory",
                    "timestamp": result_manager.timestamp
                },
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
    green_print(f"All evaluations for method '{args.method}' complete. Results saved at: {result_manager.result_path}")

if __name__ == "__main__":
    main()