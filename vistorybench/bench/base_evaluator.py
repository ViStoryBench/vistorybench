from vistorybench.data_process.dataset_process.dataset_load import StoryDataset
import os
from abc import ABC, abstractmethod


class BaseEvaluator(ABC):
    """
    Abstract base class for all evaluators.
    """
    def __init__(self, config, timestamp, mode, language):
        self.config = config
        self.timestamp = timestamp
        self.mode = mode
        self.language = language

        # Unified config access methods
        self.dataset_path = self.get_path('dataset')
        self.output_path = self.get_path('outputs')
        self.pretrain_path = self.get_path('pretrain')
        self.result_path = self.get_path('results')
        self.device = self.get_device()
        
        # Initialize story dataset
        self.story_dataset = StoryDataset(os.path.join(self.dataset_path, 'ViStory'))
    
    def get_path(self, path_type):
        """Get path from unified config; prefer CLI overrides in config['cli_args'] when provided."""
        cli = self.config.get('cli_args', {}) if isinstance(self.config, dict) else {}
        mapping = {'dataset': 'dataset_path', 'outputs': 'outputs_path', 'pretrain': 'pretrain_path', 'results': 'result_path'}
        cli_key = mapping.get(path_type)
        if cli_key and cli.get(cli_key):
            return cli.get(cli_key)
        return self.config.get('core', {}).get('paths', {}).get(path_type, f'data/{path_type}')
    
    def get_device(self):
        """Get device setting from config"""
        device_str = self.config.get('core', {}).get('runtime', {}).get('device', 'cuda')
        return device_str if isinstance(device_str, str) else str(device_str)
    
    def get_api_key(self):
        """Resolve API key from CLI overrides, YAML, or environment."""
        cli = self.config.get('cli_args', {}) if isinstance(self.config, dict) else {}
        return (
            cli.get('api_key')
            or self.config.get('api_key', '')
            or os.environ.get('OPENAI_API_KEY')
            or os.environ.get('VISTORYBENCH_API_KEY')
            or ''
        )
    
    def get_base_url(self):
        """Resolve base URL from CLI overrides or YAML."""
        cli = self.config.get('cli_args', {}) if isinstance(self.config, dict) else {}
        return cli.get('base_url') or self.config.get('base_url', '')
    
    def get_cli_arg(self, key, default=None):
        """Get a CLI argument passed via merged config without overlapping YAML keys."""
        cli = self.config.get('cli_args', {}) if isinstance(self.config, dict) else {}
        return cli.get(key, default)
    
    def get_evaluator_config(self, evaluator_name):
        """Get evaluator-specific configuration"""
        return self.config.get('evaluators', {}).get(evaluator_name, {})

    @abstractmethod
    def evaluate(self, method: str, story_id: str, **kwargs):
        """
        Main evaluation method to be implemented by subclasses.

        :param method: The method name.
        :param story_id: The story id.
        """
        pass

    def build_item_records(self, method: str, story_id: str, story_result, run_info: dict):
        """
        Construct item-level records (JSONL entries) for this evaluator based on the story_result.
        Default implementation returns an empty list; subclasses may override.

        :param method: The method name.
        :param story_id: The story id.
        :param story_result: The result returned by evaluate(...)
        :param run_info: Common run information to be embedded into each item
        :return: List of item dicts to append via ResultManager.append_items(...)
        """
        return []