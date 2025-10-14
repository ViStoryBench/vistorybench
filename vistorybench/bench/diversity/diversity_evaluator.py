from typing import Dict, List, Optional

import torch

from ..base_evaluator import BaseEvaluator
from .inception_score import calculate_inception_score
from vistorybench.dataset_loader.read_outputs import load_outputs

class DiversityEvaluator(BaseEvaluator):
    """
    Evaluates the diversity of generated images for a given method using Inception Score.
    """
    def __init__(self, config: dict, timestamp: str, mode: str, language: str, outputs_timestamp=None):
        """
        Initializes the DiversityEvaluator.

        Args:
            config (dict): The configuration dictionary.
        """
        super().__init__(config, timestamp, mode, language, outputs_timestamp)
        self.device = torch.device(self.get_device())
        self.mode = mode
        self.language = language
        self.outputs_timestamp = outputs_timestamp
        self.is_batch_size = 32
        self.is_splits = 1

    def _collect_image_paths(
        self,
        method: str,
        story_outputs: Optional[Dict[str, Dict[str, List[str]]]] = None,
    ) -> List[str]:
        """Collect all shot image paths for the specified combination."""
        if story_outputs is None:
            story_outputs = load_outputs(
            outputs_root=self.output_path,
            methods=[method],
            modes=[self.mode],
            languages=[self.language],
            timestamps=[self.outputs_timestamp],
            return_latest=False
        )

        image_paths: List[str] = []
        for story_id, data in story_outputs.items():
            shots = data.get("shots") if isinstance(data, dict) else None
            if not shots:
                print(f"Warning: No shots found for story '{story_id}' when collecting diversity inputs.")
                continue
            image_paths.extend(shots)

        return image_paths

    def evaluate(
        self,
        method: str,
        *,
        mode: Optional[str] = None,
        language: Optional[str] = None,
        timestamp: Optional[str] = None,
        **kwargs,
    ):
        """
        Calculates the Inception Score for a specific (method, mode, language, timestamp) combination.

        Args:
            method (str): The name of the method to evaluate.

        Returns:
            dict: A dictionary containing the inception score and its standard deviation.
                  Returns {'inception_score': 0.0} if no images are found.
        """
        mode = mode or kwargs.get("mode") or getattr(self, "mode", None)
        language = language or kwargs.get("language") or getattr(self, "language", None)
        timestamp = (
            timestamp
            or kwargs.get("timestamp")
            or self.get_cli_arg("timestamp")
        )

        print(
            f"Starting Diversity evaluation for method: {method}, "
            f"mode: {mode or 'ALL'}, language: {language or 'ALL'}, timestamp: {timestamp or 'ALL'}"
        )

        prefetched_outputs = kwargs.get("stories_outputs")

        image_paths = self._collect_image_paths(
            method,
            story_outputs=prefetched_outputs,
        )

        if not image_paths:
            print(
                "Warning: No images found for the specified combination. "
                "Skipping diversity evaluation."
            )
            return {'inception_score': 0.0}

        print(
            f"Found {len(image_paths)} images for method '{method}'. "
            "Calculating Inception Score..."
        )
        
        is_mean, is_std = calculate_inception_score(
            image_paths,
            batch_size=self.is_batch_size,
            splits=self.is_splits,
            device=self.device
        )

        print(f"Diversity evaluation complete for method: {method}. Inception Score: {is_mean:.4f} +/- {is_std:.4f}")

        return {
            'inception_score': is_mean,
        }
