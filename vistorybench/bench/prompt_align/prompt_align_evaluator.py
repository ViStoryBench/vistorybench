import time
import json
import requests
import re
import io
import base64
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from vistorybench.bench.base_evaluator import BaseEvaluator
from vistorybench.dataset_loader.read_outputs import load_outputs
from vistorybench.bench.prompt_align.gptv_utils import gptv_query, load_img_content


class PromptAlignEvaluator(BaseEvaluator):
    def __init__(self, config: dict, timestamp: str, mode: str, language: str, outputs_timestamp=None):
        super().__init__(config, timestamp, mode, language, outputs_timestamp)

        # Get evaluator-specific config
        pa_cfg = self.get_evaluator_config('prompt_align')
        gpt_cfg = pa_cfg.get('gpt', {}) if isinstance(pa_cfg, dict) else {}
        # CLI overrides have priority but do not mutate YAML structure
        model = self.get_model_id() or 'gpt-4.1'
        base_url = self.get_base_url()
        api_key = self.get_api_key()
        
        self.gpt_api_pkg = (model, api_key, base_url)
        self.workers = int(gpt_cfg.get('workers', 1) or 1)

        # Built-in prompts (no YAML overrides)
        vlm_bench_path = 'vistorybench/bench/prompt_align'
        self.txt_prompt_list = {
            "scene": f"{vlm_bench_path}/user_prompts/user_prompt_environment_text_align.txt",
            "character_action": f"{vlm_bench_path}/user_prompts/user_prompt_character_text_align.txt",
            "camera": f"{vlm_bench_path}/user_prompts/user_prompt_camera_text_align.txt",
        }


    def _eval_single_dimension(self, eval_dimension, prompt, image_path, character_name=None, image_mode='path'):
        model_type, api_key, base_url = self.gpt_api_pkg
        
        with open(self.txt_prompt_list[eval_dimension], "r") as f:
            eval_dimension_prompt = f.read()

        user_content = [
            {"type": "text", "text": eval_dimension_prompt},
            {"type": "text", "text": prompt},
        ]
        if character_name:
            user_content.append({"type": "text", "text": f"Evaluated Character Name is {character_name}"})
        
        user_content.append(load_img_content(image_path, image_mode))

        transcript = [{"role": "user", "content": user_content}]

        max_retry = 10
        temp_start = 0.0
        score = 0
        while max_retry > 0:
            try:
                response = gptv_query(
                    transcript,
                    top_p=0.2,
                    temp=temp_start,
                    model_type=model_type,
                    api_key=api_key,
                    base_url=base_url,
                )
                pattern = r"(score|Score):\s*[a-zA-Z]*\s*(\d+)"
                scores = re.findall(pattern, response)
                if scores:
                    score = int(scores[0][1])
                    break
                else:
                    temp_start += 0.1
                    max_retry -= 1
            except Exception as e:
                print(f"Error processing {eval_dimension} for {image_path}: {e}, retrying...")
                temp_start += 0.1
                max_retry -= 1
        return score

    def evaluate(self, method: str, story_id: str, **kwargs):
        story_data = self.story_dataset.load_story(story_id)
        all_outputs = load_outputs(
            outputs_root=self.output_path,
            methods=[method],
            modes=[self.mode],
            languages=[self.language],
            timestamps=self.outputs_timestamp,
            return_latest=False
        )
        story_outputs = all_outputs.get(story_id)

        if not story_outputs or not story_outputs.get("shots"):
            print(f"Skipping prompt alignment for {story_id}: missing outputs.")
            return None
        
        image_paths = story_outputs["shots"]
        if not story_data or not image_paths:
            print(f"Skipping prompt alignment for {story_id}: data mismatch or missing.")
            return None

        # Evaluate scores for each shot using the full image (no cropping)
        all_shots_scores = {}
        total_scores = {"scene": [], "character_action": [], "camera": []}

        shots = [shot for shot in story_data['shots'] if image_paths.get(int(shot['index']))]

        # Local worker for a single shot (makes sure image path is located by shot index)
        def eval_one(shot):
            try:
                shot_index = int(shot['index'])
                image_path = image_paths.get(shot_index)
                if not image_path:
                    return shot_index, {}

                shot_scores = {}
                prompts = {
                    "scene": shot['scene'],
                    "character_action": shot['script'],
                    "camera": shot['camera'],
                }
                for dim, prompt in prompts.items():
                    score = self._eval_single_dimension(dim, prompt, image_path, image_mode='path')
                    shot_scores[dim] = score
                return shot_index, shot_scores
            except Exception as e:
                print(f"Error evaluating shot for {story_id}: {e}")
                return shot.get('index', -1), {}

        worker_count = getattr(self, "workers", 1) or 1

        def record_scores(result):
            shot_index, shot_scores = result
            if shot_index is None or shot_index == -1 or not shot_scores:
                return
            all_shots_scores[shot_index] = shot_scores
            for dim in ("scene", "character_action", "camera"):
                if dim in shot_scores:
                    total_scores[dim].append(shot_scores[dim])

        # Parallel or serial execution
        if worker_count > 1:
            print(f"PromptAlign: evaluating {len(shots)} shots in parallel with {worker_count} workers.")
            with ThreadPoolExecutor(max_workers=worker_count) as ex:
                futures = [ex.submit(eval_one, shot) for shot in shots]
                for fut in as_completed(futures):
                    record_scores(fut.result())
        else:
            for shot in shots:
                record_scores(eval_one(shot))

        avg_scores = {dim: (sum(scores) / len(scores) if scores else 0) for dim, scores in total_scores.items()}
        
        final_result = {
            'metrics': avg_scores,
            'detailed_scores': all_shots_scores
        }

        print(f"PromptAlign evaluation complete for story: {story_id}. Average scores: {avg_scores}")
        return final_result

    def build_item_records(self, method: str, story_id: str, story_result):
        items = []
        try:
            if isinstance(story_result, dict):
                detailed = story_result.get("detailed_scores")
                if isinstance(detailed, dict):
                    for shot_idx, dims in detailed.items():
                        for dim, score in (dims or {}).items():
                            item = {
                                "metric": {"name": "prompt_align", "submetric": dim},
                                "scope": {"level": "item", "story_id": str(story_id), "shot_index": shot_idx},
                                "value": score,
                                "status": "complete",
                            }
                            items.append(item)
        except Exception as _e:
            print(f"\033[33mWarning: build_item_records failed for PromptAlign story {story_id}: {_e}\033[0m")
        return items
