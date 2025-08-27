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
from vistorybench.data_process.outputs_read.read_outputs import load_outputs


def gptv_query(transcript=None, top_p=0.2, temp=0., model_type="gpt-4o", api_key='', base_url='', seed=123, max_tokens=512, wait_time=10):

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    base_url=base_url[:-1] if base_url.endswith('/') else base_url
    requests_url = f"{base_url}/chat/completions" if base_url.endswith('/v1') else f"{base_url}/v1/chat/completions"

    data = {
        'model': model_type,
        'max_tokens': max_tokens,
        'temperature': temp,
        'top_p': top_p,
        'messages': transcript or [],
        'seed': seed,
    }

    response_text, retry, response_json = '', 0, None
    while len(response_text) < 2:
        retry += 1
        try:
            response = requests.post(url=requests_url, headers=headers, data=json.dumps(data))
            response_json = response.json()
        except Exception as e:
            print(e)
            time.sleep(wait_time)
            continue
        if response.status_code != 200:
            print(response.headers, response.content)
            time.sleep(wait_time)
            data['temperature'] = min(data['temperature'] + 0.2, 1.0)
            continue
        if 'choices' not in response_json:
            time.sleep(wait_time)
            continue
        response_text = response_json["choices"][0]["message"]["content"]
    return response_json["choices"][0]["message"]["content"]

def encode_image(image_input, image_mode='path', resize_to=(512, 512)):
    if image_mode == 'path':
        with Image.open(image_input) as img:
            img = img.resize(resize_to)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    elif image_mode == 'pil':
        img = image_input.resize(resize_to)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

def load_img_content(image_input, image_mode='path', resize_to=(512, 512), pil_detail='low'):
    base64_image = encode_image(image_input, image_mode, resize_to=resize_to)
    image_meta = "data:image/jpeg;base64"
    content = {
        "type": "image_url",
        "image_url": {"url": f"{image_meta},{base64_image}"},
    }
    if image_mode == 'pil':
        content["image_url"]["detail"] = pil_detail  # PIL image is small, configurable detail
    return content

class PromptAlignEvaluator(BaseEvaluator):
    def __init__(self, config: dict, timestamp: str, mode: str, language: str):
        super().__init__(config, timestamp, mode, language)

        # Get evaluator-specific config
        pa_cfg = self.get_evaluator_config('prompt_align')
        gpt_cfg = pa_cfg.get('gpt', {}) if isinstance(pa_cfg, dict) else {}
        # CLI overrides have priority but do not mutate YAML structure
        model = self.get_cli_arg('model_id') or gpt_cfg.get('model') or 'gpt-4o'
        base_url = gpt_cfg.get('base_url') or self.get_base_url()
        api_key = gpt_cfg.get('api_key') or self.get_api_key()
        
        self.gpt_api_pkg = (model, api_key, base_url)
        self.workers = int(gpt_cfg.get('workers', 1) or 1)

        # Built-in prompts (no YAML overrides)
        vlm_bench_path = 'vistorybench/bench/prompt_align'
        self.txt_prompt_list = {
            "scene": f"{vlm_bench_path}/user_prompts/user_prompt_environment_text_align.txt",
            "character_action": f"{vlm_bench_path}/user_prompts/user_prompt_character_text_align.txt",
            "camera": f"{vlm_bench_path}/user_prompts/user_prompt_camera_text_align.txt",
            "single_character_action": f"{vlm_bench_path}/user_prompts/user_prompt_single_character_text_align.txt"
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
            except Exception as e:
                print(f"Error processing {eval_dimension} for {image_path}: {e}, retrying...")
                temp_start += 0.1
                max_retry -= 1
        return score

    def evaluate(self, method: str, story_id: str, **kwargs):
        story_data = self.story_dataset.load_story(story_id)
        all_outputs = load_outputs(outputs_root=self.output_path, methods=[method])
        story_outputs = all_outputs.get(story_id)

        if not story_outputs or not story_outputs.get("shots"):
            print(f"Skipping prompt alignment for {story_id}: missing outputs.")
            return None
        
        image_paths = story_outputs["shots"]
        if not story_data or not image_paths or len(story_data['shots']) != len(image_paths):
            print(f"Skipping prompt alignment for {story_id}: data mismatch or missing.")
            return None

        # Evaluate scores for each shot using the full image (no cropping)
        all_shots_scores = {}
        total_scores = {"scene": [], "character_action": [], "camera": [], "single_character_action": []}

        shots = story_data['shots']

        # Local worker for a single shot
        def eval_one(shot, image_path):
            try:
                shot_index = shot['index']
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
                print(f"Error evaluating shot for {story_id} at image {image_path}: {e}")
                return shot.get('index', -1), {}

        # Parallel or serial execution
        if  getattr(self, "workers", 1) and self.workers > 1:
            print(f"PromptAlign: evaluating {len(shots)} shots in parallel with {self.workers} workers.")
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                futures = [ex.submit(eval_one, shot, img_path) for shot, img_path in zip(shots, image_paths)]
                for fut in as_completed(futures):
                    shot_index, shot_scores = fut.result()
                    if shot_index is None or shot_index == -1:
                        continue
                    all_shots_scores[shot_index] = shot_scores
                    for dim in ("scene", "character_action", "camera"):
                        if dim in shot_scores:
                            total_scores[dim].append(shot_scores[dim])

        avg_scores = {dim: (sum(scores) / len(scores) if scores else 0) for dim, scores in total_scores.items()}
        
        final_result = {
            'metrics': avg_scores,
            'detailed_scores': all_shots_scores
        }

        print(f"PromptAlign evaluation complete for story: {story_id}. Average scores: {avg_scores}")
        return final_result

    def build_item_records(self, method: str, story_id: str, story_result, run_info: dict):
        items = []
        try:
            if isinstance(story_result, dict):
                detailed = story_result.get("detailed_scores")
                if isinstance(detailed, dict):
                    for shot_idx, dims in detailed.items():
                        for dim, score in (dims or {}).items():
                            item = {
                                "run": run_info,
                                "metric": {"name": "prompt_align", "submetric": dim},
                                "scope": {"level": "item", "story_id": str(story_id), "shot_index": shot_idx},
                                "value": score,
                                "status": "complete",
                            }
                            items.append(item)
        except Exception as _e:
            print(f"\033[33mWarning: build_item_records failed for PromptAlign story {story_id}: {_e}\033[0m")
        return items