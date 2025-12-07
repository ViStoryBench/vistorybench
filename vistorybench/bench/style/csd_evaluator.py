import os
import torch
from PIL import Image
from itertools import combinations
import numpy as np
import torch.nn.functional as F
import time
from collections import OrderedDict
try:
    import torchvision.transforms.v2 as T
except ImportError:
    import torchvision.transforms as T

from vistorybench.bench.base_evaluator import BaseEvaluator
from .csd.csd_model import CSD_CLIP
from vistorybench.dataset_loader.read_outputs import load_outputs

def load_csd(w_path):
    csd_image_encoder = CSD_CLIP(only_global_token=True)
    state_dict = torch.load(w_path, map_location="cpu")
    csd_image_encoder.load_state_dict(state_dict, strict=False)
    csd_image_encoder.eval()
    for param in csd_image_encoder.parameters():
        param.requires_grad = False
    csd_image_encoder = csd_image_encoder.to(dtype=torch.float32)
    for module in csd_image_encoder.modules():
        if isinstance(module, torch.nn.LayerNorm):
            module.to(dtype=torch.float32)
    return csd_image_encoder

class CSDEvaluator(BaseEvaluator):
    def __init__(self, config: dict, timestamp: str, mode: str, language: str, outputs_timestamp=None):
        super().__init__(config, timestamp, mode, language, outputs_timestamp)
        self.device = torch.device(self.get_device())
        csd_model_path = os.path.join(self.pretrain_path, 'csd/csd_vit-large.pth')
        self.csd_cfg = self.get_evaluator_config('csd') or {}
        self.csd_cache_limit = int(self.csd_cfg.get('cache_limit', 20))

        self.csd_size = 224
        self.csd_mean = [0.48145466, 0.4578275, 0.40821073]
        self.csd_std = [0.26862954, 0.26130258, 0.27577711]

        self.csd_encoder = load_csd(csd_model_path).to(self.device)
        self._image_tensor_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._image_embed_cache: OrderedDict[str, torch.Tensor] = OrderedDict()

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        image_array = np.array(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(image_array)[None,...].permute(0,3,1,2)
        return tensor.detach()

    def _cache_put(self, cache: OrderedDict, key: str, value: torch.Tensor):
        cache[key] = value
        if self.csd_cache_limit > 0:
            while len(cache) > self.csd_cache_limit:
                cache.popitem(last=False)

    def _get_image_tensor(self, image_path: str) -> torch.Tensor:
        tensor = self._image_tensor_cache.get(image_path)
        if tensor is None:
            img = Image.open(image_path).convert('RGB')
            tensor = self._preprocess(img)
            self._cache_put(self._image_tensor_cache, image_path, tensor)
        return tensor

    def _encode(self, image_tensor):
        preprocess = T.Compose([
            T.Resize((self.csd_size, self.csd_size), interpolation=T.InterpolationMode.BICUBIC),
            T.Normalize(tuple(self.csd_mean),
                        tuple(self.csd_std)),
        ])
        input_image_tensor = preprocess(image_tensor).to(device=self.device, dtype=torch.float32)
        image_embeds = self.csd_encoder(input_image_tensor)['style']
        return image_embeds

    def _get_image_embed(self, image_path: str) -> torch.Tensor:
        embed = self._image_embed_cache.get(image_path)
        if embed is None:
            tensor = self._get_image_tensor(image_path)
            embed = self._encode(tensor).detach().cpu()
            self._cache_put(self._image_embed_cache, image_path, embed)
        return embed.clone()  # return a tensor decoupled from cache

    def _get_csd_score(self, img1_path, img2_path):
        embed1 = self._get_image_embed(img1_path)
        embed2 = self._get_image_embed(img2_path)

        cos_sim = F.cosine_similarity(embed1, embed2, dim=-1)
        return cos_sim.mean().item()

    def evaluate(self, method: str, story_id: str, **kwargs):
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
            print(f"Warning: No images found for story {story_id}, method {method}")
            return {"self_csd": 0.0, "cross_csd": 0.0}

        
        story_data = self.story_dataset.load_story(story_id)

        image_paths = story_outputs["shots"]
        
        self_csd_score = self.get_self_csd_score(image_paths.values())
        cross_csd_score, shot_details = self.get_cross_csd_score(story_data, story_outputs)

        scores = {
            "self_csd": self_csd_score,
            "cross_csd": cross_csd_score
        }
        result = {
            "metrics": scores,
            "shot_details": shot_details
        }
        
        print(f"CSD evaluation complete for story: {story_id}")
        return result

    def build_item_records(self, method: str, story_id: str, story_result):
        items = []
        try:
            if isinstance(story_result, dict):
                shot_details = story_result.get("shot_details") or story_result.get("detailed_scores")
                if isinstance(shot_details, list):
                    for sd in shot_details:
                        shot_idx = sd.get("index")
                        score = sd.get("score")
                        item = {
                            "metric": {"name": "csd", "submetric": "cross_csd"},
                            "scope": {"level": "item", "story_id": str(story_id), "shot_index": shot_idx},
                            "value": score,
                            "unit": "cosine_similarity",
                            "extras": {"ref_image": sd.get("ref_image"), "gen_image": sd.get("gen_image")},
                            "status": "complete",
                        }
                        items.append(item)
        except Exception as _e:
            print(f"\033[33mWarning: build_item_records failed for CSD story {story_id}: {_e}\033[0m")
        return items

    def get_self_csd_score(self, image_paths):
        if len(image_paths) < 2:
            return 0.0

        total_score = 0
        count = 0
        for img1_path, img2_path in combinations(image_paths, 2):
            score = self._get_csd_score(img1_path, img2_path)
            total_score += score
            count += 1
        
        return total_score / count if count > 0 else 0.0

    def get_cross_csd_score(self, story_data, outputs_data):
        # Calculate cross style similarity scores between generated images and reference images

        total_ri_score = []
        shot_details = []

        characters = story_data["characters"]
        outputs_shots = outputs_data.get('shots') or {}

        def _pick_reference_image(target_key=None):
            if target_key and target_key in characters:
                imgs = characters[target_key].get('images') or []
                if imgs:
                    return target_key, imgs[0]
            for fallback_key, info in characters.items():
                imgs = info.get('images') or []
                if imgs:
                    return fallback_key, imgs[0]
            return None, None

        for shot_info in story_data['shots']:
            shot_index = int(shot_info['index'])
            output_img_path = outputs_shots.get(shot_index) or outputs_shots.get(str(shot_index))
            if not output_img_path:
                continue

            start_time = time.time()
            per_char_scores = []
            for char_key in shot_info.get('character_key', []):
                selected_key, ref_image = _pick_reference_image(char_key)
                if not ref_image:
                    continue
                score = self._get_csd_score(ref_image, output_img_path)
                per_char_scores.append({
                    "character_key": selected_key,
                    "ref_image": ref_image,
                    "score": score
                })

            if not per_char_scores:
                # Fallback to any available character reference if the shot has no characters/images
                selected_key, ref_image = _pick_reference_image()
                if ref_image:
                    score = self._get_csd_score(ref_image, output_img_path)
                    per_char_scores.append({
                        "character_key": selected_key,
                        "ref_image": ref_image,
                        "score": score
                    })

            if not per_char_scores:
                continue

            shot_score = sum(item["score"] for item in per_char_scores) / len(per_char_scores)
            end_time = time.time()
            elapsed_time = end_time - start_time

            shot_details.append({
                "index": shot_index,
                "score": shot_score,
                "ref_image": per_char_scores[0]["ref_image"],
                "gen_image": output_img_path,
                "elapsed_time(seconds)": elapsed_time,
                "character_scores": per_char_scores
            })

            total_ri_score.append(shot_score)

        average_ri_csd = 0.0
        # Calculate and print average score (need at least two images)
        if len(total_ri_score) > 0:
            average_ri_csd = sum(total_ri_score) / len(total_ri_score)
            print(f"Average CSD score between generated images and reference images: {average_ri_csd:.4f}")
        else:
            print("Warning: Need at least one generated image to calculate cross CSD score")

        return average_ri_csd, shot_details
