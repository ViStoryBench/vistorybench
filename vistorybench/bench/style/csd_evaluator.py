import os
import torch
from PIL import Image
from itertools import combinations
import numpy as np
import torch.nn.functional as F
import time
try:
    import torchvision.transforms.v2 as T
except ImportError:
    import torchvision.transforms as T

from vistorybench.bench.base_evaluator import BaseEvaluator
from .csd.csd_model import CSD_CLIP
from vistorybench.data_process.outputs_read.read_outputs import load_outputs

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
    def __init__(self, config: dict, timestamp: str, mode: str, language: str):
        super().__init__(config, timestamp, mode, language)
        self.device = torch.device(self.get_device())
        csd_model_path = os.path.join(self.pretrain_path, 'csd/csd_vit-large.pth')

        self.csd_size = 224
        self.csd_mean = [0.48145466, 0.4578275, 0.40821073]
        self.csd_std = [0.26862954, 0.26130258, 0.27577711]

        self.csd_encoder = load_csd(csd_model_path).to(self.device)

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        image_array = np.array(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(image_array).to(self.device)[None,...].permute(0,3,1,2)
        return tensor.detach()

    def _encode(self, image_tensor):
        preprocess = T.Compose([
            T.Resize((self.csd_size, self.csd_size), interpolation=T.InterpolationMode.BICUBIC),
            T.Normalize(tuple(self.csd_mean),
                        tuple(self.csd_std)),
        ])
        input_image_tensor = preprocess(image_tensor).to(device=image_tensor.device, dtype=torch.float32)
        image_embeds = self.csd_encoder(input_image_tensor)['style']
        return image_embeds

    def _get_csd_score(self, img1_path, img2_path):
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        tensor1 = self._preprocess(img1)
        tensor2 = self._preprocess(img2)

        embed1 = self._encode(tensor1)
        embed2 = self._encode(tensor2)

        cos_sim = F.cosine_similarity(embed1, embed2, dim=-1)
        return cos_sim.mean().item()

    def evaluate(self, method: str, story_id: str, **kwargs):
        all_outputs = load_outputs(
            outputs_root=self.output_path,
            methods=[method],
        )
        story_outputs = all_outputs.get(story_id)
        
        if not story_outputs or not story_outputs.get("shots"):
            print(f"Warning: No images found for story {story_id}, method {method}")
            return {"self_csd": 0.0, "cross_csd": 0.0}

        
        story_data = self.story_dataset.load_story(story_id)

        image_paths = story_outputs["shots"]
        
        self_csd_score = self.get_self_csd_score(image_paths)
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

    def build_item_records(self, method: str, story_id: str, story_result, run_info: dict):
        items = []
        try:
            if isinstance(story_result, dict):
                shot_details = story_result.get("shot_details") or story_result.get("detailed_scores")
                if isinstance(shot_details, list):
                    for sd in shot_details:
                        shot_idx = sd.get("index")
                        score = sd.get("score")
                        item = {
                            "run": run_info,
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

        for i, (shot_info, output_img_path) in enumerate(zip(story_data['shots'], outputs_data['shots'])):
            
            characters = story_data["characters"]
            char_key = shot_info['character_key']
            if char_key:
                char_info = characters[char_key[0]]
                # Always select the first image of the character (e.g., 00.png)
                if char_info['images']:
                    char_ref_image = char_info['images'][0]
                else:
                    # Fallback if character has no images
                    characters_1_key = list(characters.keys())[0]
                    char_info = characters[characters_1_key]
                    char_ref_image = char_info['images'][0]
            else:
                # Fallback if shot has no character
                characters_1_key = list(characters.keys())[0]
                char_info = characters[characters_1_key]
                char_ref_image = char_info['images'][0]

            start_time = time.time()
            csd_score = self._get_csd_score(char_ref_image, output_img_path)
            end_time = time.time()
            elapsed_time = end_time - start_time

            # Record shot details
            shot_details.append({
                "index": shot_info['index'],
                "score": csd_score,
                "ref_image": char_ref_image,
                "gen_image": output_img_path,
                "elapsed_time(seconds)": elapsed_time
            })

            total_ri_score.append(csd_score)

        average_ri_csd = 0.0
        # Calculate and print average score (need at least two images)
        if len(total_ri_score) > 0:
            average_ri_csd = sum(total_ri_score) / len(total_ri_score)
            print(f"Average CSD score between generated images and reference images: {average_ri_csd:.4f}")
        else:
            print("Warning: Need at least one generated image to calculate cross CSD score")

        return average_ri_csd, shot_details