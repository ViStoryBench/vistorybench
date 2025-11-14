import os
import torch
from PIL import Image
from torchvision import transforms
from typing import Any

from vistorybench.bench.base_evaluator import BaseEvaluator
from vistorybench.dataset_loader.read_outputs import load_outputs
class AestheticEvaluator(BaseEvaluator):
    def __init__(self, config: dict, timestamp: str, mode: str, language: str, outputs_timestamp=None):
        super().__init__(config, timestamp, mode, language, outputs_timestamp)
        self.device = torch.device(self.get_device())

        VGO_HUB_ROOT = "vistorybench/bench/quality/aesthetic"
        weight_path = f'{self.pretrain_path}/aesthetic_predictor/aesthetic_predictor_v2_5.pth'
        self.torch_dtype = torch.float16

        aesthetic_predictor_v2_5, _ = torch.hub.load(
            os.path.join(VGO_HUB_ROOT, "aesthetic-predictor-v2-5"),
            "aesthetic_predictor_v2_5",
            predictor_name_or_path=weight_path,
            pretrain_path=self.pretrain_path,
            source="local",
            torch_dtype=self.torch_dtype,
        )
        self.aesthetic_predictor_v2_5 = aesthetic_predictor_v2_5.to(self.device)

        self.preprocess_pil = transforms.Compose([
            transforms.Resize((378, 378)),
            transforms.ToTensor(),
        ])

    @torch.inference_mode()
    def inference_aesthetic_predictor_v2_5(self, image):
        image = image.to(self.device, dtype=self.torch_dtype)
        output = self.aesthetic_predictor_v2_5(image)
        scores = [logit.item() for logit in output.logits]
        return scores[0]

    def inference_pil(self, image: Image.Image):
        image = self.preprocess_pil(image)
        image = image.unsqueeze(0).to(self.device, dtype=self.torch_dtype)
        scores = self.inference_aesthetic_predictor_v2_5(image)
        return scores

    def get_aesthetic_score(self, image_path):
        Img = Image.open(image_path)
        if Img.mode != 'RGB':
            Img = Img.convert('RGB')
        score = self.inference_pil(Img)
        return score

    def evaluate(self, method: str, story_id: str, **kwargs):
        all_outputs = load_outputs(
            outputs_root=self.output_path,
            methods=[method],
            modes=[self.mode],
            languages=[self.language],
            timestamps=[self.outputs_timestamp],
            return_latest=False
        )
        story_outputs = all_outputs.get(story_id)
        
        if not story_outputs or not story_outputs.get("shots"):
            print(f"Warning: No images found for story {story_id}, method {method}")
            return {"metrics": {"aesthetic_score": 0}, "per_image_scores": []}

        image_paths = story_outputs["shots"].values()
        
        per_image_scores = []
        if not image_paths:
            print(f"Warning: No images found for story {story_id}, method {method}")
            average_score = 0
        else:
            total_score = 0
            image_count = 0

            for image_path in image_paths:
                score = self.get_aesthetic_score(image_path)
                per_image_scores.append(score)
                total_score += score
                image_count += 1
            
            average_score = total_score / image_count if image_count > 0 else 0
        
        print(f"Aesthetic evaluation complete for story: {story_id}. Average score: {average_score:.4f}")
        return {"metrics": {"aesthetic_score": average_score}, "per_image_scores": per_image_scores}

    def build_item_records(self, method: str, story_id: str, story_result):
        items = []
        try:
            if isinstance(story_result, dict):
                per_scores = story_result.get("per_image_scores")
                if isinstance(per_scores, list):
                    for idx, score in enumerate(per_scores):
                        item = {
                            "metric": {"name": "aesthetic", "submetric": "aesthetic_score"},
                            "scope": {"level": "item", "story_id": str(story_id), "shot_index": idx},
                            "value": score,
                            "status": "complete",
                        }
                        items.append(item)
        except Exception as _e:
            print(f"\033[33mWarning: build_item_records failed for Aesthetic story {story_id}: {_e}\033[0m")
        return items