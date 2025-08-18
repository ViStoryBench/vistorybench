import megfile
import webdataset as wds
import tarfile
import json
import logging
import re
from io import BytesIO
import time
import json
import os
from tqdm import tqdm
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from typing import List, Tuple, Any
import cv2



class Aesthetic_Metric:
    def __init__(self, rank = 0, device = "cuda", pretrain_path = '/data/pretrain') -> None:
        self.rank = rank
        self.device = torch.device(device)

        VGO_HUB_ROOT = "bench/quality/aesthetic"

        aesthetic_predictor_v2_5, _ = torch.hub.load(
            os.path.join(VGO_HUB_ROOT, "aesthetic-predictor-v2-5"),
            "aesthetic_predictor_v2_5",
            # predictor_name_or_path=os.path.join(VGO_HUB_ROOT, "aesthetic-predictor-v2-5", "aesthetic_predictor_v2_5.pth"),
            predictor_name_or_path = f'{pretrain_path}/aesthetic_predictor/aesthetic_predictor_v2_5.pth',
            pretrain_path = pretrain_path,
            source="local",
            
            torch_dtype=torch.float16,
            # attn_implementation="flash_attention_2",
        )  # type: ignore


        self.aesthetic_predictor_v2_5 = aesthetic_predictor_v2_5.to(device=self.device)

        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((378,378)),
            transforms.ToTensor(),          
        ])
        self.preprocess_pil = transforms.Compose([
            transforms.Resize((378,378)),
            transforms.ToTensor(),          
        ])
        # self.threshold = 4


# class Aesthetic_Metric:
#     def __init__(self, rank=0, device="cuda") -> None:
#         self.rank = rank
#         self.device = torch.device(device)

#         # Define local paths
#         MODEL_ROOT = "/data/pretrain/aesthetic_predictor"  # Change this to your actual path
#         MODEL_NAME = "aesthetic_predictor_v2_5"
#         WEIGHTS_PATH = os.path.join(MODEL_ROOT, f"{MODEL_NAME}.pth")
        
#         # Verify the weights file exists
#         if not os.path.exists(WEIGHTS_PATH):
#             raise FileNotFoundError(f"Model weights not found at {WEIGHTS_PATH}")
        
#         # Load model architecture (assuming you have the model class definition available locally)
#         # You might need to import the model class definition from a local file
#         aesthetic_predictor_v2_5 = YourModelClass()  # Replace with actual model class
        
#         # Load weights from local file
#         state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")
#         aesthetic_predictor_v2_5.load_state_dict(state_dict)
        
#         self.aesthetic_predictor_v2_5 = aesthetic_predictor_v2_5.to(
#             device=self.device,
#             dtype=torch.float16
#         )

#         self.preprocess = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize((378, 378)),
#             transforms.ToTensor(),          
#         ])
#         self.preprocess_pil = transforms.Compose([
#             transforms.Resize((378, 378)),
#             transforms.ToTensor(),          
#         ])


    @torch.inference_mode()
    def inference_aesthetic_predictor_v2_5(self, image) -> list[float]:
        image = image.to(self.device, dtype=torch.float16)
        # [1,3,378,378]


        # processed_images = []
        # for img in image: 
        #     img = img.to(self.device, dtype=torch.float16) 
        #     img = self.preprocess(img)
        #     processed_images.append(img.to(self.device))

        # processed_images = torch.stack(processed_images)

        # print(processed_images.shape)
        # print(self.aesthetic_predictor_v2_5)

        output = self.aesthetic_predictor_v2_5(image)
        # print(output.logits.shape) [1,1]
        # print(output.logits) tensor([[3.6367]], device='cuda:0', dtype=torch.float16)


        scores = [logit.item() for logit in output.logits]

        return scores[0]




    def inference_pil(self, image: Image.Image) -> list[float]:
        image = self.preprocess_pil(image)
        image = image.unsqueeze(0).to(self.device, dtype=torch.float16)
        scores = self.inference_aesthetic_predictor_v2_5(image)
        return scores

    def __call__(self, image: Any) -> list[float]:
        if isinstance(image, torch.Tensor):
            return self.inference_aesthetic_predictor_v2_5(image)
        elif isinstance(image, Image.Image):
            return self.inference_pil(image)
        else:
            raise TypeError("Input must be a torch.Tensor or PIL.Image")


# def get_aesthetic_score(aesthetic_metric, image_path):
#     Img = Image.open(image_path)
#     if Img.mode != 'RGB':
#         Img = Img.convert('RGB')
#     scores = aesthetic_metric(Img)
#     print(f"Aesthetic Score: {scores:.2f}")
#     return scores


if __name__ == "__main__":
    # Example usage
    
    # aesthetic_metric = Aesthetic_Metric(rank=0, device="cuda")
    # image = torch.randn(1, 3, 378, 378).to("cuda")  # Example tensor
    # scores = aesthetic_metric(image)
    # print(scores)

    image_path = '../data/outputs/uno/WildStory_en/01/20250430-021913/0_0.png'
    # scores = get_aesthetic_score(image_path)

    