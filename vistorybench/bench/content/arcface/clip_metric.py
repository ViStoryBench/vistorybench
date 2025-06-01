# A class of metrics for CLIP, evaluating CLIP-I and CLIP-T (cosine similarity)
import torch
import numpy as np
# import open_clip
import transformers
from PIL import Image
from torch.nn import functional as F
import cv2

class CLIP_Metric():
    def __init__(self,device='cuda'):
        self.device = device
        # self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = transformers.CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.tokenizer = transformers.CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
        self.preprocess = transformers.CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.model = self.model.to(self.device)
        self.model.eval()
        # self.tokenizer = open_clip.get_tokenizer(model_name)
        self.cos = F.cosine_similarity
        
    def encode_image(self, image):
        """
        编码图像为特征向量，支持单张图片或批量图片输入
        """
        with torch.no_grad():
            # 处理 numpy 数组
            if isinstance(image, np.ndarray):
                if image.ndim == 3:  # 单张图片 (H, W, C)
                    image = Image.fromarray(image)
                elif image.ndim == 4:  # 批量图片 (B, H, W, C)
                    image = [Image.fromarray(img) for img in image]
            
            # 处理单张 PIL 图像
            if isinstance(image, Image.Image):
                inputs = self.preprocess(images=image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                image_features = self.model.get_image_features(**inputs)
                return F.normalize(image_features, dim=-1).squeeze(0)
            
            # 处理批量 PIL 图像
            elif isinstance(image, (list, tuple)) and all(isinstance(img, Image.Image) for img in image):
                inputs = self.preprocess(images=image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                image_features = self.model.get_image_features(**inputs)
                return F.normalize(image_features, dim=-1)
            
            # 处理 torch 张量
            elif isinstance(image, torch.Tensor):
                if image.dim() == 3:  # 单张图片 (C, H, W)
                    image = image.unsqueeze(0)
                inputs = {"pixel_values": image.to(self.device)}
                image_features = self.model.get_image_features(**inputs)
                if image.dim() == 3:  # 如果输入是单张图片，移除批次维度
                    return F.normalize(image_features, dim=-1).squeeze(0)
                return F.normalize(image_features, dim=-1)
            
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")

    def encode_text(self, text):
        """
        编码文本为特征向量，支持单段文本或批量文本输入
        """
        with torch.no_grad():
            if isinstance(text, str):
                # 单段文本
                inputs = self.tokenizer([text], padding=True,truncation=True, return_tensors="pt", max_length=77).to(self.device)
                text_features = self.model.get_text_features(**inputs)
                return F.normalize(text_features, dim=-1).squeeze(0)
            elif isinstance(text, (list, tuple)) and all(isinstance(t, str) for t in text):
                # 批量文本
                inputs = self.tokenizer(text, padding=True,truncation=True, return_tensors="pt", max_length=77).to(self.device)
                text_features = self.model.get_text_features(**inputs)
                return F.normalize(text_features, dim=-1)
            else:
                raise ValueError(f"Unsupported text type: {type(text)}")

    def compute_clip_i(self, image1, image2):
        """
        计算两张图片之间的 CLIP-I 相似度（图片-图片相似度）
        支持单张图片或批量图片输入
        """
        # 编码图像
        img1_features = self.encode_image(image1)
        img2_features = self.encode_image(image2)
        
        # 计算余弦相似度
        if img1_features.dim() == 1 and img2_features.dim() == 1:
            # 单张图片
            return self.cos(img1_features.unsqueeze(0), img2_features.unsqueeze(0)).item()
        elif img1_features.dim() == 2 and img2_features.dim() == 1:
            # 第一张是批量图片，第二张是单张图片
            img2_features = img2_features.unsqueeze(0).expand(img1_features.size(0), -1)
            return self.cos(img1_features, img2_features)
        elif img1_features.dim() == 1 and img2_features.dim() == 2:
            # 第一张是单张图片，第二张是批量图片
            img1_features = img1_features.unsqueeze(0).expand(img2_features.size(0), -1)
            return self.cos(img1_features, img2_features)
        else:
            # 都是批量图片
            if img1_features.size(0) == img2_features.size(0):
                return self.cos(img1_features, img2_features)
            else:
                raise ValueError("Batch sizes of images must match for batch processing.")

    def compute_clip_t(self, image, text):
        """
        计算图片与文本之间的 CLIP-T 相似度（图片-文本相似度）
        支持单张/批量图片和单段/批量文本输入
        """
        # 编码图像和文本
        img_features = self.encode_image(image)
        text_features = self.encode_text(text)
        
        # 计算余弦相似度
        if img_features.dim() == 1 and text_features.dim() == 1:
            # 单张图片和单段文本
            return self.cos(img_features.unsqueeze(0), text_features.unsqueeze(0)).item()
        elif img_features.dim() == 2 and text_features.dim() == 1:
            # 批量图片和单段文本
            text_features = text_features.unsqueeze(0).expand(img_features.size(0), -1)
            return self.cos(img_features, text_features)
        elif img_features.dim() == 1 and text_features.dim() == 2:
            # 单张图片和批量文本
            img_features = img_features.unsqueeze(0).expand(text_features.size(0), -1)
            return self.cos(img_features, text_features)
        else:
            # 批量图片和批量文本
            if img_features.size(0) == text_features.size(0):
                return self.cos(img_features, text_features)
            else:
                raise ValueError("Batch sizes of images and texts must match for batch processing.")


if __name__ == "__main__":
    # 测试代码
    clip_metric = CLIP_Metric()
    
    # 测试单张图片和单段文本
    image1 = cv2.imread("test1.jpg")
    image2 = cv2.imread("test0.jpg")
    text1 = "A picture of a cat"
    text2 = "A picture of a person with a cat, A picture of a person with a cat, A picture of a person with a cat, A picture of a person with a cat, A picture of a person with a cat, A picture of a person with a cat, A picture of a person with a cat, A picture of a person with a cat, A picture of a person with a cat, A picture of a person with a cat, A picture of a person with a cat, A picture of a person with a cat, A picture of a person with a cat, A picture of a person with a cat, A picture of a person with a cat, A picture of a person with a cat, A picture of a person with a cat, A picture of a person with a cat, "
    
    # clip i between two images
    clip_i = clip_metric.compute_clip_i(image1, image2)
    print("CLIP-I between two images:", clip_i)
    # clip t between image and text
    clip_t = clip_metric.compute_clip_t(image1, text2)
    print("CLIP-T between image and text:", clip_t)
