import os
import cv2
import glob
import json
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision.ops import box_convert

import insightface
from insightface.data import get_image as ins_get_image
from transformers import CLIPProcessor, CLIPModel
from groundingdino.util.inference import load_model, load_image, predict, annotate


class CRef:
    def __init__(self, pre_fix, output_dir, save_dir):
        # >>>>>>>> Modify here
        # paths
        self.DINO_MODEL = "/data/AIGC_Research/Story_Telling/StoryVisBMK/code/bench/content/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        self.DINO_WEIGHTS = "/data/pretrain/groundingdino/weights/groundingdino_swint_ogc.pth"
        self.PRE_FIX = pre_fix  # 故事级
        self.OUTPUT_DIR = output_dir  # 故事级结果
        self.SAVE_DIR = save_dir # 保存路径
        self.mid_result_dir = os.path.join(self.SAVE_DIR, "mid_result")
        os.makedirs(self.mid_result_dir, exist_ok=True)
        # thresholds
        self.BOX_TRESHOLD = 0.25
        self.TEXT_TRESHOLD = 0.25
        # [optional] character key word for dino
        self.PRE_TEXT_PROMPT = {}
        # <<<<<<<<

        # ############################################################################################################

        self.REF_DIR = os.path.join(self.PRE_FIX, "image")
        self.PROMPT_PATH = os.path.join(self.PRE_FIX, "story.json")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load models
        self.clip_model = CLIPModel.from_pretrained("/data/pretrain/openai/clip-vit-base-patch16").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("/data/pretrain/openai/clip-vit-base-patch16")
        self.arcface_model = insightface.app.FaceAnalysis(root='/data/pretrain/insightface', name="antelope", providers=['CUDAExecutionProvider'])
        self.arcface_model.prepare(ctx_id=0, det_thresh=0.45)
        self.dino_model = load_model(self.DINO_MODEL, self.DINO_WEIGHTS).to(self.device)

    def load_shots(self, prompt_shot_path: str) -> list[dict]:
        with open(prompt_shot_path, 'r') as f:
            shots = json.load(f)
        return shots["Shots"], shots["Characters"]

    def get_char_feat(self, img: Image.Image | list[Image.Image], encoder_name="arcface") -> torch.Tensor:
        if not isinstance(img, list):
            img = [img]

        if encoder_name == "clip":
            inputs = self.processor(images=[img], return_tensors='pt', padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
            image_features = F.normalize(image_features, p=2, dim=1)

        elif encoder_name == "arcface":
            image_features = []
            for _img in img:
                _img_np = cv2.cvtColor(np.array(_img), cv2.COLOR_RGB2BGR)
                _img_faces = self.arcface_model.get(_img_np)
                if len(_img_faces) > 0:
                    image_features.append(torch.from_numpy(_img_faces[0].embedding))
                else:
                    _img.save("bad_case.png")
            if image_features:
                image_features = torch.stack(image_features, dim=0)
                image_features = F.normalize(image_features, p=2, dim=1)
            else:
                image_features = None

        else:
            raise NotImplementedError

        return image_features

    def dino_detect(self, inp_img: str, inp_cap: str, box_threshold=None, text_threshold=None):
        if box_threshold is None:
            box_threshold = self.BOX_TRESHOLD
        if text_threshold is None:
            text_threshold = self.TEXT_TRESHOLD
            
        image_source, image = load_image(inp_img)
        with torch.no_grad():
            boxes, logits, phrases = predict(
                model=self.dino_model,
                image=image,
                caption=inp_cap,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        return boxes, logits, phrases, annotated_frame

    def crop_img(self, img_src: str, boxes) -> list[Image.Image]:
        img_source = Image.open(img_src)
        w, h = img_source.size
        boxes = (boxes * torch.Tensor([w, h, w, h])).int()
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        cropped_images = []
        for box in xyxy:
            x_min, y_min, x_max, y_max = box.tolist()
            cropped = img_source.crop((x_min, y_min, x_max, y_max))
            cropped_images.append(cropped)
        return cropped_images

    def main(self):
        CHARACTER = os.listdir(self.REF_DIR)
        REF_PATH = {char: os.path.join(self.REF_DIR, char) for char in CHARACTER}
        TEXT_PROMPT = {char: "people" for char in CHARACTER} if not self.PRE_TEXT_PROMPT else self.PRE_TEXT_PROMPT

        # load shots
        shots, Characters = self.load_shots(self.PROMPT_PATH)

        # get ref character features
        ref_clip_feats = {}
        for char in CHARACTER:
            enc_name = "arcface" if Characters[char]["tag"] == "realistic_human" else "clip"
            input_ref_imgs = []
            ref_imgs = sorted(glob.glob(os.path.join(REF_PATH[char], '**/*.jpg'), recursive=True))
            for img in ref_imgs:
                img_file_name = img.split('/')[-1].split('.')[0]
                boxes, _, _, _ = self.dino_detect(img, TEXT_PROMPT[char])
                cropped_imgs = self.crop_img(img, boxes)
                if len(cropped_imgs) != 0:
                    input_ref_imgs.append(cropped_imgs[0])  # assume each ref img cantains only one character
                else:
                    print(f"\033[33mNo char: {char} found in {img} with prompt: {TEXT_PROMPT[char]}, please check\033[0m")
            ref_feats = self.get_char_feat(input_ref_imgs, encoder_name=enc_name)
            ref_clip_feats[char] = ref_feats

        # get output character features
        # 存储cref检测所需返回的结果
        results = {"cref": {}}
        # 存储完成匹配的主体，用于计算cref的各项指标
        char_pil_imgs = {}
        
        # 分场景角色匹配
        for shot in shots:
            shot_results = {}
            shot_id = int(shot["index"]) - 1
            print(f'self.OUTPUT_DIR:{self.OUTPUT_DIR}')
            # target_img_path = os.path.join(self.OUTPUT_DIR, f"{shot_id}_0.png")  # HACK:为什么有_0后缀？
            target_img_path = self.OUTPUT_DIR[shot_id]
            print(f'当前分镜的生成图像：{target_img_path}')

            # 通过角色关键词检索生成图像中可能的主体
            for char in shot['Characters Appearing']['en']:
                enc_name = "arcface" if Characters[char]["tag"] == "realistic_human" else "clip"
                if char not in char_pil_imgs.keys():
                    char_pil_imgs.update({char: []})
                boxes, logits, phrases, annotated_frame = self.dino_detect(target_img_path, TEXT_PROMPT[char])
                # 从中挑选不超过当前场景所需角色数量的相关性最高的主体
                _, indices = torch.topk(logits, min(len(logits), len(shot['Characters Appearing']['en'])))
                boxes = boxes[indices]
                # cv2.imwrite(f"{shot_id}-{char}.png", annotated_frame) # NOTE: debug
                # 裁剪出对应关键词的可能主体
                cropped_imgs = self.crop_img(target_img_path, boxes)
                if len(cropped_imgs) != 0:
                    output_feats = self.get_char_feat(cropped_imgs, encoder_name=enc_name)
                    cosine_sims = {}
                    # 预先计算每个主体对每个角色参考图的相似性（取多图余弦相似系数中最大值）
                    for _char in shot['Characters Appearing']['en']:
                        cosine_sims.update({_char: (output_feats @ ref_clip_feats[_char].T).max(dim=1).values.cpu()})
                    sims_bak = cosine_sims[char].tolist()
                    matched_flag = False
                    # 从最相似的主体开始逐一检查是否有其余角色与其相似度更高；
                    # 若有，则排除该主体，检查次一等相似的主体；若无，则将该主体与该角色匹配
                    while len(sims_bak) > 0:
                        _id = torch.argmax(torch.Tensor(sims_bak)).item()
                        for _char in shot['Characters Appearing']['en']:
                            if _char != char:
                                if cosine_sims[_char][_id] > cosine_sims[char][_id]:
                                    break
                        else:
                            matched_flag = True
                            boxes_id = cosine_sims[char].tolist().index(sims_bak[_id])
                            break
                        sims_bak.pop(_id)
                    if matched_flag:
                        boxes_to_write = [round(x, 3) for x in boxes[boxes_id].tolist()]
                        shot_results.update({char: {"box": boxes_to_write}})
                        char_pil_imgs[char].append(cropped_imgs[boxes_id])
                        cropped_imgs[boxes_id].save(f"{self.mid_result_dir}/shot{shot_id:02d}-{char}.png") # NOTE:debug

                    else:
                        # 某个场景中某角色检测失败时，box数据留空
                        shot_results.update({char: {"box": "null"}})
                else:
                    print(f"\033[33mNo char: {char} found in {target_img_path} with prompt: {TEXT_PROMPT[char]}, please check\033[0m")
                    # 某个场景中某角色检测失败时，box数据留空
                    shot_results.update({char: {"box": "null"}})
            results.update({f"shot-{shot_id}": shot_results})

        # 可以解除以下代码的注释以手工检查是否正确识别了角色
        # for char in char_pil_imgs.keys():  # NOTE: for debug
        #     for idx, img in enumerate(char_pil_imgs[char]):
        #         img.save(f"{self.mid_result_dir}/{char}-{idx}.png")

        # 分别计算cref的cross指标和self指标
        for char in CHARACTER:
            enc_name = "arcface" if Characters[char]["tag"] == "realistic_human" else "clip"
            if len(char_pil_imgs[char]) > 0:
                char_feats = self.get_char_feat(char_pil_imgs[char], encoder_name=enc_name)
                cross_sim = (char_feats @ ref_clip_feats[char].T)
                cross_sim_avg = cross_sim.mean().item()
                self_sim = (char_feats @ char_feats.T)
                self_sim_avg = self_sim.mean().item()
                results["cref"].update({char: {
                    "cross": round(cross_sim_avg, 4),
                    "self": round(self_sim_avg, 4)
                }})
            else:
                # 全程未检出某个角色时，得分为0.0
                results["cref"].update({char: {
                    "cross": 0.0,
                    "self": 0.0,
                }})
        return results

def get_cref_score(pre_fix, output_dir, save_dir):

    # Initialize CRef evaluator
    cref_evaluator = CRef(pre_fix, output_dir, save_dir)
    result = cref_evaluator.main()

    return result


if __name__ == "__main__":
    # Example usage:
    # pre_fix = "/path/to/your/data"
    # output_dir = "/path/to/your/output"
    # cref = CRef(pre_fix, output_dir)
    # results = cref.get_cref_score()
    # print(results)
    pass