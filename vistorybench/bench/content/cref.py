import os
import cv2
import glob
import json
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision.ops import box_convert

from transformers import CLIPProcessor, CLIPModel
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate

# >>>>>>>> Modify here
# paths
DINO_MODEL = "/home/wujingwei/Workspace/code/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
DINO_WEIGHTS = "/data/pretrain/groundingdino/groundingdino_swint_ogc.pth"
PRE_FIX = "/home/wujingwei/Workspace/code/StorySet/Arab_folk"
# thresholds
BOX_TRESHOLD = 0.25
TEXT_TRESHOLD = 0.25
# [optional] character key word for dino
PRE_TEXT_PROMPT = {}
# <<<<<<<<

# ############################################################################################################

REF_DIR = os.path.join(PRE_FIX, "image_ref", "character")
OUTPUT_DIR = os.path.join(PRE_FIX, "image_outputs")
SHOTS = os.path.join(PRE_FIX, "prompt_shot.json")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_shots(prompt_shot_path: str):
    with open(prompt_shot_path, 'r') as f:
        shots = json.load(f)
    return shots


def get_clip_feat(img: Image.Image | list[Image.Image], clip_model, processor) -> torch.Tensor:
    if not isinstance(img, list):
        img = [img]
    inputs = processor(images=[img], return_tensors='pt', padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
    image_features = F.normalize(image_features, p=2, dim=1)

    return image_features


def dino_detect(dino_model, inp_img: str, inp_cap: str, box_threshold = BOX_TRESHOLD, text_threshold = TEXT_TRESHOLD):
    image_source, image = load_image(inp_img)
    with torch.no_grad():
        boxes, logits, phrases = predict(
            model=dino_model,
            image=image,
            caption=inp_cap,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    return boxes, logits, phrases, annotated_frame


def crop_img(img_src: str, boxes) -> list[Image.Image]:
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


def main():
    CHARACTER = os.listdir(REF_DIR) # ["Aladdin", "Genie", ...]
    REF_PATH = {char: os.path.join(REF_DIR, char) for char in CHARACTER} 
    # {"Aladdin": "/path/to/ref_img/", ...}

    TEXT_PROMPT = {char: "people" for char in CHARACTER} if not PRE_TEXT_PROMPT else PRE_TEXT_PROMPT

    # load models
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    dino_model = load_model(DINO_MODEL, DINO_WEIGHTS).to(device)

    # load shots
    shots = load_shots(SHOTS)
    # {"shot1":{"story": "xxx", "scene": "xxx", "character": ["xxx", "xxx"]}, ...}

    # get ref character features
    ref_clip_feats = {}
    for char in CHARACTER:
        input_ref_imgs = []
        
        # 某角色的所有参考图像路径
        ref_imgs = sorted(glob.glob(os.path.join(REF_PATH[char], '**/*.png'), recursive=True))
        for img in ref_imgs:
            img_file_name = img.split('/')[-1].split('.')[0]
            boxes, _, _, _ = dino_detect(dino_model, img, TEXT_PROMPT[char])
            cropped_imgs = crop_img(img, boxes)
            if len(cropped_imgs) != 0:
                input_ref_imgs.append(cropped_imgs[0]) # assume each ref img cantains only one character
                # cropped_imgs[0].save(f"{char}-{img_file_name}.png") # NOTE: debug
            else:
                print(f"\033[33mNo char: {char} found in {img} with prompt: {TEXT_PROMPT[char]}, please check\033[0m")
        ref_feats = get_clip_feat(input_ref_imgs, clip_model, processor)
        ref_clip_feats[char] = ref_feats

    # get output character features
    # 存储cref检测所需返回的结果
    results = {"cref": {}}
    # 存储完成匹配的主体，用于计算cref的各项指标
    char_pil_imgs = {}
    # 分场景角色匹配
    for shot in shots:
        shot_results = {}
        shot_id = int(shot.replace("shot", "")) - 1
        target_img_path = os.path.join(OUTPUT_DIR, f"{shot_id}_0.png") # HACK:为什么原子样本中输出的img有_0后缀？

        # 通过角色关键词检索生成图像中可能的主体
        for char in shots[shot]['character']:
            if char not in char_pil_imgs.keys():
                char_pil_imgs.update({char: []})
            boxes, logits, phrases, annotated_frame = dino_detect(dino_model, target_img_path, TEXT_PROMPT[char])
            # cv2.imwrite(f"{shot}-{char}.png", annotated_frame) # NOTE: debug
            # 裁剪出对应关键词的可能主体
            cropped_imgs = crop_img(target_img_path, boxes)
            if len(cropped_imgs) != 0:
                output_feats = get_clip_feat(cropped_imgs, clip_model, processor)
                cosine_sims = {}
                # 预先计算每个主体对每个角色参考图的相似性（取多图余弦相似系数中最大值）
                for _char in shots[shot]['character']:
                    cosine_sims.update({_char: (output_feats @ ref_clip_feats[_char].T).max(dim=1).values.cpu()})
                sims_bak = cosine_sims[char].tolist()
                matched_flag = False
                # 从最相似的主体开始逐一检查是否有其余角色与其相似度更高；
                # 若有，则排除该主体，检查次一等相似的主体；若无，则将该主体与该角色匹配
                while(len(sims_bak) > 0):
                    _id = torch.argmax(torch.Tensor(sims_bak)).item()
                    for _char in shots[shot]['character']:
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
                    shot_results.update({ char: { "box": boxes_to_write} })
                    char_pil_imgs[char].append(cropped_imgs[boxes_id])
                    # cropped_imgs[boxes_id].save(f"{shot}-{char}.png") # NOTE:debug

                else:
                    # 某个场景中某角色检测失败时，box数据留空
                    shot_results.update({ char: { "box": "null" } })
            else:
                print(f"\033[33mNo char: {char} found in {target_img_path} with prompt: {TEXT_PROMPT[char]}, please check\033[0m")
                # 某个场景中某角色检测失败时，box数据留空
                shot_results.update({ char: { "box": "null" } })
        results.update({shot: shot_results})

    # 可以解除以下代码的注释以手工检查是否正确识别了角色
    for char in char_pil_imgs.keys(): # NOTE: for debug
        for idx, img in enumerate(char_pil_imgs[char]):
            img.save(f"{char}-{idx}.png")

    # 分别计算cref的cross指标和self指标
    for char in CHARACTER:
        if char in char_pil_imgs.keys():
            char_feats = get_clip_feat(char_pil_imgs[char], clip_model, processor)
            cross_sim = (char_feats @ ref_clip_feats[char].T).mean().item()
            self_sim = (char_feats @ char_feats.T).mean().item()
            results["cref"].update({char: {
                "cross": round(cross_sim, 4),
                "self": round(self_sim, 4)
            }})
        else:
            # 全程未检出某个角色时，得分为0.0
            results["cref"].update({char: {
                "cross": 0.0,
                "self": 0.0,
            }})
    print(results)


if __name__ == "__main__":
    main()