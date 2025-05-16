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
from transformers import CLIPProcessor, CLIPModel
from groundingdino.util.inference import load_model, load_image, predict, annotate

# >>>>>>>> Modify here
# paths
DINO_MODEL = "/home/wujingwei/Workspace/code/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
DINO_WEIGHTS = "/home/wujingwei/Workspace/code/GroundingDINO/weights/groundingdino_swint_ogc.pth"
PRE_FIX = "/mnt/jfs/hypertext/wujingwei/Workspace/data/250508_storytell/data/story_data_generator/final_ds_en_and_ch/35"
OUTPUT_DIR = "/mnt/jfs/hypertext/wujingwei/Workspace/data/250508_storytell/outputs/uno/WildStory_en/35/20250430-000849"
# thresholds
BOX_TRESHOLD = 0.25
TEXT_TRESHOLD = 0.25
# [optional] character key word for dino
PRE_TEXT_PROMPT = {}
# <<<<<<<<

# ############################################################################################################

REF_DIR = os.path.join(PRE_FIX, "image")
PROMPT_PATH = os.path.join(PRE_FIX, "story.json")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load models
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
arcface_model = insightface.app.FaceAnalysis(name="antelope", providers=['CUDAExecutionProvider'])
dino_model = load_model(DINO_MODEL, DINO_WEIGHTS).to(device)


def load_shots(prompt_shot_path: str) -> list[dict]:
    with open(prompt_shot_path, 'r') as f:
        shots = json.load(f)
    return shots["Shots"], shots["Characters"]


def get_char_feat(img: Image.Image | list[Image.Image], encoder_name = "arcface", det_thresh = 0.45) -> torch.Tensor:
    if not isinstance(img, list):
        img = [img]

    if encoder_name == "clip":
        inputs = processor(images=[img], return_tensors='pt', padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
        image_features = F.normalize(image_features, p=2, dim=1)

    elif encoder_name == "arcface":
        curr_thresh = det_thresh
        image_features = []
        for _img in img:
            _img_np = cv2.cvtColor(np.array(_img), cv2.COLOR_RGB2BGR)
            while(True):
                if curr_thresh < 0.1:
                    break
                arcface_model.prepare(ctx_id=0, det_thresh=curr_thresh)
                _img_faces = arcface_model.get(_img_np)
                if len(_img_faces) > 0:
                    image_features.append(torch.from_numpy(_img_faces[0].embedding))
                    break
                else:
                    _img.save("bad_case.png")
                    print(f"No face detected, auto re-try, curr_thresh: {curr_thresh}")
                    curr_thresh -= 0.1
        if image_features:
            image_features = torch.stack(image_features, dim=0)
            image_features = F.normalize(image_features, p=2, dim=1)
        else:
            image_features = None

    else:
        raise NotImplementedError

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
    CHARACTER = os.listdir(REF_DIR)
    REF_PATH = {char: os.path.join(REF_DIR, char) for char in CHARACTER}
    TEXT_PROMPT = {char: "people" for char in CHARACTER} if not PRE_TEXT_PROMPT else PRE_TEXT_PROMPT

    # load shots
    shots, Characters = load_shots(PROMPT_PATH)

    # get ref character features
    ref_clip_feats = {}
    for char in CHARACTER:
        enc_name = "arcface" if Characters[char]["tag"] == "realistic_human" else "clip"
        input_ref_imgs = []
        ref_imgs = sorted(glob.glob(os.path.join(REF_PATH[char], '**/*.jpg'), recursive=True))
        for img in ref_imgs:
            img_file_name = img.split('/')[-1].split('.')[0]
            boxes, logits, _, _ = dino_detect(dino_model, img, TEXT_PROMPT[char])
            _, indices = torch.topk(logits, 1)
            boxes = boxes[indices]
            cropped_imgs = crop_img(img, boxes)
            if len(cropped_imgs) != 0:
                input_ref_imgs.append(cropped_imgs[0]) # assume each ref img cantains only one character
                # cropped_imgs[0].save(f"{char}-{img_file_name}.png") # NOTE: debug
            else:
                print(f"\033[33mNo char: {char} found in {img} with prompt: {TEXT_PROMPT[char]}, please check\033[0m")
        ref_feats = get_char_feat(input_ref_imgs, encoder_name=enc_name)
        assert ref_feats is not None, f"Cant get ref char: {char}, please check."
        ref_clip_feats[char] = ref_feats

    # get output character features
    # 存储cref检测所需返回的结果
    results = {"cref": {}}
    # 存储完成匹配的主体，用于计算cref的各项指标
    char_pil_imgs = {}
    matched_cnt = 0 # 角色匹配场景
    omissive_cnt = 0 # 角色遗漏场景
    superfluous_cnt = 0 # 角色多余场景
    # 分场景角色匹配
    for shot in shots:
        shot_results = {}
        shot_id = int(shot["index"]) - 1
        target_img_path = os.path.join(OUTPUT_DIR, f"{shot_id}_0.png") # HACK:为什么有_0后缀？
        is_omissive_shot = False

        # 通过角色关键词检索生成图像中可能的主体
        for char in shot['Characters Appearing']['en']:
            enc_name = "arcface" if Characters[char]["tag"] == "realistic_human" else "clip"
            if char not in char_pil_imgs.keys():
                char_pil_imgs.update({char: []})
            boxes, logits, phrases, annotated_frame = dino_detect(dino_model, target_img_path, TEXT_PROMPT[char])
            # 从中挑选不超过当前场景所需角色数量的相关性最高的主体
            _, indices = torch.topk(logits, min(len(logits), len(shot['Characters Appearing']['en'])))
            boxes = boxes[indices]
            # cv2.imwrite(f"{shot_id}-{char}.png", annotated_frame) # NOTE: debug
            # 裁剪出对应关键词的可能主体
            cropped_imgs = crop_img(target_img_path, boxes)
            if len(cropped_imgs) != 0:
                output_feats = get_char_feat(cropped_imgs, encoder_name=enc_name)
                if output_feats is None:
                    print(f"\033[33mNo char: {char} found in {target_img_path} with prompt: {TEXT_PROMPT[char]}, please check\033[0m")
                    # 某个场景中某角色检测失败时，box数据留空
                    shot_results.update({ char: { "box": "null" } })
                    continue
                cosine_sims = {}
                # 预先计算每个主体对每个角色参考图的相似性（取多图余弦相似系数中最大值）
                for _char in shot['Characters Appearing']['en']:
                    cosine_sims.update({_char: (output_feats.cuda() @ ref_clip_feats[_char].cuda().T).max(dim=1).values.cpu()})
                sims_bak = cosine_sims[char].tolist()
                matched_flag = False
                # 从最相似的主体开始逐一检查是否有其余角色与其相似度更高；
                # 若有，则排除该主体，检查次一等相似的主体；若无，则将该主体与该角色匹配
                while(len(sims_bak) > 0):
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
                    shot_results.update({ char: { "box": boxes_to_write} })
                    char_pil_imgs[char].append(cropped_imgs[boxes_id])
                    # cropped_imgs[boxes_id].save(f"{shot_id}-{char}.png") # NOTE:debug

                else:
                    # 某个场景中某角色检测失败时，box数据留空
                    shot_results.update({ char: { "box": "null" } })
            else:
                print(f"\033[33mNo char: {char} found in {target_img_path} with prompt: {TEXT_PROMPT[char]}, please check\033[0m")
                # 某个场景中某角色检测失败时，box数据留空
                shot_results.update({ char: { "box": "null" } })

        # 通过bounding box是否存在检测是否有遗漏角色
        if any([x["box"] == "null" for x in shot_results.values()]):
            omissive_cnt += 1
            is_omissive_shot = True

        # 对原本不包含角色的场景，检测是否存在多余角色
        is_superfluous_shot = False
        if len(shot['Characters Appearing']['en']) == 0:
            for char in CHARACTER:
                enc_name = "arcface" if Characters[char]["tag"] == "realistic_human" else "clip"
                boxes, logits, phrases, annotated_frame = dino_detect(dino_model, target_img_path, TEXT_PROMPT[char])
                _, indices = torch.topk(logits, min(len(logits), 5))
                boxes = boxes[indices]
                # 裁剪出对应关键词的可能主体
                cropped_imgs = crop_img(target_img_path, boxes)
                if len(cropped_imgs) != 0:
                    output_feats = get_char_feat(cropped_imgs, encoder_name=enc_name)
                    if output_feats is not None:
                        # 相似度大于0.8时，判定为该场景包含某一特定角色
                        cosine_sims = {}
                        for _char in shot['Characters Appearing']['en']:
                            cosine_sims.update({_char: (output_feats @ ref_clip_feats[_char].T).max(dim=1).values.cpu()})
                        if any([x > 0.8 for x in cosine_sims.values()]):
                            is_superfluous_shot = True
            if is_superfluous_shot:
                superfluous_cnt += 1

        # 若该场景既无遗漏也无多余，则属于匹配场景
        if is_omissive_shot is not True and is_superfluous_shot is not True:
            matched_cnt += 1

        results.update({f"shot-{shot_id}": shot_results})

    num_shots = len(shots)
    results.update(
        {
            "matched-shots": f"{matched_cnt} / {num_shots}",
            "omissive-shots": f"{omissive_cnt} / {num_shots}",
            "superfluous-shots": f"{superfluous_cnt} / {num_shots}",
        }
    )

    # 可以解除以下代码的注释以手工检查是否正确识别了角色
    # for char in char_pil_imgs.keys(): # NOTE: for debug
    #     for idx, img in enumerate(char_pil_imgs[char]):
    #         img.save(f"{char}-{idx}.png")

    # 分别计算cref的cross指标和self指标
    # 计算copy-paste指标
    copy_paste_cnt = 0
    shot_copy_paste_score = 0.0
    for char in CHARACTER:
        enc_name = "arcface" if Characters[char]["tag"] == "realistic_human" else "clip"
        if len(char_pil_imgs[char]) > 0:
            char_feats = get_char_feat(char_pil_imgs[char], encoder_name=enc_name)
            if char_feats is None:
                # 全程未检出某个角色时，得分为0.0
                results["cref"].update({char: {
                    "cross": 0.0,
                    "self": 0.0,
                }})
                continue
            cross_sim = (char_feats @ ref_clip_feats[char].T).mean().item()
            self_sim = (char_feats @ char_feats.T).mean().item()
            results["cref"].update({char: {
                "cross": round(cross_sim, 4),
                "self": round(self_sim, 4)
            }})
            # copy-paste
            if ref_clip_feats[char].shape[0] > 1:
                copy_paste_cnt += 1
                cross_sims = char_feats @ ref_clip_feats[char].T
                copy_paste_score = (cross_sims[:, 0].unsqueeze(-1) - cross_sims[:, 1:]).mean().item()
                shot_copy_paste_score += copy_paste_score
        else:
            # 全程未检出某个角色时，得分为0.0
            results["cref"].update({char: {
                "cross": 0.0,
                "self": 0.0,
            }})
    if copy_paste_cnt > 0:
        shot_copy_paste_score /= copy_paste_cnt
        results.update({"copy-paste-score": shot_copy_paste_score,})
    else:
        results.update({"copy-paste-score": "null"})
    print(results)


if __name__ == "__main__":
    main()