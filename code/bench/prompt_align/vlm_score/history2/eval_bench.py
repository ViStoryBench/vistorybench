import os
import time
import json
import random
import requests
import argparse
import pandas as pd
import copy
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import re
import cv2
import glob
from load_dataset import StoryDataset
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn.functional as F
from torchvision.ops import box_convert

from transformers import CLIPProcessor, CLIPModel
from groundingdino.util.inference import load_model, load_image, predict, annotate

DINO_MODEL = "/data/Code/models/Story_telling/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
DINO_WEIGHTS = "/mnt/jfs-test/pretrained_models/grounding-dino/groundingdino_swint_ogc.pth"
# thresholds
BOX_TRESHOLD = 0.25
TEXT_TRESHOLD = 0.25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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



def gptv_query(transcript=None, top_p=0.2, temp=0., model_type="stepfun-gpt-4.1", port=8000, seed=123):
    max_tokens = 512
    wait_time = 10

    model_configs = {
        "stepfun-gpt-4.1": {
            "headers": {
                "Content-Type": "application/json",
                "Authorization": f"Bearer ak-83d7efgh21i5jkl34mno90pqrs62tuv4k1"
            },
            "requests_url": "https://models-proxy.stepfun-inc.com/v1/chat/completions",
            "model_type": "gpt-4.1"
        },
    }

    if model_type in model_configs:
        config = model_configs[model_type]
        headers = config["headers"]
        requests_url = config["requests_url"]
        model_type = config["model_type"]

    data = {
        'model': model_type,
        'max_tokens':max_tokens, 
        'temperature': temp,
        'top_p': top_p,
        'messages':[],
        'seed': seed,
    }
    if transcript is not None:
        data['messages'] = transcript

    response_text, retry, response_json = '', 0, None
    while len(response_text)<2:
        retry += 1
        try:
            response = requests.post(url=requests_url, headers=headers, data=json.dumps(data))
            response_json = response.json()
        except Exception as e:
            if random.random()<1: print(e)
            time.sleep(wait_time)
            continue
        if response.status_code != 200:
            print(response.headers,response.content)
            if random.random()<0.01: print(f"The response status code for is {response.status_code} (Not OK)")
            time.sleep(wait_time)
            data['temperature'] = min(data['temperature'] + 0.2, 1.0)
            continue
        if 'choices' not in response_json:
            time.sleep(wait_time)
            continue
        response_text = response_json["choices"][0]["message"]["content"]
    return response_json["choices"][0]["message"]["content"]

def load_text(text_prompt):
    text_dict = {
        "type": "text",
        "text": text_prompt,
    }
    return text_dict

def encode_image(image_path):
    # Open the image file
    with Image.open(image_path) as img:
        # Resize the image to 512x512
        img = img.resize((512, 512))
        
        # Save the resized image to a bytes buffer
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        
        # Encode the image to base64
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

def load_img(image_input, image_mode='path', image_input_detail='high'):
    """
    加载图片内容为API可接受的格式
    
    Args:
        image_input: 图片路径或URL
        image_mode: 'path' 或 'url'
        image_input_detail: 图片细节级别
    """
    if image_mode == 'url':
        return {
            "type": "image_url",
            "image_url": {
                "url": image_input,
                "detail": image_input_detail,
            },
        }
    elif image_mode == 'path':
        base64_image = encode_image(image_input)
        image_meta = "data:image/png;base64" if 'png' in image_input else "data:image/jpeg;base64"
        return {
            "type": "image_url", 
            "image_url": {
                "url": f"{image_meta},{base64_image}",
                "detail": image_input_detail,
            },
        }
    elif image_mode == 'pil':
        img = image_input.resize((512, 512))
        
        # Save the resized image to a bytes buffer
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        image_meta = "data:image/png;base64"
        return {
            "type": "image_url", 
            "image_url": {
                "url": f"{image_meta},{base64_image}",
                "detail": image_input_detail,
            },
        }
    else:
        raise ValueError(f"The image_mode must be either 'url' or 'path', not {image_mode}.")

def visualize_detection(original_image_path, char_crops, save_dir):
    # 读取原图
    original_img = Image.open(original_image_path)
    original_width, original_height = original_img.size
    
    # 计算裁剪图像的大小和位置
    crop_height = original_height // 3  # 设置裁剪图像高度为原图的1/3
    max_crops_per_row = 3  # 每行最多显示的裁剪图像数
    
    # 计算总高度（原图 + 可能需要的额外行数）
    n_crops = len(char_crops)
    n_rows = (n_crops + max_crops_per_row - 1) // max_crops_per_row
    total_height = original_height + n_rows * crop_height
    
    # 创建新的画布
    result_img = Image.new('RGB', (original_width, total_height), 'white')
    result_img.paste(original_img, (0, 0))
    
    # 在原图下方排列裁剪的图像
    for i, (char_name, crop_img) in enumerate(char_crops):
        row = i // max_crops_per_row
        col = i % max_crops_per_row
        
        # 调整裁剪图像大小
        crop_width = original_width // max_crops_per_row
        crop_img = crop_img.resize((crop_width, crop_height), Image.LANCZOS)
        
        # 计算位置
        x = col * crop_width
        y = original_height + row * crop_height
        
        # 粘贴裁剪图像
        result_img.paste(crop_img, (x, y))
        
        # 添加角色名称标签
        draw = ImageDraw.Draw(result_img)
        font = ImageFont.load_default()  # 使用默认字体，也可以加载自定义字体
        draw.text((x + 5, y + 5), char_name, fill='white', font=font)
    
    # 保存结果
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, os.path.basename(original_image_path))
    result_img.save(save_path)

class StoryTellingEvalBench:
    def __init__(self, result_dir="", save_path="", device="cuda"):
        self.result_dir = result_dir
        self.save_path = save_path
        self.device = device
        self.dataset, self.story_name_list = self.load_data()
        self.txt_prompt_list = {
            "scene": "/data/Code/models/Story_telling/prompts/user_prompt_environment_text_align.txt",
            "character_action": "/data/Code/models/Story_telling/prompts/user_prompt_character_text_align.txt",
            "camera": "/data/Code/models/Story_telling/prompts/user_prompt_camera_text_align.txt",
            "single_character_action": "/data/Code/models/Story_telling/prompts/user_prompt_single_character_text_align.txt"
        }
         # load models
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.dino_model = load_model(DINO_MODEL, DINO_WEIGHTS).to(self.device)

    def load_data(self):
        data_path = os.path.dirname(os.path.abspath(__file__))
        dataset_name = 'WildStory'
        dataset_path = f"{data_path}/data/{dataset_name}"

        dataset = StoryDataset(dataset_path)
        story_name_list = dataset.get_story_name_list()
        print(f'\n故事名列表：{story_name_list}')

        language="en"
        # language="ch"

        # 加载所有故事（英文）
        stories_data = dataset.load_stories(story_name_list, language)

        return stories_data, story_name_list
    
    def check_character_number(self, character_number, shot):
        pass
        character_number_in_img = 0
        
        character_score = character_number_in_img / character_number

        return character_score
    
    def check_character_number_split_by_character(self, character_number, detected_number):
        character_score = detected_number / character_number

        return character_score
    
    ## each dimension
    def eval_multiple_dimensions(self,  multiple_dim_prompt_dict):
        eval_dimension_list = ["scene", "character_action", "camera"]
        score_dict = {}
        for eval_dimension in eval_dimension_list:
            transcript = [{ "role": "system", "content": [] }, {"role": "user", "content": []}]
            eval_dimension_prompt_path = self.txt_prompt_list[eval_dimension]
            with open(eval_dimension_prompt_path, "r") as f:
                eval_dimension_prompt = f.read()

            transcript[-1]["content"].append(load_text(eval_dimension_prompt))
            
            prompt = multiple_dim_prompt_dict[eval_dimension]
            transcript[-1]["content"].append(load_text(prompt))
            transcript[-1]["content"].append(load_img(multiple_dim_prompt_dict["image_path"]))

            max_retry = 10000
            temp_start = 0.0
            while True:
                try:
                    text_align_response = gptv_query(transcript, temp=temp_start, model_type="stepfun-gpt-4.1", port=8000, seed=123)
                    # print(f"eval_dimension: {eval_dimension}, text_align_response: {text_align_response}")
                    pattern = r"(score|Score):\s*[a-zA-Z]*\s*(\d+)"
                    score = re.findall(pattern, text_align_response)
                    score = [int(s) for _, s in score]
                    assert len(score) == 1
                    score = score[0]
                    break
                except Exception as e:
                    temp_start += 0.1
                    max_retry -= 1
                    if max_retry == 0:
                        score = 0
                        break
                    print(f"Error: {e}, retrying...")
            score_dict[eval_dimension] = score

        ## character split eval
        character_score_list = []
        character_image = multiple_dim_prompt_dict["character_image"]
        for item in character_image:
            character_key_item, character_image_item = next(iter(item.items()))
            transcript = [{ "role": "system", "content": [] }, {"role": "user", "content": []}]
            eval_dimension_prompt_path = self.txt_prompt_list["single_character_action"]
            with open(eval_dimension_prompt_path, "r") as f:
                eval_dimension_prompt = f.read()

            transcript[-1]["content"].append(load_text("Below is the system prompt"))
            transcript[-1]["content"].append(load_text(eval_dimension_prompt))

            transcript[-1]["content"].append(load_text("Below is the text prompt"))
            prompt = multiple_dim_prompt_dict["character_action"]
            transcript[-1]["content"].append(load_text(prompt))
            

            transcript[-1]["content"].append(load_text(f"Evaluated Character Name is {character_key_item}"))

            transcript[-1]["content"].append(load_img(character_image_item, image_mode="pil"))

            max_retry = 10000
            temp_start = 0.0
            while True:
                try:
                    text_align_response = gptv_query(transcript, temp=temp_start, model_type="stepfun-gpt-4.1", port=8000, seed=123)
                    # print(f"eval_dimension: {eval_dimension}, text_align_response: {text_align_response}")
                    pattern = r"(score|Score):\s*[a-zA-Z]*\s*(\d+)"
                    score = re.findall(pattern, text_align_response)
                    score = [int(s) for _, s in score]
                    assert len(score) == 1
                    score = score[0]
                    break
                except Exception as e:
                    temp_start += 0.1
                    max_retry -= 1
                    if max_retry == 0:
                        score = 0
                        break
                    print(f"Error: {e}, retrying...")
            character_score_list.append({character_key_item:score})
        score_dict["single_character_action"] = character_score_list

        return score_dict
    
    def eval_story_single_item(self, story_item):
        shots = story_item["shots"]
        story_name = story_item["name"]
        _scene_score_list = []
        _character_number_score_list = []
        _character_action_score_list = []
        _camera_score_list = []

        _all_shot_score_dict = {}
        
        for shot in tqdm(shots, desc=f"Evaluating {story_name}"):
            index = shot['index']
            index_image = index - 1
            image_path = f"{self.result_dir}/{story_name}/20250429-230021/{index_image}_0.png"
            assert os.path.exists(image_path), f"图片不存在: {image_path}"

            scene = shot['scene']
            character_key = shot['character_key']
            character_action = shot['script']
            camera = shot['camera']
            multiple_dim_prompt_dict = {
                "scene": scene,
                "character_action": character_action,
                "camera": camera,
                "image_path": image_path,
            }

            character_number = len(character_key)
            character_number_score = self.check_character_number(character_number, shot)

            _character_number_score_list.append(character_number_score)

            score_dict = self.eval_multiple_dimensions(multiple_dim_prompt_dict)
            score_dict["character_number"] = character_number_score
            _scene_score_list.append(score_dict["scene"])
            _character_action_score_list.append(score_dict["character_action"])
            _camera_score_list.append(score_dict["camera"])

            print(f"scene: {score_dict['scene']}, character_action: {score_dict['character_action']}, camera: {score_dict['camera']}")
            _all_shot_score_dict[index] = score_dict
        
        avg_score_dict = {}
        avg_score_dict["scene"] = sum(_scene_score_list) / len(_scene_score_list)
        avg_score_dict["character_action"] = sum(_character_action_score_list) / len(_character_action_score_list)
        avg_score_dict["camera"] = sum(_camera_score_list) / len(_camera_score_list)
        avg_score_dict["character_number"] = sum(_character_number_score_list) / len(_character_number_score_list)
        _all_shot_score_dict["avg_score_dict"] = avg_score_dict

        return _all_shot_score_dict
    
    def eval_story_single_item_split_by_character(self, story_item):
        shots = story_item["shots"]
        story_name = story_item["name"]
        characters = story_item["characters"]
        character_key = story_item["characters"].keys()
        _scene_score_list = []
        _character_number_score_list = []
        _character_action_score_list = []
        _character_action_global_and_local_score_list = []
        _character_action_global_and_local_score_avg_list = []
        _camera_score_list = []

        _all_shot_score_dict = {}

        TEXT_PROMPT = {char: "people" for char in character_key}
        # get ref character features
        ref_clip_feats = {}
        for char in character_key:
            input_ref_imgs = []
            ref_imgs = characters[char]["images"]
            for img in ref_imgs:
                boxes, _, _, _ = dino_detect(self.dino_model, img, TEXT_PROMPT[char])
                cropped_imgs = crop_img(img, boxes)
                if len(cropped_imgs) != 0:
                    input_ref_imgs.append(cropped_imgs[0]) # assume each ref img cantains only one character
                    # cropped_imgs[0].save(f"{char}-{img_file_name}.png") # NOTE: debug
                else:
                    print(f"\033[33mNo char: {char} found in {img} with prompt: {TEXT_PROMPT[char]}, please check\033[0m")
            ref_feats = get_clip_feat(input_ref_imgs, self.clip_model, self.processor)
            ref_clip_feats[char] = ref_feats

        results = {"cref": {}}
        # 存储完成匹配的主体，用于计算cref的各项指标
        char_pil_imgs = {}

        for shot in tqdm(shots, desc=f"Evaluating {story_name} Split by Character"):
            shot_results = {}
            index = shot['index']
            index_image = index - 1
            image_path = f"{self.result_dir}/{story_name}/20250429-230021/{index_image}_0.png"
            assert os.path.exists(image_path), f"图片不存在: {image_path}"
            scene = shot['scene']
            character_key = shot['character_key']
            character_action = shot['script']
            camera = shot['camera']

            for char in character_key:
                if char not in char_pil_imgs.keys():
                    char_pil_imgs.update({char: []})
                boxes, logits, phrases, annotated_frame = dino_detect(self.dino_model, image_path, TEXT_PROMPT[char])

                # cv2.imwrite(f"{shot}-{char}.png", annotated_frame) # NOTE: debug
                # 裁剪出对应关键词的可能主体
                cropped_imgs = crop_img(image_path, boxes)
                if len(cropped_imgs) != 0:
                    output_feats = get_clip_feat(cropped_imgs, self.clip_model, self.processor)
                    cosine_sims = {}
                    # 预先计算每个主体对每个角色参考图的相似性（取多图余弦相似系数中最大值）
                    for _char in character_key:
                        cosine_sims.update({_char: (output_feats @ ref_clip_feats[_char].T).max(dim=1).values.cpu()})
                    sims_bak = cosine_sims[char].tolist()
                    matched_flag = False
                    # 从最相似的主体开始逐一检查是否有其余角色与其相似度更高；
                    # 若有，则排除该主体，检查次一等相似的主体；若无，则将该主体与该角色匹配
                    while(len(sims_bak) > 0):
                        _id = torch.argmax(torch.Tensor(sims_bak)).item()
                        for _char in character_key:
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
                        char_pil_imgs[char].append({index_image:cropped_imgs[boxes_id]})
                        # cropped_imgs[boxes_id].save(f"{shot}-{char}.png") # NOTE:debug

                    else:
                        # 某个场景中某角色检测失败时，box数据留空
                        shot_results.update({ char: { "box": "null" } })
                else:
                    print(f"\033[33mNo char: {char} found in {image_path} with prompt: {TEXT_PROMPT[char]}, please check\033[0m")
                    # 某个场景中某角色检测失败时，box数据留空
                    shot_results.update({ char: { "box": "null" } })
            results.update({index_image: shot_results})

        # # # 可以解除以下代码的注释以手工检查是否正确识别了角色
        # for char in char_pil_imgs.keys(): # NOTE: for debug
        #     for idx, item in enumerate(char_pil_imgs[char]):
        #         tmp_index, img = next(iter(item.items()))
        #         img.save(f"{img}-{tmp_index}.png")

        # # 分别计算cref的cross指标和self指标
        # for char in character_key:
        #     if char in char_pil_imgs.keys():
        #         char_feats = get_clip_feat(char_pil_imgs[char], self.clip_model, self.processor)
        #         cross_sim = (char_feats @ ref_clip_feats[char].T).mean().item()
        #         self_sim = (char_feats @ char_feats.T).mean().item()
        #         results["cref"].update({char: {
        #             "cross": round(cross_sim, 4),
        #             "self": round(self_sim, 4)
        #         }})
        #     else:
        #         # 全程未检出某个角色时，得分为0.0
        #         results["cref"].update({char: {
        #             "cross": 0.0,
        #             "self": 0.0,
        #         }})
        # print(results)
        for shot in tqdm(shots, desc=f"Evaluating {story_name} Score"):
            index = shot['index']
            index_image = index - 1
            image_path = f"{self.result_dir}/{story_name}/20250429-230021/{index_image}_0.png"
            assert os.path.exists(image_path), f"图片不存在: {image_path}"
            

            scene = shot['scene']
            character_key = shot['character_key']
            character_action = shot['script']
            camera = shot['camera']
            character_image_new = []

            # 收集当前shot中检测到的所有角色裁剪图像
            current_shot_crops = []
            for char in character_key:
                if char in char_pil_imgs:
                    for item in char_pil_imgs[char]:
                        try:
                            current_shot_crops.append((char, item[index_image]))
                            character_image_new.append({char: item[index_image]})
                        except:
                            pass
        
            # 如果检测到了角色，则生成可视化结果
            if current_shot_crops:
                save_dir = os.path.join(self.save_path, story_name, "visualizations")
                visualize_detection(image_path, current_shot_crops, save_dir)
                    

            multiple_dim_prompt_dict = {
                "scene": scene,
                "character_action": character_action,
                "camera": camera,
                "image_path": image_path,
                "character_key":character_key,
                "character_number": len(character_key),
                "character_image":character_image_new
            }

            character_number = len(character_key)
            character_number_score = self.check_character_number_split_by_character(character_number, len(character_image_new))

            _character_number_score_list.append(character_number_score)

            score_dict = self.eval_multiple_dimensions(multiple_dim_prompt_dict)
            score_dict["character_number"] = character_number_score
            _scene_score_list.append(score_dict["scene"])
            _character_action_score_list.append(score_dict["character_action"])
            _camera_score_list.append(score_dict["camera"])
            for item in score_dict["single_character_action"]:
                character_key, character_score = next(iter(item.items()))
                _character_action_global_and_local_score_list.append(character_score)
            _character_action_global_and_local_score_list.append(score_dict["character_action"])
            avg_score = (sum(_character_action_global_and_local_score_list)/ len(_character_action_global_and_local_score_list)) * ((len(character_image_new) + 1) / (character_number + 1))
            _character_action_global_and_local_score_avg_list.append(avg_score)
            print(f"scene: {score_dict['scene']}, character_action: {score_dict['character_action']}, camera: {score_dict['camera']}, single_character_score: {score_dict['single_character_action']}")
            _all_shot_score_dict[index] = score_dict
        
        avg_score_dict = {}
        avg_score_dict["scene"] = sum(_scene_score_list) / len(_scene_score_list)
        avg_score_dict["global_character_action"] = sum(_character_action_score_list) / len(_character_action_score_list)
        avg_score_dict["single_and_global_character_action_avg"] = sum(_character_action_global_and_local_score_avg_list) / len(_character_action_global_and_local_score_avg_list)
        avg_score_dict["camera"] = sum(_camera_score_list) / len(_camera_score_list)
        avg_score_dict["character_number"] = sum(_character_number_score_list) / len(_character_number_score_list)
        _all_shot_score_dict["avg_score_dict"] = avg_score_dict
        
        
    def eval_story_multithread(self, max_workers=10):
        result_list = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for key, item in self.dataset.items():
                item["name"] = key
                future = executor.submit(self.eval_story_single_item, item)
                futures.append(future)
            
        for _ in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result:
                result_list.append(result)
                import pdb; pdb.set_trace()

        return result_list
    
    def eval_story_multithread_split_by_character(self, max_workers=10):
        result_list = []
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = []
            for key, item in self.dataset.items():
                item["name"] = key
                future = executor.submit(self.eval_story_single_item_split_by_character, item)
                futures.append(future)
            
        for _ in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result:
                result_list.append(result)
                import pdb; pdb.set_trace()

        return result_list
    
    def eval_story_by_character_single_thread(self):
        result_list = []
        for key, item in self.dataset.items():
            item["name"] = key
            result = self.eval_story_single_item_split_by_character(item)
            if result:
                result_list.append(result)
        return result_list
    
    def eval_story(self, max_workers=10):
        result_list = self.eval_story_multithread(max_workers)
        with open(self.save_path, "w") as f:
            json.dump(result_list, f)\
            
    def eval_story_by_character(self, max_workers=10):
        result_list = self.eval_story_multithread_split_by_character(max_workers)
        with open(self.save_path, "w") as f:
            json.dump(result_list, f)


    


if __name__ == "__main__":
    eval_bench = StoryTellingEvalBench(result_dir="/data/Code/models/Story_telling/outputs/outputs/uno/WildStory_en", save_path="/data/Code/models/Story_telling")
    eval_bench.eval_story_by_character_single_thread()


