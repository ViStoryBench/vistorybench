import os
import time
import json
import random
import requests
import os, argparse, pandas as pd
import copy
import time
import random
from tqdm import tqdm
from PIL import Image
import io
import json
import base64
import re

import sys
sys.path.append(f'/data/AIGC_Research/Story_Telling/StoryVisBMK/code/data_process/dataset_process/')
from dataset_load import StoryDataset

from concurrent.futures import ThreadPoolExecutor, as_completed

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
    else:
        raise ValueError(f"The image_mode must be either 'url' or 'path', not {image_mode}.")

class StoryTellingEvalBench:
    def __init__(self, result_dir="", save_path=""):
        self.result_dir = result_dir
        self.save_path = save_path
        self.stories_data, self.story_name_list = self.load_data()

        code_path = '/data/AIGC_Research/Story_Telling/StoryVisBMK/code'
        vlm_bench_path = f'{code_path}/bench/prompt_align/vlm_score'
        self.txt_prompt_list = {
            "scene": f"{vlm_bench_path}/user_prompts/user_prompt_environment_text_align.txt",
            "character_action": f"{vlm_bench_path}/user_prompts/user_prompt_character_text_align.txt",
            "camera": f"{vlm_bench_path}/user_prompts/user_prompt_camera_text_align.txt",
        }

    def load_data(self):

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
            image_path = f"{self.result_dir}/{story_name}/20250430-012249/{index_image}_0.png"
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
    
    def eval_story_multithread(self, max_workers=10):
        result_list = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for key, item in self.stories_data.items():
                item["name"] = key
                future = executor.submit(self.eval_story_single_item, item)
                futures.append(future)
            
        for _ in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result:
                result_list.append(result)
                # import pdb; pdb.set_trace()

        return result_list
    
    def eval_story(self, max_workers=10):
        result_list = self.eval_story_multithread(max_workers)
        save_json = os.path.join(self.save_path, "gpt_score.json")
        with open(save_json, "w") as f:
            json.dump(result_list, f)


    def get_vlm_score(prompt = None, image = None):
        pass


if __name__ == "__main__":

    data_path = "/data/AIGC_Research/Story_Telling/StoryVisBMK/data"
    dataset_name = 'WildStory'
    dataset_path = f"{data_path}/dataset/{dataset_name}"

    method = 'uno'
    language = 'en'

    method_path = f"{data_path}/outputs/{method}"

    result_dir= f"{method_path}/{dataset_name}_{language}"
    save_path= f"{method_path}/bench_results/prompt_align"
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    eval_bench = StoryTellingEvalBench(result_dir, save_path)
    eval_bench.eval_story(max_workers=1)
