import requests 
import json 
import megfile 
import time 
from io import BytesIO
from PIL import Image
import base64 
import magic
import os
import sys
import random
import threading
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import dataclasses
from typing import Literal
from accelerate import Accelerator
from transformers import HfArgumentParser
import itertools
import io
import torch
from bs4 import BeautifulSoup
import markdown


# ====== 全局锁 ======
last_save_lock = threading.Lock()
last_save_time = [None]


@dataclasses.dataclass
class InferenceArgs:
    prompt: str | None = None
    image_paths: list[str] | None = None
    eval_json_path: str | None = None
    offload: bool = False
    num_images_per_prompt: int = 1
    model_type: Literal["flux-dev", "flux-dev-fp8", "flux-schnell"] = "flux-dev"
    width: int = 512
    height: int = 512
    ref_size: int = -1
    num_steps: int = 25
    guidance: float = 4
    seed: int = 3407
    save_path: str = "output/inference"
    only_lora: bool = True
    concat_refs: bool = False
    lora_rank: int = 512
    data_resolution: int = 512
    pe: Literal['d', 'h', 'w', 'o'] = 'd'

sys.path.append("/data/Code/")
sys.path.append("/data/Code/models/")
# from models.evalute_dataset import EvaluationDataset, GROUPS

# Setup API credentials
appkey = "ak-83d7efgh21i5jkl34mno90pqrs62tuv4k1"
headers = {
    'Authorization': "Bearer " + appkey,
    "Content-Type": "application/json"
}

# Global lock for thread synchronization
last_save_lock = threading.Lock()
last_save_time = [None]

def convert_image_to_base64(file_content):
    # print(f'file_content:{file_content}')
    mime_type = magic.from_buffer(file_content, mime=True)
    base64_encoded_data = base64.b64encode(file_content).decode('utf-8')
    return f"data:{mime_type};base64,{base64_encoded_data}"

def convert_base64_to_image(res_item):
    img_data = base64.b64decode(res_item['data'][0]['b64_json'].split(',')[-1])
    img_bytes = BytesIO(img_data)
    img = Image.open(img_bytes).convert('RGB')
    return img

def infer(data, max_retries=3):
    base_url='https://models-proxy.stepfun-inc.com/v1'

    for attempt in range(max_retries):
        try:
            time.sleep(random.randint(1, 3))  # Add random delay between retries
            response = requests.post(
                base_url + '/images/generations',
                headers=headers,
                data=json.dumps(data),
            )
            response.raise_for_status()

            res = ""
            for line in response.iter_lines(decode_unicode=False, chunk_size=10):
                res += line.decode('utf-8')
            return json.loads(res)
        except Exception as e:
            print(f"❌ API请求失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)
            continue
    return None

def preprocess_ref(raw_image: Image.Image, long_size: int = 512):
    # 获取原始图像的宽度和高度
    image_w, image_h = raw_image.size
    # print(f'image_w, image_h:{image_w, image_h}')

    # 计算长边和短边
    if image_w >= image_h:
        new_w = long_size
        new_h = int((long_size / image_w) * image_h)
    else:
        new_h = long_size
        new_w = int((long_size / image_h) * image_w)

    # 按新的宽高进行等比例缩放
    # print(f'long_size / image_w:{long_size}|{image_w}|{long_size / image_w}')
    # print(f'(new_w, new_h):{(new_w, new_h)}')

    raw_image = raw_image.resize((new_w, new_h), resample=Image.LANCZOS)
    target_w = new_w // 16 * 16
    target_h = new_h // 16 * 16

    # 计算裁剪的起始坐标以实现中心裁剪
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    right = left + target_w
    bottom = top + target_h

    # 进行中心裁剪
    raw_image = raw_image.crop((left, top, right, bottom))

    # 转换为 RGB 模式
    raw_image = raw_image.convert("RGB")
    return raw_image


def process_image(image):
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

def process_single_item(item):
    # try:
    prompt = item['prompt']
    ref_imgs = item['ref_imgs']
    save_path = item['save_path']

    prompt = f'(Output image with width:height=16:9) {prompt}'

    print(f"📄 处理: {save_path} | 指令: {prompt}")

    # Process input image
    # input_image_bytes = process_image(ref_imgs)

    # Prepare request data
    data = {
        'model': 'gemini-2.0-flash-exp-img-gen',
        'prompt': prompt,
        # 'images': [convert_image_to_base64(img) for img in ref_imgs],
        'images': [convert_image_to_base64(process_image(img)) for img in ref_imgs],
    }
    # print(f'data:{data}')

    # Make inference request
    res = infer(data)
    if not res:
        return

    # print(f'res:{res}')

    output_image = convert_base64_to_image(res)
    # print(f'output_image:{output_image}')

    with megfile.smart_open(save_path, 'wb') as f:
        output_image.save(f, lossless=True)

    print(f"💾 已保存: {save_path}")

    with last_save_lock:
        now = time.time()
        if last_save_time[0] is not None:
            print(f"⏱️ 与上次保存间隔: {now - last_save_time[0]:.2f} 秒")
        last_save_time[0] = now

    # except Exception as e:
    #     print(f"❌ 处理错误 {save_path}: {str(e)}")

def process_dataset(dataset, save_path, max_workers=4):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for item in dataset:
            future = executor.submit(process_single_item, item, save_path)
            futures.append(future)

        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass

if __name__ == "__main__":

    data_path = "/data/AIGC_Research/Story_Telling/StoryVisBMK/data"
    dataset_name = 'WildStory_en'
    # dataset_name = 'WildStory_ch' 
    method = 'gemini'

    # dataset_path = f"{data_path}/dataset/{dataset_name}"
    processed_dataset_path = f"{data_path}/dataset_processed/{method}/{dataset_name}"

    # import sys
    # sys.path.append(processed_dataset_path)
    story_json_path = os.path.join(processed_dataset_path, "story_set.json")

    data_root = f""

    parser = HfArgumentParser([InferenceArgs])
    args = parser.parse_args_into_dataclasses()[0]

    try:
        with open(story_json_path, 'r') as f:
            story_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find story data at {story_json_path}")

    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {story_json_path}")

    WildStory = story_data.get("WildStory", {})
    print(f'{dataset_name}: {len(WildStory)} stories loaded')

    for story_name, story_info in WildStory.items():

    # if story_name in exclude_list:
    #     print(f"跳过已处理的故事 {story_name}")
    #     continue
        # if story_name <= '02':
        #     continue

        import time
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        print(f"\nProcessing story: {story_name} at {timestamp}")

        save_path = f"{data_path}/outputs/{method}/{dataset_name}/{story_name}/{timestamp}"
        os.makedirs(f'{save_path}', exist_ok=True)

        if not story_info:
            print(f'Story {story_name} is empty, skipping')
            continue

        # print(f'当前处理的故事为{story_name}，{story_info}')
        print(f'Processing story: {story_name} with {len(story_info)} shots')


        assert story_info is not None or args.prompt is not None or args.eval_json_path is not None, "Please provide either prompt or eval_json_path"

        if story_info:
            data_dicts = story_info

        for (i, data_dict), j in itertools.product(enumerate(data_dicts), range(args.num_images_per_prompt)):
            item = {}
            ref_imgs = [
                Image.open(os.path.join(img_path)) if story_info # 绝对路径
                else Image.open(os.path.join(data_root, img_path)) # 相对路径
                for img_path in data_dict["image_paths"]
            ]

            if args.ref_size==-1:
                args.ref_size = 512 if len(ref_imgs)==1 else 320

            ref_imgs = [preprocess_ref(img, args.ref_size) for img in ref_imgs] ##should be pil
            # print(f'ref_imgs:{ref_imgs}')

            item["prompt"] = data_dict["prompt"]
            item["ref_imgs"] = ref_imgs
            item["save_path"] = os.path.join(save_path, f"{i}_{j}.png")

            process_single_item(item)