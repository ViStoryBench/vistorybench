import os
import json
import base64
import magic
import requests
import time
import threading
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import os
import dataclasses
from typing import Literal
import datetime

from accelerate import Accelerator
from transformers import HfArgumentParser
from PIL import Image
import json
import itertools
import io
import torch
import megfile
from bs4 import BeautifulSoup
import sys
import random
import markdown
from concurrent.futures import ThreadPoolExecutor, as_completed


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
    run_id: str | None =None # "20250509-002833"
    max_workers: int = 64
    only_lora: bool = True
    concat_refs: bool = False
    lora_rank: int = 512
    data_resolution: int = 512
    pe: Literal['d', 'h', 'w', 'o'] = 'd'

def download_with_requests(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(
                url,
                timeout=30,
                headers={'User-Agent': 'Mozilla/5.0'},
                stream=True
            )
            response.raise_for_status()

            buffer = BytesIO()
            for chunk in response.iter_content(8192):
                if chunk:
                    buffer.write(chunk)

            buffer.seek(0)
            Image.open(buffer).verify()
            return Image.open(buffer)

        except Exception as e:
            print(f"❌ 下载失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)
            continue
    return None

def extract_md_images(text):
    try:
        html = markdown.markdown(text)
        soup = BeautifulSoup(html, 'html.parser')
        return [img['src'] for img in soup.find_all('img')]
    except Exception as e:
        print(f"❌ 提取图片链接失败: {str(e)}")
        return []

sys.path.append("/data/Code/")
sys.path.append("/data/Code/models/")

# Configuration
base_url = 'https://models-proxy.stepfun-inc.com/v1'
appkey = "ak-83d7efgh21i5jkl34mno90pqrs62tuv4k1"
headers = {
    'Authorization': f"Bearer {appkey}",
    "Content-Type": "application/json"
}

def convert_image_to_base64(image_input):
    # try:
    if isinstance(image_input, Image.Image):
        img_byte_arr = io.BytesIO()
        image_input.save(img_byte_arr, format='PNG')
        png_binary_data = img_byte_arr.getvalue()
    elif isinstance(image_input, (bytes, bytearray)):
        png_binary_data = image_input
    else:
        raise ValueError('data type not support')

    mime_type = magic.from_buffer(png_binary_data, mime=True)
    base64_encoded_data = base64.b64encode(png_binary_data).decode('utf-8')
    return f"data:{mime_type};base64,{base64_encoded_data}"
    # except Exception as e:
    #     print(f"❌ Base64转换失败: {str(e)}")
    #     return None

def make_api_request(data, max_retries=3):
    for attempt in range(max_retries):
        try:
            time.sleep(random.randint(1, 5))
            response = requests.post(
                base_url + '/chat/completions',
                headers=headers,
                json=data,
                timeout=240
            )
            response.raise_for_status()

            res = ""
            for line in response.iter_lines(decode_unicode=False, chunk_size=10):
                res += line.decode('utf-8')

            res = json.loads(res)

             # 处理响应
            image_link_list = extract_md_images(res['choices'][0]['message']['content'])
            if not image_link_list:
                print(f"❌ 未找到图片链接")

            output_image = download_with_requests(image_link_list[0])
            if not output_image:
                print(f"❌ 未找到图片链接")
            return output_image

        except Exception as e:
            print(f"  - 错误类型: {type(e).__name__}")
            print(f"❌ API请求失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
            response = response.json()
            print(response)
            if attempt < max_retries - 1:
                time.sleep(2)
            continue
    return None
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
        # input_image_bytes = [process_image(img) for img in ref_imgs]
        
        prompt = f'(Output image with width:height=16:9) {prompt}'
        
        print(f"📄 处理: {save_path} | 指令: {prompt}")

        # # 准备请求数据
        # input_image_bytes = convert_image_to_base64(ref_imgs)
        # if not input_image_bytes:
        #     return

        data = {
            'model': 'gpt-4o-all',
            'stream': False,
            "max_tokens": 4096,
            "messages": [
                {
                    'role': 'system',
                    'content': 'You are a helpful assistant that can generate images based on a given prompt.'
                },
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'text',
                            'text': prompt
                        }
                    ]
                }
            ]
        }

        for ref_img in ref_imgs:
            new_trans_data = {
                'type': 'image_url',
                'image_url': {'url': convert_image_to_base64(ref_img)}
            }
            data['messages'][1]['content'].append(new_trans_data)

            # 发送API请求
        res = make_api_request(data)
        if not res:
            print('Waring! gpt-4o generate failed')
            return

        with megfile.smart_open(save_path, 'wb') as f:
            res.save(f, lossless=True)

        print(f"💾 已保存: {save_path}")

        # Create success marker file
        if 'success_marker_path' in item and item['success_marker_path']:
            try:
                with open(item['success_marker_path'], 'w') as smf:
                    smf.write(datetime.datetime.now().isoformat()) # Store timestamp of success
                print(f"✅ Marked as success: {item['success_marker_path']}")
            except Exception as e:
                print(f"❌ Failed to create success marker {item['success_marker_path']}: {e}")

        with last_save_lock:
            now = time.time()
            if last_save_time[0] is not None:
                print(f"⏱️ 与上次保存间隔: {now - last_save_time[0]:.2f} 秒")
            last_save_time[0] = now

    # except Exception as e:
    #     print(f"❌ 意外错误: {str(e)}")

def process_dataset(dataset, save_path, max_workers=3):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for item in dataset:
            future = executor.submit(process_single_item, item, save_path)
            futures.append(future)

        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass
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


if __name__ == "__main__":
    data_path = "/data/AIGC_Research/Story_Telling/StoryVisBMK/data"
    dataset_name = 'WildStory_en'
    method = "gpt4o"
    # save_path = "s3+b://jiaqioss/benchmarks/storfy_telling_benchmark/gpt4o/" # Old save path, not used directly now
    # dataset_path = f"{data_path}/dataset/{dataset_name}"
    processed_dataset_path = f"{data_path}/dataset_processed/{method}/{dataset_name}"
    story_json_path = os.path.join(processed_dataset_path, "story_set.json")
    # data_root = f"" # Seems unused, can be removed if not needed elsewhere for relative paths

    parser = HfArgumentParser([InferenceArgs])
    args = parser.parse_args_into_dataclasses()[0]

    # Determine Run ID and Base Output Directory for this Run
    timestamp: str
    if args.run_id:
        timestamp = args.run_id
        print(f"Attempting to resume or use specified run ID: {timestamp}")
    else:
        import time
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        print(f"Starting new run with ID: {timestamp}")


    try:
        with open(story_json_path, 'r') as f:
            story_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find story data at {story_json_path}")

    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {story_json_path}")

    WildStory = story_data.get("WildStory", {})
    print(f'{dataset_name}: {len(WildStory)} stories loaded')

    for story_name, story_info in tqdm(WildStory.items(), desc="Processing Stories"):

        print(f"\nProcessing story: {story_name} at {timestamp}")

        save_path = f"{data_path}/outputs/{method}/{dataset_name}/{story_name}/{timestamp}"
        os.makedirs(f'{save_path}', exist_ok=True)

        # args.save_path =f"{data_path}/outputs/{method}/{dataset_name}/{story_name}/{timestamp}" # Old logic
        # os.makedirs(f'{args.save_path}', exist_ok=True) # Old logic

        if not story_info:
            print(f'Story {story_name} is empty, skipping')
            continue

        print(f'Processing story: {story_name} with {len(story_info)} shots')

        assert story_info is not None or args.prompt is not None or args.eval_json_path is not None, "Please provide either prompt or eval_json_path"

        data_dicts = story_info # story_info is already the list of shots for the current story

        items_to_process_for_story = []
        for (i, data_dict), j in itertools.product(enumerate(data_dicts), range(args.num_images_per_prompt)):
            shot_identifier = f"shot_{i:02d}"
            image_save_path = os.path.join(save_path, f"{shot_identifier}.png")
            success_marker_file = os.path.join(save_path, f"{shot_identifier}.success")

            if os.path.exists(success_marker_file):
                print(f"✔️ Shot {shot_identifier} for story '{story_name}' already processed (found {success_marker_file}), skipping.")
                continue

            item = {}
            # Image loading logic needs to ensure paths are correct. Assuming image_paths in data_dict are absolute.
            ref_imgs = [
                Image.open(img_path) # Assuming img_path is absolute as per original logic
                for img_path in data_dict["image_paths"]
            ]

            current_ref_size = args.ref_size
            if current_ref_size == -1:
                current_ref_size = 512 if len(ref_imgs) == 1 else 320
            
            processed_ref_imgs = [preprocess_ref(img, current_ref_size) for img in ref_imgs]

            item["prompt"] = data_dict["prompt"]
            item["ref_imgs"] = processed_ref_imgs
            item["save_path"] = image_save_path
            item["success_marker_path"] = success_marker_file
            items_to_process_for_story.append(item)

        if not items_to_process_for_story:
            print(f"No new shots to process for story '{story_name}'. All might be completed already.")
            continue

        # Process all shots for the current story using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(process_single_item, an_item): an_item for an_item in items_to_process_for_story}
            
            progress_bar = tqdm(as_completed(futures), total=len(futures), desc=f"Shots for {story_name}", leave=False)
            for future in progress_bar:
                processed_item = futures[future]
                try:
                    future.result()  # Retrieve result to raise exceptions if any occurred in thread
                except Exception as e:
                    print(f"❌ Error processing shot {processed_item['save_path']} for story '{story_name}': {e}")
                    # Note: The .success file will not be created, so it will be retried on next run.