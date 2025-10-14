# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import dataclasses
from typing import Literal

from accelerate import Accelerator
from transformers import HfArgumentParser
from PIL import Image
import json
import itertools

from uno.flux.pipeline import UNOPipeline, preprocess_ref


def horizontal_concat(images):
    widths, heights = zip(*(img.size for img in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for img in images:
        new_im.paste(img, (x_offset, 0))
        x_offset += img.size[0]

    return new_im

@dataclasses.dataclass
class InferenceArgs:
    prompt: str | None = None
    image_paths: list[str] | None = None
    eval_json_path: str | None = None
    offload: bool = False
    num_images_per_prompt: int = 1
    model_type: Literal["flux-dev", "flux-dev-fp8", "flux-schnell"] = "flux-dev"
    width: int = 1344 #512
    height: int = 768 #512
    ref_size: int = -1
    num_steps: int = 25
    guidance: float = 4
    seed: int = 42 #3407
    save_path: str = "output/inference"
    only_lora: bool = True
    concat_refs: bool = False
    lora_rank: int = 512
    data_resolution: int = 512
    pe: Literal['d', 'h', 'w', 'o'] = 'd'



import json
import os
from typing import Dict, List, Any

def model_load(args: InferenceArgs):
    accelerator = Accelerator()

    pipeline = UNOPipeline(
        args.model_type,
        accelerator.device,
        args.offload,
        only_lora=args.only_lora,
        lora_rank=args.lora_rank
    )
    return accelerator, pipeline



def story_gen(args: InferenceArgs):

    accelerator, pipeline = model_load(args)

    data_path = args.data_path 
    '''Please modify it by yourself.'''

    dataset_name = 'ViStory_en'
    # dataset_name = 'ViStory_ch' 
    method = 'uno'

    # dataset_path = f"{data_path}/dataset/{dataset_name}"
    processed_dataset_path = f"{data_path}/dataset_processed/{method}/{dataset_name}"

    # import sys
    # sys.path.append(processed_dataset_path)
    story_json_path = os.path.join(processed_dataset_path, "story_set.json")

    try:
        with open(story_json_path, 'r') as f:
            story_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find story data at {story_json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {story_json_path}")
        return

    ViStory = story_data.get("ViStory", {})
    print(f'{dataset_name}: {len(ViStory)} stories loaded')

    # exclude_list = [f"{i:02d}" for i in range(1, 41)]

    for story_name, story_info in ViStory.items():

        import time
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        print(f"\nProcessing story: {story_name} at {timestamp}")

        args.save_path = f"{data_path}/outputs/{method}/{dataset_name}/{story_name}/{timestamp}"
        os.makedirs(f'{args.save_path}', exist_ok=True)

        if not story_info:
            print(f'Story {story_name} is empty, skipping')
            continue

        print(f'Processing story: {story_name} with {len(story_info)} shots')

        main(args, story_name, story_info, accelerator, pipeline)

        # for shot_idx, shot_info in enumerate(story_info, 1):
        #     shot_prompt = shot_info["shot_prompt"]
        #     shot_image = shot_info["shot_image"]

        #     print(f"\nProcessing shot {shot_idx}/{len(story_info)}")
        #     print(f"Prompt: {shot_prompt.split(';')[0]}")  # Show just the camera info
        #     print(f"Reference images: {len(shot_image)}")

        #     args.prompt = shot_prompt
        #     args.image_paths = shot_image
        #     # args.width = 704 
        #     # args.height = 704

        #     # Assuming main() is your generation function
        #     main(args)

        

def main(args: InferenceArgs, story_name, story_info, accelerator, pipeline):


    assert story_info is not None or args.prompt is not None or args.eval_json_path is not None, \
        "Please provide either prompt or eval_json_path"
    
    if story_info:
        data_dicts = story_info

    elif args.eval_json_path is not None:
        with open(args.eval_json_path, "rt") as f:
            data_dicts = json.load(f)
        data_root = os.path.dirname(args.eval_json_path)
    else:
        data_root = "./"
        data_dicts = [{"prompt": args.prompt, "image_paths": args.image_paths}]

    for (i, data_dict), j in itertools.product(enumerate(data_dicts), range(args.num_images_per_prompt)):
        if (i * args.num_images_per_prompt + j) % accelerator.num_processes != accelerator.process_index:
            continue

        ref_imgs = [
            Image.open(os.path.join(img_path)) if story_info 
            else Image.open(os.path.join(data_root, img_path)) 
            for img_path in data_dict["image_paths"]
        ]
        if args.ref_size==-1:
            args.ref_size = 512 if len(ref_imgs)==1 else 320

        ref_imgs = [preprocess_ref(img, args.ref_size) for img in ref_imgs]

        image_gen = pipeline(
            prompt=data_dict["prompt"],
            width=args.width,
            height=args.height,
            guidance=args.guidance,
            num_steps=args.num_steps,
            seed=args.seed + j,
            ref_imgs=ref_imgs,
            pe=args.pe,
        )
        if args.concat_refs:
            image_gen = horizontal_concat([image_gen, *ref_imgs])

        os.makedirs(args.save_path, exist_ok=True)
        image_gen.save(os.path.join(args.save_path, f"{i}_{j}.png"))

        # save config and image
        args_dict = vars(args)
        args_dict['prompt'] = data_dict["prompt"]
        args_dict['image_paths'] = data_dict["image_paths"]
        with open(os.path.join(args.save_path, f"{i}_{j}.json"), 'w') as f:
            json.dump(args_dict, f, indent=4)        

if __name__ == "__main__":
    parser = HfArgumentParser([InferenceArgs])
    args = parser.parse_args_into_dataclasses()[0]
    story_gen(args)
