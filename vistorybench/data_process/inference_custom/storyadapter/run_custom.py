import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, \
    DDIMScheduler, AutoencoderKL
from PIL import Image
import numpy as np
from ip_adapter import StoryAdapterXL
import os
import random
import argparse

code_path = args.code_path 
'''Please modify it by yourself.'''

prompt_weighted_path = f'{code_path}/data_process/dataset_process'
import sys
sys.path.append(prompt_weighted_path)
from prompt_weighted import prompt_weighted_encode


def model_load():

    base_model_path = args.base_model_path
    image_encoder_path = args.image_encoder_path
    ip_ckpt = args.ip_ckpt
    device = args.device

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )

    # load SD pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        feature_extractor=None,
        safety_checker=None
    )

    # load story-adapter
    storyadapter = StoryAdapterXL(pipe, image_encoder_path, ip_ckpt, device, prompt_weighted_encode)

    return storyadapter


def story_adapter_gen(dataset_name, story_name, story, ref_images, mode):


    style = args.style

    def image_grid(imgs, rows, cols):
        assert len(imgs) == rows * cols

        w, h = imgs[0].size
        grid = Image.new('RGB', size=(cols * w, rows * h))
        grid_w, grid_h = grid.size

        for i, img in enumerate(imgs):
            grid.paste(img, box=(i % cols * w, i // cols * h))
        return grid





    character=True
    fixing_prompts = []
    for prompt, image in zip(story, ref_images):
        if character == True:
            if 'Robinson' in prompt:
                prompt = prompt.replace('Robinson', 'a man, wearing tattered sailor clothes.')
            if 'Friday' in prompt:
                prompt = prompt.replace('Friday', 'a chimpanzee.')
        fixing_prompts.append(prompt)

    prompts = fixing_prompts


    import time
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print(timestamp)

    data_path = args.data_path 
    '''Please modify it by yourself.'''

    save_dir = f"{data_path}/outputs/{method}/{mode}/{dataset_name}/{story_name}/{timestamp}"

    os.makedirs(f'{save_dir}', exist_ok=True)
    os.makedirs(f'{save_dir}/results_xl', exist_ok=True)


    # os.makedirs(f'./story', exist_ok=True)
    # os.makedirs(f'./story/results_xl', exist_ok=True)


    for i, (text, imgs) in enumerate(zip(prompts, ref_images)):

        ref_images_list = []
        for img in imgs:
            # image = Image.open(f'./story/results_xl/img_{y}.png')
            Img = Image.open(img)
            # image = image.resize((256, 256))
            ref_images_list.append(Img)

        if mode == 'text_only' or ref_images_list == []:
            images = storyadapter.generate(
                pil_image=None, 
                num_samples=1, 
                num_inference_steps=50, 
                seed=seed,
                prompt=text,
                scale=0.3, 
                use_image=False, 
                style=style
                )
        elif mode == 'img_ref' and ref_images_list:
            images = storyadapter.generate(
                pil_image=ref_images_list, 
                num_samples=1, 
                num_inference_steps=50, 
                seed=seed,
                prompt=text,
                scale=0.3, 
                use_image=True, 
                style=style
                )
        
        # images = storyadapter.generate(
        #     pil_image=None, 
        #     num_samples=1, 
        #     num_inference_steps=50, 
        #     seed=seed,
        #     prompt=text, 
        #     scale=0.3, 
        #     use_image=False, 
        #     style=style
        #     )

        grid = image_grid(images, 1, 1)
        # grid.save(f'./story/results_xl/img_{i}.png')
        grid.save(f'{save_dir}/results_xl/img_{i}.png')

    images = []
    for y in range(len(prompts)):
        # image = Image.open(f'./story/results_xl/img_{y}.png')
        image = Image.open(f'{save_dir}/results_xl/img_{y}.png')
        image = image.resize((256, 256))
        images.append(image)


    # scales = np.linspace(0.3,0.5,10)
    scales = np.linspace(0.3,0.5,5)
    print(f'scales:{scales}')

    for i, scale in enumerate(scales):
        new_images = []
        # os.makedirs(f'./story/results_xl{i+1}', exist_ok=True)
        os.makedirs(f'{save_dir}/results_xl{i+1}', exist_ok=True)
        print(f'epoch:{i+1}')
        for y, text in enumerate(prompts):
            image = storyadapter.generate(pil_image=images, num_samples=1, num_inference_steps=50, seed=seed,
                                    prompt=text, scale=scale, use_image=True, style=style)
            new_images.append(image[0].resize((256, 256)))
            grid = image_grid(image, 1, 1)
            # grid.save(f'./story/results_xl{i+1}/img_{y}.png')
            grid.save(f'{save_dir}/results_xl{i+1}/img_{y}.png')
        images = new_images




data_path = args.data_path 
'''Please modify it by yourself.'''

dataset_name = 'ViStory_en'
# dataset_name = 'ViStory_ch'
method = 'storyadapter'

dataset_path = f"{data_path}/dataset/{dataset_name}"
processed_dataset_path = f"{data_path}/dataset_processed/{method}/{dataset_name}"

import sys
sys.path.append(processed_dataset_path)
from story_list import StorySet, StorySet_image

# print(f'{dataset_name}:{StorySet}')




parser = argparse.ArgumentParser()
# parser.add_argument('--base_model_path', default=r"./RealVisXL_V4.0", type=str)
# parser.add_argument('--image_encoder_path', type=str, default=r"./IP-Adapter/sdxl_models/image_encoder")
# parser.add_argument('--ip_ckpt', default=r"./IP-Adapter/sdxl_models/ip-adapter_sdxl.bin", type=str)
parser.add_argument('--base_model_path', default=r"/data/pretrain/SG161222/RealVisXL_V4.0", type=str)
parser.add_argument('--image_encoder_path', type=str, default=r"/data/pretrain/h94/IP-Adapter/sdxl_models/image_encoder")
parser.add_argument('--ip_ckpt', default=r"/data/pretrain/h94/IP-Adapter/sdxl_models/ip-adapter_sdxl.bin", type=str)

parser.add_argument('--style', type=str, default='comic', choices=["comic","film","realistic"])
parser.add_argument('--device', default="cuda", type=str)
# parser.add_argument('--story', default=story, nargs='+', type=str)

parser.add_argument('--mode', default="img_ref", choices=['img_ref','text_only'], type=str)

args = parser.parse_args()

mode = args.mode

storyadapter = model_load()
# seed = random.randint(0, 100000)
seed = 42
print(seed)
for (story_key, story),(image_key, image) in zip(StorySet.items(),StorySet_image.items()):
    assert story_key == image_key, f"story_name不匹配：{story_name} != {story_name}"
    story_name = story_key
    
    # if story and story_name not in [f"{i:02d}" for i in range(1, 32)]:
    if story:
        print(f'Story name: {story_name}')
        story_adapter_gen(dataset_name, story_name, story, image, mode)
    else:
        print(f'Story {story_name} empty')

