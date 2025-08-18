import json
import os
from inference import test

story_id = "01" 
data_path = args.data_path 
pretrain_path = args.pretrain_path 
'''Please modify it by yourself.'''

base_data_path = f"{data_path}/dataset/ViStory"
story_path = os.path.join(base_data_path, story_id)
story_json_path = os.path.join(story_path, "story.json")
image_base_path = os.path.join(story_path, "image")
output_base_dir = f"./generated_stories_en/story_{story_id}" # 改个名区分

pretrained_model_path = f'{pretrain_path}/haoningwu/StoryGen/checkpoint_StorySalon'
num_inference_steps = 50
guidance_scale = 7.0
image_guidance_scale = 4.5
num_sample_per_prompt = 1
mixed_precision = "fp16"
stage = 'auto-regressive'

with open(story_json_path, 'r', encoding='utf-8') as f:
    story_data = json.load(f)

characters_info = story_data["Characters"]
shots = story_data["Shots"]
story_type_en = story_data["Story_type"]["ch"] 

for shot in shots:
    shot_index = shot["index"]
    print(f"--- Processing Shot {shot_index} ---")

    plot_en = shot["Plot Correspondence"]["en"]
    setting_en = shot["Setting Description"]["en"]
    perspective_en = shot["Shot Perspective Design"]["en"]
    static_desc_en = shot["Static Shot Description"]["en"]

    prompt = f"{story_type_en}. {plot_en}. {setting_en}. Perspective: {perspective_en}. Scene details: {static_desc_en}. high quality, detailed illustration."
    # prompt = f"{story_type_en}"

    print(f"Prompt: {prompt}")

    ref_image_paths = []
    ref_prompts_en = [] 
    appearing_chars_en = shot["Characters Appearing"]["en"] 

    if not appearing_chars_en:
         print("Warning: No characters listed for this shot. Using no reference images.")
         current_stage = "no"
         current_image_guidance = 0
    else:
        current_stage = stage
        current_image_guidance = image_guidance_scale
        for char_name_en in appearing_chars_en:
            char_folder = os.path.join(image_base_path, char_name_en)
            found_ref = False
            if os.path.exists(char_folder):
                for img_file in os.listdir(char_folder):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        ref_image_paths.append(os.path.join(char_folder, img_file))
                        char_info = characters_info.get(char_name_en)
                        if char_info and "prompt_en" in char_info:
                             ref_prompts_en.append(char_info["prompt_en"]) 
                        else:
                             ref_prompts_en.append(f"A picture of {char_name_en}") 
                             print(f"Warning: English prompt not found for character '{char_name_en}'. Using fallback.")
                        found_ref = True
                        break 
            if not found_ref:
                 print(f"Warning: Reference image not found for character '{char_name_en}' in folder {char_folder}")

    if len(ref_image_paths) != len(ref_prompts_en):
        print(f"ERROR: Mismatch between number of reference images ({len(ref_image_paths)}) and reference prompts ({len(ref_prompts_en)}) for shot {shot_index}. Skipping image guidance.")
        ref_image_paths = []
        ref_prompts_en = []
        current_stage = "no"
        current_image_guidance = 0


    print(f"Ref Images: {ref_image_paths}")
    print(f"Ref Prompts: {ref_prompts_en}")


    if not ref_image_paths:
         print("No valid reference images/prompts found, stage set to 'no'.")
         current_stage = "no"
         current_image_guidance = 0
         ref_prompts_en = [] 

    test(
        pretrained_model_path=pretrained_model_path,
        logdir=output_base_dir,
        prompt=prompt,               
        ref_prompt=ref_prompts_en,   
        ref_image=ref_image_paths,   
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        image_guidance_scale=current_image_guidance,
        num_sample_per_prompt=num_sample_per_prompt,
        stage=current_stage,
        mixed_precision=mixed_precision,
        image_name=f"shot_{shot_index:02d}.png"
    )

    print(f"--- Finished Shot {shot_index} ---")


print("Story generation complete (English prompts).")

