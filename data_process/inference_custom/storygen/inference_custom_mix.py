import json
import os
import collections 

from inference import test


story_id = "01" 

data_path = args.data_path 
pretrain_path = args.pretrain_path 
'''Please modify it by yourself.'''

base_data_path = f"{data_path}/dataset/ViStory"
story_path = os.path.join(base_data_path, story_id)
story_json_path = os.path.join(story_path, "story.json")
image_base_path = os.path.join(story_path, "image")
output_base_dir = f"./generated_stories_en/story_{story_id}" 

pretrained_model_path = f'{pretrain_path}/haoningwu/StoryGen/checkpoint_StorySalon'
num_inference_steps = 50
guidance_scale = 7.0
image_guidance_scale = 4.5 
num_sample_per_prompt = 1 
mixed_precision = "fp16"
sliding_window_size = 1 

with open(story_json_path, 'r', encoding='utf-8') as f:
    story_data = json.load(f)

characters_info = story_data["Characters"]
shots = story_data["Shots"]
story_type_en = story_data["Story_type"]["en"]

generated_history = collections.deque(maxlen=sliding_window_size)

if not os.path.exists(output_base_dir):
    os.makedirs(output_base_dir)

for shot_index, shot in enumerate(shots, 1): 
    print(f"--- Processing Shot {shot_index} ---")

    plot_en = shot["Plot Correspondence"]["en"]
    setting_en = shot["Setting Description"]["en"]
    perspective_en = shot["Shot Perspective Design"]["en"]
    static_desc_en = shot["Static Shot Description"]["en"]

    current_prompt = f"{story_type_en}. {plot_en}. {setting_en}. Perspective: {perspective_en}. Scene details: {static_desc_en}. high quality, detailed illustration."
    # current_prompt = f"{story_type_en}"

    print(f"Current Prompt: {current_prompt}")

    current_char_image_paths = []
    current_char_prompts_en = []
    appearing_chars_en = shot["Characters Appearing"]["en"]

    for char_name_en in appearing_chars_en:
        char_folder = os.path.join(image_base_path, char_name_en)
        found_ref = False
        if os.path.exists(char_folder):
            img_files = [f for f in os.listdir(char_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if img_files:
                current_char_image_paths.append(os.path.join(char_folder, img_files[0]))
                char_info = characters_info.get(char_name_en)
                if char_info and "prompt_en" in char_info:
                    current_char_prompts_en.append(char_info["prompt_en"])
                else:
                    current_char_prompts_en.append(f"A picture of {char_name_en}") # Fallback
                    print(f"Warning: English prompt not found for character '{char_name_en}'. Using fallback.")
                found_ref = True
        if not found_ref:
            print(f"Warning: Reference image not found for character '{char_name_en}' in folder {char_folder}")
            if char_name_en in current_char_prompts_en: 
                 pass 

    print(f"Current Char Images: {current_char_image_paths}")
    print(f"Current Char Prompts: {current_char_prompts_en}")

    final_ref_images = []
    final_ref_prompts = []
    current_stage = ""
    current_image_guidance = image_guidance_scale 

    if shot_index == 1:
        print("Processing first shot: using multi-image-condition.")
        current_stage = "multi-image-condition"
        final_ref_images = current_char_image_paths
        final_ref_prompts = current_char_prompts_en
    else:
        print(f"Processing shot {shot_index}: using auto-regressive with window size up to {sliding_window_size}.")
        current_stage = "auto-regressive"
        prev_frames_data = list(generated_history) 
        prev_image_paths = [item[0] for item in prev_frames_data]
        prev_prompts = [item[1] for item in prev_frames_data]

        print(f"Previous {len(prev_frames_data)} frames added as reference.")

        final_ref_images = current_char_image_paths + prev_image_paths
        final_ref_prompts = current_char_prompts_en + prev_prompts

    if not final_ref_images:
        print("Warning: No reference images (character or previous frames) available. Setting stage to 'no' and image guidance to 0.")
        current_stage = "no"
        current_image_guidance = 0.0
        final_ref_prompts = [] 

    output_image_name = f"shot_{shot_index:02d}.png" 

    generated_image_path = test(
        pretrained_model_path=pretrained_model_path,
        logdir=output_base_dir,      
        prompt=current_prompt,       
        ref_prompt=final_ref_prompts,
        ref_image=final_ref_images,  
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        image_guidance_scale=current_image_guidance,
        num_sample_per_prompt=num_sample_per_prompt,
        stage=current_stage,         
        mixed_precision=mixed_precision,
        image_name=output_image_name 
    )

    if generated_image_path and os.path.exists(generated_image_path):
        print(f"Generated image saved to: {generated_image_path}")
        generated_history.append((generated_image_path, current_prompt))
    else:
        print(f"Error: Failed to generate image for shot {shot_index} or path not returned correctly.")

    print(f"--- Finished Shot {shot_index} ---")
    print(f"Current history size: {len(generated_history)}")

print("Story generation with sliding window complete.")
