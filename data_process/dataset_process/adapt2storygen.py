from dataset_load import StoryDataset
import json
import os

class StoryConverter:
    def __init__(self, input_root, output_root_base, method, dataset_name, language):
        self.input_root = input_root
        self.output_root_base = output_root_base # Base for dataset_processed
        self.output_results_base = "/Story_Telling/vistorybench/outputs" # Base for outputs like images
        self.method = method
        self.dataset_name = dataset_name
        self.language = language
        self.data_set_name = f"{self.dataset_name}_{self.language}"
        # Construct the specific output path for this configuration
        self.output_root = os.path.join(self.output_root_base, self.method,  self.data_set_name)

    def story_prompt_merge(self, story_data):
        characters = story_data["characters"]
        shots = []
        for shot in story_data["shots"]:
            char_names = [c.strip() for c in shot["character"]] if shot.get("character") else []
            print(f'char_names:{char_names}')
            char_prompts = []
            char_images = []
            for name in char_names:
                if name in characters:
                    char_prompt = characters[name]['prompt']
                    # char_image = characters[name]['images'] # Take all images for each character
                    char_image = characters[name]['images'][0] # Take one image for each character
                    char_prompts.append(f'{name} is {char_prompt}')
                    char_images.append(char_image)
            shot_prompt = (
                f"{shot['camera']};"
                f"{shot['plot']};"
                f"{shot['script']};"
                f"{shot['scene']};"
                f"{';'.join(char_prompts)}"  # Use ; to separate multiple character descriptions
            )

            shots.append({
                "prompt": shot_prompt,
                "image_paths": char_images
            })

            print(f'prompt:{shot_prompt}')
            print(f'image_paths:{char_images}')

        return shots

    def convert(self, story_name_list, stories_data):
        all_story_outputs = {}
        for story_name in story_name_list:
            if story_name not in stories_data:
                print(f"Warning: Story {story_name} not found in stories_data, skipping.")
                continue

            story_data = stories_data[story_name]
            characters = story_data.get("characters", {})
            shots_data = story_data.get("shots", [])
            output_shots = {}

            print(f"Processing story: {story_name}")

            for shot in shots_data:
                shot_id = f"{shot.get('index', 'unknown'):02d}"
                char_keys = shot.get("character_key", [])

                char_prompts = []
                ref_images = []
                for key in char_keys:
                    char_info = characters.get(key)
                    if char_info:
                        prompt = char_info.get('prompt', '')
                        images = char_info.get('images', [])
                        if prompt:
                            char_prompts.append(prompt)
                        if images:
                            ref_images.append(images[0]) # Take first image for each character
                    else:
                         print(f"Warning: Key '{key}' not found in character list of story {story_name}")


                # Build main prompt
                main_prompt = (
                    f"{shot.get('camera', '')};"
                    f"{shot.get('plot', '')};"
                    f"{shot.get('script', '')};"
                    f"{shot.get('scene', '')};"
                    f"{';'.join(char_prompts)}"
                )

                # Build log_dir (pointing to future generated image directory)
                # Use placeholder {time_stamp}
                log_dir = os.path.join(
                    self.output_results_base,
                    self.method,
                    self.data_set_name,
                    story_name,
                    "{time_stamp}"
                )

                image_name = f"shot_{shot_id}.png"

                output_shots[shot_id] = {
                    "prompt": main_prompt,
                    "prev_p": char_prompts,
                    "ref_images": ref_images,
                    "log_dir": log_dir,
                    "image_name": image_name,
                    "windows_size": 1
                }
                # print(f"  Processing Shot {shot_id}: prompt={main_prompt[:50]}..., prev_p={len(char_prompts)}, ref_images={len(ref_images)}")


            # Create output directory for current story
            output_story_dir = os.path.join(self.output_root, story_name)
            os.makedirs(output_story_dir, exist_ok=True)

            # Save shots.json
            output_json_path = os.path.join(output_story_dir, "shots.json")
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(output_shots, f, indent=4, ensure_ascii=False)

            print(f"Story {story_name} shots.json saved to: {output_json_path}")
            all_story_outputs[story_name] = output_json_path # Store path for reference

        print("All stories processed.")
        return all_story_outputs


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Story Dataset Processing Tool')
    parser.add_argument('--language', type=str, choices=['en', 'ch'],
                        default='en', help='Language option: en (English) or ch (Chinese)')
    args = parser.parse_args()

    '''Please modify it by yourself.'''
    data_path = args.data_path
    dataset_name = 'ViStory'

    language=args.language
    for language in ['en', 'ch']:
        args.language = language

        method = 'storygen' # Or could be another method name

        dataset_path = f"{data_path}/dataset/{dataset_name}"
        # Base path for processed data, specific subdirs created by converter
        processed_dataset_base_path = f"{data_path}/dataset_processed"

        dataset = StoryDataset(dataset_path)

        converter = StoryConverter(
            input_root=dataset_path,
            output_root_base=processed_dataset_base_path,
            method=method,
            dataset_name=dataset_name,
            language=language
        )

        story_name_list = dataset.get_story_name_list()
        print(f'Story name list: {story_name_list}')
        stories_data = dataset.load_stories(story_name_list, language)
        # print(f'Loaded story data (partial): {json.dumps(dict(list(stories_data.items())[0:1]), indent=2, ensure_ascii=False)}') # Debug: print first story data

        output_paths = converter.convert(story_name_list, stories_data)
        print("Conversion complete, output file paths:")
        for story, path in output_paths.items():
            print(f"  {story}: {path}")
# stories_data=
# {
#     story_id(01):{
#         type:"",
#         shots:[
#             shot_id(00):{
#                 index:1,
#                 scene:'',
#                 plot:'',
#                 script:'',
#                 camera:'',
#                 character_key:['Litte Brown Rabbit',...],
#                 character_name:['',...],
#             },
#             ...
#         ],
#         characters:[
#             character_name(Litte Brown Rabbit):{
#                 name:'Litte Brown Rabbit',
#                 key:'Litte Brown Rabbit',
#                 prompt:'',
#                 tag:'',
#                 num_of_appearance:16,
#                 tag:'',
#                 images:[
#                     'data/dataset/ViStory/01/image/Little Brown Rabbit/00.jpg',
#                     ...
#                 ]
#             },
#             ...
#         ]
        
#     }
#     ...
# }

# dataset_processed:
#     storygen:
#         data_set(vistorybench_en,vistorybench_ch):
#             story_id(01):
#                 shots.json
# shots.json{
#     shot_id(01):{
#         prompt:'',
#         prev_p:[],
#         ref_iamges[],
#         log_dir:''
#         image_name:''
#         windows_size:1
#     }
# }
# outputs:
#     storygen:
#         mode("multi-image-condition", "auto-regressive","mix"):
#             data_set(vistorybench_en,vistorybench_ch):
#                 story_id(01):
#                     time_stamp(20250430-022517):
#                         shot_id.png(shot_01.png)