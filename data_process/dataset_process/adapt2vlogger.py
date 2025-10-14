import os
import json
import shutil
from collections import OrderedDict
from dataset_load import StoryDataset

class StoryConverter:
    def __init__(self, input_root, output_root, dataset):
        self.input_root = input_root
        self.output_root = output_root
        self.dataset = dataset

    def save_json(self, path, data):
        """Save JSON file, maintaining key order"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def convert(self, story_name_list, stories_data):
        for story_name in story_name_list:
            print(f"Processing story: {story_name}")
            output_dir = os.path.join(self.output_root, story_name)
            script_dir = os.path.join(output_dir, "script")
            ref_img_dir = os.path.join(output_dir, "ref_img")

            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(script_dir, exist_ok=True)
            os.makedirs(ref_img_dir, exist_ok=True)

            # Load multilingual shot data
            en_shots = sorted(self.dataset.load_shots(story_name, 'en'), key=lambda x: x['index'])
            zh_shots = sorted(self.dataset.load_shots(story_name, 'ch'), key=lambda x: x['index'])
            
            # 1. Generate video_prompts.txt (English)
            video_prompts = [{
                "video fragment id": shot['index'],
                "video fragment description": shot['script']
            } for shot in en_shots]
            # self.save_json(os.path.join(output_dir, "video_prompts.txt"), video_prompts)

            # 2. Generate zh_video_prompts.txt (Chinese)
            zh_video_prompts = [{
                "sequence_number": shot['index'],
                "description": shot['script']
            } for shot in zh_shots]
            # self.save_json(os.path.join(output_dir, "zh_video_prompts.txt"), zh_video_prompts)

            # 3. Generate protagonists_places.txt
            characters = self.dataset.load_characters(story_name, 'en')
            protagonists = []
            char_id_map = OrderedDict()
            current_id = 1
            for char_key in characters:
                char_data = characters[char_key]
                protagonists.append({
                    "id": current_id,
                    "name": char_data['key'],  # Use English identifier as name
                    "description": char_data['prompt']
                })
                char_id_map[char_key] = current_id
                current_id += 1
            # self.save_json(os.path.join(output_dir, "protagonists_places.txt"), protagonists)

            # 4. Generate protagonists_place_reference.txt
            place_refs = []
            for shot in en_shots:
                character_ids = []
                for char_key in shot.get('character_key', []):
                    if char_key in char_id_map:
                        character_ids.append(char_id_map[char_key])
                place_refs.append({
                    "video segment id": shot['index'],
                    "character/place id": character_ids if character_ids else [0]
                })
            # self.save_json(os.path.join(output_dir, "protagonists_place_reference.txt"), place_refs)

            # 5. Generate time_scripts.txt (using default duration of 3 seconds)
            time_scripts = [{
                "video fragment id": shot['index'],
                "time": 3  # Default 3 seconds, can be modified as needed
            } for shot in en_shots]
            # self.save_json(os.path.join(output_dir, "time_scripts.txt"), time_scripts)
            
            # 1~5. Save the five TXT files to script directory
            self.save_json(os.path.join(script_dir, "video_prompts.txt"), video_prompts)
            self.save_json(os.path.join(script_dir, "zh_video_prompts.txt"), zh_video_prompts)
            self.save_json(os.path.join(script_dir, "protagonists_places.txt"), protagonists)
            self.save_json(os.path.join(script_dir, "protagonists_place_reference.txt"), place_refs)
            self.save_json(os.path.join(script_dir, "time_scripts.txt"), time_scripts)

            # 6. Process reference images
            characters = self.dataset.load_characters(story_name, 'en')
            for char_key in characters:
                char_data = characters[char_key]
                if not char_data['images']:
                    print(f"Character {char_data['key']} has no reference images, skipping save")
                    continue
                
                # Get the first image
                src_img = char_data['images'][0]
                
                # Generate standardized filename
                # char_name = char_data['key'].replace(' ', '_')  # Replace spaces
                char_name = char_data['key']
                file_ext = os.path.splitext(src_img)[1]  # Keep original extension
                dst_img = os.path.join(ref_img_dir, f"{char_name}{file_ext}")
                
                try:
                    shutil.copy(src_img, dst_img)
                    print(f"Saved reference image: {os.path.basename(src_img)} -> {os.path.basename(dst_img)}")
                except Exception as e:
                    print(f"Failed to copy image: {str(e)}")



# if __name__ == "__main__":
#     # Initialize dataset
#     '''Please modify it by yourself.'''
#     data_path = args.data_path
#     dataset_name = 'ViStory'
#     dataset_path = f"{data_path}/dataset/{dataset_name}"
#     dataset = StoryDataset(dataset_path)
    
#     # Create converter
#     processed_root = f"{data_path}/dataset_processed/vlogger/{dataset_name}_en"
#     converter = StoryConverter(dataset_path, processed_root, dataset)
    
#     # Get all stories and convert
#     story_names = dataset.get_story_name_list()
#     stories_data = dataset.load_stories(story_names, 'en')
#     converter.convert(story_names, stories_data)
#     print("\nAll stories processed successfully!")



if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Story Dataset Processing Tool')
    parser.add_argument('--language', type=str, choices=['en', 'ch'], 
                        default='en', help='Language option: en (English) or ch (Chinese)')
    args = parser.parse_args()
    language=args.language

    '''Please modify it by yourself.'''
    data_path = args.data_path
    dataset_name = 'ViStory'

    method = 'vlogger'

    dataset_path = f"{data_path}/dataset/{dataset_name}"
    processed_dataset_path = f"{data_path}/dataset_processed/{method}/{dataset_name}_{language}"

    dataset = StoryDataset(dataset_path)

    input_dir = dataset_path
    output_dir = processed_dataset_path
    converter = StoryConverter(input_dir, output_dir, dataset)


    story_name_list = dataset.get_story_name_list()
    print(f'\nStory name list: {story_name_list}')  # Get all story list
    stories_data = dataset.load_stories(story_name_list,language)  # Load specified story data
    converter.convert(story_name_list, stories_data)


