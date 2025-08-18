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
        """
        Converts story data into the animdirector format and aggregates
        all stories into a single stories.json file.
        """
        # Initialize a dictionary to hold data for ALL stories
        all_stories_output_data = {}
        # The output directory is now one level up (contains stories.json)
        output_dir = self.output_root
        os.makedirs(output_dir, exist_ok=True) # Ensure the base output directory exists

        for story_id in story_name_list:
            if story_id not in stories_data:
                print(f"Warning: Story {story_id} not found in stories_data, skipping.")
                continue

            story_data = stories_data[story_id]
            input_shots = story_data.get("shots", [])
            characters_info = story_data.get("characters", {}) # Get character information

            if not input_shots:
                print(f"Warning: Story {story_id} has no shots data, skipping.")
                continue

            print(f"Processing story (aggregating to stories.json): {story_id}")

            # === Initialize data for segment2prompt ===
            segment2prompt_data = {
                "segment": "",
                "answer": "",
                "final_answer": ""
            }
            char_details = []
            setting_details = set() # Use set to store unique scenes as settings
            scene_segments_for_segment_field = []
            scene_segments_for_answer_field = []

            # --- Extract character information ---
            for key, char_info in characters_info.items():
                name = char_info.get('name', key)
                prompt = char_info.get('prompt', 'No description available.')
                char_details.append(f"{name}: {prompt}")

            # === Process each shot to build scene segments ===
            scene2image_output = {}
            segment_num = len(input_shots)
            scene2image_output["segment_num"] = segment_num

            for i, shot in enumerate(input_shots):
                segment_key = f"Scene 1 Segment {i + 1}" # Use 1-based index

                # --- Build scene2image section ---
                scene_description_for_s2i = f"{shot.get('scene', '')}"
                current_shot_char_prompts = []
                for key in shot.get('character_key', []):
                    char_info = characters_info.get(key)
                    if char_info and char_info.get('prompt'):
                        current_shot_char_prompts.append(char_info['prompt'])
                main_prompt_for_s2i = (
                    f"{shot.get('camera', '')};"
                    f"{shot.get('plot', '')};"
                    f"{shot.get('script', '')};"
                    f"{scene_description_for_s2i.strip()};"
                    f"{';'.join(current_shot_char_prompts)}"
                )
                scene2image_output[segment_key] = {
                    "scene": main_prompt_for_s2i.strip(';'),
                    "prompt": main_prompt_for_s2i.strip(';')
                }

                # --- Build scene segments for segment2prompt ---
                shot_scene = shot.get('scene', 'Unknown Location')
                setting_details.add(shot_scene) # Add to unique settings list
                char_names_in_shot = [characters_info.get(key, {}).get('name', key)
                                      for key in shot.get('character_key', [])]
                char_list_str = f"[{', '.join(char_names_in_shot)}]" if char_names_in_shot else []
                # Combine plot and script as description
                shot_description = f"{shot.get('plot', '')} {shot.get('script', '')}".strip()
                camera_info = shot.get('camera', '')
                camera_str = f"({camera_info}.)" if camera_info else ""

                # Format scene line for segment field
                scene_line_segment = f"{segment_key}: {char_list_str}[{shot_scene}] {shot_description} {camera_str}"
                scene_segments_for_segment_field.append(scene_line_segment.strip())

                # Format scene line for answer field (may need fine-tuning for character/costume details based on actual situation)
                # Here we temporarily use a format similar to segment, you can adjust this logic later as needed
                # For example, adding details like "(in pink)" or "(in white)" requires more complex logic or metadata
                scene_line_answer = f"{segment_key}: {char_list_str}[{shot_scene}] {shot_description} {camera_str}"
                scene_segments_for_answer_field.append(scene_line_answer.strip())

            # --- Assemble segment2prompt ---
            characters_section = "\n".join(char_details)
            # Currently only list scene names as settings
            settings_section = "\n".join(list(setting_details))
            scenes_section_segment = "\n".join(scene_segments_for_segment_field)
            scenes_section_answer = "\n".join(scene_segments_for_answer_field)

            segment2prompt_data["segment"] = (
                f"Characters:\n{characters_section}\n"
                f"Settings:\n{settings_section}\n"
                f"Scenes:\n{scenes_section_segment}"
            )

            # Note: answer field is usually the expected generation result, here temporarily build with extracted information
            segment2prompt_data["answer"] = f"'''{scenes_section_answer}'''"

            segment2prompt_data["final_answer"] = (
                 f"Characters:\n{characters_section}\n"
                 f"Settings:\n{settings_section}\n"
                 f"Scenes:\n'''{scenes_section_answer}'''"
            )

            # Build output structure for current story, including scene2image and segment2prompt
            story_output_data = {
                "segment2prompt": segment2prompt_data, # Add segment2prompt
                "scene2image": scene2image_output
            }
            # Add current story data to the total dictionary with story_id as key
            all_stories_output_data[story_id] = story_output_data

        # After all stories are processed, save aggregated data to stories.json
        output_json_path = os.path.join(output_dir, "stories.json")
        try:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                # Directly write aggregated all_stories_output_data
                json.dump(all_stories_output_data, f, indent=4, ensure_ascii=False)
            print(f"All stories aggregated and saved to: {output_json_path}")
            # Return dictionary containing single file path
            return {self.data_set_name: output_json_path}
        except Exception as e:
            print(f"Error: Unable to save aggregated stories.json to {output_json_path}: {e}")
            return {} # Return empty dictionary to indicate failure

        print("All stories (animdirector format) processing completed.")
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

        method = 'animdirector'

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
        # Now output_paths is a dictionary, e.g. {'vistorybench_en': '.../stories.json'}
        print(f"Conversion completed ({language}, method={method}), output file paths:")
        if output_paths:
            # output_paths dictionary has only one key-value pair
            for dataset_lang, path in output_paths.items():
                print(f"  {dataset_lang}: {path}")
        else:
            print("  No output files generated.")
