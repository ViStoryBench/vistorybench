from dataset_load import StoryDataset
import os
import json
import hashlib

class StoryConverter:
    def __init__(self, input_root, output_root):
        self.input_root = input_root
        self.output_root = output_root
        self.image_placeholder = "image/start_image.png"

    def _generate_story_id(self, story_name):
        """Generate unique numeric story ID"""
        return int(hashlib.md5(story_name.encode()).hexdigest()[:8], 16) % 1000000

    def story_format_adapt(self, story_name, story_data):
        """Process single story"""

        shots_image = dataset.story_prompt_merge(story_data, mode='image')
        chars_prompt = dataset.story_prompt_merge(story_data, mode='char_prompt')
        for shot_char in range(len(shots_image)):
            current_image = shots_image[shot_char]
            current_char_prompt = chars_prompt[shot_char]
            # Fix judgment logic: check if current_image is valid
            if current_image and current_char_prompt:  # Equivalent to current_image is not None and current_image != ""
                # print(f"Found valid image: {current_image}")
                break  # Stop after processing the first valid image
        else:
            print("Traversal ended, no image found")
    
        return {
            "id": self._generate_story_id(story_name),
            # "images": [self.image_placeholder],
            "images": [current_image[0]], # Take first image of first character
            "captions": [current_char_prompt[0]] + dataset.story_prompt_merge(story_data,mode='prompt'),
            "orders": list(range(len(story_data["shots"]))),
            "story_name": story_name
        }


    def convert(self, story_name_list, stories_data):
        """Execute conversion"""

        # Create output root directory
        os.makedirs(self.output_root, exist_ok=True)

        # Process each story
        success_count = 0
        for story_name in story_name_list:
            story_data = stories_data[story_name]
            story_data = self.story_format_adapt(story_name, story_data)
            if not story_data:
                print(f"Skipping empty data story: {story_name}")
                continue

            # Build output path
            output_dir = os.path.join(
                self.output_root,
                story_name,
                "json"
            )
            os.makedirs(output_dir, exist_ok=True)
            
            # Save JSON file
            output_path = os.path.join(output_dir, f"{story_name}.json")
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(story_data, f, ensure_ascii=False, separators=(',', ':'))
                success_count += 1
            except Exception as e:
                print(f"Save failed [{story_name}]: {str(e)}")

        print(f"Conversion complete: {success_count} successful, {len(story_name_list)-success_count} failed")



    def merge_json(self, story_name_list):
        """Merge all story JSON files into a single JSONL file"""
        merge_path = os.path.join(
            self.output_root,
            "merge.jsonl"
        )
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(merge_path), exist_ok=True)
        
        success_count = 0
        error_count = 0
        
        with open(merge_path, "w", encoding="utf-8") as merge_file:
            # Traverse all story directories
            for story_name in os.listdir(self.output_root): # This approach ensures excluding conversion failed stories
            # for story_name in story_name_list:
                
                # Skip non-directory files
                if not os.path.isdir(os.path.join(self.output_root, story_name)):
                    continue
                    
                json_path = os.path.join(
                    self.output_root, 
                    story_name,
                    "json",
                    f"{story_name}.json"
                )
                
                # Check if JSON file exists
                if not os.path.exists(json_path):
                    print(f"JSON file not found: {json_path}")
                    error_count += 1
                    continue
                
                # start_image_path = os.path.join(
                #     story_name,
                #     self.image_placeholder
                # )

                try:
                    # Read and write to merge file
                    with open(json_path, "r", encoding="utf-8") as f:
                        story_data = json.load(f)
                    story_data['story_name'] = story_name
                    merge_file.write(json.dumps(story_data, ensure_ascii=False) + "\n")
                    success_count += 1
                except Exception as e:
                    print(f"Error processing {story_name}: {str(e)}")
                    error_count += 1
        
        print(f"Merge complete: {success_count} successful, {error_count} failed")
        print(f"Merge file saved to: {merge_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Story Dataset Processing Tool')
    parser.add_argument('--language', type=str, choices=['en', 'ch'], 
                        default='en', help='Language option: en (English) or ch (Chinese)')
    args = parser.parse_args()
    language = args.language

    data_path = args.data_path
    dataset_name = 'ViStory'
    method = 'seedstory'

    dataset_path = f"{data_path}/dataset/{dataset_name}"
    processed_dataset_path = f"{data_path}/dataset_processed/{method}/{dataset_name}_{language}"

    dataset = StoryDataset(dataset_path)

    input_dir = dataset_path
    output_dir = processed_dataset_path
    converter = StoryConverter(input_dir, output_dir)

    story_name_list = dataset.get_story_name_list()
    print(f'Story name list: {story_name_list}')
    stories_data = dataset.load_stories(story_name_list, language)
    converter.convert(story_name_list, stories_data)
    converter.merge_json(story_name_list)