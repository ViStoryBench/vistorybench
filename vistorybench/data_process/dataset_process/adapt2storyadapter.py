import os
from dataset_load import StoryDataset  # Assuming original code file name is story_dataset.py

class StoryConverter:
    def __init__(self, input_root, output_root):
        self.input_root = input_root
        self.output_root = output_root

    def convert(self, story_name_list, stories_data):
        
        # Create result dictionary and final list
        shots_prompt_dict = {}
        story_image_dict = {}
        StorySet = []
        StorySet_image = []
        story_name_variate_list = []
        story_name_variate_image_list = []
        
        # Iterate through each story to generate corresponding format
        for story_name in story_name_list:

            story_data = stories_data[story_name]

            shots_prompt = dataset.story_prompt_merge(story_data,mode='prompt')
            shots_image = dataset.story_prompt_merge(story_data,mode='image')

            # Dynamically create variables and add to list
            story_name_variate = f'story_{story_name}'
            story_name_variate_image = f'story_{story_name}_image'
            exec(f"{story_name_variate} = {shots_prompt}")  # Create variable named after story name
            exec(f"{story_name_variate_image} = {shots_prompt}")  # Create variable named after story name
            StorySet.append(eval(story_name_variate))  # Add to total list
            StorySet_image.append(eval(story_name_variate_image))  # Add to total list
            story_name_variate_list.append(story_name_variate)
            story_name_variate_image_list.append(story_name_variate_image)
            
            # Also save to dictionary for debugging
            shots_prompt_dict[story_name_variate] = shots_prompt
            story_image_dict[story_name_variate_image] = shots_image
        
        # Save generated results to file
        json_path = os.path.join(
            self.output_root, 
            "story_list.py"
        )

        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        
        with open(json_path, "w", encoding="utf-8") as f:
            # Write variables for each story
            for name_prompt,name_image in zip(story_name_variate_list,story_name_variate_image_list):
                f.write(f"{name_prompt} = [\n")
                for desc in shots_prompt_dict[name_prompt]:
                    escaped_desc = desc.replace('"', r'\"').replace("'", r"\'")  # Escape double quotes and single quotes
                    f.write(f'    "{escaped_desc}",\n')
                f.write("]\n\n")
            
                f.write(f"{name_image} = [\n")
                for desc in story_image_dict[name_image]:
                    f.write(f'    {desc},\n')
                f.write("]\n\n")

            # Write total collection
            # f.write("StorySet = [\n")
            # for name in story_name_list:
            #     f.write(f"    {name},\n")
            # f.write("]\n")
                
            f.write("StorySet = {\n")
            for name, name_variate in zip(story_name_list, story_name_variate_list):
                f.write(f'    "{name}": {name_variate},\n')  # Note: use English double quotes
            f.write("}\n")

            f.write("StorySet_image = {\n")
            for name, name_variate in zip(story_name_list, story_name_variate_image_list):
                f.write(f'    "{name}": {name_variate},\n')  # Note: use English double quotes
            f.write("}\n")

        return StorySet, StorySet_image

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Story Dataset Processing Tool')
    parser.add_argument('--language', type=str, choices=['en', 'ch'], 
                        default='en', help='Language option: en (English) or ch (Chinese)')
    args = parser.parse_args()
    language=args.language

    data_path = args.data_path
    dataset_name = 'ViStory'

    method = 'storyadapter'

    dataset_path = f"{data_path}/dataset/{dataset_name}"
    processed_dataset_path = f"{data_path}/dataset_processed/{method}/{dataset_name}_{language}"


    dataset = StoryDataset(dataset_path)

    input_dir = dataset_path
    output_dir = processed_dataset_path
    converter = StoryConverter(input_dir, output_dir)


    story_name_list = dataset.get_story_name_list()
    print(f'\nStory name list: {story_name_list}')  # Get all story list
    stories_data = dataset.load_stories(story_name_list, language)  # Load specified story data
    StorySet, StorySet_image = converter.convert(story_name_list, stories_data)
    print(f"Successfully generated {len(StorySet)} story datasets")

