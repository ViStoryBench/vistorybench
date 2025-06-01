from dataset_load import StoryDataset
import json
import os

class StoryConverter:
    def __init__(self, input_root, output_root):
        self.input_root = input_root
        self.output_root = output_root

    def convert(self, story_name_list, stories_data):
        story_set = {}
        
        for story_name in story_name_list:
            story_data = stories_data[story_name]
            shots = dataset.story_prompt_merge(story_data,mode='all')
            story_set[story_name] = shots
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_root, exist_ok=True)
        
        # Save the output
        output_path = os.path.join(self.output_root, "story_set.json")
        with open(output_path, 'w') as f:
            json.dump(
                {f"{dataset_name}": story_set}, 
                f, 
                indent=4,
                ensure_ascii=False
                )
        
        return {f"{dataset_name}": story_set}
    

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Story Dataset Processing Tool')
    parser.add_argument('--language', type=str, choices=['en', 'ch'], 
                        default='en', help='Language option: en (English) or ch (Chinese)')
    args = parser.parse_args()
    language=args.language

    data_path = "/data/AIGC_Research/Story_Telling/StoryVisBMK/data"
    dataset_name = 'WildStory'

    method = 'uno'

    dataset_path = f"{data_path}/dataset/{dataset_name}"
    processed_dataset_path = f"{data_path}/dataset_processed/{method}/{dataset_name}_{language}"

    dataset = StoryDataset(dataset_path)

    input_dir = dataset_path
    output_dir = processed_dataset_path
    converter = StoryConverter(input_dir, output_dir)

    story_name_list = dataset.get_story_name_list()
    print(f'\n故事名列表：{story_name_list}')  # 获取所有故事列表
    stories_data = dataset.load_stories(story_name_list,language)  # 加载指定故事数据
    result = converter.convert(story_name_list, stories_data)
    print("Conversion completed and saved to:", os.path.join(output_dir, "story_set.json"))
