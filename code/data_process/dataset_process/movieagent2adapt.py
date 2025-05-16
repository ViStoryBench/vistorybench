from dataset_load import StoryDataset

class StoryConverter:
    def __init__(self, input_root, output_root):
        self.input_root = input_root
        self.output_root = output_root

    def convert(self, story_name_list, stories_data):
        pass


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Story Dataset Processing Tool')
    parser.add_argument('--language', type=str, choices=['en', 'ch'], 
                        default='en', help='Language option: en (English) or ch (Chinese)')
    args = parser.parse_args()
    language=args.language

    data_path = "/data/AIGC_Research/Story_Telling/StoryVisBMK/data"
    dataset_name = 'WildStory'

    method = 'movieagent'

    dataset_path = f"{data_path}/dataset/{dataset_name}"
    processed_dataset_path = f"{data_path}/dataset_processed/{method}/{dataset_name}"

    dataset = StoryDataset(dataset_path)

    input_dir = dataset_path
    output_dir = processed_dataset_path
    converter = StoryConverter(input_dir, output_dir)


    story_name_list = dataset.get_story_name_list()
    print(f'\n故事名列表：{story_name_list}')  # 获取所有故事列表
    stories_data = dataset.load_stories(story_name_list,language)  # 加载指定故事数据
    converter.convert(story_name_list, stories_data)


