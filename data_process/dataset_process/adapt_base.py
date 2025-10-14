from dataset_load import StoryDataset
import argparse
import yaml
from pathlib import Path

class StoryConverter:
    def __init__(self, input_root, output_root):
        self.input_root = input_root
        self.output_root = output_root

    def convert(self, story_name_list, stories_data):
        pass


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}  # Return empty dict if file not found

if __name__ == "__main__":

    grandparent_dir = Path(__file__).resolve().parent.parent.parent.parent
    print(f'grandparent_dir: {grandparent_dir}')

    data_path = f'{grandparent_dir}/data'
    code_path = f'{grandparent_dir}/vistorybench'
    print(f'data_path: {data_path}')
    print(f'code_path: {code_path}')
    
    base_parser = argparse.ArgumentParser(description='Application path configuration', add_help=False)
    base_parser.add_argument('--config', type=str, default=f'{code_path}/config.yaml', help='Path to configuration file (default: config.yaml)')
    base_args, _ = base_parser.parse_known_args()  # Parse only known args
    config = load_config(base_args.config)
    
    parser = argparse.ArgumentParser(
        description='Story Dataset Processing Tool',
        parents=[base_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--dataset_path', type=str, default=config.get('dataset_path') or f'{data_path}/dataset', help='Directory for datasets')
    parser.add_argument('--processed_dataset_path', type=str, default=config.get('processed_dataset_path') or f'{data_path}/dataset_processed', help='Directory for datasets')
    parser.add_argument('--language', type=str, choices=['en', 'ch'], 
                        default='en', help='Language option: en (English) or ch (Chinese)')
    args = parser.parse_args()

    # -------------------------------------------------------------------------------------

    method = 'your_method_name'
    dataset_name = 'ViStory'
    language=args.language
    
    dataset_path = args.dataset_path
    processed_dataset_path = args.processed_dataset_path
    dataset_name_path = f"{dataset_path}/{dataset_name}"
    processed_dataset_name_path = f"{processed_dataset_path}/{method}/{dataset_name}"
    print(f'dataset_name_path:{dataset_name_path}')
    print(f'processed_dataset_name_path:{processed_dataset_name_path}')

    dataset = StoryDataset(root_dir = dataset_name_path)
    converter = StoryConverter(input_root = dataset_name_path, output_root = processed_dataset_name_path)

    story_name_list = dataset.get_story_name_list()
    print(f'\nStory name list：{story_name_list}')  # Get list of all stories
    stories_data = dataset.load_stories(story_name_list,language)
    converter.convert(story_name_list, stories_data)


