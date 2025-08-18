
import os
import json
import argparse

import yaml
from pathlib import Path

class StoryDataset:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def get_story_name_list(self):
        """Get list of all story names from the dataset directory"""        
        # Sort by number (e.g. 01, 02,...)
        entries = os.listdir(self.root_dir)
        return sorted(
            [entry for entry in entries if os.path.isdir(os.path.join(self.root_dir, entry))],
            key=lambda x: int(x) if x.isdigit() else 0
        )

    def _load_story_json(self, story_name):
        story_path = os.path.join(self.root_dir, story_name)
        story_json = os.path.join(story_path, "story.json")
        with open(story_json, "r", encoding="utf-8-sig") as f:
            return json.load(f)

    def load_shots(self, story_name, language="en"):
        """Load and process shots data for a single story.
        
        Args:
            story_name (str): Identifier for the target story
            language (str): Target language code (default: 'en')
            
        Returns:
            list: Processed shot data with standardized names
        """
        story_data = self._load_story_json(story_name)
        shots = []
        
        # Field mapping: original name -> standardized name
        field_mapping = {
            "Setting Description": "scene",
            "Plot Correspondence": "plot",
            "Characters Appearing": "character_name",
            "Static Shot Description": "script",
            "Shot Perspective Design": "camera"
        }
        
        for shot in story_data.get("Shots", []):
            # Validate required name (original name) exist in source data
            original_fields = field_mapping.keys()
            if not all(field in shot for field in original_fields):
                missing = [f for f in original_fields if f not in shot]
                print(f"Warning: Shot {shot.get('index', 'unknown')} missing fields {missing}")
                continue
            
            # Build processed shot structure via standardized name
            processed_shot = {"index": shot["index"]}
            for orig_field, new_field in field_mapping.items():
                if language in shot[orig_field]:
                    processed_shot[new_field] = shot[orig_field][language]
                    processed_shot["character_key"] = shot["Characters Appearing"]['en']
                else:
                    processed_shot[new_field] = f"({language} data missing) " + str(shot[orig_field])
            
            shots.append(processed_shot)
        
        return shots


    def load_characters(self, story_name, language="en"):
        """Load and process characters data for a single story.
        
        Args:
            story_name (str): Identifier of the target story
            language (str): Language code for localization (default: 'en')
            
        Returns:
            dict: Processed character data with structure:
                {
                    char_key: {
                        "name": localized_name,
                        "key": english_name,
                        "prompt": character_description,
                        "tag": character_type,
                        "num_of_appearances": int,
                        "images": [path1, path2,...]
                    }
                }
        """
        story_data = self._load_story_json(story_name)
        characters = {}
        story_path = os.path.join(self.root_dir, story_name)
        
        # Directory containing character reference images
        image_ref_dir = os.path.join(story_path, "image")
        
        for char_key, char_data in story_data.get("Characters", {}).items():
            # Handle multilingual fields (supports variants like prompt_en/prompt_ch)
            name_key = f"name_{language}"
            prompt_key = f"prompt_{language}"
            
            # Extract all metadata fields
            char_info = {
                "name": char_data.get(name_key, ""),  # Prioritize displaying char names in the chosen language (ch/en)
                "key": char_data.get("name_en", ""),  # English name for indexing Chinese name
                "prompt": char_data.get(prompt_key, ""),  # Character description prompt
                "tag": char_data.get("tag", ""),  # non_human/realistic_human/unrealistic_human
                "num_of_appearances": char_data.get("num_of_appearances", -1),  # Default -1 if missing
                "images": []  # Will store valid image paths
            }
            
            # Scan for character images
            char_dir = os.path.join(image_ref_dir, char_key)
            if os.path.isdir(char_dir):
                for root, _, files in os.walk(char_dir):
                    for img_file in sorted(files):  # Process in sorted order
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(root, img_file)
                            # Verify file exists before adding (defensive programming)
                            if os.path.isfile(img_path):
                                char_info["images"].append(img_path)
            
            characters[char_key] = char_info
        
        return characters




    def load_type(self, story_name, language="en"):
        """Load story type"""
        story_data = self._load_story_json(story_name)
        story_type = story_data.get("Story_type", {})
        return story_type.get(language, f"({language} type not defined)")
    
    def load_story(self, story_name, language="en"):
        """Load complete story data"""
        return {
            "type": self.load_type(story_name, language),
            "shots": self.load_shots(story_name, language),
            "characters": self.load_characters(story_name, language)
        }

    def load_stories(self, story_name_list, language="en"):
        """Load multiple stories"""
        return {
            story_name: self.load_story(story_name, language)
            for story_name in story_name_list
        }


    def story_prompt_merge(self, story_data, mode=''):
            """Merge and process story shots with character prompts and images.
            
            Args:
                story_data (dict): Contains 'shots' and 'characters' data
                mode (str): Output mode selector:
                    'all' - returns full shot data
                    'prompt' - returns merged prompts only
                    'image' - returns image paths only  
                    'char_prompt' - returns character prompts only
                    
            Returns:
                Varied: Processed data according to specified mode
                
            Raises: 
                ValueError: If invalid mode is specified
            """
            shots = story_data["shots"]
            characters = story_data["characters"]

            # Initialize output containers
            shots_all = []      # Complete processed shots
            shots_prompt = []   # Merged prompt strings
            shots_image = []    # Image path collections  
            chars_prompt = []   # Individual character prompts

            for shot in shots:
                # 1. Extract character references from shot
                char_keys = []
                char_names = []
                char_prompts = []
                char_images = []
                
                for char_key, char_name in zip(shot['character_key'], shot['character_name']):
                    # Validate character exists in library
                    if char_key not in characters:
                        print(f"Warning: Character {char_key} not defined in character library")
                        continue
                    
                    char_key = char_key.strip()
                    char_keys.append(char_key)
                    char_names.append(char_name)

                    # 2. Extract character metadata
                    char_prompt = characters[char_key]['prompt']
                    # char_image = characters[char_key]['images']  # All images of this character
                    char_image = characters[char_key]['images'][0]  # First image of this character only
                    
                    # Format character description
                    char_prompts.append(f'{char_name} is {char_prompt}')
                    char_images.append(char_image)
                
                # 3. Construct merged shot prompt
                # shot_prompt = (
                #     f"{';'.join(char_prompts)}" # Separate multiple character descriptions with ;
                #     f"{shot['scene']};"
                #     f"{shot['camera']};"
                #     f"{shot['script']};"
                #     f"{shot['plot']};"
                # )
                shot_prompt = (
                    f"{shot['camera']};"
                    f"{shot['plot']};" 
                    f"{shot['script']};"
                    f"{';'.join(char_prompts)};" # Separate multiple character descriptions with ;
                    f"{shot['scene']};"
                )

                shots_all.append({
                    "prompt": shot_prompt,
                    "image_paths": char_images,
                    "char_prompt": char_prompts
                })
                shots_prompt.append(shot_prompt)
                shots_image.append(char_images)
                chars_prompt.append(char_prompts)

                # print(f'prompt:{shot_prompt}')
                # print(f'image_paths:{char_images}')

            if mode == 'all':
                return shots_all
            elif mode == 'prompt':
                return shots_prompt
            elif mode == 'image':
                return shots_image
            elif mode == 'char_prompt':
                return chars_prompt
            else:
                raise ValueError(f"Invalid mode specified: {mode}")


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}  # Return empty dict if file not found

def main():

    grandparent_dir = Path(__file__).resolve().parent.parent.parent.parent
    print(f'grandparent_dir: {grandparent_dir}')

    data_path = f'{grandparent_dir}/data'
    code_path = f'{grandparent_dir}/code'
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
    args = parser.parse_args()

    _dataset_path = args.dataset_path

    print(f'_dataset_path:{_dataset_path}')


    dataset_name = 'ViStory'
    dataset_path = f"{_dataset_path}/{dataset_name}"

    dataset = StoryDataset(dataset_path)
    story_name_list = dataset.get_story_name_list()
    print(f'\nStory name list: {story_name_list}')

    language="en"
    # language="ch"

    # Load all stories (English)
    stories_data = dataset.load_stories(story_name_list, language)
    # print(f'\nRead all story information {stories_data}')
    
    print('\n' \
    '\\\\\\\\\\\\\\\\\\\\\ Fine-grained Information Extraction Example \\\\\\\\\\\\\\\\\\\\\ ')
    # Example: Extract the first shot of the first story
    story_name = '01'  # Assuming there is a story numbered 01
    story_data = stories_data[story_name]

    type = story_data["type"]
    print(f"""
    Story {story_name} story type: {type}
    """)  # Output: "Children's Picture Books"

    shots = story_data["shots"]
    first_shot = shots[0]
    print(f"""
    First shot:
    - Index: {first_shot['index']}
    - Scene: {first_shot['scene']}
    - Subjective plot: {first_shot['plot']}
    - Character name: {first_shot['character_name']}
    - Character key: {first_shot['character_key']}
    - Objective description: {first_shot['script']}
    - Camera design: {first_shot['camera']}
    """)
    
    # Example: Extract the first character information
    characters = story_data["characters"] # Character library
    keys_list = list(characters.keys())
    values_list = list(characters.values())
    characters_1_key = keys_list[0]
    characters_1_value = values_list[0]
    print(f"""
    Character {characters_1_key}:
    - Name: {characters_1_value['name']}
    - Name key: {characters_1_value['key']}
    - Description: {characters_1_value['prompt']}
    - Human/Non-human: {characters_1_value['tag']}
    - Reference images: {characters_1_value['images']}
    """)


    # Prompt decomposition and merging
    shots_all = dataset.story_prompt_merge(story_data,mode='all')
    shots_prompt = dataset.story_prompt_merge(story_data,mode='prompt')
    shots_image = dataset.story_prompt_merge(story_data,mode='image')
    print(f"""
    Prompt decomposition and merging:
    - take the first shot of shots_all: 
        {shots_all[0].keys()}
    - take the prompt from the first shot of shots_all: 
        {shots_all[0]['prompt']}
    - take all character images from the first shot of shots_all: 
        {shots_all[0]['image_paths']}
    - take the first character image from the first shot of shots_all: 
        {shots_all[0]['image_paths'][0]}
    - take the first image of the first character from the first shot of shots_all: 
        {shots_all[0]['image_paths'][0][0]}
    \n
    - shots_prompt: {shots_prompt[0]}
    - shots_image: {shots_image[0]}
    """)



    # Define color codes
    COLOR = {
        "HEADER": "\033[95m",
        "BLUE": "\033[94m",
        "CYAN": "\033[96m",
        "GREEN": "\033[92m",
        "YELLOW": "\033[93m",
        "RED": "\033[91m",
        "BOLD": "\033[1m",
        "UNDERLINE": "\033[4m",
        "END": "\033[0m",
    }

    print(f"""
    {COLOR['HEADER']}Prompt decomposition and merging:{COLOR['END']}
    {COLOR['BLUE']}■ take first shot of shots_all:{COLOR['END']} 
        {COLOR['YELLOW']}Keys: {COLOR['GREEN']}{list(shots_all[0].keys())}{COLOR['END']}

    {COLOR['BLUE']}■ Shot description:{COLOR['END']}
        {COLOR['CYAN']}{shots_all[0]['prompt'][:50]}...{COLOR['END']}

    {COLOR['BLUE']}■ Character image paths:{COLOR['END']}
        {COLOR['YELLOW']}All characters: {COLOR['END']}
        {COLOR['UNDERLINE']}{shots_all[0]['image_paths']}{COLOR['END']}

        {COLOR['YELLOW']}First character: {COLOR['END']}
        {COLOR['BOLD']}{shots_all[0]['image_paths'][0]}{COLOR['END']}

    {COLOR['GREEN']}◆ Merge result preview:{COLOR['END']}
        {COLOR['CYAN']}Prompt: {shots_prompt[0][:100]}...{COLOR['END']}
        {COLOR['CYAN']}Images: {len(shots_image[0])} images{COLOR['END']}
    """)




if __name__ == "__main__":
    main()
