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
        """保存JSON文件，保持key顺序"""
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

            # 加载多语言分镜数据
            en_shots = sorted(self.dataset.load_shots(story_name, 'en'), key=lambda x: x['index'])
            zh_shots = sorted(self.dataset.load_shots(story_name, 'ch'), key=lambda x: x['index'])
            
            # 1. 生成 video_prompts.txt（英文）
            video_prompts = [{
                "video fragment id": shot['index'],
                "video fragment description": shot['script']
            } for shot in en_shots]
            # self.save_json(os.path.join(output_dir, "video_prompts.txt"), video_prompts)

            # 2. 生成 zh_video_prompts.txt（中文）
            zh_video_prompts = [{
                "序号": shot['index'],
                "描述": shot['script']
            } for shot in zh_shots]
            # self.save_json(os.path.join(output_dir, "zh_video_prompts.txt"), zh_video_prompts)

            # 3. 生成 protagonists_places.txt
            characters = self.dataset.load_characters(story_name, 'en')
            protagonists = []
            char_id_map = OrderedDict()
            current_id = 1
            for char_key in characters:
                char_data = characters[char_key]
                protagonists.append({
                    "id": current_id,
                    "name": char_data['key'],  # 使用英文标识符作为name
                    "description": char_data['prompt']
                })
                char_id_map[char_key] = current_id
                current_id += 1
            # self.save_json(os.path.join(output_dir, "protagonists_places.txt"), protagonists)

            # 4. 生成 protagonists_place_reference.txt
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

            # 5. 生成 time_scripts.txt（使用默认时长3秒）
            time_scripts = [{
                "video fragment id": shot['index'],
                "time": 3  # 默认3秒，可根据需要修改
            } for shot in en_shots]
            # self.save_json(os.path.join(output_dir, "time_scripts.txt"), time_scripts)
            
            # 1~5. 将五个TXT文件保存到script目录
            self.save_json(os.path.join(script_dir, "video_prompts.txt"), video_prompts)
            self.save_json(os.path.join(script_dir, "zh_video_prompts.txt"), zh_video_prompts)
            self.save_json(os.path.join(script_dir, "protagonists_places.txt"), protagonists)
            self.save_json(os.path.join(script_dir, "protagonists_place_reference.txt"), place_refs)
            self.save_json(os.path.join(script_dir, "time_scripts.txt"), time_scripts)

            # 6. 处理参考图片
            characters = self.dataset.load_characters(story_name, 'en')
            for char_key in characters:
                char_data = characters[char_key]
                if not char_data['images']:
                    print(f"角色 {char_data['key']} 没有参考图片，跳过保存")
                    continue
                
                # 获取第一张图片
                src_img = char_data['images'][0]
                
                # 生成标准化文件名
                # char_name = char_data['key'].replace(' ', '_')  # 替换空格
                char_name = char_data['key']
                file_ext = os.path.splitext(src_img)[1]  # 保留原始扩展名
                dst_img = os.path.join(ref_img_dir, f"{char_name}{file_ext}")
                
                try:
                    shutil.copy(src_img, dst_img)
                    print(f"保存参考图: {os.path.basename(src_img)} -> {os.path.basename(dst_img)}")
                except Exception as e:
                    print(f"复制图片失败: {str(e)}")



# if __name__ == "__main__":
#     # 初始化数据集
#     data_path = "/data/AIGC_Research/Story_Telling/StoryVisBMK/data"
#     dataset_name = 'WildStory'
#     dataset_path = f"{data_path}/dataset/{dataset_name}"
#     dataset = StoryDataset(dataset_path)
    
#     # 创建转换器
#     processed_root = f"{data_path}/dataset_processed/vlogger/{dataset_name}_en"
#     converter = StoryConverter(dataset_path, processed_root, dataset)
    
#     # 获取所有故事并转换
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

    data_path = "/data/AIGC_Research/Story_Telling/StoryVisBMK/data"
    dataset_name = 'WildStory'

    method = 'vlogger'

    dataset_path = f"{data_path}/dataset/{dataset_name}"
    processed_dataset_path = f"{data_path}/dataset_processed/{method}/{dataset_name}_{language}"

    dataset = StoryDataset(dataset_path)

    input_dir = dataset_path
    output_dir = processed_dataset_path
    converter = StoryConverter(input_dir, output_dir, dataset)


    story_name_list = dataset.get_story_name_list()
    print(f'\n故事名列表：{story_name_list}')  # 获取所有故事列表
    stories_data = dataset.load_stories(story_name_list,language)  # 加载指定故事数据
    converter.convert(story_name_list, stories_data)


