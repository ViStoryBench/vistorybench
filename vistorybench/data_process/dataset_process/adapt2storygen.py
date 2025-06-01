from dataset_load import StoryDataset
import json
import os

class StoryConverter:
    def __init__(self, input_root, output_root_base, method, dataset_name, language):
        self.input_root = input_root
        self.output_root_base = output_root_base # Base for dataset_processed
        self.output_results_base = "/data/AIGC_Research/Story_Telling/StoryVisBMK/outputs" # Base for outputs like images
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
                    # char_image = characters[name]['images'] # 每个角色取全部
                    char_image = characters[name]['images'][0] # 每个角色取一张
                    char_prompts.append(f'{name} is {char_prompt}')
                    char_images.append(char_image)
            shot_prompt = (
                f"{shot['camera']};"
                f"{shot['plot']};"
                f"{shot['script']};"
                f"{shot['scene']};"
                f"{';'.join(char_prompts)}"  # 用;分隔多个角色描述
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
                print(f"警告：在 stories_data 中找不到故事 {story_name}，跳过。")
                continue

            story_data = stories_data[story_name]
            characters = story_data.get("characters", {})
            shots_data = story_data.get("shots", [])
            output_shots = {}

            print(f"正在处理故事: {story_name}")

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
                            ref_images.append(images[0]) # 每个角色取第一张图
                    else:
                         print(f"警告：在故事 {story_name} 的角色列表中找不到键 '{key}'")


                # 构建主 prompt
                main_prompt = (
                    f"{shot.get('camera', '')};"
                    f"{shot.get('plot', '')};"
                    f"{shot.get('script', '')};"
                    f"{shot.get('scene', '')};"
                    f"{';'.join(char_prompts)}"
                )

                # 构建 log_dir (指向未来生成图像的目录)
                # 使用占位符 {time_stamp}
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
                # print(f"  处理 Shot {shot_id}: prompt={main_prompt[:50]}..., prev_p={len(char_prompts)}, ref_images={len(ref_images)}")


            # 为当前故事创建输出目录
            output_story_dir = os.path.join(self.output_root, story_name)
            os.makedirs(output_story_dir, exist_ok=True)

            # 保存 shots.json
            output_json_path = os.path.join(output_story_dir, "shots.json")
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(output_shots, f, indent=4, ensure_ascii=False)

            print(f"故事 {story_name} 的 shots.json 已保存至: {output_json_path}")
            all_story_outputs[story_name] = output_json_path # Store path for reference

        print("所有故事处理完毕。")
        return all_story_outputs


if __name__ == "__main__":

    data_path = "/data/AIGC_Research/Story_Telling/StoryVisBMK/data"
    dataset_name = 'WildStory'
    import argparse
    parser = argparse.ArgumentParser(description='Story Dataset Processing Tool')
    parser.add_argument('--language', type=str, choices=['en', 'ch'],
                        default='en', help='Language option: en (English) or ch (Chinese)')
    args = parser.parse_args()
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
        print(f'故事名列表：{story_name_list}')
        stories_data = dataset.load_stories(story_name_list, language)
        # print(f'加载的故事数据 (部分): {json.dumps(dict(list(stories_data.items())[0:1]), indent=2, ensure_ascii=False)}') # Debug: print first story data

        output_paths = converter.convert(story_name_list, stories_data)
        print("转换完成，输出文件路径:")
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
#                     '/data/AIGC_Research/Story_Telling/StoryVisBMK/data/dataset/WildStory/01/image/Little Brown Rabbit/00.jpg',
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
#         data_set(WildStory_en,WildStory_ch):
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
#             data_set(WildStory_en,WildStory_ch):
#                 story_id(01):
#                     time_stamp(20250430-022517):
#                         shot_id.png(shot_01.png)