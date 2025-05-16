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
                print(f"警告：在 stories_data 中找不到故事 {story_id}，跳过。")
                continue

            story_data = stories_data[story_id]
            input_shots = story_data.get("shots", [])
            characters_info = story_data.get("characters", {}) # 获取角色信息

            if not input_shots:
                print(f"警告：故事 {story_id} 没有找到 shots 数据，跳过。")
                continue

            print(f"正在处理故事 (聚合到 stories.json): {story_id}")

            # === 初始化用于 segment2prompt 的数据 ===
            segment2prompt_data = {
                "segment": "",
                "answer": "",
                "final_answer": ""
            }
            char_details = []
            setting_details = set() # 使用集合来存储唯一的场景作为设置
            scene_segments_for_segment_field = []
            scene_segments_for_answer_field = []

            # --- 提取角色信息 ---
            for key, char_info in characters_info.items():
                name = char_info.get('name', key)
                prompt = char_info.get('prompt', 'No description available.')
                char_details.append(f"{name}: {prompt}")

            # === 处理每个 shot 来构建 scene segments ===
            scene2image_output = {}
            segment_num = len(input_shots)
            scene2image_output["segment_num"] = segment_num

            for i, shot in enumerate(input_shots):
                segment_key = f"Scene 1 Segment {i + 1}" # 使用 1-based index

                # --- 构建 scene2image 部分 ---
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

                # --- 构建 segment2prompt 的 scene segments ---
                shot_scene = shot.get('scene', 'Unknown Location')
                setting_details.add(shot_scene) # 添加到唯一设置列表
                char_names_in_shot = [characters_info.get(key, {}).get('name', key)
                                      for key in shot.get('character_key', [])]
                char_list_str = f"[{', '.join(char_names_in_shot)}]" if char_names_in_shot else []
                # 结合 plot 和 script 作为描述
                shot_description = f"{shot.get('plot', '')} {shot.get('script', '')}".strip()
                camera_info = shot.get('camera', '')
                camera_str = f"({camera_info}.)" if camera_info else ""

                # 格式化 segment 字段的场景行
                scene_line_segment = f"{segment_key}: {char_list_str}[{shot_scene}] {shot_description} {camera_str}"
                scene_segments_for_segment_field.append(scene_line_segment.strip())

                # 格式化 answer 字段的场景行 (可能需要根据实际情况微调角色/服装等细节)
                # 这里我们暂时使用与 segment 相似的格式，您可以后续根据需要调整此逻辑
                # 例如，添加 "(in pink)" 或 "(in white)" 等细节需要更复杂的逻辑或元数据
                scene_line_answer = f"{segment_key}: {char_list_str}[{shot_scene}] {shot_description} {camera_str}"
                scene_segments_for_answer_field.append(scene_line_answer.strip())

            # --- 组装 segment2prompt ---
            characters_section = "\n".join(char_details)
            # 目前仅列出场景名称作为设置
            settings_section = "\n".join(list(setting_details))
            scenes_section_segment = "\n".join(scene_segments_for_segment_field)
            scenes_section_answer = "\n".join(scene_segments_for_answer_field)

            segment2prompt_data["segment"] = (
                f"Characters:\n{characters_section}\n"
                f"Settings:\n{settings_section}\n"
                f"Scenes:\n{scenes_section_segment}"
            )

            # 注意：answer 字段通常是期望的生成结果，这里暂时用提取的信息构建
            segment2prompt_data["answer"] = f"'''{scenes_section_answer}'''"

            segment2prompt_data["final_answer"] = (
                 f"Characters:\n{characters_section}\n"
                 f"Settings:\n{settings_section}\n"
                 f"Scenes:\n'''{scenes_section_answer}'''"
            )

            # 构建当前故事的输出结构,包含 scene2image 和 segment2prompt
            story_output_data = {
                "segment2prompt": segment2prompt_data, # 添加 segment2prompt
                "scene2image": scene2image_output
            }
            # 将当前故事的数据添加到总的字典中，以 story_id 为键
            all_stories_output_data[story_id] = story_output_data

        # 所有故事处理完毕后，保存聚合后的数据到 stories.json
        output_json_path = os.path.join(output_dir, "stories.json")
        try:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                # 直接写入聚合后的 all_stories_output_data
                json.dump(all_stories_output_data, f, indent=4, ensure_ascii=False)
            print(f"所有故事已聚合保存至: {output_json_path}")
            # 返回包含单个文件路径的字典
            return {self.data_set_name: output_json_path}
        except Exception as e:
            print(f"错误：无法保存聚合后的 stories.json 到 {output_json_path}: {e}")
            return {} # 返回空字典表示失败

        print("所有故事 (animdirector format) 处理完毕。")
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
        print(f'故事名列表：{story_name_list}')
        stories_data = dataset.load_stories(story_name_list, language)
        # print(f'加载的故事数据 (部分): {json.dumps(dict(list(stories_data.items())[0:1]), indent=2, ensure_ascii=False)}') # Debug: print first story data

        output_paths = converter.convert(story_name_list, stories_data)
        # 现在 output_paths 是一个字典，例如 {'WildStory_en': '.../stories.json'}
        print(f"转换完成 ({language}, method={method})，输出文件路径:")
        if output_paths:
            # output_paths 字典只有一个键值对
            for dataset_lang, path in output_paths.items():
                print(f"  {dataset_lang}: {path}")
        else:
            print("  没有生成任何输出文件。")
