
import os
import re
from dataset_load import StoryDataset

class StoryConverter:
    def __init__(self, input_root, output_root):
        self.input_root = input_root
        self.output_root = output_root

    def replace_character_names(self, text, characters):
        """将文本中的角色名称替换为[角色名]格式"""
        # 按名称长度降序排序，避免短名称误替换
        sorted_chars = sorted(characters, key=lambda x: -len(x))
        for char in sorted_chars:
            pattern = re.compile(r'\b{}\b'.format(re.escape(char)))
            text = pattern.sub(f'[{char}]', text)
        return text

    def array2string(self, arr):
        stringtmp = ""
        for i, part in enumerate(arr):
            if i != len(arr) - 1:
                stringtmp += part + "\n"
            else:
                stringtmp += part
        return stringtmp


    def story_prompt_merge(self, shots, trigger):
        # 构建prompt_array
        prompt_array = []
        for shot in shots:
            chars_in_shot = shot['character_name']
            # 处理角色部分
            if not chars_in_shot:
                roles = "[NC]"
            else:
                name_list = [f"{name}" for name in chars_in_shot]
                name_and_name = " and ".join(name_list) if len(name_list) > 1 else name_list[0]
                name_and_name_img = name_and_name + trigger

                name_is_name_list = [f"{name} is [{name}]" for name in chars_in_shot]
                name_is_name_and_name = " and ".join(name_is_name_list) if len(name_is_name_list) > 1 else name_is_name_list[0]
                roles = name_and_name_img + ', ' + name_is_name_and_name
                # role_part = roles[0] # 记得改

            # 组合场景、剧本和故事描述
            camera = shot.get('camera', '')
            scene = shot.get('scene', '')
            script = shot.get('script', '')
            plot = shot.get('plot', '')
            # scene = self.replace_character_names(scene, chars_in_shot)
            # script = self.replace_character_names(script, chars_in_shot)
            # plot = self.replace_character_names(plot, chars_in_shot)

            # 输入文本（prompt）不能超出模型的最大序列长度限制（77个token）
            prompt_merge_mode = 'plot_first'
            if prompt_merge_mode == 'full':
                front_part = f"{roles}; {scene}; {camera}; {script}; {plot}.".strip(', ').strip()
            elif prompt_merge_mode == 'plot+script':
                front_part = f"{roles}; {camera}; {script}; {plot}.".strip(', ').strip()
            elif prompt_merge_mode == 'plot':
                front_part = f"{roles}; {camera}; {plot}.".strip(', ').strip()
            elif prompt_merge_mode == 'script':
                front_part = f"{roles}; {camera}; {script}.".strip(', ').strip()
            elif prompt_merge_mode == 'plot_first':
                front_part = f"{camera}; {plot}; {script}; {roles}; {scene}.".strip(', ').strip()

            # prompt_element = f"{front_part} # {plot}" if plot else front_part
            prompt_element = front_part
            prompt_array.append(prompt_element)
        return prompt_array


    def ref_img_extract(self, shots, characters):
        image_paths = []
        general_prompt_lines = []

        for char_key, char_info in characters.items(): # 角色库
            char_path = characters[f'{char_key}']['images'][0] #取第一张角色图
            if os.path.exists(char_path):
                image_paths.append(char_path)
            else:
                print(f"警告: {char_key} 参考图缺失: {char_path}")

            line = f"[{char_key}] {characters[char_key]['prompt']}"
            general_prompt_lines.append(line)

        # for shot in shots:
        #     for char_name in shot['character_key']:
        #         # 验证角色是否存在
        #         if char_name not in characters:
        #             print(f"警告: 角色 {char_name} 未在角色库中定义")
        #             continue
                
        #         # 生成角色图片路径
        #         # sanitized_name = char_name.replace(" ", "_")
        #         char_path = characters[f'{char_name}']['images'][0] #取第一张角色图
        #         if os.path.exists(char_path):
        #             image_paths.append(char_path)
        #         else:
        #             print(f"警告: {char_name} 参考图缺失: {char_path}")
            
        #     # 改进后的general_prompt生成（添加img触发词）
        #         line = f"[{char_name}] {characters[char_name]['prompt']}"
        #         general_prompt_lines.append(line)
        
        return image_paths, general_prompt_lines


    def convert(self, story_name_list, stories_data):
        examples = []
        for story_name in story_name_list:
            story_data = stories_data[story_name]
            shots = story_data["shots"]
            characters = story_data["characters"] # 总的角色表

            # 判断模型类型 + 触发词判断逻辑
            has_ref_images = any(len(char['images']) > 0 for char in characters.values())
            # has_ref_images = False # 记得改，has_ref_images为True时，使用参考图，为False时，只使用文本
            model_type = "Using Ref Images" if has_ref_images else "Only Using Textual Description"

            # 根据参考图存在性添加img标记
            if model_type == "Using Ref Images":
                trigger = " img"
            elif model_type == "Only Using Textual Description":
                trigger = ""

            # 收集所有图片路径
            image_paths, general_prompt_lines = self.ref_img_extract(shots, characters)

            general_prompt = "\n".join(general_prompt_lines)

            # 固定negative_prompt
            negative_prompt = "bad anatomy, bad hands, missing fingers, extra fingers, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, three crus, fused feet, fused thigh, extra crus, ugly fingers, horn, cartoon, cg, 3d, unreal, animate, amputation, disconnected limbs"

            prompt_array = self.story_prompt_merge(shots, trigger)
            
            prompt_array_str = self.array2string(prompt_array)

            # 其他参数
            seed_ = 42
            sa32_ = 0.5
            sa64_ = 0.5
            id_length_ = 1 # 记得改
            style = "(No style)"
            G_height = 768
            G_width = 1344

            # 构建example条目
            example = [
                seed_,
                sa32_,
                sa64_,
                id_length_,
                dataset_name, 
                story_name,
                general_prompt,
                negative_prompt,
                prompt_array_str,
                style,
                model_type,
                image_paths,
                G_height,
                G_width
            ]
            examples.append(example)
        
        # 保存生成的examples
        self.save_examples(examples)



    def save_examples(self, examples):
        os.makedirs(self.output_root, exist_ok=True)
        output_path = os.path.join(self.output_root, 'examples.py')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("examples = [\n")
            for ex in examples:
                f.write("    [\n")
                # 数值型参数
                f.write(f"        {ex[0]},  # seed_\n")
                f.write(f"        {ex[1]},  # sa32_\n")
                f.write(f"        {ex[2]},  # sa64_\n")
                f.write(f"        {ex[3]},  # id_length_ 登场角色id在分镜中至少出现的次数（不要为0，至少为1）\n")
                
                # 文本型参数
                f.write(f"        {repr(ex[4])},  # dataset_name\n")
                f.write(f"        {repr(ex[5])},  # story_name\n")

                f.write(f"        {repr(ex[6])},  # general_prompt\n")
                f.write(f"        {repr(ex[7])},  # negative_prompt\n")
                
                # 处理多行prompt（直接写入处理后的字符串）
                f.write(f"        {repr(ex[8])},  # prompt_array\n")
                
                # 其他参数
                f.write(f"        {repr(ex[9])},  # style\n")
                f.write(f"        {repr(ex[10])},  # model_type\n")
                f.write(f"        {repr(ex[11])},  # files\n")  # 直接写入路径列表
                f.write(f"        {ex[12]},  # G_height\n")
                f.write(f"        {ex[13]},  # G_width\n")
                f.write("    ],\n")
            f.write("]\n")


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Story Dataset Processing Tool')
    parser.add_argument('--language', type=str, choices=['en', 'ch'], 
                        default='en', help='Language option: en (English) or ch (Chinese)')
    args = parser.parse_args()
    language=args.language

    data_path = "/data/AIGC_Research/Story_Telling/StoryVisBMK/data"
    dataset_name = 'WildStory'

    method = 'storydiffusion'

    dataset_path = f"{data_path}/dataset/{dataset_name}"
    processed_dataset_path = f"{data_path}/dataset_processed/{method}/{dataset_name}_{language}"

    dataset = StoryDataset(dataset_path)

    input_dir = dataset_path
    output_dir = processed_dataset_path
    converter = StoryConverter(input_dir, output_dir)

    story_name_list = dataset.get_story_name_list()
    stories_data = dataset.load_stories(story_name_list, language)
    converter.convert(story_name_list, stories_data)