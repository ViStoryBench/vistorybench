import os
from dataset_load import StoryDataset  # 假设原始代码文件名为 story_dataset.py

class StoryConverter:
    def __init__(self, input_root, output_root):
        self.input_root = input_root
        self.output_root = output_root

    def convert(self, story_name_list, stories_data):
        
        # 创建结果字典和最终列表
        shots_prompt_dict = {}
        story_image_dict = {}
        StorySet = []
        StorySet_image = []
        story_name_variate_list = []
        story_name_variate_image_list = []
        
        # 遍历每个故事生成对应格式
        for story_name in story_name_list:

            story_data = stories_data[story_name]

            shots_prompt = dataset.story_prompt_merge(story_data,mode='prompt')
            shots_image = dataset.story_prompt_merge(story_data,mode='image')

            # 动态创建变量并添加到列表
            story_name_variate = f'story_{story_name}'
            story_name_variate_image = f'story_{story_name}_image'
            exec(f"{story_name_variate} = {shots_prompt}")  # 创建以故事名命名的变量
            exec(f"{story_name_variate_image} = {shots_prompt}")  # 创建以故事名命名的变量
            StorySet.append(eval(story_name_variate))  # 添加到总列表
            StorySet_image.append(eval(story_name_variate_image))  # 添加到总列表
            story_name_variate_list.append(story_name_variate)
            story_name_variate_image_list.append(story_name_variate_image)
            
            # 同时保存到字典便于调试
            shots_prompt_dict[story_name_variate] = shots_prompt
            story_image_dict[story_name_variate_image] = shots_image
        
        # 保存生成结果到文件
        json_path = os.path.join(
            self.output_root, 
            "story_list.py"
        )

        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        
        with open(json_path, "w", encoding="utf-8") as f:
            # 写入每个故事的变量
            for name_prompt,name_image in zip(story_name_variate_list,story_name_variate_image_list):
                f.write(f"{name_prompt} = [\n")
                for desc in shots_prompt_dict[name_prompt]:
                    escaped_desc = desc.replace('"', r'\"').replace("'", r"\'")  # 转义双引号和单引号
                    f.write(f'    "{escaped_desc}",\n')
                f.write("]\n\n")
            
                f.write(f"{name_image} = [\n")
                for desc in story_image_dict[name_image]:
                    f.write(f'    {desc},\n')
                f.write("]\n\n")

            # 写入总集合
            # f.write("StorySet = [\n")
            # for name in story_name_list:
            #     f.write(f"    {name},\n")
            # f.write("]\n")
                
            f.write("StorySet = {\n")
            for name, name_variate in zip(story_name_list, story_name_variate_list):
                f.write(f'    "{name}": {name_variate},\n')  # 注意使用英文双引号
            f.write("}\n")

            f.write("StorySet_image = {\n")
            for name, name_variate in zip(story_name_list, story_name_variate_image_list):
                f.write(f'    "{name}": {name_variate},\n')  # 注意使用英文双引号
            f.write("}\n")

        return StorySet, StorySet_image

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Story Dataset Processing Tool')
    parser.add_argument('--language', type=str, choices=['en', 'ch'], 
                        default='en', help='Language option: en (English) or ch (Chinese)')
    args = parser.parse_args()
    language=args.language

    data_path = "/data/AIGC_Research/Story_Telling/StoryVisBMK/data"
    dataset_name = 'WildStory'

    method = 'storyadapter'

    dataset_path = f"{data_path}/dataset/{dataset_name}"
    processed_dataset_path = f"{data_path}/dataset_processed/{method}/{dataset_name}_{language}"


    dataset = StoryDataset(dataset_path)

    input_dir = dataset_path
    output_dir = processed_dataset_path
    converter = StoryConverter(input_dir, output_dir)


    story_name_list = dataset.get_story_name_list()
    print(f'\n故事名列表：{story_name_list}')  # 获取所有故事列表
    stories_data = dataset.load_stories(story_name_list, language)  # 加载指定故事数据
    StorySet, StorySet_image = converter.convert(story_name_list, stories_data)
    print(f"成功生成 {len(StorySet)} 个故事数据集")

