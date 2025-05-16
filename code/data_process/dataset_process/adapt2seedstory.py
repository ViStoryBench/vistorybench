from dataset_load import StoryDataset
import os
import json
import hashlib

class StoryConverter:
    def __init__(self, input_root, output_root):
        self.input_root = input_root
        self.output_root = output_root
        self.image_placeholder = "image/start_image.png"

    def _generate_story_id(self, story_name):
        """生成唯一的数字故事ID"""
        return int(hashlib.md5(story_name.encode()).hexdigest()[:8], 16) % 1000000

    def story_format_adapt(self, story_name, story_data):
        """处理单个故事"""

        shots_image = dataset.story_prompt_merge(story_data, mode='image')
        chars_prompt = dataset.story_prompt_merge(story_data, mode='char_prompt')
        for shot_char in range(len(shots_image)):
            current_image = shots_image[shot_char]
            current_char_prompt = chars_prompt[shot_char]
            # 修正判断逻辑：检查 current_image 是否有效
            if current_image and current_char_prompt:  # 等效于 current_image is not None and current_image != ""
                # print(f"找到有效图片: {current_image}")
                break  # 处理第一个有效图片后停止
        else:
            print("遍历结束，未找到图片")
    
        return {
            "id": self._generate_story_id(story_name),
            # "images": [self.image_placeholder],
            "images": [current_image[0]], # 取第一个角色的第一张图
            "captions": [current_char_prompt[0]] + dataset.story_prompt_merge(story_data,mode='prompt'),
            "orders": list(range(len(story_data["shots"]))),
            "story_name": story_name
        }


    def convert(self, story_name_list, stories_data):
        """执行转换"""

        # 创建输出根目录
        os.makedirs(self.output_root, exist_ok=True)

        # 处理每个故事
        success_count = 0
        for story_name in story_name_list:
            story_data = stories_data[story_name]
            story_data = self.story_format_adapt(story_name, story_data)
            if not story_data:
                print(f"跳过空数据故事: {story_name}")
                continue

            # 构建输出路径
            output_dir = os.path.join(
                self.output_root,
                story_name,
                "json"
            )
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存JSON文件
            output_path = os.path.join(output_dir, f"{story_name}.json")
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(story_data, f, ensure_ascii=False, separators=(',', ':'))
                success_count += 1
            except Exception as e:
                print(f"保存失败 [{story_name}]: {str(e)}")

        print(f"转换完成: 成功{success_count}个，失败{len(story_name_list)-success_count}个")



    def merge_json(self, story_name_list):
        """合并所有故事的JSON文件到单个JSONL文件"""
        merge_path = os.path.join(
            self.output_root,
            "merge.jsonl"
        )
        
        # 确保目录存在
        os.makedirs(os.path.dirname(merge_path), exist_ok=True)
        
        success_count = 0
        error_count = 0
        
        with open(merge_path, "w", encoding="utf-8") as merge_file:
            # 遍历所有故事目录
            for story_name in os.listdir(self.output_root): # 这个写法可以确保排除掉转换失败的故事
            # for story_name in story_name_list:
                
                # 跳过非目录文件
                if not os.path.isdir(os.path.join(self.output_root, story_name)):
                    continue
                    
                json_path = os.path.join(
                    self.output_root, 
                    story_name,
                    "json", 
                    f"{story_name}.json"
                )

                start_image_path = os.path.join(
                    story_name,
                    self.image_placeholder
                )

                # 读取并写入数据
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    data['story_name'] = story_name
                        
                    # 写入JSONL（每行一个JSON对象）
                    merge_file.write(json.dumps(data, ensure_ascii=False) + "\n")
                    success_count += 1
                except Exception as e:
                    print(f"合并失败 [{story_name}]: {str(e)}")
                    error_count += 1
        
        print(f"合并完成: 成功{success_count}条，失败{error_count}条")



if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Story Dataset Processing Tool')
    parser.add_argument('--language', type=str, choices=['en', 'ch'], 
                        default='en', help='Language option: en (English) or ch (Chinese)')
    args = parser.parse_args()
    language=args.language

    data_path = "/data/AIGC_Research/Story_Telling/StoryVisBMK/data"
    dataset_name = 'WildStory'

    method = 'seedstory'

    dataset_path = f"{data_path}/dataset/{dataset_name}"
    processed_dataset_path = f"{data_path}/dataset_processed/{method}/{dataset_name}_{language}"

    dataset = StoryDataset(dataset_path)

    input_dir = dataset_path
    output_dir = processed_dataset_path
    converter = StoryConverter(input_dir, output_dir)


    story_name_list = dataset.get_story_name_list()
    print(f'\n故事名列表：{story_name_list}')  # 获取所有故事列表
    stories_data = dataset.load_stories(story_name_list, language)  # 加载指定故事数据
    converter.convert(story_name_list, stories_data) # 执行转换
    converter.merge_json(story_name_list) # 执行合并