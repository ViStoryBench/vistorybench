
import os
import json

class StoryDataset:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def get_story_name_list(self):
        """获取所有故事名称列表（按数字排序）"""
        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f"根目录 {self.root_dir} 不存在")
        
        # 按目录名数字排序 (e.g. 01, 02,...)
        entries = os.listdir(self.root_dir)
        return sorted(
            [entry for entry in entries if os.path.isdir(os.path.join(self.root_dir, entry))],
            key=lambda x: int(x) if x.isdigit() else 0
        )

    def _load_story_json(self, story_name):
        story_path = os.path.join(self.root_dir, story_name)
        if not os.path.exists(story_path):
            raise FileNotFoundError(f"故事目录 {story_path} 不存在")
        
        story_json = os.path.join(story_path, "story.json")
        if not os.path.isfile(story_json):
            raise FileNotFoundError(f"story.json 不存在于 {story_path}")
        
        try:
            with open(story_json, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"错误: {story_json} 不是有效的JSON文件")
            return {}
        except Exception as e:
            print(f"读取文件失败: {str(e)}")
            return {}

    def load_shots(self, story_name, language="en"):
        """读取分镜数据（支持多语言）"""
        story_data = self._load_story_json(story_name)
        shots = []
        
        # 字段映射关系：原始字段 -> 新字段名
        field_mapping = {
            "Setting Description": "scene",
            "Plot Correspondence": "plot",
            "Characters Appearing": "character_name",
            "Static Shot Description": "script",
            "Shot Perspective Design": "camera"
        }
        
        for shot in story_data.get("Shots", []):
            # 校验原始字段是否存在
            original_fields = field_mapping.keys()
            if not all(field in shot for field in original_fields):
                missing = [f for f in original_fields if f not in shot]
                print(f"警告: 分镜 {shot.get('index', '未知')} 缺失字段 {missing}")
                continue
            
            # 构建新字段结构
            processed_shot = {"index": shot["index"]}
            for orig_field, new_field in field_mapping.items():
                if language in shot[orig_field]:
                    processed_shot[new_field] = shot[orig_field][language]
                    processed_shot["character_key"] = shot["Characters Appearing"]['en']
                else:
                    processed_shot[new_field] = f"({language} 数据缺失) " + str(shot[orig_field])
            
            shots.append(processed_shot)
        
        return shots

    def load_characters(self, story_name, language="en"):
        """读取角色数据（支持多语言和图片）"""
        story_data = self._load_story_json(story_name)
        characters = {}
        story_path = os.path.join(self.root_dir, story_name)
        
        image_ref_dir = os.path.join(story_path, "image")
        
        for char_key, char_data in story_data.get("Characters", {}).items():
            # 提取多语言描述（兼容 prompt_zh/prompt_cn 等变体）
            name_key = f"name_{language}"
            prompt_key = f"prompt_{language}"
            
            # 提取所有元数据
            char_info = {
                "name": char_data.get(name_key, ""),  # 优先显示对应语言的名字
                "key": char_data.get("name_en", ""),  # 用于中文故事的角色索引
                "prompt": char_data.get(prompt_key, ""),
                "tag": char_data.get("tag", ""),
                "num_of_appearances": char_data.get("num_of_appearances", -1),
                "images": []
            }
            
            # 收集角色图片
            char_dir = os.path.join(image_ref_dir, char_key)
            if os.path.isdir(char_dir):
                for root, _, files in os.walk(char_dir):
                    for img_file in sorted(files):
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(root, img_file)
                            if os.path.isfile(img_path):  # 增加文件存在性校验
                                char_info["images"].append(img_path)
            
            characters[char_key] = char_info
        
        return characters

    def load_type(self, story_name, language="en"):
        """读取故事类型（支持多语言）"""
        story_data = self._load_story_json(story_name)
        story_type = story_data.get("Story_type", {})
        
        # 获取对应语言的类型描述
        return story_type.get(language, f"({language} 类型未定义)")
    
    def load_story(self, story_name, language="en"):
        return {
            "type": self.load_type(story_name, language),
            "shots": self.load_shots(story_name, language),
            "characters": self.load_characters(story_name, language)
        }

    def load_stories(self, story_name_list, language="en"):
        return {
            story_name: self.load_story(story_name, language)
            for story_name in story_name_list
        }


    def story_prompt_merge(self, story_data, mode = ''):

        characters = story_data["characters"]

        shots_all = []
        shots_prompt = []
        shots_image = []
        chars_prompt = []

        for shot in story_data["shots"]:
            
            char_keys = []
            char_names = []
            char_prompts = []
            char_images = []
            # 1. 获取当前分镜的角色列表
            for char_key,char_name in zip(shot['character_key'],shot['character_name']):
                # 验证角色是否存在
                if char_key not in characters:
                    print(f"警告: 角色 {char_key} 未在角色库中定义")
                    continue
                char_key = char_key.strip()
                char_keys.append(char_key)
                char_names.append(char_name)

            # 2. 提取这些角色的描述词及图像路径
                char_prompt = characters[char_key]['prompt']
                # char_image = characters[char_key]['images']  # 取所有角色图
                char_image = characters[char_key]['images'][0]  # 取第一张角色图
                char_prompts.append(f'{char_name} is {char_prompt}')
                char_images.append(char_image)
            
            # 3. 将角色描述词拼接到分镜信息中
            # shot_prompt = (
            #     f"{';'.join(char_prompts)}"  # 用;分隔多个角色描述
            #     f"{shot['scene']};"
            #     f"{shot['camera']};"
            #     f"{shot['script']};"
            #     f"{shot['plot']};"
            # )

            shot_prompt = (
                f"{shot['camera']};"
                f"{shot['plot']};"
                f"{shot['script']};"
                f"{';'.join(char_prompts)};"  # 用;分隔多个角色描述
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
            raise

def main():
    data_path = "/data/AIGC_Research/Story_Telling/StoryVisBMK/data"
    dataset_name = 'WildStory'
    dataset_path = f"{data_path}/dataset/{dataset_name}"

    dataset = StoryDataset(dataset_path)
    story_name_list = dataset.get_story_name_list()
    print(f'\n故事名列表：{story_name_list}')

    language="en"
    # language="ch"

    # 加载所有故事（英文）
    stories_data = dataset.load_stories(story_name_list, language)
    # print(f'\n读取所有故事信息{stories_data}')
    
    print('\n' \
    '\\\\\\\\\\\\\\\\\\\\\ 细粒度信息提取示例 \\\\\\\\\\\\\\\\\\\\\ ')
    # 示例：提取第一个故事的第一个分镜
    story_name = '01'  # 假设存在编号为01的故事
    story_data = stories_data[story_name]

    type = story_data["type"]
    print(f"""
    故事 {story_name} 的故事类型: {type}
    """)  # 输出: "Children's Picture Books"

    shots = story_data["shots"]
    first_shot = shots[0]
    print(f"""
    第一个分镜:
    - 索引: {first_shot['index']}
    - 布景: {first_shot['scene']}
    - 主观剧情: {first_shot['plot']}
    - 登场角色: {first_shot['character_name']}
    - 登场角色关键词: {first_shot['character_key']}
    - 客观描述: {first_shot['script']}
    - 镜头设计: {first_shot['camera']}
    """)
    
    # 示例：提取第一个角色信息
    characters = story_data["characters"] # 角色库
    keys_list = list(characters.keys())
    values_list = list(characters.values())
    characters_1_key = keys_list[0]
    characters_1_value = values_list[0]
    print(f"""
    角色 {characters_1_key}:
    - 名字: {characters_1_value['name']}
    - 名字关键词: {characters_1_value['key']}
    - 描述: {characters_1_value['prompt']}
    - 人/非人: {characters_1_value['tag']}
    - 参考图: {characters_1_value['images']}
    """)


    # 提示词拆解与合并
    shots_all = dataset.story_prompt_merge(story_data,mode='all')
    shots_prompt = dataset.story_prompt_merge(story_data,mode='prompt')
    shots_image = dataset.story_prompt_merge(story_data,mode='image')
    print(f"""
    提示词拆解与合并:
    - shots_all取第一个镜头: 
        {shots_all[0].keys()}
    - shots_all取第一个镜头的prompt: 
        {shots_all[0]['prompt']}
    - shots_all取第一个镜头的所有角色的image: 
        {shots_all[0]['image_paths']}
    - shots_all取第一个镜头的第一个角色的image: 
        {shots_all[0]['image_paths'][0]}
    - shots_all取第一个镜头的第一个角色的第一张image: 
        {shots_all[0]['image_paths'][0][0]}
    \n
    - shots_prompt: {shots_prompt[0]}
    - shots_image: {shots_image[0]}
    """)



    # 定义颜色代码
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
    {COLOR['HEADER']}提示词拆解与合并:{COLOR['END']}
    {COLOR['BLUE']}■ shots_all取第一个镜头:{COLOR['END']} 
        {COLOR['YELLOW']}Keys: {COLOR['GREEN']}{list(shots_all[0].keys())}{COLOR['END']}

    {COLOR['BLUE']}■ 镜头描述:{COLOR['END']}
        {COLOR['CYAN']}{shots_all[0]['prompt'][:50]}...{COLOR['END']}

    {COLOR['BLUE']}■ 角色图像路径:{COLOR['END']}
        {COLOR['YELLOW']}所有角色: {COLOR['END']}
        {COLOR['UNDERLINE']}{shots_all[0]['image_paths']}{COLOR['END']}

        {COLOR['YELLOW']}首个角色: {COLOR['END']}
        {COLOR['RED']}{shots_all[0]['image_paths'][0]}{COLOR['END']}

        {COLOR['YELLOW']}具体文件: {COLOR['END']}
        {COLOR['BOLD']}{shots_all[0]['image_paths'][0][0]}{COLOR['END']}

    {COLOR['GREEN']}◆ 合并结果预览:{COLOR['END']}
        {COLOR['CYAN']}Prompt: {shots_prompt[0][:30]}...{COLOR['END']}
        {COLOR['CYAN']}Images: {len(shots_image[0])}张{COLOR['END']}
    """)




if __name__ == "__main__":
    main()
