import json
import math

def get_single_character_action_avg(data):
    """
    读取JSON文件并计算single_character_action字段的平均值
    :param json_path: JSON文件路径
    :return: 平均值结果
    """

    
    
    # 假设数据结构为列表，每个元素是包含"single_character_action"键的字典
    # 例如：data = [{"single_character_action": 5}, {"single_character_action": 3}]
    # actions = [
    #     item.get("single_character_action") 
    #     for item in data 
    #     if isinstance(item.get("single_character_action"), (int, float))
    # ]
    
    _single_character_action_score_avg_list = []

    for shot, score_dict in data.items():

        
        
        if shot not in ['avg_score_dict','story_name']:
            print(f'shot:{shot}')
            _single_character_action_score_list = []
            for item in score_dict["single_character_action"]:
                print(f'item:{item}')
                character_key, character_score = next(iter(item.items()))
                print(f'character_score:{character_score}')
                _single_character_action_score_list.append(character_score)

            # avg_score = (sum(_single_character_action_score_list)/ len(_single_character_action_score_list)) * ((len(character_image_new)) / (character_number))
            
            # 处理第一部分：全局和局部评分的平均值
            print(f'_single_character_action_score_list:{_single_character_action_score_list}')
            epsilon = math.exp(-6)
            avg_score = (sum(_single_character_action_score_list)/ (len(_single_character_action_score_list) + epsilon)) 
            # * ((len(character_image_new)) / (character_number + epsilon))
            
            _single_character_action_score_avg_list.append(avg_score)
        elif shot in ['avg_score_dict','story_name']:
            continue
            
    print(f'_single_character_action_score_avg_list:{_single_character_action_score_avg_list}')
    single_character_action_avg = round(sum(_single_character_action_score_avg_list) / len(_single_character_action_score_avg_list), 4)
    

    return single_character_action_avg

# 使用示例
if __name__ == "__main__":

    json_path = '/data/AIGC_Research/Story_Telling/StoryVisBMK/data/outputs/naive_baseline/bench_results/prompt_align/20250515_181733/06/06_gpt_score_prompt_align-WildStory_en_80.json'
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)  # 读取整个JSON文件
        
        average = get_single_character_action_avg(data)
        print(f"single_character_action_avg = {average:.2f}")
