import json
import math

def get_single_character_action_avg(data):
    """
    Read JSON file and calculate the average value of single_character_action field
    :param json_path: JSON file path
    :return: Average result
    """

    # Assume data structure is a list, each element is a dictionary containing "single_character_action" key
    # Example: data = [{"single_character_action": 5}, {"single_character_action": 3}]
    # actions = [
    #     item.get("single_character_action") 
    #     for item in data 
    #     if isinstance(item.get("single_character_action"), (int, float))
    # ]
    
    _single_character_action_score_avg_list = []

    for shot, score_dict in data.items():

        if shot not in ['avg_score_dict','story_name','elapsed_time(seconds)']:
            print(f'shot:{shot}')
            _single_character_action_score_list = []
            for item in score_dict["single_character_action"]:
                print(f'item:{item}')
                character_key, character_score = next(iter(item.items()))
                print(f'character_score:{character_score}')
                _single_character_action_score_list.append(character_score)

            # avg_score = (sum(_single_character_action_score_list)/ len(_single_character_action_score_list)) * ((len(character_image_new)) / (character_number))
            
            # Process first part: average of global and local scores
            print(f'_single_character_action_score_list:{_single_character_action_score_list}')
            epsilon = math.exp(-6)
            avg_score = (sum(_single_character_action_score_list)/ (len(_single_character_action_score_list) + epsilon)) 
            # * ((len(character_image_new)) / (character_number + epsilon))
            
            _single_character_action_score_avg_list.append(avg_score)
        elif shot in ['avg_score_dict','story_name','elapsed_time(seconds)']:
            continue

    print(f'_single_character_action_score_avg_list:{_single_character_action_score_avg_list}')
    single_character_action_avg = round(sum(_single_character_action_score_avg_list) / len(_single_character_action_score_avg_list), 4)
    
    return single_character_action_avg

if __name__ == "__main__":

    json_path = 'data/outputs/naive_baseline/bench_results/prompt_align/20250515_181733/06/06_gpt_score_prompt_align-WildStory_en_80.json'
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)  # Read the entire JSON file
        
        average = get_single_character_action_avg(data)
        print(f"single_character_action_avg = {average:.2f}")
