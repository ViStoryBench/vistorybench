

import os
import json
from statistics import mean
import re

def aesthetic_total_avg(path):


    def extract_story_name(filename):
        """
        精准提取故事编号（如 "01"），忽略时间戳等其他数字。
        匹配以下模式：
        - WildStory_en_<编号> 或 WildStory_en_lite_<编号>
        - 编号后接 _、. 或时间戳
        """
        match = re.search(r'WildStory_en_(?:lite_)?(\d{2})(?=[._])', filename)
        return match.group(1) if match else None


    print(f'get aesthetic_total_avg ......')

    timestamp = get_timestamp_name(path)
    path = f'{path}/{timestamp}'

    if method in ['storyadapter_img_ref_results_xl', 'storyadapter_text_only_results_xl']:
        json_path = os.path.join(path, 'results_xl','json')
    elif method in ['storyadapter_img_ref_results_xl5', 'storyadapter_text_only_results_xl5']:
        json_path = os.path.join(path, 'results_xl5','json')
    else:
        json_path = os.path.join(path, 'json')
    
    save_path = os.path.join(path, 'total_avg')

    # 创建保存目录（如果不存在）
    os.makedirs(save_path, exist_ok=True)


    all_avg = {
        "score": [],
    }
    total_avg = {
        "total_avg": None,
    }


    # 遍历JSON文件并提取average_score
    for filename in os.listdir(json_path):
        if filename.endswith('.json'):
            story_name = extract_story_name(filename)
            if story_name in CHOICE_DATASET:

                print(f'-------------now filename:{filename}')
                filepath = os.path.join(json_path, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    score = data.get('average_score')

                    if score is not None:
                        all_avg['score'].append(float(score))


    # 计算总平均值
    total_avg['total_avg'] = mean(all_avg['score']) if all_avg else 0.0

    # 保存结果
    result = {
        "check_timestamp": timestamp,
        "all_average_scores": all_avg,
        "total_average_score": total_avg,
        "processed_files_count": len(all_avg['score'])
    }
    output_file = os.path.join(save_path, 'total_average.json')
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=4)

    print(f"Total average score calculated: {total_avg['total_avg']:.4f}")
    print(f"Result saved to: {output_file}")

    return result

def csd_total_avg(path, mode =''):


    def extract_story_name(filename):
        """
        精准提取故事编号（如 "01"），忽略时间戳等其他数字。
        匹配以下模式：
        - WildStory_en_<编号> 或 WildStory_en_lite_<编号>
        - 编号后接 _、. 或时间戳
        """
        match = re.search(r'WildStory_en_(?:lite_)?(\d{2})(?=[._])', filename)
        return match.group(1) if match else None


    print(f'get csd_total_avg ......')

    timestamp = get_timestamp_name(path)
    path = f'{path}/{timestamp}'

    if method in ['storyadapter_img_ref_results_xl', 'storyadapter_text_only_results_xl']:
        json_path = os.path.join(path, 'results_xl','json')
    elif method in ['storyadapter_img_ref_results_xl5', 'storyadapter_text_only_results_xl5']:
        json_path = os.path.join(path, 'results_xl5','json')
    else:
        json_path = os.path.join(path, 'json')

    save_path = os.path.join(path, 'total_avg')

    # 创建保存目录（如果不存在）
    os.makedirs(save_path, exist_ok=True)


    all_avg = {
        "score": [],
    }
    total_avg = {
        "total_avg": None,
    }



    # 遍历JSON文件并提取average_score
    for filename in os.listdir(json_path):
        if filename.endswith('.json'):
            story_name = extract_story_name(filename)
            if story_name in CHOICE_DATASET:

                print(f'-------------now filename:{filename}')
                filepath = os.path.join(json_path, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    score = data.get('average_score')

                    if score is not None:
                        all_avg['score'].append(float(score))



    # 计算总平均值
    total_avg['total_avg'] = mean(all_avg['score']) if all_avg else 0.0

    # 保存结果
    if mode == 'self':
        result = {
            "check_timestamp": timestamp,
            "all_average_self_sref_scores": all_avg,
            "total_average_self_sref_score": total_avg,
            "processed_files_count(self)": len(all_avg['score'])
        }
    elif mode == 'cross':
        result = {
            "check_timestamp": timestamp,
            "all_average_cross_sref_scores": all_avg,
            "total_average_cross_sref_score": total_avg,
            "processed_files_count(cross)": len(all_avg['score'])
        }

    output_file = os.path.join(save_path, 'total_average.json')
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=4)

    print(f"Total average score calculated: {total_avg['total_avg']:.4f}")
    print(f"Result saved to: {output_file}")

    return result


def cref_total_avg(path):

    print(f'get cref_total_avg ......')

    timestamp = get_timestamp_name(path)
    path = f'{path}/{timestamp}'

    save_path = os.path.join(path, 'total_avg')
    
    # 创建保存目录（如果不存在）
    os.makedirs(save_path, exist_ok=True)

    all_avg = {
        "cross": [],
        "self": [],
        "copy-paste-score":[]
    }
    total_avg = {
        "total_cross_avg": None,
        "total_self_avg": None,
        "total_avg_copy-paste-score": None
    }

    # 遍历01到80的子目录
    for story_name in CHOICE_DATASET:

        if method in ['storyadapter_img_ref_results_xl', 'storyadapter_text_only_results_xl']:
            json_dir = os.path.join(path, story_name, 'results_xl','json')
        elif method in ['storyadapter_img_ref_results_xl5', 'storyadapter_text_only_results_xl5']:
            json_dir = os.path.join(path, story_name, 'results_xl5','json')
        else:
            json_dir = os.path.join(path, story_name, 'json')
        
        # 检查JSON目录是否存在
        if not os.path.exists(json_dir):
            print(f"Directory {json_dir} does not exist, skipping.")
            continue
        
        # 遍历JSON目录下的所有JSON文件
        for filename in os.listdir(json_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(json_dir, filename)

                with open(filepath, 'r') as f:
                    data = json.load(f)
                    cref_data = data.get('cref', {})
                    copy_paste_score = data.get('copy-paste-score')

                    if isinstance(copy_paste_score, (int, float)) and copy_paste_score != "null":
                        all_avg['copy-paste-score'].append(float(copy_paste_score))

                    for char, score in cref_data.items():
                        cross = score.get('cross')
                        self = score.get('self')  # 避免与关键字self冲突

                        if cross is not None:
                            all_avg['cross'].append(float(cross))
                        if self is not None:
                            all_avg['self'].append(float(self))
                    
    # 计算总平均值
    total_avg['total_cross_avg'] = mean(all_avg['cross']) if all_avg else 0.0
    total_avg['total_self_avg'] = mean(all_avg['self']) if all_avg else 0.0
    total_avg['total_avg_copy-paste-score'] = mean(all_avg['copy-paste-score']) if all_avg else 0.0
            
    # 保存结果
    result = {
        "check_timestamp": timestamp,
        "all_average_cref_scores": all_avg,
        "total_average_cref_score": total_avg,
        "processed_files_count(cross)": len(all_avg['cross']),
        "processed_files_count(self)": len(all_avg['self'])
    }
    output_file = os.path.join(save_path, 'total_average_cref.json')
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=4)

    print(f"Total average cref score calculated: cross is {total_avg['total_cross_avg']:.4f} and self is {total_avg['total_self_avg']:.4f}")
    print(f"Result saved to: {output_file}")

    return result




def prompt_align_total_avg(path):

    print(f'get prompt_align_total_avg ......')

    timestamp = get_timestamp_name(path)
    path = f'{path}/{timestamp}'

    save_path = os.path.join(path, 'total_avg')
    
    # 创建保存目录（如果不存在）
    os.makedirs(save_path, exist_ok=True)

    all_avg = {
        "scene": [],
        "global_character_action": [],
        "single_character_action_avg": [],
        "camera": [],
        "character_number": []

    }
    total_avg = {
        "total_avg_scene": None,
        "total_avg_global_character_action": None,
        "total_avg_single_character_action_avg": None,
        "total_avg_camera": None,
        "total_avg_character_number": None
    }

    # 遍历01到80的子目录
    for story_name in CHOICE_DATASET:

        json_dir = os.path.join(path, story_name)
    
        # 检查JSON目录是否存在
        if not os.path.exists(json_dir):
            print(f"Directory {json_dir} does not exist, skipping.")
            continue
        
        # 遍历JSON目录下的所有JSON文件
        for filename in os.listdir(json_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(json_dir, filename)

                with open(filepath, 'r') as f:
                    data = json.load(f)
                    from single_char_action_avg import get_single_character_action_avg
                    single_character_action_avg = get_single_character_action_avg(data)
                    avg_data = data.get('avg_score_dict', {})
                    for score_name, score_value in avg_data.items():
                        if score_value is not None:
                            if score_name == "single_character_action_avg":
                                all_avg[f'{score_name}'].append(float(single_character_action_avg))
                            else:
                                all_avg[f'{score_name}'].append(float(score_value))

                
    # 计算总平均值
    for score_name, score_list in all_avg.items():
        total_avg[f'total_avg_{score_name}'] = mean(score_list) if all_avg[f'{score_name}'] else 0.0

            
    # 保存结果
    result = {
        "check_timestamp": timestamp,
        "all_average_scores": all_avg,
        "total_average_score": total_avg,
        "processed_files_count(scene)": len(all_avg['scene']),
    }
    output_file = os.path.join(save_path, 'total_average.json')
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=4)

    print(f"Total average score calculated: {total_avg}")
    print(f"Result saved to: {output_file}")

    return result



def inception_score_total_avg(path):

    print(f'get inception_score_total_avg ......')

    timestamp = get_timestamp_name(path)
    path = f'{path}/{timestamp}'

    inception_scores = []

    # 直接遍历目标路径（无需子目录）
    for filename in os.listdir(path):
        if filename.endswith('.json'):
            filepath = os.path.join(path, filename)

            with open(filepath, 'r') as f:
                data = json.load(f)
                # 深度提取inception_score
                score = data.get('aggregate_scores', {})\
                                .get('generated_diversity', {})\
                                .get('inception_score')
                if score is not None:
                    inception_scores.append(float(score))


    # 计算平均值（实际只有一个分数）
    total_avg = sum(inception_scores) / len(inception_scores) if inception_scores else 0.0

    # 保存结果
    result = {
        "check_timestamp": timestamp,
        "inception_scores": inception_scores,
        "total_average_inception_score": total_avg,
        "processed_files_count": len(inception_scores)
    }

    return result



def get_timestamp_name(path):

    # 获取所有日期时间子目录
    import re
    from datetime import datetime
    datetime_pattern = re.compile(r'^\d{8}[_-]\d{6}$')
    datetime_dirs = [d for d in os.listdir(path) 
                    if os.path.isdir(os.path.join(path, d)) 
                    and datetime_pattern.match(d)]
    datetime_dirs.sort(key=lambda x: datetime.strptime(x.replace('_', '-'), "%Y%m%d-%H%M%S"))
    # 遍历每个日期时间目录
    datetime_dirs = datetime_dirs[-1]  # 只取最后一个目录

    print(f'timestamp:{datetime_dirs}')

    return datetime_dirs


if __name__ == "__main__":

    STORY_IMG = ['uno', 
                 'seedstory', 
                 'storygen_auto-regressive', 'storygen_mix', 'storygen_multi-image-condition',
                 'storydiffusion_original', 'storydiffusion_Photomaker',
                 'storyadapter_img_ref_results_xl', 'storyadapter_img_ref_results_xl5', 'storyadapter_text_only_results_xl', 'storyadapter_text_only_results_xl5',
                 'theatergen']
    
    STORY_VIDEO = ['movieagent_ROICtrl', 'movieagent_SD-3', 'animdirector', 
                   'vlogger_img_ref', 'vlogger_text_only',
                   'mmstoryagent']
    CLOSED_SOURCE = ['gemini', 'gpt4o']
    BUSINESS = ['moki', 'morphic_studio', 'bairimeng_ai', 'shenbimaliang', 'xunfeihuiying', 'doubao']
    NAIVE = ['naive_baseline']

    ALL_METHODS = STORY_IMG + STORY_VIDEO + CLOSED_SOURCE + BUSINESS + NAIVE



    FULL_DATA = [
        '01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
        '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
        '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
        '31', '32', '33', '34', '35', '36', '37', '38', '39', '40',
        '41', '42', '43', '44', '45', '46', '47', '48', '49', '50',
        '51', '52', '53', '54', '55', '56', '57', '58', '59', '60',
        '61', '62', '63', '64', '65', '66', '67', '68', '69', '70',
        '71', '72', '73', '74', '75', '76', '77', '78', '79', '80'
    ]

    LITE_DATA = [
        '01', '08', '09', '15', '17', '19', '24', '27', '28', '29', 
        '32', '41', '52', '53', '55', '57', '60', '64', '68', '79'
    ]

    REAL_DATA = [
        '03', '04', '05', '06', '07', '11', '12', '15', '16', '20', 
        '21', '24', '25', '27', '30', '35', '36', '39', '41', '42', 
        '49', '51', '52', '53', '54', '55', '59', '60', '64', '65', 
        '66', '67', '70', '71', '72', '73', '74', '75', '79'
    ]

    UNREAL_DATA = [x for x in FULL_DATA if x not in REAL_DATA]
    

    DATA_DICT = {
        'Full': FULL_DATA,
        'Lite': LITE_DATA,
        'Real': REAL_DATA,
        'UnReal': UNREAL_DATA
    }
    

    # uno_self_csd_path = f'{outputs_path}/uno/bench_results/csd_score(ii)/20250511_201127'
    # uno_cross_csd_path = f'{outputs_path}/uno/bench_results/csd_score(ri)/20250511_201127'
    


    # morphic_studio_cref = f'{outputs_path}/morphic_studio/bench_results/cref_score/20250513_214921'
    # morphic_studio_aesthetic = f'{outputs_path}/morphic_studio/bench_results/aesthetic_score/20250512_192901'
    
    # bairimeng_ai_cref = f'{outputs_path}/bairimeng_ai/bench_results/cref_score/20250513_215317'
    # bairimeng_ai_aesthetic = f'{outputs_path}/bairimeng_ai/bench_results/aesthetic_score/20250512_194655'
    
    # shenbimaliang_cref  = f'{outputs_path}/shenbimaliang/bench_results/cref_score/20250513_215332'
    # shenbimaliang_aesthetic  = f'{outputs_path}/shenbimaliang/bench_results/aesthetic_score/20250512_194704'
    
    # xunfeihuiying_cref = f'{outputs_path}/xunfeihuiying/bench_results/cref_score/20250513_215343'
    # xunfeihuiying_aesthetic = f'{outputs_path}/xunfeihuiying/bench_results/aesthetic_score/20250512_195201'
    
    # doubao_cref = f'{outputs_path}/doubao/bench_results/cref_score/20250513_215356'
    # doubao_aesthetic = f'{outputs_path}/doubao/bench_results/aesthetic_score/20250512_195208'



    # doubao_path = f'{outputs_path}/doubao/bench_results/inception_score/20250512_195208'




    
    for label in ['Full', 'Lite', 'Real', 'UnReal']:

        CHOICE_DATASET = DATA_DICT[f'{label}'] # 选择你需要统计的数据集

        print(f'------------------CHOICE_DATASET:{CHOICE_DATASET}')


        data_path = '/data/AIGC_Research/Story_Telling/StoryVisBMK/data'
        outputs_path = f'{data_path}/outputs'
        outputs_score_path = f'{data_path}/outputs_total_avgs/in_{label}'


        excluded = [
            'storygen_multi-image-condition',
            'storyadapter_text_only_results_xl5',

            'movieagent_SD-3',

            ]
        # ['vlogger_img_ref', 'vlogger_text_only'] ['storygen_auto-regressive', 'storygen_mix'] ['storyadapter_img_ref_results_xl', 'storyadapter_img_ref_results_xl5']
        # for method in [m for m in ALL_METHODS if m not in excluded]:
        for method in ['storyadapter_text_only_results_xl']:
        
        # for method in ALL_METHODS: # 选择你需要统计的方法集

            print(f'method now:{method}')

            # method = 'moki'

            path = { # moki
                "cref": f'{outputs_path}/{method}/bench_results/cref_score', # 20250513_211428
                "self_csd": f'{outputs_path}/{method}/bench_results/csd_score(ii)', # 20250512_193355
                "cross_csd": f'{outputs_path}/{method}/bench_results/csd_score(ri)', # 20250512_193355
                "aesthetic": f'{outputs_path}/{method}/bench_results/aesthetic_score', # 20250512_193355
                "inception_score": f'{outputs_path}/{method}/bench_results/inception_score', # 20250512_193355
                "prompt_align": f'{outputs_path}/{method}/bench_results/prompt_align'
            }
            

            # self_csd = csd_total_avg(path['self_csd'], mode='self')
            # cross_csd = csd_total_avg(path['cross_csd'], mode='cross')
            # aesthetic = aesthetic_total_avg(path['aesthetic'])
            # cref = cref_total_avg(path['cref'])
            prompt_align = prompt_align_total_avg(path['prompt_align'])
            # inception_score = inception_score_total_avg(path['inception_score'])


            BIG_TABLE = {
                "method": method,
                # "aesthetic": aesthetic,
                # "self_csd": self_csd,
                # "cross_csd": cross_csd,
                # "cref": cref,
                "prompt_align": prompt_align,
                # "inception_score":inception_score
            }


            os.makedirs(outputs_score_path,exist_ok=True)
            output_file = os.path.join(outputs_score_path, f'total_avg_of_{method}_in_{label}.json')
            with open(output_file, 'w') as f:
                json.dump(BIG_TABLE, f, indent=4)
            print(f'All save in {output_file}')
                
            # path = '/data/AIGC_Research/Story_Telling/StoryVisBMK/data/outputs/uno/bench_results/cref_score/20250509_000437'
            # cref_total_avg(path)


            # path = '/data/AIGC_Research/Story_Telling/StoryVisBMK/data/outputs/uno/bench_results/prompt_align_history/20250512_151300'
            # prompt_align_total_avg(path)


    