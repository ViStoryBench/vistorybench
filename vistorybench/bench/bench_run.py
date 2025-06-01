from style.csd.csd_test import get_csd_score, load_csd
from quality.aesthetic.aesthetic_metric import get_aesthetic_score, Aesthetic_Metric
import os
import sys
import itertools
import argparse
import pandas as pd
from datetime import datetime
import re
from PIL import Image
import json


def load_dataset(data_path, dataset_name, language):
    from dataset_load import StoryDataset
    dataset_path = f"{data_path}/dataset/{dataset_name}"
    # processed_dataset_path = f"{data_path}/dataset_processed/{method}/{dataset_name}_{language}"

    dataset = StoryDataset(dataset_path)
    story_name_list = dataset.get_story_name_list()
    print(f'\n故事名列表：{story_name_list}')

    # 加载所有故事（英文）
    stories_data = dataset.load_stories(story_name_list, language)
    # print(f'\n读取所有故事信息{stories_data}')

    return stories_data
    
def load_outputs(data_path, code_path, method, dataset_name, language):
    outputs_read_path = f'{code_path}/data_process/outputs_read'
    sys.path.append(outputs_read_path)

    # ///////////////////// Image /////////////////////
    if method == 'uno':
        from read_outputs import read_uno_outputs
        stories_image_paths = read_uno_outputs(
            data_path, method, dataset_name, language)

    if method == 'storygen':
        from read_outputs import read_storygen_outputs
        stories_image_paths = read_storygen_outputs(
            data_path, method, dataset_name, language,
            model_mode)

    if method == 'storyadapter':
        from read_outputs import read_storyadapter_outputs
        stories_image_paths = read_storyadapter_outputs(
            data_path, method, dataset_name, language,
            content_mode,
            scale_stage)

    if method == 'storydiffusion':
        from read_outputs import read_storydiffusion_outputs
        stories_image_paths = read_storydiffusion_outputs(
            data_path, method, dataset_name, language,
            content_mode,
            style_mode)

    if method == 'seedstory':
        from read_outputs import read_seedstory_outputs
        stories_image_paths = read_seedstory_outputs(
            data_path, method, dataset_name, language)

    if method == 'theatergen':
        from read_outputs import read_theatergen_outputs
        stories_image_paths = read_theatergen_outputs(
            data_path, method, dataset_name, language)

    # ///////////////////// Video /////////////////////
    if method == 'movieagent':
        from read_outputs import read_movieagent_outputs
        stories_image_paths = read_movieagent_outputs(
            data_path, method, dataset_name, language,
            model_type)
        
    if method == 'animdirector':
        from read_outputs import read_animdirector_outputs
        stories_image_paths = read_animdirector_outputs(
            data_path, method, dataset_name, language,
            model_type)
    
    if method == 'mmstoryagent':
        from read_outputs import read_mmstoryagent_outputs
        stories_image_paths = read_mmstoryagent_outputs(
            data_path, method, dataset_name, language)
        
    if method == 'vlogger':
        from read_outputs import read_vlogger_outputs
        stories_image_paths = read_vlogger_outputs(
            data_path, method, dataset_name, language,
            content_mode)

    # ///////////////////// Closed source /////////////////////
    if method in CLOSED_SOURCE:
        from read_outputs import read_mllm_outputs
        stories_image_paths = read_mllm_outputs(
            data_path, method, dataset_name, language)

    # ///////////////////// Business /////////////////////
    if method in BUSINESS:
        from read_outputs import read_business_outputs
        stories_image_paths = read_business_outputs(
            data_path, method, dataset_name, language)
        
    # ///////////////////// Naive baseline /////////////////////
    if method in NAIVE:
        from read_outputs import read_naive_outputs
        stories_image_paths = read_naive_outputs(
            data_path, method, dataset_name, language)

    return stories_image_paths



import json
def input_data_json_save(story_data, outputs_data):
    # Create directory for JSON files if it doesn't exist
    json_dir = os.path.join(data_path, 'json_data')
    os.makedirs(json_dir, exist_ok=True)
    
    # Save story_data to JSON
    story_json_path = os.path.join(json_dir, f'{method}_{story_name}_story_data.json')
    with open(story_json_path, 'w', encoding='utf-8') as f:
        json.dump(story_data, f, ensure_ascii=False, indent=2)
    
    # Save outputs_data to JSON
    outputs_json_path = os.path.join(json_dir, f'{method}_{story_name}_outputs_data.json')
    with open(outputs_json_path, 'w', encoding='utf-8') as f:
        json.dump(outputs_data, f, ensure_ascii=False, indent=2)


#///////////////////////////////////////////////////////////////////////////////////////////

def extract_timestamp(parts):
    """识别符合时间戳格式的路径组件"""
    # 定义时间戳正则表达式（示例格式：20250429-234922）
    timestamp_pattern = r"\d{8}[_-]\d{6}"  # 匹配8位日期+6位时间
    
    for part in reversed(parts):  # 从后往前找更高效
        if re.fullmatch(timestamp_pattern, part):
            try:
                # 验证是否为合法时间
                datetime.strptime(part.replace('_', '-'), "%Y%m%d-%H%M%S")
                return part
            except ValueError:
                continue
    return None



def save_filename_extra(shot_scores, method, score_type): # 通用函数

    # 从第一个生成图像路径提取文件名组件
    if shot_scores:
        # 示例路径：/data/.../outputs/uno/WildStory_en/12/20250429-234922/2_0.png
        if score_type == 'cref_score':
            sample_path = shot_scores[0]
        if score_type == 'csd_score(ri)':
            sample_path = shot_scores[0]["gen_image"]
        if score_type == 'csd_score(ii)':
            sample_path = shot_scores[0]["gen_image_1"]
        if score_type == 'aesthetic_score':
            sample_path = shot_scores[0]["gen_image"]

        if score_type == 'prompt_align':
            sample_path = shot_scores[0]
        if score_type == 'inception_score':
            sample_path = shot_scores[0]

        # 分割路径组件
        path_parts = sample_path.split('/')

        if method in BUSINESS + NAIVE or method in ['mmstoryagent','theatergen']:
            method_index = path_parts.index(f"{method}")
            story_name_index = path_parts.index(f"{story_name}")
            components = path_parts[method_index+1:story_name_index+1] # story_name_index+1 不能改，否则多故事的结果会覆盖
            print(f'components:{components}')
        else:
            # 找到method在路径中的位置（outputs目录的下一个）
            method_index = path_parts.index(f"{method}")
            # story_index = path_parts.index(f"{story_name}")
            timestamp_part = extract_timestamp(path_parts)
            print(f'timestamp_part:{timestamp_part}')
            timestamp_index = path_parts.index(f"{timestamp_part}")
            # 提取从method到timestamp的部分
            # 路径结构：outputs/[method]/[dataset]/[story_id]/[timestamp]/...
            components = path_parts[method_index+1:timestamp_index+1]
            print(f'components:{components}')
        
        # 拼接文件名组件
        filename_base = "_".join(components)
        if score_type == 'cref_score':
            filename = f"cref-{filename_base}"
        if score_type == 'csd_score(ri)' or score_type == 'csd_score(ii)':
            filename = f"csd-{filename_base}"
        if score_type == 'aesthetic_score':
            filename = f"aesthetic-{filename_base}"

        if score_type == 'prompt_align':
            filename = f"prompt_align-{filename_base}"
        if score_type == 'inception_score':
            filename = f"inception_score-{filename_base}"

    return filename

#///////////////////////////////////////////////////////////////////////////////////////////

def save_csd_to_excel_and_json(shot_scores, average_score, method, language, data_path, csd_type):
    """将生成图像之间的CSD得分保存到Excel和JSON文件"""
    
    # 创建结果目录结构
    base_dir = f'{data_path}/outputs/{method}/bench_results/{csd_type}/{timestamp}'
    if method == 'storyadapter':
        base_dir = f'{base_dir}/{scale_stage}'
    
    # 创建不同格式的保存目录
    excel_dir = os.path.join(base_dir, 'excel')
    json_dir = os.path.join(base_dir, 'json')
    os.makedirs(excel_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    
    # 生成文件名
    filename_base = save_filename_extra(shot_scores, method, csd_type)
    excel_filepath = os.path.join(excel_dir, filename_base + '.xlsx')
    json_filepath = os.path.join(json_dir, filename_base + '.json')
    
    # 构建数据框架 
    if csd_type == 'csd_score(ri)':
        data = {
            "分镜索引": [s["index"] for s in shot_scores],
            "CSD得分": [s["score"] for s in shot_scores],
            "参考图像路径": [s["ref_image"] for s in shot_scores],
            "生成图像路径": [s["gen_image"] for s in shot_scores]
        }
        avg_data = {
            "分镜索引": "平均分",
            "CSD得分": average_score,
            "参考图像路径": "",
            "生成图像路径": ""
        }

    elif csd_type == 'csd_score(ii)':
        data = {
            "组合索引": [s["index"] for s in shot_scores],
            "CSD得分": [s["score"] for s in shot_scores],
            "生成图像1路径": [s["gen_image_1"] for s in shot_scores],
            "生成图像2路径": [s["gen_image_2"] for s in shot_scores]
        }
        avg_data = {
            "组合索引": "平均分",
            "CSD得分": average_score,
            "生成图像1路径": "",
            "生成图像2路径": ""
        }
    
    # 保存Excel文件
    df = pd.DataFrame(data)
    df_avg = pd.DataFrame([avg_data])
    final_df = pd.concat([df, df_avg], ignore_index=True)
    final_df.to_excel(excel_filepath, index=False)
    print(f"\nExcel结果已保存至：{excel_filepath}")
    
    # 保存JSON文件
    json_data = {
        "metadata": {
            "method": method,
            "language": language,
            "csd_type": csd_type,
            "timestamp": timestamp
        },
        "average_score": average_score,
        "scores": shot_scores
    }
    
    with open(json_filepath, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)
    print(f"JSON结果已保存至：{json_filepath}")



def get_ri_csd_score(csd_encoder, story_data, outputs_data):
    # 计算生成图像与参考图像之间的风格一致性得分

    total_ri_score = []
    shot_details = []

    for i, (shot_info, output_img_path) in enumerate(zip(story_data['shots'], outputs_data)):
        # print(f'分镜{i+1}的文本提示: {text}')
        # print(f'分镜{i+1}的参考图像: {imgs}')

        print(f"""
        第{i+1}个分镜:
        - 索引: {shot_info['index']}
        - 布景: {shot_info['scene']}
        - 主观剧情: {shot_info['plot']}
        - 登场角色: {shot_info['character_name']}
        - 登场角色关键词: {shot_info['character_key']}
        - 客观描述: {shot_info['script']}
        - 镜头设计: {shot_info['camera']}
        """)
        
        characters = story_data["characters"]
        char_key = shot_info['character_key']
        if char_key != []:
            char_info = characters[char_key[0]]
            char_ref_image = char_info['images'][0]
        else:
            characters_1_key = list(characters.keys())[0]
            char_info = characters[characters_1_key]
            char_ref_image = char_info['images'][0]

        print(f'''
        第{i+1}个分镜的：
        - 参考图像: {char_ref_image}
        - 生成图像: {output_img_path}
        ''')

        # 计算参考图与生成图之间的风格一致性得分
        csd_score_tensor = get_csd_score(csd_encoder, char_ref_image, output_img_path)
        csd_score = csd_score_tensor[1].item()
        print(f'第{i+1}个分镜的风格一致性得分: {csd_score}')

        # 记录分镜详情
        shot_details.append({
            "index": shot_info['index'],
            "score": csd_score,
            "ref_image": char_ref_image,
            "gen_image": output_img_path
        })
        if method == 'seedstory' and shot_info['index'] == 1:
            assert shot_info['index'] == i+1
            shot_details.append({
                "index": shot_info['index'],
                "score": csd_score,
                "ref_image": char_ref_image,
                "gen_image": output_img_path,
                "PS": "In SEED-Story, the-first-shot-image and the-first-shot-image is the start of MLLM. \
We didn't consider using them effectively, so the information in this part is unreliable, please skip it. \
(SEED-Story is a visual story continuation task, which requires the-first-shot-image to be given in advance based on the-first-shot-prompt.)"
            })
        else:
            shot_details.append({
                "index": shot_info['index'],
                "score": csd_score,
                "ref_image": char_ref_image,
                "gen_image": output_img_path
            })

        print(f'即将保存的分镜信息：{shot_details}')

        total_ri_score.append(csd_score)

    print(f'所有分镜的风格一致性得分: {total_ri_score}')
    # 计算并打印平均分（至少需要两张图片）
    if len(total_ri_score) > 0:
        average_ri_csd = sum(total_ri_score) / len(total_ri_score)
        print(f"生成图像与参考图像之间的平均CSD得分: {average_ri_csd:.4f}")
    else:
        print("警告: 需要至少两张生成图像来计算两两分数")

    return shot_details, average_ri_csd

def get_ii_csd_score(csd_encoder, outputs_data):
    # 计算生成图像之间的风格一致性得分

    total_ii_score = []
    shot_details = []

    pairs = itertools.combinations(outputs_data, 2)
    print(f'所有生成图像的组合数: {len(list(pairs))}')
    for i, (output_img_path1, output_img_path2) in enumerate(itertools.combinations(outputs_data, 2)):
        print(f'''第{i+1}个组合：
              生成图像1: {output_img_path1}
              生成图像2: {output_img_path2}''')

        # 计算参考图与生成图之间的风格一致性得分
        csd_score_tensor = get_csd_score(csd_encoder, output_img_path1, output_img_path2)
        csd_score = csd_score_tensor[1].item()
        print(f'组合{i+1}的风格一致性得分: {csd_score}')

        # 记录分镜详情
        if method == 'seedstory':
            shot_details.append({
                "index": i+1,
                "score": csd_score,
                "gen_image_1": output_img_path1,
                "gen_image_2": output_img_path2,
                "PS": "In SEED-Story, the-first-shot-image and the-first-shot-image is the start of MLLM. \
    We didn't consider using them effectively, so the information in this part is unreliable, please skip it. \
    (SEED-Story is a visual story continuation task, which requires the-first-shot-image to be given in advance based on the-first-shot-prompt.)"
            })
        else:
            shot_details.append({
                "index": i+1,
                "score": csd_score,
                "gen_image_1": output_img_path1,
                "gen_image_2": output_img_path2
            })

        print(f'即将保存的分镜信息：{shot_details}')

        total_ii_score.append(csd_score)

    # 计算并打印平均分（至少需要两张图片）
    print(f'所有组合的风格一致性得分: {total_ii_score}')
    if len(total_ii_score) > 0:
        average_ii_csd = sum(total_ii_score) / len(total_ii_score)
        print(f"生成图像之间的平均CSD得分: {average_ii_csd:.4f}")
    else:
        print("警告: 需要至少两张生成图像来计算两两分数")


    return shot_details, average_ii_csd

#///////////////////////////////////////////////////////////////////////////////////////////

def save_aesthetic_to_excel_and_json(shot_scores, average_score, method, language, data_path, score_type):
    """保存美学评分结果到Excel和JSON文件"""
    
    # 创建结果目录结构
    base_dir = f'{data_path}/outputs/{method}/bench_results/aesthetic_score/{timestamp}'
    if method == 'storyadapter':
        base_dir = f'{base_dir}/{scale_stage}'
    
    # 创建不同格式的保存目录
    excel_dir = os.path.join(base_dir, 'excel')
    json_dir = os.path.join(base_dir, 'json')
    os.makedirs(excel_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    
    # 生成文件名
    filename_base = save_filename_extra(shot_scores, method, score_type)
    excel_filepath = os.path.join(excel_dir, filename_base + '.xlsx')
    json_filepath = os.path.join(json_dir, filename_base + '.json')
    
    # 构建数据结构
    data = {
        "分镜索引": [s["index"] for s in shot_scores],
        "美学评分": [s["score"] for s in shot_scores],
        "生成图像路径": [s["gen_image"] for s in shot_scores]
    }
    
    avg_data = {
        "分镜索引": "平均分",
        "美学评分": average_score,
        "生成图像路径": ""
    }
    
    # 保存Excel文件
    df = pd.DataFrame(data)
    df_avg = pd.DataFrame([avg_data])
    final_df = pd.concat([df, df_avg], ignore_index=True)
    final_df.to_excel(excel_filepath, index=False)
    print(f"\nExcel美学评分结果已保存至：{excel_filepath}")
    
    # 保存JSON文件
    json_data = {
        "metadata": {
            "method": method,
            "language": language,
            "score_type": score_type,
            "timestamp": timestamp,
            "scale_stage": scale_stage if method == 'storyadapter' else None
        },
        "average_score": average_score,
        "scores": shot_scores
    }
    
    with open(json_filepath, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)
    print(f"JSON美学评分结果已保存至：{json_filepath}")



def get_i_aesthetic_score(aesthetic_metric, outputs_data):
    """计算生成图像的美学评分"""
    total_score = []
    shot_details = []

    for idx, img_path in enumerate(outputs_data):

        score = get_aesthetic_score(aesthetic_metric, img_path)
        print(f"分镜{idx+1}美学评分: {score:.4f}")
        
        # 记录分镜详情
        if method == 'seedstory':
            shot_details.append({
                "index": idx+1,
                "score": float(score),
                "gen_image": img_path,
                "PS": "In SEED-Story, the-first-shot-image and the-first-shot-image is the start of MLLM. \
    We didn't consider using them effectively, so the information in this part is unreliable, please skip it. \
    (SEED-Story is a visual story continuation task, which requires the-first-shot-image to be given in advance based on the-first-shot-prompt.)"
            })
        else:
            shot_details.append({
                "index": idx+1,
                "score": float(score),
                "gen_image": img_path
            })

        print(f'即将保存的分镜信息：{shot_details}')

        total_score.append(score)

    # 计算平均分
    avg_score = sum(total_score)/len(total_score) if len(total_score) > 0 else 0
    print(f"\n平均美学评分: {avg_score:.4f}")
    
    return shot_details, avg_score

#///////////////////////////////////////////////////////////////////////////////////////////

def save_cref_to_excel(shot_scores, average_score, method, language, data_path, cref_type):
    pass

def get_ri_cref_score(story_data, outputs_data):

    cref_score = get_cref_score(char_ref_image, output_img_path)

    return shot_details, avg_ri_cref

def get_ii_cref_score(story_data, outputs_data):

    cref_score = get_cref_score(output_img_path1, output_img_path2)

    return shot_details, avg_ii_cref


def get_cref_score_and_save(ori_story_data, outputs_data, cref_type = 'cref_score'):
    """Calculate character reference (cref) scores between generated images and reference images"""
    cref_score_path = f'{code_path}/bench/content'
    sys.path.append(cref_score_path)

    # 创建结果目录结构
    base_save_dir = f'{data_path}/outputs/{method}/bench_results/{cref_type}/{timestamp}/{story_name}'
    if method == 'storyadapter':
        base_save_dir = f'{base_save_dir}/{scale_stage}'

    from cref_final import get_cref_score  # Import the CRef class we completed earlier
    cref_results = get_cref_score(ori_story_data, outputs_data, base_save_dir) # 单个故事的结果
    print(f'cref_results check story:{cref_results["cref"]}')
    
    # 创建不同格式的保存目录
    os.makedirs(base_save_dir, exist_ok=True)
    excel_dir = os.path.join(base_save_dir, 'excel')
    json_dir = os.path.join(base_save_dir, 'json')
    os.makedirs(excel_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    
    # 生成文件名
    filename_base = save_filename_extra(outputs_data, method, cref_type)
    excel_filepath = os.path.join(excel_dir, filename_base + '.xlsx')
    json_filepath = os.path.join(json_dir, filename_base + '.json')
    
    with open(json_filepath, 'w', encoding='utf-8') as f:
        json.dump(cref_results, f, ensure_ascii=False, indent=4)
    print(f"CRef json results saved to: {json_filepath}")



    # # Initialize results storage
    # total_cross_score = []
    # total_self_score = []
    # shot_details = []
    
    # # Prepare paths for CRef
    # pre_fix = os.path.join(data_path, "dataset", dataset_name, story_name)
    # output_dir = os.path.join(data_path, "outputs", method, dataset_name, story_name.split('_')[0], timestamp)
    
    
    # # Process each shot
    # for i, (shot_info, output_img_path) in enumerate(zip(story_data['shots'], outputs_data)):
    #     print(f"\nProcessing shot {i+1} for cref evaluation...")
        
    #     # Get character info
    #     characters = story_data["characters"]
    #     char_key = shot_info['character_key']
        
    #     if char_key:
    #         # Use the first character if multiple exist
    #         char_name = char_key[0]
    #         char_info = characters[char_name]
    #         char_ref_image = char_info['images'][0]
    #     else:
    #         # Fallback to first character if no character specified
    #         char_name = list(characters.keys())[0]
    #         char_info = characters[char_name]
    #         char_ref_image = char_info['images'][0]
        
    #     print(f"Character: {char_name}")
    #     print(f"Reference image: {char_ref_image}")
    #     print(f"Generated image: {output_img_path}")
        
    #     # Calculate cref scores
    #     cref_results = get_cref_score(pre_fix, output_dir)
        
    #     # Extract scores for this character
    #     char_scores = cref_results["cref"].get(char_name, {"cross": 0.0, "self": 0.0})
        
    #     # Record shot details
    #     shot_details.append({
    #         "index": shot_info['index'],
    #         "character": char_name,
    #         "cross_score": char_scores["cross"],
    #         "self_score": char_scores["self"],
    #         "ref_image": char_ref_image,
    #         "gen_image": output_img_path
    #     })
        
    #     # Accumulate scores
    #     total_cross_score.append(char_scores["cross"])
    #     total_self_score.append(char_scores["self"])
        
    #     print(f"Cref scores for shot {i+1}: Cross={char_scores['cross']:.4f}, Self={char_scores['self']:.4f}")
    
    # # Calculate average scores
    # avg_cross = sum(total_cross_score) / len(total_cross_score) if total_cross_score else 0.0
    # avg_self = sum(total_self_score) / len(total_self_score) if total_self_score else 0.0
    
    # print(f"\nAverage cref scores:")
    # print(f"Cross (reference-gen): {avg_cross:.4f}")
    # print(f"Self (gen-gen): {avg_self:.4f}")
    
    # # Prepare combined results
    # combined_results = {
    #     "cross": avg_cross,
    #     "self": avg_self,
    #     "details": shot_details
    # }
    
    # return shot_details, combined_results

# def save_cref_to_excel(shot_scores, average_score, method, language, data_path, cref_type):
#     """Save CRef scores to JSON file"""
    
#     # Create save directory
#     save_dir = f'{data_path}/outputs/{method}/bench_results/cref_score/{timestamp}'
#     if method == 'storyadapter':
#         save_dir = f'{save_dir}/{scale_stage}'
#     os.makedirs(save_dir, exist_ok=True)
    
#     # Get filename components
#     filename = save_filename_extra(shot_scores, method, cref_type).replace('.xlsx', '.json')
#     filepath = os.path.join(save_dir, filename)
    
#     # Prepare data structure
#     results_data = {
#         "shot_scores": shot_scores,
#         "average_scores": average_score,
#         "metadata": {
#             "method": method,
#             "language": language,
#             "timestamp": timestamp,
#             "cref_type": cref_type
#         }
#     }
    
#     # Save to JSON
#     with open(filepath, 'w', encoding='utf-8') as f:
#         json.dump(results_data, f, ensure_ascii=False, indent=2)
    
#     print(f"\nCRef results saved to: {filepath}")

#///////////////////////////////////////////////////////////////////////////////////////////


def save_prompt_align_to_excel(shot_scores, average_score, method, language, data_path, score_type):
    pass


def get_ti_vlm_score_2(story_data, outputs_data):

    gpt_score = get_vlm_score(input_prompt, output_img_path)
    
    return shot_details, avg_prompt_align

def get_ti_vlm_score(story_data, outputs_data):
    # 计算生成图像与参考图像之间的风格一致性得分

    total_ri_score = []
    shot_details = []

    for i, (shot_info, output_img_path) in enumerate(zip(story_data['shots'], outputs_data)):
        # print(f'分镜{i+1}的文本提示: {text}')
        # print(f'分镜{i+1}的参考图像: {imgs}')

        print(f"""
        第{i+1}个分镜:
        - 索引 Index: {shot_info['index']}
        - 布景 Scene Setting: {shot_info['scene']}
        - 主观剧情 Subjective Plot: {shot_info['plot']}
        - 登场角色 Characters Appearing: {shot_info['character_name']}
        - 登场角色关键词 Characters Appearing (key): {shot_info['character_key']}
        - 客观描述 Objective Description: {shot_info['script']}
        - 镜头设计 Camera Design: {shot_info['camera']}
        """)
        
        characters = story_data["characters"]
        char_key = shot_info['character_key']
        if char_key != []:
            char_info = characters[char_key[0]]
            char_ref_image = char_info['images'][0]
        else:
            characters_1_key = list(characters.keys())[0]
            char_info = characters[characters_1_key]
            char_ref_image = char_info['images'][0]

        print(f'''
        第{i+1}个分镜的：
        - 参考图像 Ref Img: {char_ref_image}
        - 生成图像 Gen Img: {output_img_path}
        ''')

        # 计算参考图与生成图之间的风格一致性得分
        csd_score_tensor = get_csd_score(char_ref_image, output_img_path)
        csd_score = csd_score_tensor[1].item()
        print(f'第{i+1}个分镜的风格一致性得分: {csd_score}')

        # 记录分镜详情
        shot_details.append({
            "index": shot_info['index'],
            "score": csd_score,
            "ref_image": char_ref_image,
            "gen_image": output_img_path
        })

        print(f'即将保存的分镜信息：{shot_details}')

        total_ri_score.append(csd_score)

    print(f'所有分镜的风格一致性得分: {total_ri_score}')
    # 计算并打印平均分（至少需要两张图片）
    if len(total_ri_score) > 0:
        average_ri_csd = sum(total_ri_score) / len(total_ri_score)
        print(f"生成图像与参考图像之间的平均CSD得分: {average_ri_csd:.4f}")
    else:
        print("警告: 需要至少两张生成图像来计算两两分数")

    return shot_details, average_ri_csd



if __name__ == "__main__":

    # Define method categories
    STORY_IMG = ['uno', 'seedstory', 'storygen', 'storydiffusion', 'storyadapter', 'theatergen']
    STORY_VIDEO = ['movieagent', 'animdirector', 'vlogger', 'mmstoryagent']
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


    parser = argparse.ArgumentParser(description='Story Dataset Processing Tool')

    parser.add_argument('--cref', action='store_true', help='get Ref-Gen & Gen-Gen Image CRef Score? (default: False)')
    parser.add_argument('--cref_cross', action='store_true', help='get Ref-Gen Image CRef Score? (default: False)')
    parser.add_argument('--cref_self', action='store_true', help='get Gen-Gen Image CRef Score? (default: False)')
    
    parser.add_argument('--csd_cross', action='store_true', help='get Ref-Gen Image CSD Score? (default: False)')
    parser.add_argument('--csd_self', action='store_true', help='get Gen-Gen Image CSD Score? (default: False)')
    
    parser.add_argument('--aesthetic', action='store_true', help='get generative image aesthetic Score? (default: False)')
    
    parser.add_argument('--prompt_align', action='store_true', help='get Text-Image alignment VLM Score? (default: False)')
    parser.add_argument('--prompt_align2', action='store_true', help='get Text-Image alignment VLM Score? (default: False)')

    parser.add_argument('--diversity', action='store_true', help='get Generative Image Inception Score? (default: False)')


    parser.add_argument('--language', type=str, choices=['en', 'ch'], default='en', help='Language option: en (English) or ch (Chinese)')
    parser.add_argument('--method', type=str, required=True, choices=ALL_METHODS, help='Method option')

    # for storydiffusion, storyadapter
    parser.add_argument('--content_mode', type=str, 
                        choices=['original', 'Photomaker', 'img_ref', 'text_only'], 
                        default='Photomaker', help='')
    
    # for storydiffusion
    parser.add_argument('--style_mode', type=str, 
                        choices=['(No style)'], 
                        default='(No style)', help='')
    
    # for storyadapter
    parser.add_argument('--scale_stage', type=str, 
                        choices=['results_xl', 'results_xl1', 'results_xl2', 'results_xl3', 'results_xl4', 'results_xl5'], 
                        default='results_xl5', help='')

    # for movieagent and animdirector
    parser.add_argument('--model_type', type=str, 
                        choices=['ROICtrl', 'SD-3', 'sd3'], 
                        default='ROICtrl', help='')

    # for storygen
    parser.add_argument('--model_mode', type=str, 
                        choices=['auto-regressive', 'multi-image-condition', 'mix'], 
                        default='mix', help='')

    args = parser.parse_args()

    data_path = '/data/AIGC_Research/Story_Telling/StoryVisBMK/data'
    code_path = '/data/AIGC_Research/Story_Telling/StoryVisBMK/code'
    dataset_process_path = f'{code_path}/data_process/dataset_process'
    sys.path.append(dataset_process_path)

    method=args.method
    dataset_name = 'WildStory'
    language=args.language
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


    # Image
    if method == 'uno':
        pass
    elif method == 'seedstory':
        pass
    elif method == 'storygen':
        model_mode = args.model_mode
        assert model_mode in ['auto-regressive', 'multi-image-condition', 'mix'], f"Invalid model_mode: {model_mode}"
    elif method == 'storydiffusion':
        content_mode = args.content_mode
        assert content_mode in ['original', 'Photomaker'], f"Invalid content_mode: {content_mode}"
        style_mode = args.style_mode
    elif method == 'storyadapter':
        content_mode = args.content_mode
        assert content_mode in ['img_ref', 'text_only'], f"Invalid content_mode: {content_mode}"
        scale_stage = args.scale_stage
    elif method == 'theatergen':
        pass


    # Video
    elif method == 'movieagent':
        model_type = args.model_type
        assert model_type in ['ROICtrl', 'SD-3'], f"Invalid model_type: {model_type}"
    elif method == 'animdirector':
        model_type = args.model_type
        assert model_type in ['sd3'], f"Invalid model_type: {model_type}"
    elif method == 'vlogger':
        content_mode = args.content_mode
        assert content_mode in ['img_ref', 'text_only'], f"Invalid content_mode: {content_mode}"
    elif method == 'mmstoryagent':
        pass

    # Closed source
    elif method in CLOSED_SOURCE:
        pass

    # Business
    elif method in BUSINESS:
        pass
    
    # Naive baseline
    elif method in NAIVE:
        pass


    
    stories_data = load_dataset(
        data_path, 
        dataset_name,
        language,
        )
    
    stories_outputs = load_outputs( 
        data_path, 
        code_path, 
        method,
        dataset_name, 
        language,
        )

    enable_cref_score = args.cref
    # enable_ri_cref_score = args.cref_cross
    # enable_ii_cref_score = args.cref_self
    
    enable_ri_csd_score = args.csd_cross
    enable_ii_csd_score = args.csd_self

    enable_aesthetic_score = args.aesthetic

    enable_prompt_align_vlm_score = args.prompt_align
    enable_prompt_align_vlm_score2 = args.prompt_align2

    enable_inception_score = args.diversity


    # model_init:
    if enable_ri_csd_score or enable_ii_csd_score:
        csd_encoder = load_csd().cuda()
    if enable_aesthetic_score:
        aesthetic_metric = Aesthetic_Metric(rank=0, device="cuda")
    if enable_prompt_align_vlm_score2:
        from transformers import CLIPProcessor, CLIPModel
        from groundingdino.util.inference import load_model, load_image, predict, annotate
        DINO_MODEL = "/data/AIGC_Research/Story_Telling/StoryVisBMK/code/bench/content/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        DINO_WEIGHTS = "/data/pretrain/groundingdino/weights/groundingdino_swint_ogc.pth"
         # load models
        clip_model = CLIPModel.from_pretrained("/data/pretrain/openai/clip-vit-base-patch16")
        clip_processor = CLIPProcessor.from_pretrained("/data/pretrain/openai/clip-vit-base-patch16")
        dino_model = load_model(DINO_MODEL, DINO_WEIGHTS)
        model_pkg = (clip_model, clip_processor, dino_model)

    # for (dataset_story_name, story_data),(outputs_story_name, outputs_data) in zip(stories_data.items(), stories_outputs.items()):
        
    #     if method not in BUSINESS:
    #         assert dataset_story_name == outputs_story_name, f"story_name不匹配：{dataset_story_name} != {outputs_story_name}"
    #         story_name = dataset_story_name
    #     elif method in BUSINESS:
    #         pass

    for outputs_story_name, outputs_data in stories_outputs.items():
        for dataset_story_name, story_data in stories_data.items():

            if (dataset_story_name == outputs_story_name 
                # and outputs_story_name not in ['09','16'] # for seed story
                # and outputs_story_name in LITE_DATA # for gpt/gemini
                ):

                story_name = dataset_story_name

                print(f'''
                    当前方法 Method：{method}
                    故事名 Story name: {story_name}
                    源故事数据分镜数 Shot length of source dataset: {len(story_data['shots'])}
                    生成图像数据分镜数 Shot length of generated data: {len(outputs_data)}
                    ''')
                

                assert len(story_data['shots']) == len(outputs_data), f"生成的分镜数与原始数据分镜数不匹配：{len(story_data['shots'])} != {len(outputs_data)}"

                if len(story_data['shots']) != len(outputs_data):
                    print(f"生成的分镜数与原始数据分镜数不匹配：{len(story_data['shots'])} != {len(outputs_data)}")



                # if story_name < '35':
                #     continue

                # input_data_json_save(story_data, outputs_data)

                # if enable_ri_cref_score:
                #     print(f'计算生成图像与参考图像、关键词之间的cref得分')
                #     # 计算生成图像与参考图像之间的内容一致性得分
                #     ri_cref_shot_details, avg_ri_cref = get_ri_cref_score(story_data, outputs_data)
                #     save_cref_to_excel(ri_cref_shot_details, avg_ri_cref, method, language, data_path, 'cref_score(ri)')

                # if enable_ii_cref_score:
                #     print(f'计算生成图像与参考图像、关键词之间的cref得分')
                #     # 计算生成图像与参考图像之间的内容一致性得分
                #     ii_cref_shot_details, avg_ii_cref = get_ii_cref_score(story_data, outputs_data)
                #     save_cref_to_excel(ii_cref_shot_details, avg_ii_cref, method, language, data_path, 'cref_score(ii)')

                if enable_cref_score:
                    print(f'计算生成图像与参考图像、关键词之间的cref得分')
                    # 计算生成图像与参考图像之间的内容一致性得分
                    ori_story_data = f"{data_path}/dataset/{dataset_name}/{story_name}"
                    get_cref_score_and_save(ori_story_data, outputs_data, 'cref_score')
                    # save_cref_to_excel(cref_shot_details, avg_cref, method, language, data_path, 'cref_score(ri+ii)')


                if enable_ri_csd_score:
                    print(f'计算生成图像与参考图像之间的CSD得分')
                    # 计算生成图像与参考图像之间的风格一致性得分
                    ri_csd_shot_details, avg_ri_csd = get_ri_csd_score(csd_encoder, story_data, outputs_data)
                    save_csd_to_excel_and_json(ri_csd_shot_details, avg_ri_csd, method, language, data_path, 'csd_score(ri)')

                if enable_ii_csd_score:
                    print(f'计算生成图像之间的CSD得分')
                    # 计算生成图像之间的风格一致性得分
                    ii_csd_shot_details, avg_ii_csd = get_ii_csd_score(csd_encoder, outputs_data)
                    save_csd_to_excel_and_json(ii_csd_shot_details, avg_ii_csd, method, language, data_path, 'csd_score(ii)')


                if enable_aesthetic_score: 
                    print(f'计算生成图像的美学评分')
                    aesthetic_shot_details, avg_aesthetic = get_i_aesthetic_score(aesthetic_metric, outputs_data)
                    save_aesthetic_to_excel_and_json(aesthetic_shot_details, avg_aesthetic, method, language, data_path, 'aesthetic_score')


                if enable_prompt_align_vlm_score: 
                    print(f'计算生成图像与提示词对齐程度的VLM评分')
                    prompt_align_shot_details, avg_prompt_align = get_ti_vlm_score(story_data, outputs_data)
                    save_prompt_align_to_excel(prompt_align_shot_details, avg_prompt_align, method, language, data_path, 'vlm_score')
            
            else:
                continue



    if enable_prompt_align_vlm_score2:

        vlm_score_path = f'{code_path}/bench/prompt_align/vlm_score/'
        sys.path.append(vlm_score_path)
        # from gpt_bench2 import run_vlm_bench
        from gpt_eval import run_vlm_bench

        print(f'当前方法 Method：{method}')
        save_dir= f"{data_path}/outputs/{method}/bench_results/prompt_align/{timestamp}"

        story_case = stories_outputs[f'{story_name}']
        filename_base = save_filename_extra(story_case, method, 'prompt_align')
        # filename_base = 'prompt_align'

        run_vlm_bench(
            model_pkg,
            stories_data, 
            stories_outputs,
            save_dir,
            filename_base,
            method
            )
    
    if enable_inception_score:

        labels = ['Real', 'Full', 'Lite', 'Real', 'UnReal']
        for label in labels:
            CHOICE_DATASET = DATA_DICT[f'{label}'] # 选择你需要统计的数据集

            inception_score_path = f'{code_path}/bench/diversity/'
            sys.path.append(inception_score_path)
            # from gpt_bench2 import run_vlm_bench
            from inception_score import get_inception_score_and_save

            print(f'当前方法 Method：{method}')
            save_dir= f"{data_path}/outputs/{method}/bench_results/inception_score/{timestamp}"

            story_case = stories_outputs[f'{story_name}']
            print(f'story_case:{story_case}')
            filename_base = save_filename_extra(story_case, method, 'inception_score')

            get_inception_score_and_save(
                method, stories_outputs, 
                CHOICE_DATASET, label,
                save_dir, filename_base)
            