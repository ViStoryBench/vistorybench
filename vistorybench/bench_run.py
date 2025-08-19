import os
import sys
import itertools
import argparse
import pandas as pd
from datetime import datetime
import re
from PIL import Image
import json
import time

import yaml
from pathlib import Path
from data_process.outputs_read.read_outputs import load_outputs

def blue_print(text, bright=True):
    color_code = "\033[94m" if bright else "\033[34m"
    print(f"{color_code}{text}\033[0m")

def yellow_print(text):
    print(f"\033[93m{text}\033[0m")

def green_print(text):
    print(f"\033[92m{text}\033[0m")

def load_dataset(_dataset_path, dataset_name, language):

    from data_process.dataset_process.dataset_load import StoryDataset
    dataset_path = f"{_dataset_path}/{dataset_name}"

    dataset = StoryDataset(dataset_path)
    story_name_list = dataset.get_story_name_list()
    print(f'\nStory name list：{story_name_list}')

    # loading all stories
    stories_data = dataset.load_stories(story_name_list, language)
    # print(f'\nAll stories: {stories_data}')

    return stories_data

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
    """Identify path components that match timestamp format"""
    # Define timestamp regex pattern (example format: 20250429-234922)
    timestamp_pattern = r"\d{8}[_-]\d{6}" # Match 8-digit date + 6-digit time
    
    for part in reversed(parts): # Search from back to front for efficiency
        if re.fullmatch(timestamp_pattern, part):
            try:
                # Validate if it's a legal timestamp
                datetime.strptime(part.replace('_', '-'), "%Y%m%d-%H%M%S")
                return part
            except ValueError:
                continue
    return None



def save_filename_extra(shot_scores, method, score_type):

    # Extract filename components from the first generated image path
    # Example path: /data/.../outputs/uno/vistorybench_en/12/20250429-234922/2_0.png
    if shot_scores:
        if score_type == 'cids_score':
            sample_path = shot_scores[0]
        if score_type == 'cross_csd':
            sample_path = shot_scores[0]["gen_image"]
        if score_type == 'self_csd':
            sample_path = shot_scores[0]["gen_image_1"]
        if score_type == 'aesthetic_score':
            sample_path = shot_scores[0]["gen_image"]

        if score_type == 'prompt_align':
            sample_path = shot_scores[0]
        if score_type == 'inception_score':
            sample_path = shot_scores[0]

        # Split path component
        path_parts = sample_path.split('/')

        # first_image_path = list(shot_scores.keys())[0]
        # path_parts = first_image_path.split('/')

        if method in BUSINESS + NAIVE or method in ['mmstoryagent','theatergen']:
            method_index = path_parts.index(f"{method}")
            story_name_index = path_parts.index(f"{story_name}")
            components = path_parts[method_index+1:story_name_index+1] # story_name_index+1 cannot be changed, otherwise multi-story results will be overwritten
            print(f'components:{components}')
        else:
            # Find the position of 'method' in the path (The next one in outputs directory)
            method_index = path_parts.index(f"{method}")
            # story_index = path_parts.index(f"{story_name}")
            timestamp_part = extract_timestamp(path_parts)
            print(f'timestamp_part:{timestamp_part}')
            timestamp_index = path_parts.index(f"{timestamp_part}")
            # Extract from method to timestamp
            # Path structure: outputs/[method]/[dataset]/[story_id]/[timestamp]/...
            components = path_parts[method_index+1:timestamp_index+1]
            print(f'components:{components}')
        
        # Concatenate filename components
        filename_base = "_".join(components)
        if score_type == 'cids_score':
            filename = f"cids-{filename_base}"
        if score_type == 'cross_csd' or score_type == 'self_csd':
            filename = f"csd-{filename_base}"
        if score_type == 'aesthetic_score':
            filename = f"aesthetic-{filename_base}"

        if score_type == 'prompt_align':
            filename = f"prompt_align-{filename_base}"
        if score_type == 'inception_score':
            filename = f"inception_score-{filename_base}"

    return filename

#///////////////////////////////////////////////////////////////////////////////////////////

def save_csd_to_excel_and_json(shot_scores, average_score, method, language, outputs_path, csd_type):
    """Save CSD scores to Excel and JSON files"""
    
    # Create result directory structure
    base_dir = f'{outputs_path}/{method}/bench_results/{csd_type}/{timestamp}'
    if method == 'storyadapter':
        base_dir = f'{base_dir}/{scale_stage}'
    
    # Create save directories for different formats
    excel_dir = os.path.join(base_dir, 'excel')
    json_dir = os.path.join(base_dir, 'json')
    os.makedirs(excel_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    # Generate filename
    filename_base = save_filename_extra(shot_scores, method, csd_type)
    excel_filepath = os.path.join(excel_dir, filename_base + '.xlsx')
    json_filepath = os.path.join(json_dir, filename_base + '.json')
    
    # Build data structure
    if csd_type == 'cross_csd':
        data = {
            "Shot Index": [s["index"] for s in shot_scores],
            "CSD Score": [s["score"] for s in shot_scores],
            "Reference Image Path": [s["ref_image"] for s in shot_scores],
            "Generated Image Path": [s["gen_image"] for s in shot_scores]
        }
        avg_data = {
            "Shot Index": "Average Score",
            "CSD Score": average_score,
            "Reference Image Path": "",
            "Generated Image Path": ""
        }

    elif csd_type == 'self_csd':
        data = {
            "Combination Index": [s["index"] for s in shot_scores],
            "CSD Score": [s["score"] for s in shot_scores],
            "Generated Image 1 Path": [s["gen_image_1"] for s in shot_scores],
            "Generated Image 2 Path": [s["gen_image_2"] for s in shot_scores]
        }
        avg_data = {
            "Combination Index": "Average Score",
            "CSD Score": average_score,
            "Generated Image 1 Path": "",
            "Generated Image 2 Path": ""
        }
    
    # Save Excel file
    df = pd.DataFrame(data)
    df_avg = pd.DataFrame([avg_data])
    final_df = pd.concat([df, df_avg], ignore_index=True)
    final_df.to_excel(excel_filepath, index=False)
    print(f"\nExcel results saved to: {excel_filepath}")
    
    # Save JSON file
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
    print(f"JSON results saved to: {json_filepath}")



def get_cross_csd_score(csd_encoder, story_data, outputs_data):
    # Calculate cross style similarity scores between generated images and reference images

    total_ri_score = []
    shot_details = []

    for i, (shot_info, output_img_path) in enumerate(zip(story_data['shots'], outputs_data['shots'])):
        # print(f'Shot {i+1} text prompt: {text}')
        # print(f'Shot {i+1} reference image: {imgs}')

        print(f"""
        Shot {i+1}:
        - Index: {shot_info['index']}
        - Scene Setting: {shot_info['scene']}
        - Subjective Plot: {shot_info['plot']}
        - Characters Appearing: {shot_info['character_name']}
        - Characters Appearing Keywords: {shot_info['character_key']}
        - Objective Description: {shot_info['script']}
        - Camera Design: {shot_info['camera']}
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
        The {i+1}th Shot:
        - Reference Image: {char_ref_image}
        - Generated Image: {output_img_path}
        ''')

        start_time = time.time()
        csd_score_tensor = get_csd_score(csd_encoder, char_ref_image, output_img_path)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Cross-CSD Score Calculation Time-consuming: {elapsed_time:.4f} seconds")
        csd_score = csd_score_tensor[1].item()
        print(f'The {i+1}th Shot Style Similarity Score: {csd_score}')

        # Record shot details
        shot_details.append({
            "index": shot_info['index'],
            "score": csd_score,
            "ref_image": char_ref_image,
            "gen_image": output_img_path,
            "elapsed_time(seconds)": elapsed_time
        })
        if method == 'seedstory' and shot_info['index'] == 1:
            assert shot_info['index'] == i+1
            shot_details.append({
                "index": shot_info['index'],
                "score": csd_score,
                "ref_image": char_ref_image,
                "gen_image": output_img_path,
                "elapsed_time(seconds)": elapsed_time
            })
        else:
            shot_details.append({
                "index": shot_info['index'],
                "score": csd_score,
                "ref_image": char_ref_image,
                "gen_image": output_img_path,
                "elapsed_time(seconds)": elapsed_time
            })

        print(f'Shot details to be saved (Only show the first one): {shot_details[0]}')

        total_ri_score.append(csd_score)

    print(f'All Shot Style Similarity Scores: {total_ri_score}')
    # Calculate and print average score (need at least two images)
    if len(total_ri_score) > 0:
        average_ri_csd = sum(total_ri_score) / len(total_ri_score)
        print(f"Average CSD score between generated images and reference images: {average_ri_csd:.4f}")
    else:
        print("Warning: Need at least two generated images to calculate pairwise scores")

    return shot_details, average_ri_csd

def get_self_csd_score(csd_encoder, outputs_data):
    # Calculate self style similarity scores between generated images

    total_ii_score = []
    shot_details = []

    pairs = itertools.combinations(outputs_data, 2)
    print(f'Total combinations of generated images: {len(list(pairs))}')
    for i, (output_img_path1, output_img_path2) in enumerate(itertools.combinations(outputs_data, 2)):
        print(f'''Combination {i+1}th:
              Generated Image 1: {output_img_path1}
              Generated Image 2: {output_img_path2}''')

        start_time = time.time()
        csd_score_tensor = get_csd_score(csd_encoder, output_img_path1, output_img_path2)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Self-CSD Score Calculation Time-consuming: {elapsed_time:.4f} seconds")
        csd_score = csd_score_tensor[1].item()
        print(f'The {i+1}th Shot Style Similarity Score: {csd_score}')

        # 记录分镜详情
        shot_details.append({
            "index": i+1,
            "score": csd_score,
            "gen_image_1": output_img_path1,
            "gen_image_2": output_img_path2,
            "elapsed_time(seconds)": elapsed_time
        })
        print(f'Shot details to be saved (Only show the first one): {shot_details[0]}')

        total_ii_score.append(csd_score)

    # Calculate and print average score (need at least two images)
    print(f'All combination style consistency scores: {total_ii_score}')
    if len(total_ii_score) > 0:
        average_ii_csd = sum(total_ii_score) / len(total_ii_score)
        print(f"Average CSD score between generated images: {average_ii_csd:.4f}")
    else:
        print("Warning: Need at least two generated images to calculate pairwise scores")


    return shot_details, average_ii_csd

#///////////////////////////////////////////////////////////////////////////////////////////

def save_aesthetic_to_excel_and_json(shot_scores, average_score, method, language, outputs_path, score_type):
    """Save aesthetic scoring results to Excel and JSON files"""
    
    # Create result directory structure
    base_dir = f'{outputs_path}/{method}/bench_results/aesthetic_score/{timestamp}'
    if method == 'storyadapter':
        base_dir = f'{base_dir}/{scale_stage}'
    
    # Create save directories for different formats
    excel_dir = os.path.join(base_dir, 'excel')
    json_dir = os.path.join(base_dir, 'json')
    os.makedirs(excel_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    
    # Generate filename
    filename_base = save_filename_extra(shot_scores, method, score_type)
    excel_filepath = os.path.join(excel_dir, filename_base + '.xlsx')
    json_filepath = os.path.join(json_dir, filename_base + '.json')
    
    # Build data structure
    data = {
        "Shot Index": [s["index"] for s in shot_scores],
        "Aesthetic Score": [s["score"] for s in shot_scores],
        "Generated Image Path": [s["gen_image"] for s in shot_scores]
    }
    
    avg_data = {
        "Shot Index": "Average Score",
        "Aesthetic Score": average_score,
        "Generated Image Path": ""
    }
    
    # Save Excel file
    df = pd.DataFrame(data)
    df_avg = pd.DataFrame([avg_data])
    final_df = pd.concat([df, df_avg], ignore_index=True)
    final_df.to_excel(excel_filepath, index=False)
    print(f"\nExcel aesthetic scoring results saved to: {excel_filepath}")
    
    # Save JSON file
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
    print(f"JSON aesthetic scoring results saved to: {json_filepath}")



def get_aesthetic_score(aesthetic_metric, outputs_data):
    """Calculate aesthetic scores for generated images"""
    total_score = []
    shot_details = []

    for idx, img_path in enumerate(outputs_data):

        start_time = time.time()

        Img = Image.open(img_path)
        if Img.mode != 'RGB':
            Img = Img.convert('RGB')
        score = aesthetic_metric(Img)
        print(f"Shot {idx+1} aesthetic score: {score:.4f}")
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Single-image Aesthetic Score Calculation Time-consuming: {elapsed_time:.4f} seconds")

        # Record shot details
        shot_details.append({
            "index": idx+1,
            "score": float(score),
            "gen_image": img_path,
            "elapsed_time(seconds)": elapsed_time
        })

        print(f'Shot details to be saved (Only show the first one): {shot_details[0]}')

        total_score.append(score)

    # Calculate average score
    avg_score = sum(total_score)/len(total_score) if len(total_score) > 0 else 0
    print(f"\nAverage aesthetic score: {avg_score:.4f}")
    
    return shot_details, avg_score

#///////////////////////////////////////////////////////////////////////////////////////////


def get_cids_score_and_save(model_pkg, pretrain_path, 
                            story_data_path, outputs_data, 
                            cids_type = 'cids_score', ref_mode = 'origin'):

    """Calculate character ID consistency scores between generated images and reference images"""
    cids_score_path = f'{code_path}/bench/content'
    sys.path.append(cids_score_path)

    # Create result directory structure
    if ref_mode == 'origin':
        base_save_dir = f'{result_path}/{method}/bench_results/{cids_type}/{timestamp}/{story_name}'
    elif ref_mode == 'mid-gen':
        base_save_dir = f'{result_path}/{method}/bench_results/{cids_type}_{ref_mode}/{timestamp}/{story_name}'

    # if method == 'storyadapter':
    #     base_save_dir = f'{base_save_dir}/{scale_stage}'

    
    from bench.content.cids import CIDS  # Import the CIDS class we completed earlier
    start_time = time.time()

    print(f'cids_mode:{ref_mode}')
    # Initialize CIDS evaluator
    cids_evaluator = CIDS(model_pkg, pretrain_path, story_data_path, outputs_data, base_save_dir, ref_mode)
    cids_results = cids_evaluator.main() # Single story results
    print(f'cids_results check story:{cids_results["cids"]}')

    end_time = time.time()
    elapsed_time = end_time - start_time
    cids_results.update({"elapsed_time(seconds)": elapsed_time})
    print(f"Single-story CIDS Score Calculation Time-consuming: {elapsed_time:.4f} seconds")


    # Create save directories for different formats
    os.makedirs(base_save_dir, exist_ok=True)
    excel_dir = os.path.join(base_save_dir, 'excel')
    json_dir = os.path.join(base_save_dir, 'json')
    os.makedirs(excel_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    
    # Generate filename
    filename_base = save_filename_extra(outputs_data['shots'], method, cids_type)
    filename_base = f'{filename_base}_({ref_mode}_ref)'
    excel_filepath = os.path.join(excel_dir, filename_base + '.xlsx')
    json_filepath = os.path.join(json_dir, filename_base + '.json')
    
    with open(json_filepath, 'w', encoding='utf-8') as f:
        json.dump(cids_results, f, ensure_ascii=False, indent=4)
    print(f"JSON CIDS results saved to: {json_filepath}")



    # # Initialize results storage
    # total_cross_score = []
    # total_self_score = []
    # shot_details = []
    
    # # Prepare paths for CIDS
    # pre_fix = os.path.join(data_path, "dataset", dataset_name, story_name)
    # output_dir = os.path.join(data_path, "outputs", method, dataset_name, story_name.split('_')[0], timestamp)
    
    
    # # Process each shot
    # for i, (shot_info, output_img_path) in enumerate(zip(story_data['shots'], outputs_data)):
    #     print(f"\nProcessing shot {i+1} for cids evaluation...")
        
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
        
    #     # Calculate cids scores
    #     cids_results = get_cids_score(pre_fix, output_dir)
        
    #     # Extract scores for this character
    #     char_scores = cids_results["cids"].get(char_name, {"cross": 0.0, "self": 0.0})
        
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
        
    #     print(f"CIDS scores for shot {i+1}: Cross={char_scores['cross']:.4f}, Self={char_scores['self']:.4f}")
    
    # # Calculate average scores
    # avg_cross = sum(total_cross_score) / len(total_cross_score) if total_cross_score else 0.0
    # avg_self = sum(total_self_score) / len(total_self_score) if total_self_score else 0.0
    
    # print(f"\nAverage cids scores:")
    # print(f"Cross (reference-gen): {avg_cross:.4f}")
    # print(f"Self (gen-gen): {avg_self:.4f}")
    
    # # Prepare combined results
    # combined_results = {
    #     "cross": avg_cross,
    #     "self": avg_self,
    #     "details": shot_details
    # }
    
    # return shot_details, combined_results

# def save_cids_to_excel(shot_scores, average_score, method, language, data_path, cids_type):
#     """Save CIDS scores to JSON file"""
    
#     # Create save directory
#     save_dir = f'{data_path}/outputs/{method}/bench_results/cids_score/{timestamp}'
#     if method == 'storyadapter':
#         save_dir = f'{save_dir}/{scale_stage}'
#     os.makedirs(save_dir, exist_ok=True)
    
#     # Get filename components
#     filename = save_filename_extra(shot_scores, method, cids_type).replace('.xlsx', '.json')
#     filepath = os.path.join(save_dir, filename)
    
#     # Prepare data structure
#     results_data = {
#         "shot_scores": shot_scores,
#         "average_scores": average_score,
#         "metadata": {
#             "method": method,
#             "language": language,
#             "timestamp": timestamp,
#             "cids_type": cids_type
#         }
#     }
    
#     # Save to JSON
#     with open(filepath, 'w', encoding='utf-8') as f:
#         json.dump(results_data, f, ensure_ascii=False, indent=2)
    
#     print(f"\nCIDS results saved to: {filepath}")

#///////////////////////////////////////////////////////////////////////////////////////////




def get_prompt_align_score_and_save(story_data, outputs_data):
    # Calculate prompt consistency scores between generated images and given prompt

    from bench.prompt_align.gpt_eval import StoryTellingEvalBench

    print(f'Current Method：{method}')
    save_dir= f"{result_path}/{method}/bench_results/prompt_align/{timestamp}"

    story_case = stories_outputs[f'{story_name}']['shots']
    filename_base = save_filename_extra(story_case, method, 'prompt_align')
    # filename_base = 'prompt_align'

    os.makedirs(save_dir, exist_ok=True)
    eval_bench = StoryTellingEvalBench(model_pkg, gpt_api_pkg,
                                    story_data, outputs_data, code_path,
                                    save_dir, filename_base, method)
    eval_bench.eval_story_by_character(story_name, max_workers=1)


def save_outputs_in_same_format(oss_imgs_path, outputs_data):

    import os
    import shutil

    print(f'oss_imgs_path:{oss_imgs_path}\noutputs_data:{outputs_data}')

    os.makedirs(oss_imgs_path, exist_ok=True)

    # Copy and rename the files in sequence
    for index, src_path in enumerate(outputs_data, 1):  # Count from 1
        # Generate new file name (e.g., shot_01.png)
        new_filename = f"shot_{index:02d}.png"
        dest_path = os.path.join(oss_imgs_path, new_filename)
        
        # Copy the file to the target path
        shutil.copy(src_path, dest_path)
        print(f"Copied: {src_path} -> {dest_path}")

    print("All files copied successfully!")


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}  # Return empty dict if file not found


if __name__ == "__main__":

    # //////////////////// Pre-define method categories ////////////////////
    STORY_IMG = ['uno', 'seedstory', 'storygen','storydiffusion', 'storyadapter', 'theatergen']
    STORY_VIDEO = ['movieagent', 'animdirector', 'vlogger', 'mmstoryagent']
    CLOSED_SOURCE = ['gemini', 'gpt4o']
    BUSINESS = ['moki', 'morphic_studio', 'bairimeng_ai', 'shenbimaliang', 'xunfeihuiying', 'doubao']
    NAIVE = ['naive_baseline']

    ALL_METHODS = STORY_IMG + STORY_VIDEO + CLOSED_SOURCE + BUSINESS + NAIVE

    METHODS_DICT = {
        'STORY_IMG': STORY_IMG,
        'STORY_VIDEO': STORY_VIDEO,
        'CLOSED_SOURCE': CLOSED_SOURCE,
        'BUSINESS': BUSINESS,
        'NAIVE': NAIVE,
    }
    # ////////////////////////////////////////////////////////////////////////

    # //////////////////// Pre-define dataset categories ////////////////////

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

    CUSTOM_DATA = ['01']
    
    DATA_DICT = {
        'Full': FULL_DATA,
        'Lite': LITE_DATA,
        'Real': REAL_DATA,
        'UnReal': UNREAL_DATA,
        'Custom': CUSTOM_DATA
    }
    # ////////////////////////////////////////////////////////////////////////

    grandparent_dir = Path(__file__).resolve().parent.parent
    print(f'grandparent_dir: {grandparent_dir}')

    code_path = f'{grandparent_dir}/vistorybench'
    data_path = f'{grandparent_dir}/data'
    print(f'code_path: {code_path}')
    print(f'data_path: {data_path}')
    
    base_parser = argparse.ArgumentParser(description='Application path configuration', add_help=False)
    base_parser.add_argument('--config', type=str, default=f'{code_path}/config.yaml', help='Path to configuration file (default: config.yaml)')
    base_args, _ = base_parser.parse_known_args()  # Parse only known args
    config = load_config(base_args.config)
    
    parser = argparse.ArgumentParser(
        description='Story Dataset Processing Tool',
        parents=[base_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--dataset_path', type=str, default=config.get('dataset_path') or f'{data_path}/dataset', help='Directory for datasets')
    parser.add_argument('--outputs_path', type=str, default=config.get('outputs_path') or f'{data_path}/outputs', help='Directory for generated outputs')
    parser.add_argument('--pretrain_path', type=str, default=config.get('pretrain_path') or f'{data_path}/pretrain', help='Directory for pretrained models')
    parser.add_argument('--result_path', type=str,default=config.get('result_path') or f'{data_path}/results', help='Directory for results')
    parser.add_argument('--model_id', type=str, default=config.get('model_id') or 'gpt-4.1', help='model_id of gpt api for prompt align score')
    parser.add_argument('--api_key', type=str, default=config.get('api_key') or '', help='api_key of gpt api for prompt align score')
    parser.add_argument('--base_url', type=str, default=config.get('base_url') or '', help='base_url of gpt api for prompt align score')

    parser.add_argument('--save_format', action='store_true', help='resave the generated results in the same format? (default: False)')

    parser.add_argument('--cids', action='store_true', help='get Ref-Gen & Gen-Gen Image CIDS Score? (default: False)')
    parser.add_argument('--cids_cross', action='store_true', help='get Ref-Gen Image CIDS Score? (default: False)')
    parser.add_argument('--cids_self', action='store_true', help='get Gen-Gen Image CIDS Score? (default: False)')
    parser.add_argument('--csd_cross', action='store_true', help='get Ref-Gen Image CSD Score? (default: False)')
    parser.add_argument('--csd_self', action='store_true', help='get Gen-Gen Image CSD Score? (default: False)')
    parser.add_argument('--aesthetic', action='store_true', help='get generative image aesthetic Score? (default: False)')
    parser.add_argument('--prompt_align', action='store_true', help='get Text-Image alignment VLM Score? (default: False)')
    parser.add_argument('--diversity', action='store_true', help='get Generative Image Inception Score? (default: False)')

    parser.add_argument('--language', type=str, choices=['en', 'ch'],nargs='+', default='en', help='Language option: en (English) or ch (Chinese)')
    parser.add_argument('--method', type=str, nargs='+', required=True, 
                        help='Method option (accepts one or multiple values)')
                        # choices=ALL_METHODS, help='Method option')
                        # choices=ALL_METHODS+['STORY_IMG','STORY_VIDEO','CLOSED_SOURCE','BUSINESS','NAIVE'], help='Method option')
    parser.add_argument('--timestamp', type=str, default=None,nargs='+', help='Timestamp for the output files. %Y%m%d_%H%M%S')
    parser.add_argument('--mode', type=str, default=None, nargs='+',help='Mode for method.')
    parser.add_argument('--last_timestamp', type=bool, default=False, help='Only use the last timestamp for the output files.')

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

    # -------------------------------------------------------------------------------------


    # if args_method in ['STORY_IMG','STORY_VIDEO','CLOSED_SOURCE','BUSINESS','NAIVE']:
    #     method_list = METHODS_DICT[args_method]
    # elif args_method in ALL_METHODS:
    #     method_list = [args_method]

    method_list = args.method
    print(f'method_list:{method_list}')

    dataset_path = args.dataset_path
    outputs_path = args.outputs_path
    pretrain_path = args.pretrain_path
    result_path = args.result_path

    print(f'dataset_path:{dataset_path}')
    print(f'outputs_path:{outputs_path}')
    print(f'pretrain_path:{pretrain_path}')


    for method in method_list:

        dataset_process_path = f'{code_path}/data_process/dataset_process'
        sys.path.append(dataset_process_path)

        dataset_name = 'ViStory'
        language=args.language
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        BASE_MODE_CATEGORIES = {
            'uno',
            'seedstory',
            'theatergen',
            'mmstoryagent',
            *CLOSED_SOURCE,
            *BUSINESS,
            *NAIVE
        }

        # Image
        if method == 'storygen':
            model_mode = args.model_mode
            assert model_mode in ['auto-regressive', 'multi-image-condition', 'mix'], f"Invalid model_mode: {model_mode}"
            mode_name_for_oss = model_mode
        elif method == 'storydiffusion':
            content_mode = args.content_mode
            assert content_mode in ['original', 'Photomaker'], f"Invalid content_mode: {content_mode}"
            style_mode = args.style_mode
            mode_name_for_oss = content_mode
        elif method == 'storyadapter':
            content_mode = args.content_mode
            assert content_mode in ['img_ref', 'text_only'], f"Invalid content_mode: {content_mode}"
            scale_stage = args.scale_stage
            mode_name_for_oss = f'{content_mode}_{scale_stage}'
        # Video
        elif method == 'movieagent':
            model_type = args.model_type
            assert model_type in ['ROICtrl', 'SD-3'], f"Invalid model_type: {model_type}"
            mode_name_for_oss = model_type
        elif method == 'animdirector':
            model_type = args.model_type
            assert model_type in ['sd3'], f"Invalid model_type: {model_type}"
            mode_name_for_oss = model_type
        elif method == 'vlogger':
            content_mode = args.content_mode
            assert content_mode in ['img_ref', 'text_only'], f"Invalid content_mode: {content_mode}"
            mode_name_for_oss = content_mode
        # Other
        elif method in BASE_MODE_CATEGORIES:
            mode_name_for_oss = 'base'
        else:
            mode_name_for_oss = 'base'
        
        stories_data = load_dataset(dataset_path, dataset_name, language)
        stories_outputs = load_outputs(outputs_root=outputs_path, methods=method,languages=language,modes=args.mode,return_latest=args.last_timestamp,timestamps=args.timestamp)

        enable_cids_score = args.cids
        enable_cross_csd_score = args.csd_cross
        enable_self_csd_score = args.csd_self
        enable_aesthetic_score = args.aesthetic
        enable_prompt_align_score = args.prompt_align
        enable_inception_score = args.diversity
        enable_restore = args.save_format

        # model_init:
        if enable_cross_csd_score or enable_self_csd_score:
            from bench.style.csd.csd_test import get_csd_score, load_csd
            CSD_WEIGHTS = f'{pretrain_path}/csd/csd_vit-large.pth'
            csd_encoder = load_csd(CSD_WEIGHTS).cuda()
        if enable_aesthetic_score:
            from bench.quality.aesthetic.aesthetic_metric import Aesthetic_Metric
            aesthetic_metric = Aesthetic_Metric(rank=0, device="cuda", pretrain_path=pretrain_path)
        if enable_cids_score or enable_prompt_align_score:
            from transformers import CLIPProcessor, CLIPModel
            # from groundingdino.util.inference import load_model, load_image, predict, annotate
            from bench.content.groundingdino_util import load_model
            # DINO_MODEL = "bench/content/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
            DINO_MODEL = 'bench/content/GroundingDINO_SwinT_OGC.py'
            DINO_WEIGHTS = f"{pretrain_path}/groundingdino/weights/groundingdino_swint_ogc.pth"
            # load models
            clip_version = "clip-vit-large-patch14" # clip-vit-base-patch16
            clip_model = CLIPModel.from_pretrained(f"{pretrain_path}/openai/{clip_version}")
            clip_processor = CLIPProcessor.from_pretrained(f"{pretrain_path}/openai/{clip_version}")
            dino_model = load_model(DINO_MODEL, DINO_WEIGHTS, text_encoder_type = f"{pretrain_path}/google-bert/bert-base-uncased")
            model_pkg = (clip_model, clip_processor, dino_model)

            gpt_api_pkg = (args.model_id, args.api_key, args.base_url)


        for (dataset_story_name, story_data),(outputs_story_name, outputs_data) in zip(stories_data.items(), stories_outputs.items()):
            
            # assert dataset_story_name == outputs_story_name, f"story_name mismatch：{dataset_story_name} != {outputs_story_name}"
            if dataset_story_name != outputs_story_name:
                yellow_print(f"Story number in generated results mismatch that in full source dataset: story_name mismatch: {dataset_story_name} != {outputs_story_name}")

        for outputs_story_name, outputs_data in stories_outputs.items():
            for dataset_story_name, story_data in stories_data.items():

                if (dataset_story_name == outputs_story_name):

                    story_name = dataset_story_name

                    blue_print(f'''
                        Current method: {method}
                        Current story name: {story_name}
                        Shot number of current story in source dataset: {len(story_data['shots'])}
                        Shot number of current story in generated results: {len(outputs_data['shots'])}
                        Character number of current story in generated results: {len(outputs_data['chars'])}
                        ''')
                    
                    assert len(story_data['shots']) == len(outputs_data['shots']), f"Shot number of current story in generated results mismatch that in source dataset: {len(story_data['shots'])} != {len(outputs_data['shots'])}"

                    if len(story_data['shots']) != len(outputs_data['shots']):
                        yellow_print(f"Shot number of current story in generated results mismatch that in source dataset: {len(story_data['shots'])} != {len(outputs_data['shots'])}")

                    if enable_cids_score:
                        green_print(f'[Cross and self CIDS Score] Calculating the character consistency')
                        story_data_path = f"{dataset_path}/{dataset_name}/{story_name}"
                        for ref_mode in ['origin']: 
                        # for ref_mode in ['origin','mid-gen']:
                            get_cids_score_and_save(model_pkg, pretrain_path, story_data_path, outputs_data, 'cids_score', ref_mode) 

                    if enable_cross_csd_score:
                        green_print(f'[Cross-CSD Score] Calculating the style similarity between the reference and generated images...')
                        cross_csd_shot_details, avg_cross_csd = get_cross_csd_score(csd_encoder, story_data, outputs_data)
                        save_csd_to_excel_and_json(cross_csd_shot_details, avg_cross_csd, method, language, result_path, 'cross_csd')

                    if enable_self_csd_score:
                        green_print(f'[Self-CSD Score] Calculating the style similarity between the generated images...')
                        self_csd_shot_details, avg_self_csd = get_self_csd_score(csd_encoder, outputs_data['shots'])
                        save_csd_to_excel_and_json(self_csd_shot_details, avg_self_csd, method, language, result_path, 'self_csd')

                    if enable_aesthetic_score: 
                        green_print(f'[Aesthetic Score] Calculating the aesthetic score of the generated image...')
                        aesthetic_shot_details, avg_aesthetic = get_aesthetic_score(aesthetic_metric, outputs_data['shots'])
                        save_aesthetic_to_excel_and_json(aesthetic_shot_details, avg_aesthetic, method, language, result_path, 'aesthetic_score')

                    if enable_prompt_align_score: 
                        green_print(f'[Prompt Align and OCCM] Calculating the VLM score for image-prompt alignment...')
                        get_prompt_align_score_and_save(story_data, outputs_data)
                        # save_prompt_align_to_excel(prompt_align_shot_details, avg_prompt_align, method, language, outputs_path, 'vlm_score')
                
                else:
                    continue
        

        if enable_inception_score:

            green_print(f'[Inception Score] Calculating the inception score of the generated images...')

            labels = ['Full', 'Lite', 'Real', 'UnReal', 'Custom']
            for label in labels:
                CHOICE_DATASET = DATA_DICT[f'{label}']

                inception_score_path = f'{code_path}/bench/diversity/'
                sys.path.append(inception_score_path)
                from bench.diversity.inception_score import get_inception_score_and_save

                print(f'Current Method：{method}')
                save_dir= f"{outputs_path}/{method}/bench_results/inception_score/{timestamp}"

                story_case = stories_outputs[f'{story_name}']['shots']
                print(f'story_case:{story_case}')
                filename_base = save_filename_extra(story_case, method, 'inception_score')

                get_inception_score_and_save(
                    method, stories_outputs, data_path,
                    CHOICE_DATASET, label,
                    save_dir, filename_base)
                