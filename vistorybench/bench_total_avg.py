

import os
import json
from statistics import mean
import re
import argparse

import yaml
from pathlib import Path


def blue_print(text, bright=True):
    color_code = "\033[94m" if bright else "\033[34m"
    print(f"{color_code}{text}\033[0m")

def yellow_print(text):
    print(f"\033[93m{text}\033[0m")

def green_print(text):
    print(f"\033[92m{text}\033[0m")

def aesthetic_total_avg(path):

    def extract_story_name(filename):
        """
        Precisely extract story number (such as "01"), ignoring timestamps and other numbers.
        Matches the following patterns:
        - ViStory_en_<number>
        - Number followed by _, . or timestamp
        """
        match = re.search(r'ViStory_en_(?:lite_)?(\d{2})(?=[._])', filename)
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

    # Create save directory (if it doesn't exist)
    os.makedirs(save_path, exist_ok=True)


    all_avg = {
        "score": [],
    }
    total_avg = {
        "total_avg": None,
    }


    # Traverse JSON files and extract average_score
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


    # Calculate total average
    total_avg['total_avg'] = mean(all_avg['score']) if all_avg else 0.0

    # Save results
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
        Precisely extract story number (such as "01"), ignoring timestamps and other numbers.
        Matches the following patterns:
        - ViStory_en_<number>
        - Number followed by _, . or timestamp
        """
        match = re.search(r'ViStory_en_(?:lite_)?(\d{2})(?=[._])', filename)
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

    # Create save directory (if it doesn't exist)
    os.makedirs(save_path, exist_ok=True)

    all_avg = {
        "score": [],
    }
    total_avg = {
        "total_avg": None,
    }

    # Traverse JSON files and extract average_score
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

    # Calculate total average
    total_avg['total_avg'] = mean(all_avg['score']) if all_avg else 0.0

    # Save results
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


def cids_total_avg(path):

    print(f'get cids_total_avg ......')

    timestamp = get_timestamp_name(path)
    path = f'{path}/{timestamp}'

    save_path = os.path.join(path, 'total_avg')
    
    # Create save directory (if it doesn't exist)
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

    # Traverse subdirectories from 01 to 80
    for story_name in CHOICE_DATASET:

        if method in ['storyadapter_img_ref_results_xl', 'storyadapter_text_only_results_xl']:
            json_dir = os.path.join(path, story_name, 'results_xl','json')
        elif method in ['storyadapter_img_ref_results_xl5', 'storyadapter_text_only_results_xl5']:
            json_dir = os.path.join(path, story_name, 'results_xl5','json')
        else:
            json_dir = os.path.join(path, story_name, 'json')
        
        # Check if JSON directory exists
        if not os.path.exists(json_dir):
            print(f"Directory {json_dir} does not exist, skipping.")
            continue
        
        # Traverse all JSON files in JSON directory
        for filename in os.listdir(json_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(json_dir, filename)

                with open(filepath, 'r') as f:
                    data = json.load(f)
                    cids_data = data.get('cids', {})
                    copy_paste_score = data.get('copy-paste-score')

                    if isinstance(copy_paste_score, (int, float)) and copy_paste_score != "null":
                        all_avg['copy-paste-score'].append(float(copy_paste_score))

                    for char, score in cids_data.items():
                        cross = score.get('cross')
                        self = score.get('self')  # Avoid conflict with keyword self

                        if cross is not None:
                            all_avg['cross'].append(float(cross))
                        if self is not None:
                            all_avg['self'].append(float(self))
                    
    # Calculate total average
    total_avg['total_cross_avg'] = mean(all_avg['cross']) if all_avg else 0.0
    total_avg['total_self_avg'] = mean(all_avg['self']) if all_avg else 0.0
    total_avg['total_avg_copy-paste-score'] = mean(all_avg['copy-paste-score']) if all_avg else 0.0
            
    # Save results
    result = {
        "check_timestamp": timestamp,
        "all_average_cids_scores": all_avg,
        "total_average_cids_score": total_avg,
        "processed_files_count(cross)": len(all_avg['cross']),
        "processed_files_count(self)": len(all_avg['self'])
    }
    output_file = os.path.join(save_path, 'total_average_cids.json')
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=4)

    print(f"Total average cids score calculated: cross is {total_avg['total_cross_avg']:.4f} and self is {total_avg['total_self_avg']:.4f}")
    print(f"Result saved to: {output_file}")

    return result


def prompt_align_total_avg(path):

    print(f'get prompt_align_total_avg ......')

    timestamp = get_timestamp_name(path)
    path = f'{path}/{timestamp}'

    save_path = os.path.join(path, 'total_avg')
    
    # Create save directory (if it doesn't exist)
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

    # Traverse subdirectories from 01 to 80
    for story_name in CHOICE_DATASET:

        json_dir = os.path.join(path, story_name)
    
        # Check if JSON directory exists
        if not os.path.exists(json_dir):
            print(f"Directory {json_dir} does not exist, skipping.")
            continue
        
        # Traverse all JSON files in JSON directory
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

    # Calculate total average
    for score_name, score_list in all_avg.items():
        total_avg[f'total_avg_{score_name}'] = mean(score_list) if all_avg[f'{score_name}'] else 0.0

    # Save results
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

    # Directly traverse target path (no subdirectories needed)
    for filename in os.listdir(path):
        if filename.endswith('.json'):
            filepath = os.path.join(path, filename)

            with open(filepath, 'r') as f:
                data = json.load(f)
                # Deep extract inception_score
                score = data.get('aggregate_scores', {})\
                                .get('generated_diversity', {})\
                                .get('inception_score')
                if score is not None:
                    inception_scores.append(float(score))

    # Calculate average (actually only one score)
    total_avg = sum(inception_scores) / len(inception_scores) if inception_scores else 0.0

    # Save results
    result = {
        "check_timestamp": timestamp,
        "inception_scores": inception_scores,
        "total_average_inception_score": total_avg,
        "processed_files_count": len(inception_scores)
    }

    return result


def get_timestamp_name(path):

    # Get all datetime subdirectories
    import re
    from datetime import datetime
    datetime_pattern = re.compile(r'^\d{8}[_-]\d{6}$')
    datetime_dirs = [d for d in os.listdir(path) 
                    if os.path.isdir(os.path.join(path, d)) 
                    and datetime_pattern.match(d)]
    datetime_dirs.sort(key=lambda x: datetime.strptime(x.replace('_', '-'), "%Y%m%d-%H%M%S"))
    # Traverse each datetime directory
    datetime_dirs = datetime_dirs[-1]  # Only take the last directory

    print(f'timestamp:{datetime_dirs}')

    return datetime_dirs


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}  # Return empty dict if file not found


if __name__ == "__main__":

    # //////////////////// Pre-define method categories ////////////////////
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

    parser.add_argument('--dataset_path', type=str, default=f'{data_path}/dataset', help='Directory for datasets')
    parser.add_argument('--outputs_path', type=str, default=f'{data_path}/outputs', help='Directory for generated outputs')
    parser.add_argument('--avg_score_path', type=str, default=f'{data_path}/avg_score', help='Directory for all method avg score')
    parser.add_argument('--pretrain_path', type=str, default=f'{data_path}/pretrain', help='Directory for pretrained weights')

    parser.add_argument('--full', action='store_true', help='Average scores on Full Dataset')
    parser.add_argument('--lite', action='store_true', help='Average scores on Lite Dataset')
    parser.add_argument('--real', action='store_true', help='Average scores on Real Dataset')
    parser.add_argument('--unreal', action='store_true', help='Average scores on UnReal Dataset')
    parser.add_argument('--custom', action='store_true', help='Average scores on Custom Dataset')

    parser.add_argument('--method', type=str, nargs='+', required=True, 
                        help='Method option (accepts one or multiple values)')

    args = parser.parse_args()

    # -------------------------------------------------------------------------------------

    # method_list = ['moki']
    # method_list = BUSINESS
    method_list = args.method
    print(f'method_list:{method_list}')

    dataset_path = args.dataset_path
    outputs_path = args.outputs_path
    pretrain_path = args.pretrain_path
    avg_score_path = args.avg_score_path

    print(f'dataset_path:{dataset_path}')
    print(f'outputs_path:{outputs_path}')
    print(f'pretrain_path:{pretrain_path}')
    print(f'avg_score_path:{avg_score_path}')

    dataset_label_list = []
    if args.full:
        dataset_label_list.append('Full')
    if args.lite:
        dataset_label_list.append('Lite')
    if args.real:
        dataset_label_list.append('Real')
    if args.unreal:
        dataset_label_list.append('UnReal')
    if args.custom:
        dataset_label_list.append('Custom')

    print(f'dataset_label_list:{dataset_label_list}')

    for dataset_label in dataset_label_list:

        CHOICE_DATASET = DATA_DICT[f'{dataset_label}'] # Select the dataset you need to count
        blue_print(f'CHOICE_DATASET:{CHOICE_DATASET}')

        for method in method_list:

            blue_print(f'Current method: {method}')

            path = {
                "cids": f'{outputs_path}/{method}/bench_results/cids_score', 
                "cids_mid": f'{outputs_path}/{method}/bench_results/cids_score_mid-gen',
                "self_csd": f'{outputs_path}/{method}/bench_results/self_csd', 
                "cross_csd": f'{outputs_path}/{method}/bench_results/cross_csd',
                "aesthetic": f'{outputs_path}/{method}/bench_results/aesthetic_score', 
                "inception_score": f'{outputs_path}/{method}/bench_results/inception_score', 
                "prompt_align": f'{outputs_path}/{method}/bench_results/prompt_align'
            }

            self_csd = cross_csd = aesthetic = cids = cids_mid = prompt_align = inception_score = None

            if os.path.exists(path['self_csd']):
                self_csd = csd_total_avg(path['self_csd'], mode='self')
            if os.path.exists(path['cross_csd']):
                cross_csd = csd_total_avg(path['cross_csd'], mode='cross')
            if os.path.exists(path['aesthetic']):
                aesthetic = aesthetic_total_avg(path['aesthetic'])
            if os.path.exists(path['cids']):
                cids = cids_total_avg(path['cids'])
            if os.path.exists(path['cids_mid']):
                cids_mid = cids_total_avg(path['cids_mid'])
            if os.path.exists(path['prompt_align']):
                prompt_align = prompt_align_total_avg(path['prompt_align'])
            if os.path.exists(path['inception_score']):
                inception_score = inception_score_total_avg(path['inception_score'])

            BIG_TABLE = {
                "method": method,
                "aesthetic": aesthetic,
                "self_csd": self_csd,
                "cross_csd": cross_csd,
                "cids": cids,
                "cids_mid": cids_mid,
                "prompt_align": prompt_align,
                "inception_score":inception_score
            }

            os.makedirs(avg_score_path,exist_ok=True)
            output_file = os.path.join(avg_score_path, f'total_avg_of_{method}_in_{dataset_label}.json')
            with open(output_file, 'w') as f:
                json.dump(BIG_TABLE, f, indent=4)
            green_print(f'All save in {output_file}')
                
            # path = 'data/outputs/uno/bench_results/cids_score/20250509_000437'
            # cids_total_avg(path)

            # path = 'data/outputs/uno/bench_results/prompt_align_history/20250512_151300'
            # prompt_align_total_avg(path)

