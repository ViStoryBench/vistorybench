import os
import re
from datetime import datetime
from natsort import natsorted

def find_all_image_paths(method, base_path, datetime_pattern, image_extensions, 
                         scale_stage_for_storyadapter=None,
                         style_mode_for_storydiffusion=None,
                         model_type_for_movieagent=None,
                         content_mode_for_vlogger=None,
                         model_mode_for_storygen=None
                         ):
    
    stories_image_paths = {}

    # Get all subdirectories under base_path as group_id
    all_groups_id = [d for d in os.listdir(base_path) 
                    if os.path.isdir(os.path.join(base_path, d))]
    all_groups_id.sort(key=lambda x: int(x))

    # Iterate through all existing group directories
    print(f'all_groups_id: {all_groups_id}') # 01 02 ... 80
    for group_id in all_groups_id:

        stories_image_paths[group_id] = {}

        shot_image_paths = []
        char_image_paths = []

        if style_mode_for_storydiffusion:
            group_dir = os.path.join(base_path, group_id, style_mode_for_storydiffusion)
        else:
            group_dir = os.path.join(base_path, group_id)
        
        # Get all datetime subdirectories
        datetime_dirs = [d for d in os.listdir(group_dir) 
                        if os.path.isdir(os.path.join(group_dir, d)) 
                        and datetime_pattern.match(d)]
        datetime_dirs.sort(key=lambda x: datetime.strptime(x.replace('_', '-'), "%Y%m%d-%H%M%S"))
        # Iterate through each datetime directory
        datetime_dirs = [datetime_dirs[-1]] # Only take the last directory
        # print(f'datetime_dirs: {datetime_dirs}')
        for datetime_dir in datetime_dirs:
            if scale_stage_for_storyadapter:
                datetime_path = os.path.join(group_dir, datetime_dir, scale_stage_for_storyadapter)
            elif method == 'vlogger': 
                datetime_path = os.path.join(group_dir, datetime_dir, 'first_frames')
                if not os.path.exists(datetime_path):
                    from video2image import process_videos_in_folder
                    input_folder = f"{os.path.join(group_dir, datetime_dir)}/video/origin_video"
                    output_folder = f"{os.path.join(group_dir, datetime_dir)}/first_frames"
                    process_videos_in_folder(input_folder, output_folder)
            else:
                datetime_path = os.path.join(group_dir, datetime_dir)
            # Collect all image files
            for filename in natsorted(os.listdir(datetime_path)):
                file_path = os.path.join(datetime_path, filename)

                if method == 'seedstory':
                    if (os.path.isfile(file_path) and 
                        filename.lower().endswith(image_extensions) and 
                        # New filename prefix condition
                        (
                            # filename.startswith('00') or  # First two digits are 00
                            filename[:3].lower() == 'ori')): # First three characters are 'ori' (case insensitive)
                        shot_image_paths.append(file_path)

                elif model_type_for_movieagent == 'ROICtrl':
                    if (os.path.isfile(file_path) and 
                        filename.lower().endswith(image_extensions) and
                        not filename.lower().endswith('_vis.jpg')):
                        shot_image_paths.append(file_path)
                        
                elif method == 'vlogger':
                    if (os.path.isfile(file_path) and 
                        filename.lower().endswith(image_extensions) and
                        not filename.lower().endswith('result.jpg')):
                        shot_image_paths.append(file_path)

                elif method == 'storydiffusion':
                    if (os.path.isfile(file_path) and 
                        filename.lower().endswith(image_extensions) and
                        not filename.lower().endswith('_0.png')):
                        shot_image_paths.append(file_path)

                else:
                    if (os.path.isfile(file_path) and 
                        filename.lower().endswith(image_extensions)):
                        shot_image_paths.append(file_path)

        # stories_image_paths[group_id] = natsorted(shot_image_paths)  # Use actual directory name as key
        stories_image_paths[group_id]['shots'] = natsorted(shot_image_paths)
        stories_image_paths[group_id]['chars'] = char_image_paths  # Use actual directory name as key
        
    return stories_image_paths, all_groups_id




def find_all_image_paths_without_timestamp(method, base_path, image_extensions):
    
    stories_image_paths = {}

    # Get all subdirectories under base_path as group_id
    all_groups_id = [d for d in os.listdir(base_path) 
                    if os.path.isdir(os.path.join(base_path, d))]
    all_groups_id.sort(key=lambda x: int(x))

    # Iterate through all existing group directories
    print(f'all_groups_id: {all_groups_id}')  # 01 02 ... 80
    for group_id in all_groups_id:

        stories_image_paths[group_id] = {}

        shot_image_paths = []  # Initialize each group separately
        char_image_paths = []

        group_dir = os.path.join(base_path, group_id)
        shot_dir = group_dir
        
        # Collect image files in current shot directory
        for filename in natsorted(os.listdir(shot_dir)):
            file_path = os.path.join(shot_dir, filename)

            if (os.path.isfile(file_path) and 
                filename.lower().endswith(image_extensions)):
                shot_image_paths.append(file_path)

        # stories_image_paths[group_id] = shot_image_paths  # Use actual directory name as key
        stories_image_paths[group_id]['shots'] = shot_image_paths  # Use actual directory name as key
        stories_image_paths[group_id]['chars'] = char_image_paths  # Use actual directory name as key

    return stories_image_paths, all_groups_id





def find_all_image_paths_for_business(method, base_path, image_extensions):
    
    stories_image_paths = {}

    # Get all subdirectories under base_path as group_id
    all_groups_id = [d for d in os.listdir(base_path) 
                    if os.path.isdir(os.path.join(base_path, d))]
    all_groups_id.sort(key=lambda x: int(x))

    # Iterate through all existing group directories
    print(f'all_groups_id: {all_groups_id}')  # 01 02 ... 80
    for group_id in all_groups_id:

        stories_image_paths[group_id] = {}

        shot_image_paths = []  # Initialize each group separately
        char_image_paths = []

        group_dir = os.path.join(base_path, group_id)
        
        # Find subdirectories that may contain shots or character images.
        # Some projects name these folders in English ('shot(s)', 'chars') or Chinese ('分镜', '角色').
        shot_dir = None
        char_dir = None
        for d in os.listdir(group_dir):
            dir_path = os.path.join(group_dir, d)
            if os.path.isdir(dir_path) and d.endswith('分镜'):
                shot_dir = dir_path
            elif os.path.isdir(dir_path) and d.endswith('角色'):
                char_dir = dir_path
            # if os.path.isdir(dir_path) and d.endswith('20250500_000000'):
            #     shot_dir = dir_path

        # Check if shot directory exists
        if not shot_dir:
            print(f"Warning: Shot directory does not exist in group directory - {group_dir}")
            stories_image_paths[group_id] = []  # Add empty list
            continue

        if not char_dir:
            print(f"Warning: Shot directory does not exist in group directory - {group_dir}")
            stories_image_paths[group_id]['chars'] = []  # Add empty list
            continue

        # if method in ("moki", "xunfeihuiying"):
        #     name_suffix = "_1"
        # elif method == "morphic_studio":
        #     name_suffix = "-1"
        # else:
        #     name_suffix = ""

        if method in ("moki", "xunfeihuiying", "morphic_studio"):
            def suffix_check(base_name):
                return base_name.endswith("_1") or base_name.endswith("-1")
        else:
            def suffix_check(_):
                return True
            
        # Collect image files in current shot directory
        for filename in natsorted(os.listdir(shot_dir)):
            file_path = os.path.join(shot_dir, filename)

            if os.path.isfile(file_path):
                base, ext = os.path.splitext(filename)
                ext = ext.lower()
                
                if ext in image_extensions and suffix_check(base):
                    shot_image_paths.append(file_path)


        # Collect image files in current character directory
        for filename in natsorted(os.listdir(char_dir)):
            file_path = os.path.join(char_dir, filename)

            if os.path.isfile(file_path):
                char_image_paths.append(file_path)


            # if (os.path.isfile(file_path) and 
            #     filename.lower().endswith(image_extensions)):
            #     shot_image_paths.append(file_path)

        stories_image_paths[group_id]['shots'] = shot_image_paths  # Use actual directory name as key
        stories_image_paths[group_id]['chars'] = char_image_paths  # Use actual directory name as key
    
    return stories_image_paths, all_groups_id