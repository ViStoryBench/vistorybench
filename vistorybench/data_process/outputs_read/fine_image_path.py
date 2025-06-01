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
    
    story_image_paths = {}

    # 获取base_path下所有子目录作为group_id
    all_groups_id = [d for d in os.listdir(base_path) 
                    if os.path.isdir(os.path.join(base_path, d))]
    all_groups_id.sort(key=lambda x: int(x))

    # 遍历所有存在的组目录
    print(f'all_groups_id: {all_groups_id}') # 01 02 ... 80
    for group_id in all_groups_id:
        shot_image_paths = []  # 每个组单独初始化
        if style_mode_for_storydiffusion:
            group_dir = os.path.join(base_path, group_id, style_mode_for_storydiffusion)
        else:
            group_dir = os.path.join(base_path, group_id)
        
        # 获取所有日期时间子目录
        datetime_dirs = [d for d in os.listdir(group_dir) 
                        if os.path.isdir(os.path.join(group_dir, d)) 
                        and datetime_pattern.match(d)]
        datetime_dirs.sort(key=lambda x: datetime.strptime(x.replace('_', '-'), "%Y%m%d-%H%M%S"))
        # 遍历每个日期时间目录
        datetime_dirs = [datetime_dirs[-1]]  # 只取最后一个目录
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
            # 收集所有图片文件
            for filename in natsorted(os.listdir(datetime_path)):
                file_path = os.path.join(datetime_path, filename)

                if method == 'seedstory':
                    if (os.path.isfile(file_path) and 
                        filename.lower().endswith(image_extensions) and 
                        # 新增文件名前缀条件
                        (
                            # filename.startswith('00') or  # 前两位是00
                            filename[:3].lower() == 'ori')):  # 前三位是ori（不区分大小写）
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

        story_image_paths[group_id] = natsorted(shot_image_paths)  # 使用实际目录名作为key
    
    return story_image_paths, all_groups_id




def find_all_image_paths_without_timestamp(method, base_path, image_extensions):
    
    story_image_paths = {}

    # 获取base_path下所有子目录作为group_id
    all_groups_id = [d for d in os.listdir(base_path) 
                    if os.path.isdir(os.path.join(base_path, d))]
    all_groups_id.sort(key=lambda x: int(x))

    # 遍历所有存在的组目录
    print(f'all_groups_id: {all_groups_id}')  # 01 02 ... 80
    for group_id in all_groups_id:
        shot_image_paths = []  # 每个组单独初始化

        group_dir = os.path.join(base_path, group_id)
        shot_dir = group_dir
        
        # 收集当前分镜目录下的图片文件
        for filename in natsorted(os.listdir(shot_dir)):
            file_path = os.path.join(shot_dir, filename)

            if (os.path.isfile(file_path) and 
                filename.lower().endswith(image_extensions)):
                shot_image_paths.append(file_path)

        story_image_paths[group_id] = shot_image_paths  # 使用实际目录名作为key
    
    return story_image_paths, all_groups_id





def find_all_image_paths_for_business(method, base_path, image_extensions):
    
    story_image_paths = {}

    # 获取base_path下所有子目录作为group_id
    all_groups_id = [d for d in os.listdir(base_path) 
                    if os.path.isdir(os.path.join(base_path, d))]
    all_groups_id.sort(key=lambda x: int(x))

    # 遍历所有存在的组目录
    print(f'all_groups_id: {all_groups_id}')  # 01 02 ... 80
    for group_id in all_groups_id:
        shot_image_paths = []  # 每个组单独初始化

        group_dir = os.path.join(base_path, group_id)
        
        # 查找所有以"分镜"结尾的子目录
        shot_dir = None
        for d in os.listdir(group_dir):
            dir_path = os.path.join(group_dir, d)
            if os.path.isdir(dir_path) and d.endswith('分镜'):
                shot_dir = dir_path
        
        # 检查是否存在分镜目录
        if not shot_dir:
            print(f"Warning: 分镜目录不存在于组目录 - {group_dir}")
            story_image_paths[group_id] = []  # 添加空列表
            continue
            
        # if method in ("moki", "xunfeihuiying"):
        #     name_suffix = "_1"
        # elif method == "morphic_studio":
        #     name_suffix = "-1"
        # else:
        #     name_suffix = ""

        if method in ("moki", "xunfeihuiying", "morphic_studio"):
            # 双重后缀验证闭包
            def suffix_check(base_name):
                return base_name.endswith("_1") or base_name.endswith("-1")
        else:
            # 其他业务不验证后缀
            def suffix_check(_):
                return True
            
        # 收集当前分镜目录下的图片文件
        for filename in natsorted(os.listdir(shot_dir)):
            file_path = os.path.join(shot_dir, filename)

            if os.path.isfile(file_path):
                base, ext = os.path.splitext(filename)
                ext = ext.lower()
                
                # 双重验证：扩展名匹配 + 文件名后缀匹配
                if ext in image_extensions and suffix_check(base):
                    shot_image_paths.append(file_path)

            # if (os.path.isfile(file_path) and 
            #     filename.lower().endswith(image_extensions)):
            #     shot_image_paths.append(file_path)

        story_image_paths[group_id] = shot_image_paths  # 使用实际目录名作为key
    
    return story_image_paths, all_groups_id