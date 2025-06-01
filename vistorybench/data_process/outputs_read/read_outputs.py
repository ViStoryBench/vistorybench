import os
import re
from datetime import datetime
from fine_image_path import find_all_image_paths, find_all_image_paths_for_business, find_all_image_paths_without_timestamp

# ///////////////////// Image /////////////////////
def read_uno_outputs(
        data_path= '/data/AIGC_Research/Story_Telling/StoryVisBMK/data', 
        method = 'uno', 
        dataset_name = 'WildStory', 
        language = 'en',
        **kwargs
    ):
    
    base_path = f"{data_path}/outputs/{method}/{dataset_name}_{language}"
    datetime_pattern = re.compile(r'^\d{8}-\d{6}$')  # 匹配日期时间目录格式
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')  # 支持的图片格式

    # 批量读取输出数据
    stories_image_paths, all_groups_id = find_all_image_paths(
        method,
        base_path, 
        datetime_pattern, 
        image_extensions
        )
    print(f"共找到 {len(stories_image_paths)} 组图片")
    
    for group_id, (story_id, image_paths) in zip(all_groups_id, stories_image_paths.items()):
        assert group_id == story_id, f"组ID不匹配：{group_id} != {story_id}"
    
    return stories_image_paths

def read_storygen_outputs(
        data_path= '/data/AIGC_Research/Story_Telling/StoryVisBMK/data', 
        method = 'storygen', 
        dataset_name = 'WildStory', 
        language = 'en',
        model_mode = 'mix',  # 'auto-regressive' or 'multi-image-condition' or 'mix'
        **kwargs
    ):
    
    base_path = f"{data_path}/outputs/{method}/{model_mode}/{dataset_name}_{language}"
    datetime_pattern = re.compile(r'^\d{8}-\d{6}$')  # 匹配日期时间目录格式
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')  # 支持的图片格式

    # 批量读取输出数据
    stories_image_paths, all_groups_id = find_all_image_paths(
        method,
        base_path, 
        datetime_pattern, 
        image_extensions,
        model_mode_for_storygen = model_mode
        )
    print(f"共找到 {len(stories_image_paths)} 组图片")
    
    for group_id, (story_id, image_paths) in zip(all_groups_id, stories_image_paths.items()):
        assert group_id == story_id, f"组ID不匹配：{group_id} != {story_id}"
    
    return stories_image_paths

def read_seedstory_outputs(
        data_path= '/data/AIGC_Research/Story_Telling/StoryVisBMK/data', 
        method = 'seedstory', 
        dataset_name = 'WildStory', 
        language = 'en',
        **kwargs
    ):
    
    base_path = f"{data_path}/outputs/{method}/{dataset_name}_{language}"
    datetime_pattern = re.compile(r'^\d{8}-\d{6}$')  # 匹配日期时间目录格式
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')  # 支持的图片格式

    # 批量读取输出数据
    stories_image_paths, all_groups_id = find_all_image_paths(
        method,
        base_path, 
        datetime_pattern, 
        image_extensions
        )
    print(f"共找到 {len(stories_image_paths)} 组图片")
    
    for group_id, (story_id, image_paths) in zip(all_groups_id, stories_image_paths.items()):
        assert group_id == story_id, f"组ID不匹配：{group_id} != {story_id}"
    
    return stories_image_paths

def read_storydiffusion_outputs(
        data_path= '/data/AIGC_Research/Story_Telling/StoryVisBMK/data', 
        method = 'storydiffusion', 
        dataset_name = 'WildStory', 
        language = 'en',
        content_mode = 'original',  # 'original' or 'Photomaker'
        style_mode = '(No style)',
        **kwargs
    ):
    
    base_path = f"{data_path}/outputs/{method}/{content_mode}/{dataset_name}_{language}"
    datetime_pattern = re.compile(r'^\d{8}-\d{6}$')  # 匹配日期时间目录格式
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')  # 支持的图片格式

    # 批量读取输出数据
    stories_image_paths, all_groups_id = find_all_image_paths(
        method,
        base_path, 
        datetime_pattern, 
        image_extensions,
        style_mode_for_storydiffusion = style_mode
        )
    print(f"共找到 {len(stories_image_paths)} 组图片")
    
    for group_id, (story_id, image_paths) in zip(all_groups_id, stories_image_paths.items()):
        assert group_id == story_id, f"组ID不匹配：{group_id} != {story_id}"
    
    return stories_image_paths

def read_storyadapter_outputs(
        data_path= '/data/AIGC_Research/Story_Telling/StoryVisBMK/data', 
        method = 'storyadapter', 
        dataset_name = 'WildStory', 
        language = 'en',
        content_mode = 'img_ref',  # 'text_only' or 'img_ref'
        scale_stage = 'results_xl5',
        **kwargs
    ):
    
    # 第一段路径
    base_path = f"{data_path}/outputs/{method}/{content_mode}/{dataset_name}_{language}"
    datetime_pattern = re.compile(r'^\d{8}-\d{6}$')  # 匹配日期时间目录格式
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')  # 支持的图片格式

    # 批量读取输出数据
    stories_image_paths, all_groups_id = find_all_image_paths(
        method,
        base_path, 
        datetime_pattern, 
        image_extensions,
        scale_stage_for_storyadapter = scale_stage
        )
    print(f"共找到 {len(stories_image_paths)} 组图片")
    
    for group_id, (story_id, image_paths) in zip(all_groups_id, stories_image_paths.items()):
        assert group_id == story_id, f"组ID不匹配：{group_id} != {story_id}"
    
    return stories_image_paths

def read_theatergen_outputs(
        data_path= '/data/AIGC_Research/Story_Telling/StoryVisBMK/data', 
        method = 'theatergen', 
        dataset_name = 'WildStory', 
        language = 'en',
        **kwargs
    ):
    
    base_path = f"{data_path}/outputs/{method}/{dataset_name}_{language}"
    datetime_pattern = re.compile(r'^\d{8}-\d{6}$')  # 匹配日期时间目录格式
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')  # 支持的图片格式

    # 批量读取输出数据
    stories_image_paths, all_groups_id = find_all_image_paths_without_timestamp(
        method,
        base_path, 
        # datetime_pattern, 
        image_extensions
        )
    print(f"共找到 {len(stories_image_paths)} 组图片")
    
    for group_id, (story_id, image_paths) in zip(all_groups_id, stories_image_paths.items()):
        assert group_id == story_id, f"组ID不匹配：{group_id} != {story_id}"
    
    return stories_image_paths


# ///////////////////// Video /////////////////////
def read_movieagent_outputs(
        data_path= '/data/AIGC_Research/Story_Telling/StoryVisBMK/data', 
        method = 'movieagent', 
        dataset_name = 'WildStory', 
        language = 'en',
        model_type = 'ROICtrl',  # 'ROICtrl' or 'SD-3'
        **kwargs
    ):
    
    base_path = f"{data_path}/outputs/{method}/{model_type}/{dataset_name}_{language}"
    datetime_pattern = re.compile(r'^\d{8}-\d{6}$')  # 匹配日期时间目录格式
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')  # 支持的图片格式

    # 批量读取输出数据
    stories_image_paths, all_groups_id = find_all_image_paths(
        method,
        base_path, 
        datetime_pattern, 
        image_extensions,
        model_type_for_movieagent = model_type
        )
    print(f"共找到 {len(stories_image_paths)} 组图片")
    
    for group_id, (story_id, image_paths) in zip(all_groups_id, stories_image_paths.items()):
        assert group_id == story_id, f"组ID不匹配：{group_id} != {story_id}"
    
    return stories_image_paths

def read_animdirector_outputs(
        data_path= '/data/AIGC_Research/Story_Telling/StoryVisBMK/data', 
        method = 'animdirector', 
        dataset_name = 'WildStory', 
        language = 'en',
        model_type = 'sd3', 
        **kwargs
    ):
    
    base_path = f"{data_path}/outputs/{method}/{model_type}/{dataset_name}_{language}"
    # datetime_pattern = re.compile(r'^\d{8}-\d{6}$')  
    datetime_pattern = re.compile(r'^\d{8}[_-]\d{6}$')  # 同时匹配 "xxxxxxxx-xxxxxx" 和 "xxxxxxxx_xxxxxx" 日期时间目录格式
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')  # 支持的图片格式

    # 批量读取输出数据
    stories_image_paths, all_groups_id = find_all_image_paths(
        method,
        base_path, 
        datetime_pattern, 
        image_extensions,
        # model_type_for_animdirector = model_type
        )
    print(f"共找到 {len(stories_image_paths)} 组图片")
    
    for group_id, (story_id, image_paths) in zip(all_groups_id, stories_image_paths.items()):
        assert group_id == story_id, f"组ID不匹配：{group_id} != {story_id}"
    
    return stories_image_paths

def read_mmstoryagent_outputs(
        data_path= '/data/AIGC_Research/Story_Telling/StoryVisBMK/data', 
        method = 'mmstoryagent', 
        dataset_name = 'WildStory', 
        language = 'en',
        **kwargs
    ):
    
    base_path = f"{data_path}/outputs/{method}/{dataset_name}_{language}"
    datetime_pattern = re.compile(r'^\d{8}-\d{6}$')  # 匹配日期时间目录格式
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')  # 支持的图片格式

    # 批量读取输出数据
    stories_image_paths, all_groups_id = find_all_image_paths_without_timestamp(
        method,
        base_path, 
        # datetime_pattern, 
        image_extensions
        )
    print(f"共找到 {len(stories_image_paths)} 组图片")
    
    for group_id, (story_id, image_paths) in zip(all_groups_id, stories_image_paths.items()):
        assert group_id == story_id, f"组ID不匹配：{group_id} != {story_id}"
    
    return stories_image_paths


def read_vlogger_outputs(
        data_path= '/data/AIGC_Research/Story_Telling/StoryVisBMK/data', 
        method = 'vlogger', 
        dataset_name = 'WildStory',
        language = 'en',
        content_mode = 'img_ref',  # 'text_only' or 'img_ref'
        **kwargs
    ):
    
    base_path = f"{data_path}/outputs/{method}/{content_mode}/{dataset_name}_{language}"
    datetime_pattern = re.compile(r'^\d{8}-\d{6}$')  # 匹配日期时间目录格式
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')  # 支持的图片格式

    # 批量读取输出数据
    stories_image_paths, all_groups_id = find_all_image_paths(
        method,
        base_path, 
        datetime_pattern, 
        image_extensions,
        content_mode_for_vlogger = content_mode
        )
    print(f"共找到 {len(stories_image_paths)} 组图片")
    
    for group_id, (story_id, image_paths) in zip(all_groups_id, stories_image_paths.items()):
        assert group_id == story_id, f"组ID不匹配：{group_id} != {story_id}"
    
    return stories_image_paths


# ///////////////////// Closed source /////////////////////
def read_mllm_outputs(
        data_path= '/data/AIGC_Research/Story_Telling/StoryVisBMK/data', 
        method = 'gemini', # gpt4o
        dataset_name = 'WildStory', 
        language = 'en',
        **kwargs
    ):
    
    # base_path = f"{data_path}/outputs/vlm/{method}/{dataset_name}"
    base_path = f"{data_path}/outputs/{method}/{dataset_name}_{language}_lite"
    datetime_pattern = re.compile(r'^\d{8}-\d{6}$')  # 匹配日期时间目录格式
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')  # 支持的图片格式

    # 批量读取输出数据
    stories_image_paths, all_groups_id = find_all_image_paths(
        method,
        base_path, 
        datetime_pattern, 
        image_extensions
        )
    print(f"共找到 {len(stories_image_paths)} 组图片")
    
    for group_id, (story_id, image_paths) in zip(all_groups_id, stories_image_paths.items()):
        assert group_id == story_id, f"组ID不匹配：{group_id} != {story_id}"
    
    return stories_image_paths


def read_business_outputs(
        data_path= '/data/AIGC_Research/Story_Telling/StoryVisBMK/data', 
        method = 'moki', 
        dataset_name = 'WildStory', 
        language = 'en',
        **kwargs
    ):
    
    base_path = f"{data_path}/outputs/{method}/{dataset_name}_{language}"
    # datetime_pattern = re.compile(r'^\d{8}-\d{6}$')  # 匹配日期时间目录格式
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')  # 支持的图片格式

    # 批量读取输出数据
    stories_image_paths, all_groups_id = find_all_image_paths_for_business(
        method,
        base_path, 
        image_extensions
        )
    print(f"共找到 {len(stories_image_paths)} 组图片")
    
    for group_id, (story_id, image_paths) in zip(all_groups_id, stories_image_paths.items()):
        assert group_id == story_id, f"组ID不匹配：{group_id} != {story_id}"
    
    return stories_image_paths


def read_naive_outputs(
        data_path= '/data/AIGC_Research/Story_Telling/StoryVisBMK/data', 
        method = 'naive_baseline', 
        dataset_name = 'WildStory', 
        language = 'en',
        **kwargs
    ):
    
    base_path = f"{data_path}/outputs/{method}/{dataset_name}_{language}"
    # datetime_pattern = re.compile(r'^\d{8}-\d{6}$')  # 匹配日期时间目录格式
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')  # 支持的图片格式

    # 批量读取输出数据
    stories_image_paths, all_groups_id = find_all_image_paths_without_timestamp(
        method,
        base_path, 
        # datetime_pattern, 
        image_extensions
        )
    print(f"共找到 {len(stories_image_paths)} 组图片")
    
    for group_id, (story_id, image_paths) in zip(all_groups_id, stories_image_paths.items()):
        assert group_id == story_id, f"组ID不匹配：{group_id} != {story_id}"
    
    return stories_image_paths




if __name__ == "__main__":
    
    story_image_paths, all_groups_id = find_all_image_paths(
        method,
        base_path, 
        datetime_pattern, 
        image_extensions,
        **kwargs
        )
    print(f"共找到 {len(story_image_paths)} 组图片")

    story_id = all_groups_id[0]
    image_path = story_image_paths[f'{story_id}'][0]
    print(f'故事{story_id}的第一个分镜图{image_path}')

    print("\n前10个图片路径示例：")
    for name, path in story_image_paths.items():
        # 只打印前10个路径
        print(f'故事{name}：有{len(path)}张图片，前两张的路径为：')
        for p in path[:2]:
            print(f'    {p}')

    # read_vlogger_outputs()