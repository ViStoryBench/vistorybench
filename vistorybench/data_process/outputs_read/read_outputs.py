import os
import re
from datetime import datetime
from .fine_image_path import find_all_image_paths, find_all_image_paths_for_business, find_all_image_paths_without_timestamp

# ///////////////////// Custom /////////////////////
def read_custom_outputs(
        outputs_path= '', 
        method = 'my_method', 
        dataset_name = 'ViStory', 
        language = 'en',
        **kwargs
    ):
    
    base_path = f"{outputs_path}/{method}/{dataset_name}_{language}"
    datetime_pattern = re.compile(r'^\d{8}-\d{6}$') # Match datetime directory format
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp') # Supported image formats

    # Batch read output data
    stories_image_paths, all_groups_id = find_all_image_paths(
        method,
        base_path, 
        datetime_pattern, 
        image_extensions
        )
    print(f"Found {len(stories_image_paths)} groups of images")
    
    for group_id, (story_id, image_paths) in zip(all_groups_id, stories_image_paths.items()):
        assert group_id == story_id, f"Group ID mismatch: {group_id} != {story_id}"
    
    return stories_image_paths


# ///////////////////// Image /////////////////////
def read_uno_outputs(
        outputs_path= '', 
        method = 'uno', 
        dataset_name = 'ViStory', 
        language = 'en',
        **kwargs
    ):
    
    base_path = f"{outputs_path}/{method}/{dataset_name}_{language}"
    datetime_pattern = re.compile(r'^\d{8}-\d{6}$')
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')

    stories_image_paths, all_groups_id = find_all_image_paths(
        method,
        base_path, 
        datetime_pattern, 
        image_extensions
        )
    print(f"Found {len(stories_image_paths)} groups of images")
    
    for group_id, (story_id, image_paths) in zip(all_groups_id, stories_image_paths.items()):
        assert group_id == story_id, f"Group ID mismatch: {group_id} != {story_id}"
    
    return stories_image_paths

def read_storygen_outputs(
        outputs_path= '', 
        method = 'storygen', 
        dataset_name = 'ViStory', 
        language = 'en',
        model_mode = 'mix',  # 'auto-regressive' or 'multi-image-condition' or 'mix'
        **kwargs
    ):
    
    base_path = f"{outputs_path}/{method}/{model_mode}/{dataset_name}_{language}"
    datetime_pattern = re.compile(r'^\d{8}-\d{6}$')
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')

    stories_image_paths, all_groups_id = find_all_image_paths(
        method,
        base_path, 
        datetime_pattern, 
        image_extensions,
        model_mode_for_storygen = model_mode
        )
    print(f"Found {len(stories_image_paths)} groups of images")
    
    for group_id, (story_id, image_paths) in zip(all_groups_id, stories_image_paths.items()):
        assert group_id == story_id, f"Group ID mismatch: {group_id} != {story_id}"
    
    return stories_image_paths

def read_seedstory_outputs(
        outputs_path= '', 
        method = 'seedstory', 
        dataset_name = 'ViStory', 
        language = 'en',
        **kwargs
    ):
    
    base_path = f"{outputs_path}/{method}/{dataset_name}_{language}"
    datetime_pattern = re.compile(r'^\d{8}-\d{6}$')
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')

    stories_image_paths, all_groups_id = find_all_image_paths(
        method,
        base_path, 
        datetime_pattern, 
        image_extensions
        )
    print(f"Found {len(stories_image_paths)} groups of images")
    
    for group_id, (story_id, image_paths) in zip(all_groups_id, stories_image_paths.items()):
        assert group_id == story_id, f"Group ID mismatch: {group_id} != {story_id}"
    
    return stories_image_paths

def read_storydiffusion_outputs(
        outputs_path= '', 
        method = 'storydiffusion', 
        dataset_name = 'ViStory', 
        language = 'en',
        content_mode = 'original',  # 'original' or 'Photomaker'
        style_mode = '(No style)',
        **kwargs
    ):
    
    base_path = f"{outputs_path}/{method}/{content_mode}/{dataset_name}_{language}"
    datetime_pattern = re.compile(r'^\d{8}-\d{6}$')
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')

    stories_image_paths, all_groups_id = find_all_image_paths(
        method,
        base_path, 
        datetime_pattern, 
        image_extensions,
        style_mode_for_storydiffusion = style_mode
        )
    print(f"Found {len(stories_image_paths)} groups of images")
    
    for group_id, (story_id, image_paths) in zip(all_groups_id, stories_image_paths.items()):
        assert group_id == story_id, f"Group ID mismatch: {group_id} != {story_id}"
    
    return stories_image_paths

def read_storyadapter_outputs(
        outputs_path= '', 
        method = 'storyadapter', 
        dataset_name = 'ViStory', 
        language = 'en',
        content_mode = 'img_ref',  # 'text_only' or 'img_ref'
        scale_stage = 'results_xl5',
        **kwargs
    ):
    
    base_path = f"{outputs_path}/{method}/{content_mode}/{dataset_name}_{language}"
    datetime_pattern = re.compile(r'^\d{8}-\d{6}$')
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')

    stories_image_paths, all_groups_id = find_all_image_paths(
        method,
        base_path, 
        datetime_pattern, 
        image_extensions,
        scale_stage_for_storyadapter = scale_stage
        )
    print(f"Found {len(stories_image_paths)} groups of images")
    
    for group_id, (story_id, image_paths) in zip(all_groups_id, stories_image_paths.items()):
        assert group_id == story_id, f"Group ID mismatch: {group_id} != {story_id}"
    
    return stories_image_paths

def read_theatergen_outputs(
        outputs_path= '', 
        method = 'theatergen', 
        dataset_name = 'ViStory', 
        language = 'en',
        **kwargs
    ):
    
    base_path = f"{outputs_path}/{method}/{dataset_name}_{language}"
    datetime_pattern = re.compile(r'^\d{8}-\d{6}$')
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')

    stories_image_paths, all_groups_id = find_all_image_paths_without_timestamp(
        method,
        base_path, 
        # datetime_pattern, 
        image_extensions
        )
    print(f"Found {len(stories_image_paths)} groups of images")
    
    for group_id, (story_id, image_paths) in zip(all_groups_id, stories_image_paths.items()):
        assert group_id == story_id, f"Group ID mismatch: {group_id} != {story_id}"
    
    return stories_image_paths


# ///////////////////// Video /////////////////////
def read_movieagent_outputs(
        outputs_path= '', 
        method = 'movieagent', 
        dataset_name = 'ViStory', 
        language = 'en',
        model_type = 'ROICtrl',  # 'ROICtrl' or 'SD-3'
        **kwargs
    ):
    
    base_path = f"{outputs_path}/{method}/{model_type}/{dataset_name}_{language}"
    datetime_pattern = re.compile(r'^\d{8}-\d{6}$')
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')

    stories_image_paths, all_groups_id = find_all_image_paths(
        method,
        base_path, 
        datetime_pattern, 
        image_extensions,
        model_type_for_movieagent = model_type
        )
    print(f"Found {len(stories_image_paths)} groups of images")
    
    for group_id, (story_id, image_paths) in zip(all_groups_id, stories_image_paths.items()):
        assert group_id == story_id, f"Group ID mismatch: {group_id} != {story_id}"
    
    return stories_image_paths

def read_animdirector_outputs(
        outputs_path= '', 
        method = 'animdirector', 
        dataset_name = 'ViStory', 
        language = 'en',
        model_type = 'sd3', 
        **kwargs
    ):
    
    base_path = f"{outputs_path}/{method}/{model_type}/{dataset_name}_{language}"
    # datetime_pattern = re.compile(r'^\d{8}-\d{6}$')  
    datetime_pattern = re.compile(r'^\d{8}[_-]\d{6}$')  # Match both "xxxxxxxx-xxxxxx" and "xxxxxxxx_xxxxxx" datetime directory formats
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')

    stories_image_paths, all_groups_id = find_all_image_paths(
        method,
        base_path, 
        datetime_pattern, 
        image_extensions,
        # model_type_for_animdirector = model_type
        )
    print(f"Found {len(stories_image_paths)} groups of images")
    
    for group_id, (story_id, image_paths) in zip(all_groups_id, stories_image_paths.items()):
        assert group_id == story_id, f"Group ID mismatch: {group_id} != {story_id}"
    
    return stories_image_paths

def read_mmstoryagent_outputs(
        outputs_path= '', 
        method = 'mmstoryagent', 
        dataset_name = 'ViStory', 
        language = 'en',
        **kwargs
    ):
    
    base_path = f"{outputs_path}/{method}/{dataset_name}_{language}"
    datetime_pattern = re.compile(r'^\d{8}-\d{6}$')
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')

    stories_image_paths, all_groups_id = find_all_image_paths_without_timestamp(
        method,
        base_path, 
        # datetime_pattern, 
        image_extensions
        )
    print(f"Found {len(stories_image_paths)} groups of images")
    
    for group_id, (story_id, image_paths) in zip(all_groups_id, stories_image_paths.items()):
        assert group_id == story_id, f"Group ID mismatch: {group_id} != {story_id}"
    
    return stories_image_paths

def read_vlogger_outputs(
        outputs_path= '', 
        method = 'vlogger', 
        dataset_name = 'ViStory',
        language = 'en',
        content_mode = 'img_ref',  # 'text_only' or 'img_ref'
        **kwargs
    ):
    
    base_path = f"{outputs_path}/{method}/{content_mode}/{dataset_name}_{language}"
    datetime_pattern = re.compile(r'^\d{8}-\d{6}$')
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')

    stories_image_paths, all_groups_id = find_all_image_paths(
        method,
        base_path, 
        datetime_pattern, 
        image_extensions,
        content_mode_for_vlogger = content_mode
        )
    print(f"Found {len(stories_image_paths)} groups of images")
    
    for group_id, (story_id, image_paths) in zip(all_groups_id, stories_image_paths.items()):
        assert group_id == story_id, f"Group ID mismatch: {group_id} != {story_id}"
    
    return stories_image_paths


# ///////////////////// Closed source /////////////////////
def read_mllm_outputs(
        outputs_path= '', 
        method = 'gemini', # gpt4o
        dataset_name = 'ViStory', 
        language = 'en',
        **kwargs
    ):
    
    # base_path = f"{outputs_path}/vlm/{method}/{dataset_name}"
    base_path = f"{outputs_path}/{method}/{dataset_name}_{language}_lite"
    datetime_pattern = re.compile(r'^\d{8}-\d{6}$')
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')

    stories_image_paths, all_groups_id = find_all_image_paths(
        method,
        base_path, 
        datetime_pattern, 
        image_extensions
        )
    print(f"Found {len(stories_image_paths)} groups of images")
    
    for group_id, (story_id, image_paths) in zip(all_groups_id, stories_image_paths.items()):
        assert group_id == story_id, f"Group ID mismatch: {group_id} != {story_id}"
    
    return stories_image_paths

 
def read_business_outputs(
        outputs_path= '', 
        method = 'moki', 
        dataset_name = 'ViStory', 
        language = 'en',
        **kwargs
    ):
    
    base_path = f"{outputs_path}/{method}/{dataset_name}_{language}"
    # datetime_pattern = re.compile(r'^\d{8}-\d{6}$')
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')

    stories_image_paths, all_groups_id = find_all_image_paths_for_business(
        method,
        base_path, 
        image_extensions
        )
    print(f"Found {len(stories_image_paths)} groups of images, and {len(stories_image_paths)} groups of character images")
    
    for group_id, (story_id, image_paths) in zip(all_groups_id, stories_image_paths.items()):
        assert group_id == story_id, f"Group ID mismatch: {group_id} != {story_id}"
    
    return stories_image_paths


def read_naive_outputs(
        outputs_path= '', 
        method = 'naive_baseline', 
        dataset_name = 'ViStory', 
        language = 'en',
        **kwargs
    ):
    
    base_path = f"{outputs_path}/{method}/{dataset_name}_{language}"
    # datetime_pattern = re.compile(r'^\d{8}-\d{6}$')
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')

    stories_image_paths, all_groups_id = find_all_image_paths_without_timestamp(
        method,
        base_path, 
        # datetime_pattern, 
        image_extensions
        )
    print(f"Found {len(stories_image_paths)} groups of images")
    
    for group_id, (story_id, image_paths) in zip(all_groups_id, stories_image_paths.items()):
        assert group_id == story_id, f"Group ID mismatch: {group_id} != {story_id}"
    
    return stories_image_paths



if __name__ == "__main__":
    
    story_image_paths, all_groups_id = find_all_image_paths(
        method,
        base_path, 
        datetime_pattern, 
        image_extensions,
        **kwargs
        )
    print(f"Found {len(story_image_paths)} groups of images")

    story_id = all_groups_id[0]
    image_path = story_image_paths[f'{story_id}'][0]
    print(f'The first shot image of Story {story_id}: {image_path}')

    print("\nThe first 10 image path examples:")
    for name, path in story_image_paths.items():
        # Only print first 10 paths
        print(f'Story {name} : There are {len(path)} pictures. The paths of the first 2 pictures are here:')
        for p in path[:2]:
            print(f'    {p}')
