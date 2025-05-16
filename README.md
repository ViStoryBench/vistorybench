# 🐻🐻ViStoryBench: Comprehensive Benchmark Suite for Story Visualization🐱🐱

**This repository is the official repository of the paper *ViStoryBench: Comprehensive Benchmark Suite for Story Visualization*.**

Welcome to our [ViStoryBench](https://vistorybench.github.io/) page!!!

![ViStoryBench](./asset/ViStoryBench_logo.png)


## Download
```bash
git clone --recursive
cd ViStoryBench
```
## Environment
```bash
conda create -n storyvisbmk python=3.11
conda activate storyvisbmk

conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia

pip install -r requirements.txt
```

## Dataset Preparation

### About Datasets🐻
**80** stories, both *Chinese* and *English*,
including **Plot Correspondence**, **Setting Description**, **Shot Perspective Design**, **Characters Appearing** and **Static Shot Description**

### Download...
📥 Download our [ViStoryBench Datasets](https://huggingface.co/datasets/ViStoryBench/ViStoryBench) (🤗huggingface) dataset
and put it into your local path
```bash
</path/to/your/dataset>
# example
./data/dataset/
```


## Dataset Process (adapt to the corresponding method)
### Loading
Loading our WildStory dataset by 
[`dataset_load.py`](https://github.com/Stepfun/StoryVisBMK/blob/main/code/data_process/dataset_process/dataset_load.py) 
### Adapting and Running
```bash
# image
# //////////////////////////////////////////////////////////////
# Adapting seedstory
python /data/AIGC_Research/Story_Telling/StoryVisBMK/code/data_process/dataset_process/adapt2seedstory.py \
--language 'en'
python /data/AIGC_Research/Story_Telling/StoryVisBMK/code/data_process/dataset_process/adapt2seedstory.py \
--language 'ch'
# Running seedstory
conda activate seed_story
cd /data/AIGC_Research/Story_Telling/Image/SEED-Story/
python3 src/inference/vis_custom_sink.py
# //////////////////////////////////////////////////////////////

# //////////////////////////////////////////////////////////////
# Adapting storyadapter
python /data/AIGC_Research/Story_Telling/StoryVisBMK/code/data_process/dataset_process/adapt2storyadapter.py \
--language 'en'
python /data/AIGC_Research/Story_Telling/StoryVisBMK/code/data_process/dataset_process/adapt2storyadapter.py \
--language 'ch'
# Running storyadapter
conda activate NPRDreamer
cd /data/AIGC_Research/Story_Telling/Image/story-adapter/
python run_custom.py
# //////////////////////////////////////////////////////////////

# //////////////////////////////////////////////////////////////
# Adapting storydiffusion
python /data/AIGC_Research/Story_Telling/StoryVisBMK/code/data_process/dataset_process/adapt2storydiffusion.py \
--language 'en'
python /data/AIGC_Research/Story_Telling/StoryVisBMK/code/data_process/dataset_process/adapt2storydiffusion.py \
--language 'ch'
# Running storydiffusion
conda activate storydiffusion
cd /data/AIGC_Research/Story_Telling/Image/StoryDiffusion/
python gradio_app_sdxl_specific_id_low_vram_custom.py
# //////////////////////////////////////////////////////////////

# //////////////////////////////////////////////////////////////
# Adapting storygen
python /data/AIGC_Research/Story_Telling/StoryVisBMK/code/data_process/dataset_process/adapt2storygen.py \
--language 'en'
python code/data_process/dataset_process/adapt2storygen.py \
--language 'ch'
# Running storygen
pass
# //////////////////////////////////////////////////////////////

# //////////////////////////////////////////////////////////////
# Adapting storyweaver
python /data/AIGC_Research/Story_Telling/StoryVisBMK/code/data_process/dataset_process/adapt2storyweaver.py \
--language 'en'
python /data/AIGC_Research/Story_Telling/StoryVisBMK/code/data_process/dataset_process/adapt2storyweaver.py \
--language 'ch'
# Running storyweaver
pass
# //////////////////////////////////////////////////////////////

# //////////////////////////////////////////////////////////////
# Adapting uno
python /data/AIGC_Research/Story_Telling/StoryVisBMK/code/data_process/dataset_process/adapt2uno.py \
--language 'en'
python /data/AIGC_Research/Story_Telling/StoryVisBMK/code/data_process/dataset_process/adapt2uno.py \
--language 'ch'
# Running uno
conda activate UNO
cd /data/AIGC_Research/Story_Telling/Image/UNO/
python inference_custom.py
# //////////////////////////////////////////////////////////////


# video
# //////////////////////////////////////////////////////////////
# Adapting vlogger
python /data/AIGC_Research/Story_Telling/StoryVisBMK/code/data_process/dataset_process/adapt2vlogger.py \
--language 'en'
python /data/AIGC_Research/Story_Telling/StoryVisBMK/code/data_process/dataset_process/adapt2vlogger.py \
--language 'ch'
# Running vlogger
conda activate vlogger
cd /data/AIGC_Research/Story_Telling/Video/Vlogger
python sample_scripts/vlog_read_script_sample_custom.py
# //////////////////////////////////////////////////////////////

# //////////////////////////////////////////////////////////////
# Adapting animdirector
no
# Running animdirector
# //////////////////////////////////////////////////////////////

# //////////////////////////////////////////////////////////////
# Adapting movieagent
no
# Running movieagent
conda activate movieagent
cd /data/AIGC_Research/Story_Telling/Movie/MovieAgent/movie_agent
sh script/run_custom.sh
... 
```


## Read Outputs (for evaluation)
```bash
python /data/AIGC_Research/Story_Telling/StoryVisBMK/code/data_process/outputs_process/read-movieagent.py \

python /data/AIGC_Research/Story_Telling/StoryVisBMK/code/data_process/outputs_process/read-seedstory.py \

... ...
```

## 🐻Evaluation!
```bash

cd /data/AIGC_Research/Story_Telling/StoryVisBMK/code/bench
# (UNO for example)
sh bench_run.sh 'uno' # Run it for data integrity check
sh bench_run.sh 'uno' --all # Run it for all evaluation
sh bench_run.sh 'uno' --cref # Run it for content consistency eval
sh bench_run.sh 'uno' --cref --csd_cross --csd_self # Run it for both content and style consistency eval

# suggestion

```
### all metrics are as follow:
```bash
--cref 
--csd_cross 
--csd_self 
--aesthetic 
--prompt_align2
--diversity
```
### all methods are as follow:
```bash
STORY_IMG = ['uno', 'seedstory', 'storygen', 'storydiffusion', 'storyadapter', 'theatergen']
STORY_VIDEO = ['movieagent', 'animdirector', 'vlogger', 'mmstoryagent']
CLOSED_SOURCE = ['gemini', 'gpt4o']
BUSINESS = ['moki', 'morphic_studio', 'bairimeng_ai', 'shenbimaliang', 'xunfeihuiying', 'doubao']
```


```bash
# example
sh bench_run.sh 'uno' --csd_cross --csd_self --aesthetic --diversity
sh bench_run.sh 'uno' --cref
sh bench_run.sh 'uno' --prompt_align2

sh bench_run.sh 'uno' --csd_cross --csd_self --aesthetic --diversity
sh bench_run.sh 'seedstory' --csd_cross --csd_self --aesthetic --diversity
sh bench_run.sh 'storygen' --csd_cross --csd_self --aesthetic --diversity
sh bench_run.sh 'storydiffusion' --csd_cross --csd_self --aesthetic --diversity
sh bench_run.sh 'storyadapter' --csd_cross --csd_self --aesthetic --diversity
sh bench_run.sh 'theatergen' --csd_cross --csd_self --aesthetic --diversity

sh bench_run.sh 'movieagent' --csd_cross --csd_self --aesthetic --diversity
sh bench_run.sh 'animdirector' --csd_cross --csd_self --aesthetic --diversity
sh bench_run.sh 'vlogger' --csd_cross --csd_self --aesthetic --diversity
sh bench_run.sh 'mmstoryagent' --csd_cross --csd_self --aesthetic --diversity

sh bench_run.sh 'gemini' --csd_cross --csd_self --aesthetic --diversity --cref
sh bench_run.sh 'gpt4o' --csd_cross --csd_self --aesthetic --diversity --cref

sh bench_run.sh 'moki' --csd_cross --csd_self --aesthetic --diversity
sh bench_run.sh 'morphic_studio' --csd_cross --csd_self --aesthetic --diversity
sh bench_run.sh 'bairimeng_ai' --csd_cross --csd_self --aesthetic --diversity
sh bench_run.sh 'shenbimaliang' --csd_cross --csd_self --aesthetic --diversity
sh bench_run.sh 'xunfeihuiying' --csd_cross --csd_self --aesthetic --diversity
sh bench_run.sh 'doubao' --csd_cross --csd_self --aesthetic --diversity

sh bench_run.sh 'naive_baseline' --csd_cross --csd_self --aesthetic --diversity --cref --prompt_align2 
```


使用一个全局配置文件（例如 config.yaml）来管理共享信息：
模型路径 (ArcFace, CLIP, VLM models, Aesthetic Predictor)
参考数据根目录
评估参数 (阈值, VLM选择)
输出目录结构模板
...


## About WildStory dataset and output image sequence
### WildStory dataset
```
/path/to/dataset/WildStory/
├── 01/
│   ├── story.json
│   ├── image/
│   └── ...
├── 02/
│   ├── story.json
│   ├── image/
│   └── ...
└── ...

/path/to/dataset/WildStory/01/image/
├── Little Brown Rabbit/
│   ├── 00.png
│   ├── 01.png
│   └── ...
├── Big Brown Rabbit/
│   ├── 00.png
│   ├── 01.png
│   └── ...
└── ...

```
One of `story.json` example
```json
{
    "Story_type": {
        "en": "Children's Picture Books",
        "ch": "儿童绘本"
    },
    "Characters": {
        "Little Brown Rabbit": {
            "tag": "non_human",
            "name_ch": "小栗色兔子",
            "name_en": "Little Brown Rabbit",
            "prompt_ch": "一只可爱的小兔子，毛色栗色，眼睛明亮，小巧可爱，活泼好动。他的表情总是充满好奇。",
            "prompt_en": "An adorable little rabbit with chestnut fur, bright eyes, small and cute, lively and active. His expression is always full of curiosity.",
            "num_of_appearances": 16
        },
        "Big Brown Rabbit": {
            "tag": "non_human",
            "name_ch": "大栗色兔子",
            "name_en": "Big Brown Rabbit",
            "prompt_ch": "一只温暖而慈祥的大兔子，毛色栗色，高大强壮。他的表情总是充满关爱和耐心。",
            "prompt_en": "A warm and kind big rabbit with chestnut fur, tall and strong. His expression is always full of love and patience.",
            "num_of_appearances": 10
        }
    },
    "Shots": [
        {
            "index": 1,
            "Plot Correspondence": {
                "ch": "小兔子该上床睡觉了，可是他紧紧的抓住大兔子的耳朵不放。",
                "en": "It's time for the little rabbit to go to bed, but he tightly holds onto the big rabbit's ears and refuses to let go."
            },
            "Setting Description": {
                "ch": "夜晚，小兔子的卧室，温馨的氛围，柔软的床铺，床头柜上放着一盏小夜灯，墙上挂着一幅星空画，窗外可以看到月亮和星星，柔和的黄色光线",
                "en": "Nighttime, the little rabbit's bedroom, cozy atmosphere, soft bedding, a small night light on the bedside table, a starry sky painting on the wall, the moon and stars visible outside the window, soft yellow lighting"
            },
            "Shot Perspective Design": {
                "ch": "中景，平视镜头",
                "en": "Medium shot, eye level shot"
            },
            "Characters Appearing": {
                "ch": [
                    "小栗色兔子",
                    "大栗色兔子"
                ],
                "en": [
                    "Little Brown Rabbit",
                    "Big Brown Rabbit"
                ]
            },
            "Static Shot Description": {
                "ch": "小栗色兔子坐在床上，紧紧抓住大栗色兔子的耳朵，表情调皮可爱。大栗色兔子坐在床边，表情温柔，微微低头看着小兔子",
                "en": "The little brown rabbit sits on the bed, tightly holding the big brown rabbit's ears, with a playful and cute expression. The big brown rabbit sits by the bed, looking gentle, slightly lowering his head to look at the little rabbit"
            }
        },
        ...
    ]
}
```

### Shot-level image sequence output

```
/path/to/processed_outputs/WildStory_en/
├── 01/
│   ├── timestamp_01
│   ├── timestamp_02
│   └── ...
├── 02/
│   ├── timestamp_01
│   ├── timestamp_02
│   └── ...
└── ...

/path/to/processed_outputs/WildStory_en/01/timestamp_001/
├── image_outputs/
│   ├── shot_01.png
│   ├── shot_02.png
│   └── ...
├── eval_outputs/
│   ├── content_consistency
│   ├── style_consistency
│   ├── prompt_alignment 
│   ├── quality_assessment 
│   └── ...
└── ...

```




### output format example
```json
{
  "method_name": "seedstory",
  "story_set": "WildStory_en",
  "story_name": "01",
  "timestamp": "2025-04-24_23-06-10_103",
  "scores": [
    {
      "shot_id": "shot_01.png",
      "cref": {
        "arcface_similarity": 0.85, // 假设有多个角色，可以用更复杂的结构
        "confustion_score": 0.1,
        "clip_similarity": 0.75
        // ... 其他 cref 相关分数
      }
    },
    {
      "shot_id": "shot_02.png",
      // ... 分数
    }
    // ... 其他 shot
  ],
  "aggregate_scores": { // 可选的序列级别聚合分数
    "cref": {
      "average_arcface_similarity": 0.82,
      "sequence_confustion_score": 0.15
    }
  }
}

```

# Story Image Generation

A tool for generating images from story descriptions using either Gemini or GPT-4o models.

## Features

- Support for multiple AI image generation models (Gemini and GPT-4o)
- Story-level multithreading for concurrent processing
- Progress bar display for both stories and shots
- Checkpoint support (resume generation from a specific timestamp)
- Reference images for character visualization
- Sliding window history to maintain story continuity
- Success markers to track completed shots

## Usage

```bash
python story_img_gen.py --model [gpt4o|gemini] [options]
```

### Required Arguments

- `--model`: Model to use for image generation (choices: "gpt4o", "gemini")

### Optional Arguments

- `--data_path`: Path to the dataset (default: "/data/AIGC_Research/Story_Telling/StoryVisBMK/data")
- `--dataset_name`: Name of the dataset (default: "WildStory_en")
- `--timestamp`: Timestamp for resuming generation (format: YYYYMMDD-HHMMSS)
- `--history_window`: Number of previous shots to include in history (default: 3)
- `--max_workers`: Maximum number of concurrent story processing threads (default: 4)

## Examples

### Generate new images with GPT-4o

```bash
python story_img_gen.py --model gpt4o
```

### Generate new images with Gemini

```bash
python story_img_gen.py --model gemini
```

### Resume generation from a specific timestamp

```bash
python story_img_gen.py --model gpt4o --timestamp 20250430-022517
```

### Adjust history window size (for story continuity)

```bash
python story_img_gen.py --model gpt4o --history_window 5
```

### Increase concurrent processing

```bash
python story_img_gen.py --model gpt4o --max_workers 8
```

## Output Structure

```
outputs/
  model_name/
    dataset_name/
      story_id/
        timestamp/
          shot_XX.png
          shot_XX.success
```

- `model_name`: The model used (gpt4o or gemini)
- `dataset_name`: The dataset used (e.g., WildStory_en)
- `story_id`: The story identifier (e.g., 01, 02, etc.)
- `timestamp`: Generation run timestamp (YYYYMMDD-HHMMSS)
- `shot_XX.png`: Generated image for the shot
- `shot_XX.success`: Success marker file with timestamp of completion
