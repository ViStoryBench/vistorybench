#!/bin/bash

# Create directory
mkdir -p data/weights
export HF_ENDPOINT=https://hf-mirror.com
# Download weights
echo "📥 Downloading ViStory weights..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='ViStoryBench/VistoryBench_pretrain',
    local_dir='data/pretrain',
    local_dir_use_symlinks=False
)
"
echo "✅ Done! weights saved to: data/pretrain"