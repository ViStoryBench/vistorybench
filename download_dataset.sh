#!/bin/bash

# Create directory
mkdir -p data/dataset
export HF_ENDPOINT=https://hf-mirror.com
# Download dataset
echo "📥 Downloading ViStory dataset..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='ViStoryBench/ViStoryBench',
    repo_type='dataset',
    local_dir='data/dataset',
    local_dir_use_symlinks=False
)
"

# Rename folder
echo "Renaming ViStoryBench to ViStory..."
mv "data/dataset/ViStoryBench" "data/dataset/ViStory" 2>/dev/null

echo "✅ Done! Dataset saved to: data/dataset/ViStory"