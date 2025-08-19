#!/bin/bash

# Function to read YAML file
get_yaml_value() {
    local yaml_file=$1
    local key=$2
    grep "^${key}:" "$yaml_file" | awk '{print $2}' | sed 's/"//g'
}

# Get pretrain directory from config
CONFIG_FILE="vistorybench/config.yaml"
PRETRAIN_DIR=$(get_yaml_value "$CONFIG_FILE" "pretrain_path")
if [ -z "$PRETRAIN_DIR" ]; then
    echo "Error: pretrain_dir not found in config.yaml"
    exit 1
fi

echo "Using pretrain directory: $PRETRAIN_DIR"

# Create necessary directories
mkdir -p "${PRETRAIN_DIR}/groundingdino/weights"
mkdir -p "${PRETRAIN_DIR}/insightface/models/antelopev2"
mkdir -p "${PRETRAIN_DIR}/google/siglip-so400m-patch14-384"
mkdir -p "${PRETRAIN_DIR}/google-bert/bert-base-uncased"
mkdir -p "${PRETRAIN_DIR}/csd"
mkdir -p "${PRETRAIN_DIR}/aesthetic_predictor"
mkdir -p "${PRETRAIN_DIR}/adaface"

echo "⭐ Downloading weights for CIDS Score and Prompt Align Score..."

# a. GroundingDINO weights
echo "Downloading groundingdino_swint_ogc.pth..."
# wget -c -P "${PRETRAIN_DIR}/groundingdino/weights" https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
aria2c -x 16 -s 16 -k 1M -c -d "${PRETRAIN_DIR}/groundingdino/weights" "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"

# b. Antelopev2 weights
echo "Downloading antelopev2.zip..."
aria2c -x 16 -s 16 -k 1M -c -d "${PRETRAIN_DIR}/insightface/models" https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip

# c. SigLIP weights
echo "Downloading siglip-so400m-patch14-384..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='google/siglip-so400m-patch14-384',
    repo_type='model',
    local_dir='${PRETRAIN_DIR}/google/siglip-so400m-patch14-384',
    local_dir_use_symlinks=False
)
"
# d. openai_clip weights
echo "Downloading openai_clip..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='openai/clip-vit-large-patch14',
    repo_type='model',
    local_dir='${PRETRAIN_DIR}/google/clip-vit-large-patch14',
    local_dir_use_symlinks=False
)
"

# e. BERT weights
echo "Downloading bert-base-uncased..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='google-bert/bert-base-uncased',
    repo_type='model',
    local_dir='${PRETRAIN_DIR}/google-bert/bert-base-uncased',
    local_dir_use_symlinks=False
)
"
# Aesthetic predictor weights
echo "Downloading aesthetic_predictor_v2_5.pth..."
aria2c -x 16 -s 16 -k 1M -c -d "${PRETRAIN_DIR}/aesthetic_predictor" https://github.com/discus0434/aesthetic-predictor-v2-5/raw/main/models/aesthetic_predictor_v2_5.pth

# f. AdaFace weights
echo "Downloading adaface_ir101_webface12m.ckpt..."
gdown --continue "https://drive.google.com/uc?id=1dswnavflETcnAuplZj1IOKKP0eM8ITgT" -O "${PRETRAIN_DIR}/adaface/adaface_ir101_webface12m.ckpt"

echo "⭐ Downloading weights for CSD Score..."

# CSD weights
echo "Downloading csd_vit-large.pth..."
gdown --continue "https://drive.google.com/uc?id=1SETgjkj6oUIbjgwxgtXw2I2t4quRzG-3" -O "${PRETRAIN_DIR}/csd/csd_vit-large.pth"


echo "Unzipping antelopev2..."
unzip "${PRETRAIN_DIR}/insightface/models/antelopev2.zip" -d "${PRETRAIN_DIR}/insightface/models"
rm "${PRETRAIN_DIR}/insightface/models/antelopev2.zip"

echo "✅ All weights downloaded successfully!"