import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
import json
from tqdm import tqdm
from math import floor
from scipy.stats import entropy
import datetime

# --- 数据集类 ---
class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"Warning: Could not load image {img_path}. Skipping. Error: {e}")
            # Return a dummy tensor or handle appropriately
            # For simplicity, returning None and handling in collate_fn or main loop
            return None

# --- Inception Score 计算函数 ---
def calculate_inception_score(image_paths, batch_size=32, splits=1, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    计算给定图像路径列表的 Inception Score。

    Args:
        image_paths (list): 包含图像文件路径的列表。
        batch_size (int): 处理图像的批量大小。
        splits (int): 用于计算 IS 稳定性的分割次数。
        device (str): 使用的设备 ('cuda' or 'cpu')。

    Returns:
        tuple: (平均 IS, IS 标准差)
               如果图像数量不足或加载失败，则返回 (0.0, 0.0)。
    """
    if not image_paths:
        print("Warning: No valid image paths provided for IS calculation.")
        return 0.0, 0.0

    # 加载预训练的 Inception v3 模型
    inception_model = models.inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()

    # 定义图像预处理变换
    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImageDataset(image_paths, transform=preprocess)
    # Filter out None items resulting from loading errors
    dataset.image_paths = [p for i, p in enumerate(dataset.image_paths) if dataset[i] is not None]

    if len(dataset) == 0:
        print("Warning: No images could be loaded successfully for IS calculation.")
        return 0.0, 0.0
    if len(dataset) < batch_size * splits:
         print(f"Warning: Number of images ({len(dataset)}) is potentially too small for stable IS calculation with {splits} splits and batch size {batch_size}. Results might be unreliable.")


    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    preds = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Getting Inception Predictions", leave=False):
            if batch is None: continue # Skip if batch is None (due to loading errors)
            batch = batch.to(device)
            outputs = inception_model(batch)
            if isinstance(outputs, tuple): # InceptionV3 returns tuple in training mode, handle AuxLogits
                 outputs = outputs[0]
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds.append(probs)

    if not preds:
        print("Warning: No predictions generated. Cannot calculate IS.")
        return 0.0, 0.0

    preds = np.concatenate(preds, axis=0)
    num_images = preds.shape[0]
    if num_images == 0:
         print("Warning: Zero predictions obtained after processing.")
         return 0.0, 0.0


    # 计算 Inception Score
    scores = []
    for i in range(splits):
        part = preds[i * (num_images // splits): (i + 1) * (num_images // splits), :]
        if part.shape[0] == 0: continue # Skip empty splits

        # 计算 p(y|x) 的 KL 散度
        kl_divs = []
        for j in range(part.shape[0]):
            pyx = part[j, :]
            py = np.mean(part, axis=0)
            kl_div = entropy(pyx, py)
            kl_divs.append(kl_div)

        # 计算分割的 IS
        split_score = np.exp(np.mean(kl_divs))
        scores.append(split_score)

    if not scores:
        print("Warning: No scores calculated, possibly due to insufficient images per split.")
        return 0.0, 0.0

    mean_is = np.mean(scores)
    std_is = np.std(scores)

    return float(mean_is), float(std_is)

# --- 目录处理和主函数 ---
def process_directory(root_dir, device): # Pass device
    """
    遍历指定的目录结构，计算每个 method 下所有图像的聚合 IS。

    目录结构假定为: root_dir/method/**/shot_*.png (递归查找)

    Args:
        root_dir (str): 包含所有方法输出的根目录 (e.g., 'outputs')。
        device (str): 使用的设备 ('cuda' or 'cpu')。

    Returns:
        list: 包含每个 method 结果的字典列表。
    """
    results = []
    if not os.path.isdir(root_dir):
        print(f"错误：根目录 '{root_dir}' 未找到。")
        return results

    # 获取所有 method 目录
    try:
        # 仅考虑根目录下的一级目录作为方法名
        method_names = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        method_names=['storygen']
    except FileNotFoundError:
        print(f"错误：无法访问根目录 '{root_dir}'。")
        return results
    except Exception as e:
        print(f"列出方法目录时出错：{e}")
        return results

    if not method_names:
        print(f"警告：在 '{root_dir}' 中未找到方法目录。")
        return results

    print(f"找到方法: {method_names}")

    # 收集每个方法的所有图像路径
    method_images = {}
    print("开始收集图像文件路径...")
    for method in tqdm(method_names, desc="扫描方法"):
        method_path = os.path.join(root_dir, method)
        current_method_images = []

        # 使用 os.walk 递归查找所有符合条件的图像文件
        for dirpath, _, filenames in os.walk(method_path):
            for filename in filenames:
                # 检查文件名是否以 'shot_' 开头并具有支持的扩展名
                if filename.lower().startswith('shot_') and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(dirpath, filename)
                    current_method_images.append(img_path)

        # 可选：对每个方法的图像排序，虽然 IS 计算本身不依赖顺序
        current_method_images.sort()
        method_images[method] = current_method_images # 存储收集到的图像
        print(f"方法 '{method}' 找到 {len(current_method_images)} 张图像。")

    print("\n开始计算每个方法的 Inception Score...")
    # 为每个方法计算 IS
    for method, image_files in method_images.items():
        if not image_files:
            print(f"警告：方法 '{method}' 没有找到有效的 shot 图像，跳过 IS 计算。")
            continue

        print(f"\n计算方法 '{method}' 的 IS ({len(image_files)} 张图像)")

        # --- 计算 Inception Score ---
        # 传入 batch_size 和 device
        # 注意：如果图像数量非常大，batch_size 可能需要调整
        is_mean, is_std = calculate_inception_score(image_files, batch_size=32, device=device)
        print(f"  -> 方法 '{method}' IS 平均值: {is_mean:.4f}, IS 标准差: {is_std:.4f}")

        # --- 构建结果字典 ---
        result_entry = {
            "method_name": method,
            "total_images_processed": len(image_files), # 添加处理的图像总数
            "aggregate_scores": {
                "generated_diversity": {
                    "inception_score": is_mean, # 聚合分数使用 IS 均值
                    "inception_score_std": is_std # 加入标准差
                }
                # 在这里可以添加其他聚合指标
            }
            # 不再需要原有的 "scores" 列表，因为是聚合计算
        }
        results.append(result_entry)

    return results

def inception_score_for_folder(image_dir, data_path, method, 
                               CHOICE_DATASET, label,
                               save_dir, filename_base, 
                               batch_size=32, splits=1, device=None):
    """
    递归遍历 image_dir 下所有图片，计算 Inception Score，并保存结果到指定目录。

    Args:
        image_dir (str): 需要递归遍历的图片文件夹。
        data_path (str): 数据根目录，用于输出路径拼接。
        method (str): 方法名，用于输出路径拼接。
        batch_size (int): 批量大小。
        splits (int): IS分割数。
        device (str): 设备（可选）。
    Returns:
        dict: 结果字典。
        str: 保存的json路径。
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 递归收集所有图片
    # image_files = []
    # for dirpath, _, filenames in os.walk(image_dir):
    #     for filename in filenames:
    #         if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
    #             image_files.append(os.path.join(dirpath, filename))
    # image_files.sort()
    # print(f"共找到 {len(image_files)} 张图片用于 IS 计算。")


    print(f'image_dir:{image_dir}')

    # 递归收集所有图片（安全排除 bench results 目录）
    image_files = []
    if isinstance(image_dir, str):
        exclude_dir = "bench results"  # 要排除的目录名
        for dirpath, dirs, filenames in os.walk(image_dir):
            # 存在性检查 + 排除逻辑
            try:
                if exclude_dir in dirs and os.path.join(dirpath, exclude_dir):
                    dirs.remove(exclude_dir)  # 阻止 os.walk 进入该目录
            except Exception as e:
                raise
            
            # 收集图片文件
            for filename in filenames:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(dirpath, filename))

        image_files.sort()
        print(f"共找到 {len(image_files)} 张图片用于 IS 计算（已排除 {exclude_dir} 目录）。")
        
    elif isinstance(image_dir, dict):
        for story_name, image_paths in image_dir.items():
            if story_name in CHOICE_DATASET:
                image_files.extend(image_paths)  # 将每个场景的路径列表合并到总列表

        image_files.sort()
        # 验证结果
        print(f"共合并 {len(image_files)} 张图片路径:")
        for path in image_files[:3]:  # 打印前3条路径示意
            print(f" - {path}")

    print(f'all image_files:{image_files}')





    is_mean, is_std = calculate_inception_score(image_files, batch_size=batch_size, splits=splits, device=device)
    # 生成时间戳
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if not save_dir:
        save_dir = os.path.join(data_path, 'outputs', method, 'bench_results', 'inception_score', timestamp)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{filename_base}_in_{label}_data.json')
    result = {
        "method_name": method,
        "total_images_processed": len(image_files),
        "aggregate_scores": {
            "generated_diversity": {
                "inception_score": is_mean,
                "inception_score_std": is_std
            }
        }
    }
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"IS 结果已保存至 {save_path}")
    return result, save_path



def get_inception_score_and_save(
        method, stories_outputs, 
        CHOICE_DATASET, label,
        save_dir, filename_base):
    # --- 配置 ---
    # root_output_directory = "processed_outputs"  # 示例根目录
    data_path = "/data/AIGC_Research/Story_Telling/StoryVisBMK/data" # 使用实际的根目录
    method_path= f'{data_path}/outputs/{method}'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    inception_score_for_folder(
        stories_outputs, data_path, method, 
        CHOICE_DATASET, label,
        save_dir, filename_base,
        batch_size=32, splits=1, device=device)



if __name__ == "__main__":
    # --- 配置 ---
    # root_output_directory = "processed_outputs"  # 示例根目录
    data_path = "/data/AIGC_Research/Story_Telling/StoryVisBMK/data" # 使用实际的根目录
    method = 'storygen'
    method_path= f'{data_path}/outputs/{method}'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    inception_score_for_folder(
        method_path,
        data_path, 
        method, 
        batch_size=32, 
        splits=1, 
        device=device)

