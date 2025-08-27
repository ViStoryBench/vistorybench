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
import time

# --- Dataset class ---
class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# --- Inception Score calculation function ---
def calculate_inception_score(image_paths, batch_size=32, splits=1, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Calculate Inception Score for a given list of image paths.

    Args:
        image_paths (list): List containing image file paths.
        batch_size (int): Batch size for processing images.
        splits (int): Number of splits for calculating IS stability.
        device (str): Device to use ('cuda' or 'cpu').

    Returns:
        tuple: (Average IS, IS standard deviation)
               Returns (0.0, 0.0) if insufficient images or loading failure.
    """
    if not image_paths:
        print("Warning: No valid image paths provided for IS calculation.")
        return 0.0, 0.0

    # Load pre-trained Inception v3 model
    inception_model = models.inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()

    # Define image preprocessing transforms
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


    # Calculate Inception Score
    scores = []
    for i in range(splits):
        part = preds[i * (num_images // splits): (i + 1) * (num_images // splits), :]
        if part.shape[0] == 0: continue # Skip empty splits

        # Calculate KL divergence of p(y|x)
        kl_divs = []
        for j in range(part.shape[0]):
            pyx = part[j, :]
            py = np.mean(part, axis=0)
            kl_div = entropy(pyx, py)
            kl_divs.append(kl_div)

        # Calculate split IS
        split_score = np.exp(np.mean(kl_divs))
        scores.append(split_score)

    if not scores:
        print("Warning: No scores calculated, possibly due to insufficient images per split.")
        return 0.0, 0.0

    mean_is = np.mean(scores)
    std_is = np.std(scores)

    return float(mean_is), float(std_is)

# --- Directory processing and main functions ---
def process_directory(root_dir, device): # Pass device
    """
    Traverse the specified directory structure and calculate aggregated IS for all images under each method.
    
    Directory structure assumed to be: root_dir/method/**/shot_*.png (recursive search)
    
    Args:
        root_dir (str): Root directory containing all method outputs (e.g., 'outputs').
        device (str): Device to use ('cuda' or 'cpu').
    
    Returns:
        list: List of dictionaries containing results for each method.
    """
    results = []
    if not os.path.isdir(root_dir):
        print(f"Error: Root directory '{root_dir}' not found.")
        return results

    # Get all method directories
    try:
        # Only consider first-level directories under root as method names
        method_names = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        method_names=['storygen']
    except FileNotFoundError:
        print(f"Error: Cannot access root directory '{root_dir}'.")
        return results
    except Exception as e:
        print(f"Error listing method directories: {e}")
        return results

    if not method_names:
        print(f"Warning: No method directories found in '{root_dir}'.")
        return results

    print(f"Found methods: {method_names}")

    # Collect all image paths for each method
    method_images = {}
    print("Starting to collect image file paths...")
    for method in tqdm(method_names, desc="Scanning methods"):
        method_path = os.path.join(root_dir, method)
        current_method_images = []

        # Use os.walk to recursively find all image files that meet the criteria
        for dirpath, _, filenames in os.walk(method_path):
            for filename in filenames:
                # Check if filename starts with 'shot_' and has supported extension
                if filename.lower().startswith('shot_') and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(dirpath, filename)
                    current_method_images.append(img_path)

        # Optional: sort images for each method, though IS calculation itself doesn't depend on order
        current_method_images.sort()
        method_images[method] = current_method_images # Store collected images
        print(f"Method '{method}' found {len(current_method_images)} images.")

    print("\nStarting to calculate Inception Score for each method...")
    # Calculate IS for each method
    for method, image_files in method_images.items():
        if not image_files:
            print(f"Warning: Method '{method}' has no valid shot images, skipping IS calculation.")
            continue

        print(f"\nCalculating IS for method '{method}' ({len(image_files)} images)")

        # --- Calculate Inception Score ---
        # Pass batch_size and device
        # Note: if image count is very large, batch_size may need adjustment
        is_mean, is_std = calculate_inception_score(image_files, batch_size=32, device=device)
        print(f"  -> Method '{method}' IS mean: {is_mean:.4f}, IS std: {is_std:.4f}")

        # --- Build result dictionary ---
        result_entry = {
            "method_name": method,
            "total_images_processed": len(image_files), # Add total processed images
            "aggregate_scores": {
                "generated_diversity": {
                    "inception_score": is_mean, # Aggregate score uses IS mean
                    "inception_score_std": is_std # Include standard deviation
                }
                # Can add other aggregate metrics here
            }
            # No longer need original "scores" list, since it's aggregate calculation
        }
        results.append(result_entry)

    return results

def inception_score_for_folder(image_dir, data_path, method,
                               CHOICE_DATASET, label,
                               result_manager,
                               batch_size=32, splits=1, device=None):
    """
    Recursively traverse all images under image_dir, calculate Inception Score, and save results using ResultManager.

    Args:
        image_dir (str or dict): Image folder or dictionary of image paths.
        data_path (str): Data root directory.
        method (str): Method name.
        CHOICE_DATASET (list): List of story names for the current dataset split.
        label (str): The label for the dataset split (e.g., 'Full', 'Lite').
        result_manager (ResultManager): The result manager instance.
        batch_size (int): Batch size.
        splits (int): IS split count.
        device (str): Device (optional).
    Returns:
        dict: Result dictionary.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    image_files = []
    if isinstance(image_dir, str):
        exclude_dir = "bench_results"
        for dirpath, dirs, filenames in os.walk(image_dir):
            if exclude_dir in dirs:
                dirs.remove(exclude_dir)
            for filename in filenames:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(dirpath, filename))
        image_files.sort()
        print(f"Found {len(image_files)} images for IS calculation (excluded '{exclude_dir}' directory).")
    elif isinstance(image_dir, dict):
        for story_name, image_paths in image_dir.items():
            if story_name in CHOICE_DATASET:
                image_files.extend(image_paths['shots'])
        image_files.sort()
        print(f"Combined {len(image_files)} image paths for dataset '{label}'.")

    if not image_files:
        print(f"No images found for method '{method}' and dataset '{label}'. Skipping IS calculation.")
        return None

    start_time = time.time()
    is_mean, is_std = calculate_inception_score(image_files, batch_size=batch_size, splits=splits, device=device)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time-consuming for IS Score on '{label}' dataset: {elapsed_time:.4f} seconds")

    result_dict = {
        "method_name": method,
        "dataset_label": label,
        "total_images_processed": len(image_files),
        "aggregate_scores": {
            "generated_diversity": {
                "inception_score": is_mean,
                "inception_score_std": is_std
            }
        },
        "elapsed_time(seconds)": elapsed_time
    }

    result_manager.save_metric_result(f"diversity_{label}", scores=result_dict)
    print(f"IS results for '{label}' dataset saved via ResultManager.")
    
    return result_dict



if __name__ == "__main__":
    # --- Configuration ---
    # root_output_directory = "processed_outputs"  # Example root directory
    data_path = "ViStoryBench/data" # Use actual root directory
    method = 'storygen'
    method_path= f'{data_path}/outputs/{method}'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    inception_score_for_folder(
        method_path,
        data_path, 
        method, 
        batch_size=32, 
        splits=1, 
        device=device)

