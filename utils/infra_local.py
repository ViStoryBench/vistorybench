import os
import glob
import joblib
import json
import re
import torch
import numpy as np
import math
from collections import defaultdict
from torch.multiprocessing import Pool, set_start_method

# vLLM 只能在子进程内部导入，或者在 spawn 模式下导入，
# 但为了代码结构清晰，我们在 worker 函数内部导入 LLM

# ================= 配置区域 =================
MODEL_PATH = "/gemini/platform/public/aigc/zhuangcailin/pretrain/Qwen/Qwen3-VL-8B-Instruct/"
INPUT_ROOT = "prompt_align2"
OUTPUT_JSON = "inference_results_averaged2.json"
DETAILS_JSON = "inference_details2.json"

def parse_score(text):
    """尝试从模型输出中提取最后一个数字作为分数"""
    try:
        # 匹配整数或浮点数
        matches = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        if matches:
            return float(matches[-1])
        return None
    except:
        return None

def load_file_paths(root_dir):
    """加载所有 bin 文件的路径并排序"""
    files = glob.glob(os.path.join(root_dir, "**/*.bin"), recursive=True)
    files.sort() # 排序保证分配确定性
    print(f"Total files found: {len(files)}")
    return files

def run_worker(gpu_id, file_paths, model_path):
    """
    这是在每个子进程中运行的函数
    gpu_id: 当前进程使用的 GPU ID (0, 1, 2...)
    file_paths: 分配给该 GPU 的文件列表
    model_path: 模型路径
    """
    # 1. 设置环境变量，隔离显卡
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 强制使用 vLLM V1 并关闭 V1 的内部多进程（因为我们已经在外部做了多进程）
    os.environ['VLLM_USE_V1'] = '1'
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    
    # 在子进程内导入 vLLM，避免 CUDA 初始化冲突
    from vllm import LLM, SamplingParams
    
    print(f"[GPU {gpu_id}] Initializing vLLM on device {gpu_id} with {len(file_paths)} tasks...")

    # 2. 初始化 LLM (Tensor Parallel = 1)
    try:
        llm = LLM(
            model=model_path,
            tensor_parallel_size=1, # 单卡运行
            enable_prefix_caching=True,
            trust_remote_code=True,
            seed=42, # V1 基础种子
            kv_cache_dtype="fp8",
            gpu_memory_utilization=0.80 # 防止 OOM，适当调整
        )
    except Exception as e:
        print(f"[GPU {gpu_id}] Error initializing model: {e}")
        return []

    # 3. 准备数据
    prompts_data = []
    sampling_params_list = []
    req_infos = []

    for fp in file_paths:
        try:
            data = joblib.load(fp)
            
            # 解析路径元数据
            # path format: prompt_align/{method}/{mode}/{meric}/{story}/{shot}/request_{id}.bin
            rel_path = os.path.relpath(fp, INPUT_ROOT)
            parts = rel_path.split(os.sep)
            
            if len(parts) >= 5:
                info = {
                    "method": parts[0],
                    "mode": parts[1],
                    "metric": parts[2],
                    "story": parts[3],
                    "shot": parts[4],
                    "id": os.path.splitext(parts[-1])[0],
                    "file_path": fp
                }
                
                # 构建 SamplingParams
                # 注意：这里使用 bin 文件里的 seed 保证复现性
                sp = SamplingParams(
                    temperature=data.get('temperature', 0.0),
                    max_tokens=data.get('max_tokens', 512),
                    seed=data.get('seed', 42), 
                    top_p=data.get('top_p', 1.0) if data.get('top_p') is not None else 1.0,
                )

                prompts_data.append(data.get('messages', []))
                sampling_params_list.append(sp)
                req_infos.append(info)
        except Exception as e:
            print(f"[GPU {gpu_id}] Error loading {fp}: {e}")

    if not prompts_data:
        return []

    # 4. 执行批量推理
    print(f"[GPU {gpu_id}] Starting inference...")
    outputs = llm.chat(messages=prompts_data, sampling_params=sampling_params_list)

    # 5. 整理结果
    worker_results = []
    for i, output in enumerate(outputs):
        text = output.outputs[0].text
        score = parse_score(text)
        
        res = req_infos[i].copy()
        res["output"] = text
        res["parsed_score"] = score
        worker_results.append(res)
        
    print(f"[GPU {gpu_id}] Finished. Processed {len(worker_results)} items.")
    return worker_results

def main():
    # 设置启动方法为 spawn，这对于 CUDA 多进程是必须的
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    if not os.path.exists(INPUT_ROOT):
        print(f"Error: {INPUT_ROOT} not found.")
        return

    # 1. 获取所有任务文件
    all_files = load_file_paths(INPUT_ROOT)
    if not all_files:
        return

    # 2. 检测 GPU 数量
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs. Splitting tasks...")

    if num_gpus == 0:
        print("No GPUs found!")
        return

    # 3. 分配任务 (Round-robin 或 Chunk 分割)
    # 使用 chunk 分割，让每个 GPU 处理连续的一块文件
    chunk_size = math.ceil(len(all_files) / num_gpus)
    file_chunks = []
    for i in range(num_gpus):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(all_files))
        if start < end:
            file_chunks.append(all_files[start:end])
        else:
            # 如果任务少于 GPU 数
            file_chunks.append([])

    # 4. 启动多进程池
    # pool_args 格式: [(gpu_id, files, model_path), ...]
    pool_args = []
    for i in range(num_gpus):
        if file_chunks[i]:
            pool_args.append((i, file_chunks[i], MODEL_PATH))

    print(f"Launching {len(pool_args)} worker processes...")
    
    # 使用 starmap 启动
    with Pool(processes=num_gpus) as pool:
        # worker_results_list 是一个列表，包含每个进程返回的列表
        worker_results_list = pool.starmap(run_worker, pool_args)

    # 5. 聚合所有结果
    flat_results = []
    for res_list in worker_results_list:
        flat_results.extend(res_list)

    print(f"All workers done. Total results: {len(flat_results)}")

    # 6. 计算指标 Average
    # 结构: aggregation[story][method][mode][metric] = list(scores)
    aggregation = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    
    for item in flat_results:
        story = item['story']
        method = item['method']
        mode = item['mode']
        metric = item['metric']
        score = item['parsed_score']
        
        aggregation[story][method][mode][metric].append(score)

    final_output = {}
    for story, methods in aggregation.items():
        final_output[story] = {}
        for method, modes in methods.items():
            final_output[story][method] = {}
            for mode, metrics in modes.items():
                final_output[story][method][mode] = {}
                for metric, scores in metrics.items():
                    valid_scores = [s for s in scores if s is not None]
                    if valid_scores:
                        avg = sum(valid_scores) / len(valid_scores)
                    else:
                        avg = "N/A"
                    final_output[story][method][mode][metric] = avg

    # 7. 保存文件
    print(f"Saving to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
    
    print(f"Saving details to {DETAILS_JSON}...")
    with open(DETAILS_JSON, 'w', encoding='utf-8') as f:
        json.dump(flat_results, f, indent=2, ensure_ascii=False)

    print("Inference completed successfully.")
# def gptv_query(transcript=None, top_p=0.2, temp=0., model_type="gpt-4.1", api_key='', base_url='', seed=123, max_tokens=512, wait_time=10,method='',metric='',story='',shot='',id='',mode=''):
#     headers = {
#         "Content-Type": "application/json",
#         "Authorization": f"Bearer {api_key}"
#     }
#     base_url = base_url[:-1] if base_url.endswith('/') else base_url
#     requests_url = f"{base_url}/chat/completions" if base_url.endswith('/v1') else f"{base_url}/v1/chat/completions"

#     data = {
#         'model': model_type,
#         'max_tokens': max_tokens,
#         'temperature': temp,
#         'messages': transcript or [],
#         'seed': seed,
#     }
#     path=f'prompt_align2/{method}/{mode}/{metric}/{story}/{shot}'
#     os.makedirs(path,exist_ok=True)
#     joblib.dump(data,f'{path}/request_{id}.bin')
#     return ''
if __name__ == "__main__":
    main()
