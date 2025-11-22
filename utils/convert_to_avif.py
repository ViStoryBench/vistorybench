import os
from pathlib import Path
import subprocess # 导入 subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import shutil

# --- 1. 用户配置 (重要！) ---

# "NVIDIA", "INTEL", "AMD" 之一。 
# 必须与你的硬件和 ffmpeg 构建相匹配！
ENCODER_TYPE = "NVIDIA" 

# ffmpeg 可执行文件的路径。
# 如果 ffmpeg 已经在你的系统 PATH 中 (推荐)，保留为 "ffmpeg"
# 否则, 设为完整路径, e.g., "C:/ffmpeg/bin/ffmpeg.exe"
FFMPEG_PATH = "ffmpeg" 

INPUT_DIR = "data/outputs"   # 替换为你的源图片文件夹
OUTPUT_DIR = "data/outputs_avif"      # 替换为你的目标输出文件夹

# GPU 编码质量 (取决于编码器, 20-30 是个不错的范围)
# - Nvenc (NVIDIA): 使用 -cq (Constant Quality), 0-51 (越低越好), 推荐 25
# - QSV (Intel) / AMF (AMD): 使用 -q:v (Global Quality), 1-51 (越低越好), 推荐 25
GPU_QUALITY = 51

# --- 2. 高级配置 ---

# 同时运行的 ffmpeg 实例数。
# !! 不要设置得太高 (例如 CPU 核心数)，
# !! 因为你只有一个 GPU。 2, 4, 或 6 通常是最佳选择。
MAX_WORKERS = 4

# 支持转换的图片格式
SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']

# --------------------------------

def get_ffmpeg_cmd(encoder_type, input_file, output_file, quality):
    """
    根据选择的 GPU 生成 ffmpeg 命令
    """
    # 强制覆盖输出, 隐藏不必要的日志。
    # 这里统一在进入编码器前，把分辨率「填充」到偶数宽高，
    # 避免某些编码器（例如 libsvtav1 / AV1 4:2:0）对奇数高度报错。
    base_args = [
        FFMPEG_PATH, 
        '-y',               # 覆盖已存在的文件 (虽然我们的脚本会跳过, 但这是个保险)
        '-i', str(input_file),
        '-loglevel', 'error', # 只在真正出错时显示日志
        # 确保宽高都是偶数: pad=ceil(iw/2)*2:ceil(ih/2)*2
        # 这样对原图只会在需要时补 1 像素边缘，而不会裁剪内容。
        '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2'
    ]
    # -still-picture 1 是将单张图片编码为 AVIF (AV1 'still') 的关键！
    if encoder_type == "NVIDIA":
        return base_args + [
            '-c:v', 'libsvtav1',     # 使用 NVIDIA AV1 编码器
            # '-preset', '0',         # p1 (最快) -> p7 (最慢/最好)
            '-crf', str(quality),     # Constant Quality (越低越好)
            '-still-picture', '1',
            '-pix_fmt', 'yuv444p',
            '-f', 'avif',
            str(output_file)
        ]
    elif encoder_type == "INTEL":
        return base_args + [
            '-c:v', 'av1_qsv',       # 使用 Intel QSV AV1 编码器
            '-preset', 'slower',     # preset
            '-q:v', str(quality),    # Global Quality (越低越好)
            '-still-picture', '1',
            str(output_file)
        ]
    elif encoder_type == "AMD":
        return base_args + [
            '-c:v', 'av1_amf',       # 使用 AMD AMF AV1 编码器
            '-quality', 'speed',     # 质量预设 (speed, balanced, quality)
            '-q:v', str(quality),    # Global Quality (越低越好)
            '-still-picture', '1',
            str(output_file)
        ]
    else:
        # Fallback 或 抛出错误
        raise ValueError(f"不支持的 ENCODER_TYPE: {encoder_type}")


def convert_image(file_path, input_base, output_base):
    """
    调用 ffmpeg 转换单个图片文件。
    """
    try:
        # 1. 计算和创建路径 (与之前相同)
        relative_path = file_path.relative_to(input_base)
        output_path = output_base / relative_path
        output_path_avif = output_path.with_suffix('.avif')
        output_path_avif.parent.mkdir(parents=True, exist_ok=True)

        # 2. 生成 ffmpeg 命令
        cmd = get_ffmpeg_cmd(ENCODER_TYPE, file_path, output_path_avif, GPU_QUALITY)
        
        # 3. 执行命令
        # capture_output=True 会捕获 stdout 和 stderr
        # text=True 会将它们解码为字符串
        result = subprocess.run(
        cmd,
        check=False,  # 或者直接不写 check
        capture_output=True,
        text=True,
        encoding='utf-8',
        )

        if result.returncode != 0:
            print("命令执行失败！")
            print("返回码：", result.returncode)
            print("标准输出：", result.stdout)
            print("错误输出：", result.stderr)
            print("执行的命令：", ' '.join(cmd))
    
        return (str(file_path), "Success")

    except subprocess.CalledProcessError as e:
        # 如果 ffmpeg 返回非 0 退出码 (即失败)
        error_message = f"Failed: ffmpeg 错误.\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}"
        return (str(file_path), error_message)
    except Exception as e:
        # 捕获其他 Python 错误 (如路径问题)
        return (str(file_path), f"Failed: {e}")

def check_ffmpeg():
    """检查 ffmpeg 是否可用"""
    if shutil.which(FFMPEG_PATH) is None:
        print(f"❌ 严重错误: 未找到 'ffmpeg'。")
        print(f"  请确保 ffmpeg 已安装，并且 '{FFMPEG_PATH}' 是正确的路径。")
        print("  如果已安装，请将其添加到系统 PATH 或修改脚本中的 FFMPEG_PATH 变量。")
        return False
    
    # 可以在此添加更复杂的检查, 比如检查 av1_nvenc 是否真的存在
    print(f"✅ 'ffmpeg' 已找到: {shutil.which(FFMPEG_PATH)}")
    return True

def batch_convert(input_dir, output_dir):
    """
    主函数：查找文件并使用线程池 + ffmpeg 进行转换。
    """
    if not check_ffmpeg():
        return

    start_time = time.time()
    input_base = Path(input_dir)
    output_base = Path(output_dir)

    if not input_base.is_dir():
        print(f"❌ 错误: 输入目录 '{input_dir}' 不存在或不是一个文件夹。")
        return

    print(f"📁 正在从 '{input_base}' 查找图片...")

    # 1. 查找所有文件 (与之前相同)
    files_to_convert = []
    for ext in SUPPORTED_EXTENSIONS:
        files_to_convert.extend(input_base.rglob(f'*{ext}'))
        files_to_convert.extend(input_base.rglob(f'*{ext.upper()}'))

    if not files_to_convert:
        print("🟡 未找到任何支持的图片文件。")
        return

    total_found = len(files_to_convert)
    print(f"🖼️ 总共找到 {total_found} 个图片文件。")

    # 2. 检查已存在的文件 (与之前相同)
    print("🔍 正在检查哪些文件需要转换 (跳过已存在的文件)...")
    tasks_to_submit = []
    skipped_count = 0
    for file_path in files_to_convert:
        relative_path = file_path.relative_to(input_base)
        output_path_avif = (output_base / relative_path).with_suffix('.avif')
        if output_path_avif.exists():
            if os.stat(output_path_avif).st_size > 0:
                skipped_count += 1
            else:
                os.remove(output_path_avif)
                tasks_to_submit.append(file_path)
        else:
            tasks_to_submit.append(file_path)

    print(f"✅ {skipped_count} 个文件已被跳过 (目标 AVIF 文件已存在)。")
    if not tasks_to_submit:
        print("✨ 所有文件均已转换，无需操作。")
        return
    
    print(f"🚀 准备转换 {len(tasks_to_submit)} 个新文件...")

    # 3. 设置线程池 (注意：这里是 ThreadPoolExecutor)
    tasks = []
    results = []
    print(f"🚀 开始转换... (使用 {MAX_WORKERS} 个并发 ffmpeg 线程, GPU: {ENCODER_TYPE})")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for file_path in tasks_to_submit:
            task = executor.submit(convert_image, file_path, input_base, output_base)
            tasks.append(task)
        
        for future in tqdm(as_completed(tasks), total=len(tasks_to_submit), desc="转换进度"):
            results.append(future.result())

    # 5. 打印总结 (与之前相同)
    end_time = time.time()
    success_count = 0
    failed_files = []

    for (file, status) in results:
        if status == "Success":
            success_count += 1
        else:
            failed_files.append((file, status))
            
    print("\n--- ✨ 转换完成 ✨ ---")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    print("\n--- 📊 本次运行总结 ---")
    print(f"总共找到源文件: {total_found}")
    print(f"跳过 (已存在): {skipped_count}")
    print(f"尝试转换:       {len(tasks_to_submit)}")
    print(f"  - 成功:         {success_count}")
    
    if failed_files:
        print(f"  - 失败:         {len(failed_files)}")
        print("\n失败文件列表 (及 ffmpeg 错误):")
        for file, error in failed_files:
            print(f"  - {file}\n    原因: {error}")
    else:
         print(f"  - 失败:         0")

# 运行主程序
if __name__ == "__main__":
    if INPUT_DIR == "./my_image_folder":
        print("⚠️ 警告: 请先修改脚本中的 'INPUT_DIR' 和 'OUTPUT_DIR' 变量！")
        print("⚠️ 警告: 同时必须检查 'ENCODER_TYPE' 和 'FFMPEG_PATH' 是否配置正确！")
    else:
        batch_convert(INPUT_DIR, OUTPUT_DIR)
