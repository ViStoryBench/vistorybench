from huggingface_hub import HfApi
from dotenv import load_dotenv
import os
load_dotenv()
api = HfApi(token=os.getenv("HF_TOKEN"))


api.upload_large_folder(
    folder_path="/root/vistorybench/data/outputs",
    repo_id="ViStoryBench/VistoryBenchResultv2",
    repo_type="dataset",  # 根据你的仓库类型修改
)

print("文件上传完成！")