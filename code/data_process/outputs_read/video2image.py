import os
import cv2

def extract_first_frame(video_path, output_folder):
    """
    提取视频的第一帧并保存为图片
    
    参数:
        video_path (str): 视频文件路径
        output_folder (str): 输出图片的文件夹路径
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 读取视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return
    
    # 读取第一帧
    ret, frame = cap.read()
    if not ret:
        print(f"无法读取视频帧: {video_path}")
        return
    
    # 生成输出文件名（与视频同名，扩展名改为.jpg）
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_folder, f"{video_name}.jpg")
    
    # 保存第一帧
    cv2.imwrite(output_path, frame)
    print(f"已保存: {output_path}")
    
    # 释放视频资源
    cap.release()

def process_videos_in_folder(folder_path, output_folder):
    """
    处理文件夹中的所有MP4视频文件
    
    参数:
        folder_path (str): 包含视频的文件夹路径
        output_folder (str): 输出图片的文件夹路径
    """
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.mp4'):
            video_path = os.path.join(folder_path, filename)
            extract_first_frame(video_path, output_folder)

if __name__ == "__main__":
    # 输入文件夹路径
    input_folder = "/data/AIGC_Research/Story_Telling/StoryVisBMK/data/outputs/vlogger/img_ref/WildStory_en/04/20250507-115229/video/origin_video"
    
    # 输出文件夹路径（可以修改为你想保存的位置）
    output_folder = "/data/AIGC_Research/Story_Telling/StoryVisBMK/data/outputs/vlogger/img_ref/WildStory_en/04/20250507-115229/first_frames"
    
    # 处理所有视频
    process_videos_in_folder(input_folder, output_folder)