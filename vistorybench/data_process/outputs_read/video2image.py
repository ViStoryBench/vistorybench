import os
import cv2

def extract_first_frame(video_path, output_folder):
    """
    Extract the first frame of a video and save it as an image
    
    Parameters:
        video_path (str): Video file path
        output_folder (str): Output image folder path
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Read video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video file: {video_path}")
        return
    
    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print(f"Cannot read video frame: {video_path}")
        return
    
    # Generate output filename (same name as video, extension changed to .jpg)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_folder, f"{video_name}.jpg")
    
    # Save first frame
    cv2.imwrite(output_path, frame)
    print(f"Saved: {output_path}")
    
    # Release video resources
    cap.release()

def process_videos_in_folder(folder_path, output_folder):
    """
    Process all MP4 video files in a folder
    
    Parameters:
        folder_path (str): Folder path containing videos
        output_folder (str): Output image folder path
    """
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.mp4'):
            video_path = os.path.join(folder_path, filename)
            extract_first_frame(video_path, output_folder)

if __name__ == "__main__":
    # Input folder path
    input_folder = "data/outputs/vlogger/img_ref/vistorybench_en/04/20250507-115229/video/origin_video"
    
    # Output folder path (can be modified to the desired location)
    output_folder = "data/outputs/vlogger/img_ref/vistorybench_en/04/20250507-115229/first_frames"
    
    # Process all videos
    process_videos_in_folder(input_folder, output_folder)