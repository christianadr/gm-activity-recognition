from moviepy.editor import VideoFileClip
from pathlib import Path
from tqdm import tqdm
from constants import globals as g

def crop_to_4_3(input_path, output_path):
    video = VideoFileClip(str(input_path))

    original_width, original_height = video.size

    new_height = original_width * 3 / 4
    
    if new_height > original_height:
        new_height = original_height
        new_width = new_height * 4 / 3
    else:
        new_width = original_width
    
    # Calculate the position to start cropping (center cropping)
    x_center = original_width / 2
    y_center = original_height / 2
    crop_x1 = x_center - new_width / 2
    crop_x2 = x_center + new_width / 2
    crop_y1 = y_center - new_height / 2
    crop_y2 = y_center + new_height / 2
    
    # Crop the video
    cropped_video = video.crop(x1=crop_x1, y1=crop_y1, x2=crop_x2, y2=crop_y2)
    
    # Write the cropped video to the output file
    cropped_video.write_videofile(str(output_path))

# Define input and output directories
input_dir = Path(g.RAW_DIR)
output_dir = Path(g.PROCESSED_DIR)

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# Process all video files in the input directory
video_files = list(input_dir.glob("*.mp4"))  # Adjust the glob pattern if needed

for input_video_path in tqdm(video_files, desc="Processing videos"):
    output_video_path = output_dir / input_video_path.name
    crop_to_4_3(input_video_path, output_video_path)
