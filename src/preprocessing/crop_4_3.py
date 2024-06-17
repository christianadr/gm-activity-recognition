from pathlib import Path

from constants import globals as g
from moviepy.editor import VideoFileClip
from tqdm import tqdm


def center_crop(input_path, output_path, aspect_ratio=(4, 3)):
    video = VideoFileClip(str(input_path))

    original_width, original_height = video.size

    # Calculate new dimensions based on the aspect ratio
    target_width, target_height = aspect_ratio
    new_height = original_width * target_height / target_width

    if new_height > original_height:
        new_height = original_height
        new_width = new_height * target_width / target_height
    else:
        new_width = original_width

    # Calculate center cropping box
    x_center = original_width / 2
    y_center = original_height / 2
    crop_x1 = x_center - new_width / 2
    crop_x2 = x_center + new_width / 2
    crop_y1 = y_center - new_height / 2
    crop_y2 = y_center + new_height / 2

    # Perform the cropping
    cropped_video = video.crop(x1=crop_x1, y1=crop_y1, x2=crop_x2, y2=crop_y2)

    # Write the processed video to the output file
    cropped_video.write_videofile(str(output_path))


# Define input and output directories
input_dir = g.RAW_DIR
output_dir = g.PROCESSED_DIR

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# Process all video files in the input directory
video_files = list(input_dir.glob("*.mp4"))  # Adjust the glob pattern if needed

for input_video_path in tqdm(video_files, desc="Processing videos"):
    output_video_path = output_dir / input_video_path.name
    center_crop(input_video_path, output_video_path)
