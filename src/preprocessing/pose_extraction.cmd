@echo off
setlocal enabledelayexpansion

rem Set the directory containing video files
set "video_dir=D:\Users\Ainsley\Desktop\project-design\gm-activity-recognition\data\raw"

rem Iterate over each video file in the directory
for %%F in ("%video_dir%\*.mp4") do (
    rem Extract file name without extension
    set "video_name=%%~nF"

    rem Execute the Python script for each video
    C:\Users\drchr\.conda\envs\grossmotor\python.exe D:\Users\Ainsley\Desktop\project-design\gm-activity-recognition\mmaction2\tools\data\skeleton\ntu_pose_extraction.py "!video_directory!\!video_name!.mp4" "!video_directory!\!video_name!.pkl"
)

echo All videos processed.