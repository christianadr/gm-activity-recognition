import pathlib

import cv2
from tqdm import tqdm

from constants import globals as g


def extract_frames(
    src: str,
    dest: str,
    # activity: str,
    max_frames: int = 30,
):
    """Extract frames from video.

    Parameters
    ----------
    src : str
        Source path of video.
    dest : str
        Destination folder for saving extracted frames
    max_frames : int, optional
        Number of frames to be extracted, by default 24
    """
    # print("\\".join(dest.split("\\")[:-1]))
    pathlib.Path("\\".join(dest.split("\\")[:-1])).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(src)

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret or frame_count >= max_frames:
            break

        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        output = pathlib.Path(f"{dest}_frame-{frame_count}.jpg")
        cv2.imwrite(output.__str__(), frame)

        frame_count += 1

    cap.release()


def extraction_on_multiple_vids(src: pathlib.Path):
    """Iterates video clips from raw directory.

    Parameters
    ----------
    src : pathlib.Path
        Source path of raw clips.
    """
    raw_clips = list(src.iterdir())

    for vid in tqdm(
        raw_clips,
        desc=f"Extracting frames from {src.__str__()}",
        total=len(raw_clips),
        leave=True,
    ):
        # print(vid)
        # print(vid.name.split(".")[0].split("-")[0])
        extract_frames(
            vid.__str__(),
            pathlib.Path(
                g.PROCESSED_DIR,
                vid.name.split(".")[0].split("-")[0],
                vid.name.split(".")[0],
            ).__str__(),
        )


def main():
    # vid = "center-1_gallop_1_1.mp4"
    # src = pathlib.Path(g.RAW_DIR, vid).__str__()
    # dest = pathlib.Path(g.PROCESSED_DIR, vid.split(".")[0]).__str__()
    # extract_frames(src, dest)
    extraction_on_multiple_vids(pathlib.Path(g.RAW_DIR, "gm-activities"))


if __name__ == "__main__":
    main()
