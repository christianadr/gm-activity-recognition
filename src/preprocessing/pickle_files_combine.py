import pickle
from pathlib import Path

from constants import globals as g
from sklearn.model_selection import train_test_split


def combine_pickles_from_directory(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)

    combined_data = {
        "split": {"train": [], "val": []},
        "annotations": [],
    }

    # Iterate through files in the src
    video_files = list(src.iterdir())

    train_data, val_data = train_test_split(video_files, test_size=0.2, random_state=42)

    for video_file in train_data:
        with open(video_file, "rb") as f:
            data = pickle.load(f)
            # print(data)
            combined_data["split"]["train"].append(data["frame_dir"])
            combined_data["annotations"].append(data)

    for video_file in val_data:
        with open(video_file, "rb") as f:
            data = pickle.load(f)
            # print(data)
            combined_data["split"]["val"].append(data["frame_dir"])
            combined_data["annotations"].append(data)

    desired_key_arrangement = [
        "frame_dir",
        "label",
        "img_shape",
        "original_shape",
        "total_frames",
        "keypoint",
        "keypoint_score",
    ]
    # for idx in enumerate(combined_data["annotations"]):
    combined_data["annotations"][0] = {
        key: combined_data["annotations"][0][key]
        for key in desired_key_arrangement
        if key in combined_data["annotations"][0]
    }

    target_path = Path(dst, "grossmotor_2d_v2.pkl").__str__()
    with open(target_path, "wb") as f:
        pickle.dump(combined_data, f)


def main():
    combine_pickles_from_directory(
        src=g.GROSSMOTOR_DIR_PK2, dst=g.GROSSMOTOR_DIR_PK_COMBINED
    )


if __name__ == "__main__":
    main()
