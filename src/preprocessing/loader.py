from pathlib import Path
from shutil import copyfile

from constants import globals as g
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


def video_splitting(
    src: Path,
    train_dir: Path,
    val_dir: Path,
    val_size: float = 0.2,
):
    # Create train and validation directories
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # Check if train_dir and val_dir already contain files
    if any(train_dir.iterdir()) or any(val_dir.iterdir()):
        print("Train or validation directories already contain files. Skipping...")
        return

    # videos_files = [path.__str__() for path in list(src.iterdir())]

    videos_files = list(src.iterdir())
    labels = [vid.__str__().split("_")[1] for vid in videos_files]

    # Encode labels
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)

    # Split data into train val while keeping labels balanced
    train_files, val_files, train_labels, val_labels = train_test_split(
        videos_files,
        encoded_labels,
        test_size=val_size,
        stratify=encoded_labels,
        random_state=42,
    )

    for video, label in tqdm(
        zip(train_files, train_labels),
        desc="Copying videos to train",
        total=len(train_files),
        leave=True,
    ):
        source = Path(src, video.name)
        destination = Path(train_dir, video.name)
        copyfile(src=source, dst=destination)

    for video, label in tqdm(
        zip(val_files, val_labels),
        desc="Copying videos to val",
        total=len(val_files),
        leave=True,
    ):
        source = Path(src, video.name)
        destination = Path(val_dir, video.name)
        copyfile(src=source, dst=destination)


def generate_annotation_file(src, annotation_file):
    video_files = [vid.name.__str__() for vid in list(src.iterdir())]

    labels = [vid.__str__().split("_")[1] for vid in video_files]

    # Encode labels
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)

    with open(annotation_file, "w") as f:
        for video, label in tqdm(
            zip(video_files, encoded_labels),
            desc="Generating annotation file",
            total=len(video_files),
            leave=True,
        ):
            f.write(f"{video}\t{label}\n")


def main():
    src = g.RAW_DIR
    train_dir = g.TRAIN_DIR
    val_dir = g.VAL_DIR

    video_splitting(src=src, train_dir=train_dir, val_dir=val_dir)

    train_annotation_file = "data/grossmotor/grossmotor_locomotor_train_video.txt"
    val_annotation_file = "data/grossmotor/grossmotor_locomotor_val_video.txt"

    generate_annotation_file(train_dir, train_annotation_file)
    generate_annotation_file(val_dir, val_annotation_file)


if __name__ == "__main__":
    main()
