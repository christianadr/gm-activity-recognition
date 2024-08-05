from pathlib import Path

from constants import globals as g
from tqdm import tqdm


def main():
    total_files = sum(
        len(list(Path(g.RAW_DIR, action).iterdir())) for action in g.CLASS_NAMES.keys()
    )
    non_matching_files = []

    with tqdm(total=total_files, desc="Checking files", unit="file") as pbar:
        for action in g.CLASS_NAMES.keys():
            target_path = Path(g.RAW_DIR, action)

            for item in target_path.iterdir():
                if action not in item.name:
                    non_matching_files.append(item)
                pbar.update(1)

    if non_matching_files:
        print("Files not belonging to the specified action:")
        for file in non_matching_files:
            print(file)
    else:
        print("All files matched!")


if __name__ == "__main__":
    main()
