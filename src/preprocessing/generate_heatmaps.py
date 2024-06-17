import io
import pickle
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
from constants import globals as g
from PIL import Image  # Add this import
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


def generate_heatmap(src: str, dst: str = "./"):
    with open(src, "rb") as f:
        pkl_data = pickle.load(f)

        frames = []
        num_frames = len(pkl_data["keypoint"][0])

        for frame_index in range(num_frames):
            keypoints = pkl_data["keypoint"][0][frame_index]

            heatmap = np.zeros(pkl_data["img_shape"][:2])

            spread = 10
            for joint in keypoints:
                print(joint[0], joint[1])
                x, y = int(joint[0]), int(joint[1])
                if 0 <= x < heatmap.shape[1] and 0 <= y < heatmap.shape[0]:
                    heatmap[y, x] = 1

            heatmap_blurred = gaussian_filter(heatmap, sigma=spread)
            heatmap_normalized = heatmap_blurred / heatmap_blurred.max()

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(heatmap_normalized, interpolation="nearest")
            ax.axis("off")

            # Save the frame to a buffer
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
            buf.seek(0)
            plt.close(fig)

            # Resize the image to 224x224
            image = Image.open(buf)
            image = image.resize((224, 224), Image.Resampling.LANCZOS)
            frames.append(np.array(image))
            # buf = io.BytesIO()
            # plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
            # buf.seek(0)
            # frames.append(imageio.imread(buf))
            # plt.close(fig)

        imageio.mimsave(
            Path(dst, f"{pkl_data['frame_dir']}.gif").__str__(), frames, duration=1
        )


def convert_to_heatmaps(src: Path, dst: Path):
    pkl_files = list(src.iterdir())
    dst.mkdir(parents=True, exist_ok=True)

    for pkl in tqdm(
        pkl_files,
        total=len(pkl_files),
        desc="Converting clips to heatmaps",
    ):
        generate_heatmap(pkl.__str__(), Path(dst.__str__()))


def main():
    src_dir = Path(g.GROSSMOTOR_DIR_PK, "center-cropped")
    dst_dir = g.GROSSMOTOR_DIR_HEATMAPS
    convert_to_heatmaps(src_dir, dst_dir)


if __name__ == "__main__":
    main()
