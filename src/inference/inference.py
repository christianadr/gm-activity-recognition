import pickle
import tempfile
from pathlib import Path
from typing import Tuple

import cv2

# import mmcv
import mmengine
import numpy as np
import settings
import torch
from constants import globals as g
from mmaction.apis import (
    detection_inference,
    inference_skeleton,
    init_recognizer,
    pose_inference,
)

# from mmaction.registry import VISUALIZERS
from mmaction.utils import frame_extract

# from mmengine.utils import track_iter_progress

try:
    import moviepy.video.io.ImageSequenceClip as mpy
except ImportError:
    raise ImportError("Please install moviepy to enable output file")


def run_inference(
    video_path: str,
    output_path: str = "./vid",
    **kwargs,
) -> Tuple[str, float]:
    """Run action recognition inference.

    Parameters
    ----------
    video_path : str
        Source path of video data.
    output_path : str, optional
        Destination path of video, if visualization is true, by default "./vid"

    Returns
    -------
    Tuple[str, float]
        Returns predicted label with corresponding confidence score.

    Raises
    ------
    ValueError
        Returns ValueError if source video cannot be read.
    """

    if output_path == "./vid":
        Path(output_path).mkdir(
            parents=True,
            exist_ok=True,
        )

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    tmp_dir = tempfile.TemporaryDirectory()

    def clipped_one_sec() -> str:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        duration = total_frames / fps

        start_time = (duration / 2) - 0.5
        start_frame = int(start_time * fps)

        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for _ in range(fps):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()

        clip_path = f"{tmp_dir.name}/1-sec-clip.mp4"
        out = cv2.VideoWriter(
            clip_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (
                frames[0].shape[1],
                frames[0].shape[0],
            ),
        )

        for frame in frames:
            out.write(frame)

        out.release()
        return clip_path

    src_clipped_vid = clipped_one_sec()

    frame_paths, frames = frame_extract(
        src_clipped_vid,
        short_side=480,
        out_dir=tmp_dir.name,
    )

    h, w, _ = frames[0].shape

    torch.cuda.empty_cache()
    det_results, _ = detection_inference(
        **kwargs["det_config"],
        frame_paths=frame_paths,
        device=device,
    )

    # print(np.array(det_results).shape)
    torch.cuda.empty_cache()
    pose_results, pose_data_samples = pose_inference(
        **kwargs["pose_config"],
        frame_paths=frame_paths,
        det_results=det_results,
        device=device,
    )
    print(pose_results)
    with open("./pose-results", "wb") as f:
        pickle.dump(pose_results, f)

    torch.cuda.empty_cache()
    config = mmengine.Config.fromfile(kwargs["model_config"])
    config.merge_from_dict({})

    model = init_recognizer(
        config=config,
        checkpoint=kwargs["model_checkpoint"],
        device=device,
    )

    result = inference_skeleton(
        model=model,
        pose_results=pose_results,
        img_shape=(h, w),
    )

    max_idx = result.pred_score.argmax().item()
    scores = result.pred_score.cpu().numpy()

    tmp_dir.cleanup()

    action_label = kwargs["label_map"][max_idx]
    action_score = scores[max_idx]

    return action_label, action_score


def main():
    filename = "test-gallop.mp4"
    src_vid = Path(g.TEST_DATA_DIR, filename)

    label, score = run_inference(
        video_path=src_vid,
        model_config=settings.MODEL_CONFIG,
        model_checkpoint=settings.MODEL_CHECKPOINT,
        det_config=settings.DET_CONFIG,
        pose_config=settings.POSE_CONFIG,
        label_map=settings.LABEL_MAP,
    )

    print(
        f"Predicted action: {label.upper()} with confidence score of {round((score*100), 4)}%."
    )


if __name__ == "__main__":
    main()
