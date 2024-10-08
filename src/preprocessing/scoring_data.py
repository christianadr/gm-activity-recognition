from pathlib import Path

import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from constants import globals as g

model_path = Path(g.MODELS_PATH, "skeleton", "pose_landmarker_heavy.task").__str__()

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options, output_segmentation_masks=True
)
detector = vision.PoseLandmarker.create_from_options(options)


class CreateScoringDataset:
    def __init__(self, src: Path):
        self.src = src

    def read_all_data(self):
        return list(self.src.iterdir())

    def detect_keypoints(self, image_path: str):
        image = mp.Image.create_from_file(image_path)
        return detector.detect(image)

    def create_dataframe(self):
        keypoint_data = []
        for image_path in self.read_all_data():
            _landmarks = {}
            _results = self.detect_keypoints(image_path.__str__())
            # print(f"Path: {image_path}")
            # print(_results)
            try:
                for idx in range(len(_results.pose_landmarks[0])):
                    _landmark = _results.pose_landmarks[0][idx]

                    _landmarks[f"{g.ANNOT[idx]}_x"] = _landmark.x
                    _landmarks[f"{g.ANNOT[idx]}_y"] = _landmark.y
                    _landmarks[f"{g.ANNOT[idx]}_z"] = _landmark.z

                _landmarks["image"] = image_path
                keypoint_data.append(_landmarks)
            except:
                print(f"Landmarks not detected on image {image_path}")

        return pd.DataFrame(keypoint_data)
