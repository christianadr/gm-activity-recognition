from pathlib import Path

PROJECT_DIR = Path(*Path(__file__).parts[:-3])

# DATASETS
RAW_DIR = Path(PROJECT_DIR, "data", "raw")
PROCESSED_DIR = Path(PROJECT_DIR, "data", "processed")
TEST_DATA_DIR = Path(PROJECT_DIR, "data", "tests")

GROSSMOTOR_DIR = Path(PROJECT_DIR, "data", "grossmotor")
TRAIN_DIR = Path(GROSSMOTOR_DIR, "train")
VAL_DIR = Path(GROSSMOTOR_DIR, "val")

GROSSMOTOR_DIR_PK = Path(PROJECT_DIR, "data", "grossmotor-pose")
GROSSMOTOR_DIR_PK2 = Path(PROJECT_DIR, "data", "grossmotor-pose-v2")
GROSSMOTOR_DIR_HEATMAPS = Path(PROJECT_DIR, "data", "heatmap")
GROSSMOTOR_DIR_PK_COMBINED = Path(PROJECT_DIR, "data", "grossmotor-pose-combined")

# CLASS NAMES
CLASS_NAMES = {
    "gallop": 0,
    "hop": 1,
    "jump": 2,
    "leap": 3,
    "run": 4,
    # "skip": 5,
    "slide": 5,
}

LABEL_MAP_PATH = Path(PROJECT_DIR, "src", "inference", "label_map_grossmotor.txt")

MODELS_PATH = Path(PROJECT_DIR, "models")

# ANNOT = {
#     0: "nose",
#     1: "left_eye",
#     2: "right_eye",
#     3: "left_ear",
#     4: "right_ear",
#     5: "left_shoulder",
#     6: "right_shoulder",
#     7: "left_elbow",
#     8: "right_elbow",
#     9: "left_wrist",
#     10: "right_wrist",
#     11: "left_hip",
#     12: "right_hip",
#     13: "left_knee",
#     14: "right_knee",
#     15: "left_ankle",
#     16: "right_ankle",
# }

ANNOT = {
    0: "nose",
    1: "left_eye_inner",
    2: "left_eye",
    3: "left_eye_outer",
    4: "right_eye_inner",
    5: "right_eye",
    6: "right_eye_outer",
    7: "left_ear",
    8: "right_ear",
    9: "mouth_left",
    10: "mouth_right",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    17: "left_pinky",
    18: "right_pinky",
    19: "left_index",
    20: "right_index",
    21: "left_thumb",
    22: "right_thumb",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index",
}
