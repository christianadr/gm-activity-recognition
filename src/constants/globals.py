from pathlib import Path

PROJECT_DIR = Path(*Path(__file__).parts[:-3])

# DATASETS
RAW_DIR = Path(PROJECT_DIR, "data", "raw")
PROCESSED_DIR = Path(PROJECT_DIR, "data", "processed")

GROSSMOTOR_DIR = Path(PROJECT_DIR, "data", "grossmotor")
TRAIN_DIR = Path(GROSSMOTOR_DIR, "train")
VAL_DIR = Path(GROSSMOTOR_DIR, "val")

GROSSMOTOR_DIR_PK = Path(PROJECT_DIR, "data", "grossmotor-pose")
GROSSMOTOR_DIR_PK2 = Path(PROJECT_DIR, "data", "grossmotor-pose-v2")
GROSSMOTOR_DIR_PK_COMBINED = Path(PROJECT_DIR, "data", "grossmotor-pose-combined")

# CLASS NAMES
CLASS_NAMES = {
    "gallop": 0,
    "hop": 1,
    "jump": 2,
    "leap": 3,
    "run": 4,
    "skip": 5,
    "slide": 6,
}
