from pathlib import Path

PROJECT_DIR = Path(*Path(__file__).parts[:-3])

# DATASETS
RAW_DIR = Path(PROJECT_DIR, "data", "raw")
PROCESSED_DIR = Path(PROJECT_DIR, "data", "processed")

GROSSMOTOR_DIR = Path(PROJECT_DIR, "data", "grossmotor")
TRAIN_DIR = Path(GROSSMOTOR_DIR, "train")
VAL_DIR = Path(GROSSMOTOR_DIR, "val")
