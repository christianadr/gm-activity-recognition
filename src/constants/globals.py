from pathlib import Path

PROJECT_DIR = Path(*Path(__file__).parts[:-3])
RAW_DIR = Path(PROJECT_DIR, "data", "raw")
PROCESSED_DIR = Path(PROJECT_DIR, "data", "processed")
