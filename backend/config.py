from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
CHECKPOINTS_DIR = ROOT / 'checkpoints'
MODELS_DIR = ROOT / 'models'
UTILS_DIR = ROOT / 'utils'
