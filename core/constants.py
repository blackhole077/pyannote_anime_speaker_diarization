import os
from pathlib import Path
from dotenv import load_dotenv

### File Path Constants
REPO_ROOT = Path(__file__).parent.parent
CONFIG_ROOT = Path(REPO_ROOT / "configs")
ENV_PATH = REPO_ROOT / ".env"

# Load the environment variables for subsequent constants
load_dotenv(dotenv_path=ENV_PATH, override=True)
DATASET_ROOT = Path(os.environ["ANIME_DATA_ROOT"])
AUDIO_ROOT = DATASET_ROOT / "audio"
ALL_LST = AUDIO_ROOT / "all_items.lst"
RTTM_ROOT = DATASET_ROOT / "rttm"
ALL_UEM = RTTM_ROOT / "all_items.uem"
ALL_RTTM = RTTM_ROOT / "all_items.rttm"

### Model Constants
HF_AUTH_TOKEN = os.environ["HF_AUTH_TOKEN"]
SAMPLE_RATE = 16000  # 16 kHz
FRAME_LENGTH = 25  # 25 millisecond frames
FRAME_HOP = 10  # 10 millisecond
FFT_SIZE = 512
MEL_BIN = 64
EPSILON: float = 1e-8

### ETL Constants
RESERVED_LABELS = frozenset({"overlap", "unknown", "ignore"})
