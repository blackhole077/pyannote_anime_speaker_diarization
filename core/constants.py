import os
from pathlib import Path

SAMPLE_RATE = 16000  # 16 kHz
FRAME_LENGTH = 25  # 25 millisecond frames
FRAME_HOP = 10  # 10 millisecond
FFT_SIZE = 512
MEL_BIN = 64
DATASET_ROOT = Path("/data/voice_activity_detection_dataset/")
AUDIO_ROOT = DATASET_ROOT / "audio"
RTTM_ROOT = DATASET_ROOT / "rttm"
HF_AUTH_TOKEN = os.environ.get("HF_AUTH_TOKEN")

EPSILON: float = 1e-8

### File Path Constants
REPO_ROOT = Path(__file__).parent.parent
CONFIG_ROOT = Path(REPO_ROOT / "configs")
