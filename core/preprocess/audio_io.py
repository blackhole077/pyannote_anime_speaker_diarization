import numpy as np
import soundfile as sf

from core.constants import SAMPLE_RATE
from core.preprocess.utils import convert_to_mono_and_resample


def read_file_and_get_duration(
    file_path: str,
) -> tuple[np.ndarray, int, float]:
    """Read a Waveform Audio File (WAV) and return the data as a Numpy array, alongside
    the sample rate and duration of the file in seconds.

    Args:
        file_path (str): The path to the file, including its extension.

    Returns:
        tuple[np.ndarray, int, float]: Returns a tuple of the following:
            - data (np.ndarray): The WAV data as np.float32 values. Its expected shape
            is (num_samples, num_channels).
            - sample_rate (int): The sample rate of the WAV file.
            - duration_in_seconds (float): The duration of the WAV file in seconds.
            The decimal places indicate the number of milliseconds.
    """

    data: np.ndarray
    sample_rate: int
    data, sample_rate = sf.read(file_path, dtype="float32")
    duration_in_seconds: float = len(data) / sample_rate  # expect up to 4 sig. figures
    return data, sample_rate, duration_in_seconds


def load_mono_resampled(file_path, sample_rate=SAMPLE_RATE):
    data, file_sample_rate, _ = read_file_and_get_duration(file_path)
    return convert_to_mono_and_resample(data, file_sample_rate, sample_rate=sample_rate)


def write_wav(
    data: np.ndarray,
    output_file: str,
    sample_rate=SAMPLE_RATE,
):
    sf.write(
        output_file,
        data,
        samplerate=sample_rate,
        format="wav",
        subtype="PCM_16",
    )
