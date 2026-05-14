import librosa
import numpy as np

from core.constants import SAMPLE_RATE


def convert_to_mono_and_resample(
    data: np.ndarray,
    original_sample_rate: int,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Convert audio data from stereo output to mono output,
    resample to a new sample rate.

    Args:
        data (np.ndarray): The WAV data as np.float32 values. Its expected shape
        is (num_samples, num_channels).
        original_sample_rate (int): The original sample rate of the audio data.
        sample_rate (int, optional): The sample rate to use for resampling.
        Defaults to SAMPLE_RATE.
    """

    # If it's already in mono and in the expected format then skip it.
    if data.ndim == 1 and original_sample_rate == sample_rate:
        return data
    # Check if the audio is stereo (has more than 1 dimension and 2 channels)
    if data.ndim > 1 and data.shape[1] >= 2:
        # Average the channels to create a mono signal
        # The axis=1 argument ensures averaging along the channel dimension
        data = np.mean(data, axis=1)
    elif data.ndim == 1:
        # The file is already mono, so no conversion is needed
        pass
    data = data.T
    resampled_data = librosa.resample(
        data, orig_sr=original_sample_rate, target_sr=sample_rate
    )
    resampled_data = resampled_data.T
    return resampled_data
