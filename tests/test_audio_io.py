import numpy as np
import numpy.testing as npt
import pytest

from core.preprocess.utils import convert_to_mono_and_resample


class TestConvertToMonoAndResample:

    @pytest.mark.parametrize(
        "channels,input_sr,target_sr",
        [(1, 16000, 16000), (1, 44100, 16000), (2, 16000, 16000), (2, 44100, 16000)],
    )
    def test_convert(self, channels, input_sr, target_sr):
        duration = 1  # 1 second
        frequency = 1000  # 1 KHz
        # Generate wave data
        t = np.linspace(0, duration, int(input_sr * duration), endpoint=False).astype(
            np.float32
        )
        audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)  # Creates a mono wave
        if channels > 1:
            audio_data = np.stack([audio_data] * channels, axis=1)  # Make a stereo wave
        resampled_data = convert_to_mono_and_resample(audio_data, input_sr, target_sr)
        # Basic checks to verify shape and data
        assert resampled_data.shape == (target_sr * duration,)
        assert resampled_data.dtype == np.float32
        # Verify whether the peak FFT bin is the frequency of the wave
        freqs = np.fft.rfftfreq(len(resampled_data), d=1 / target_sr)
        peak_bin = np.argmax(np.abs(np.fft.rfft(resampled_data)))
        resampled_peak_freq = freqs[peak_bin]

        assert abs(resampled_peak_freq - frequency) < 1.0

    def test_stereo_averages_channels(self):
        left_channel = np.ones(1000, dtype=np.float32)
        right_channel = -np.ones(1000, dtype=np.float32)
        stereo_wave = np.stack([left_channel, right_channel], axis=1)
        out = convert_to_mono_and_resample(stereo_wave, 16000, 16000)
        npt.assert_allclose(out, np.zeros(1000), atol=1e-7)
