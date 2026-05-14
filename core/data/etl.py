import abc
import threading
from glob import glob
from pathlib import Path
from typing import List, Optional

import librosa
import numpy as np
import soundfile as sf

from core.constants import AUDIO_ROOT, DATASET_ROOT, RTTM_ROOT, SAMPLE_RATE


class AudioDataManager(metaclass=abc.ABCMeta):
    _instances: dict[str, "AudioDataManager"] = {}
    _lock = threading.Lock()  # For thread safety

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            if abc.ABCMeta.__abstractmethods__(cls):
                raise TypeError(
                    f"Class {cls} is an abstract class and therefore cannot be instantiated."
                )
            cls._instances[cls] = super(AudioDataManager, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

    def __new__(cls, *args, **kwargs):
        with cls._lock:  # Ensure only one thread can create the instance
            if not cls in __class__._instances:
                singleton = super().__new__(cls, *args, **kwargs)
                __class__._instances[cls] = singleton
            return __class__._instances[cls]

    @abc.abstractmethod
    def add_entry(self, *args, **kwargs) -> None: ...

    @abc.abstractmethod
    def remove_entry(self, *args, **kwargs) -> None: ...

    @abc.abstractmethod
    def read_entries(self, *args, **kwargs) -> None: ...

    @abc.abstractmethod
    def write_entries(self, *args, **kwargs) -> None: ...

    @abc.abstractmethod
    def split_entries(self, *args, **kwargs) -> None: ...


# NOTE: Need to write tests to verify that the singleton pattern is working as expected
class LSTManager(AudioDataManager):
    def __init__(self, entries: set[str] = None):
        self.entries: set[str] = set()
        if entries:
            self.entries = entries
        else:
            self.read_entries()

    def add_entry(self, entry: Path) -> None:
        _entry_to_append: str = str(entry.relative_to(AUDIO_ROOT).with_suffix(""))
        self.entries.add(_entry_to_append)

    # NOTE: might be better to have entires as a set instead to enforce uniqueness
    def read_entries(
        self, directory: Path = AUDIO_ROOT, file_to_use: str = "all_items.lst"
    ) -> None:
        # TEST: file_to_use needs to exist inside of directory and be a lst file
        # TEST: Need to test if all_lst_files returns nothing
        if not file_to_use:
            # Fetch all files in this directory
            all_lst_files: list[str] = glob((directory / "*.lst"))
        else:
            all_lst_files = [directory / file_to_use]
        for lst_file in all_lst_files:
            with open(lst_file, "r", encoding="utf-8") as _file:
                self.entries.update([x.strip() for x in _file.readlines()])

    def write_entries(self, output_file_path: Path = None):
        if not output_file_path:
            output_file_path = AUDIO_ROOT / "all_items.lst"
        self.entries: list[str] = sorted(self.entries)
        with open(output_file_path, "w", encoding="utf-8") as _file:
            _file.write("\n".join(self.entries))

    def remove_entry(self, *args, **kwargs):
        pass

    def split_entries(self, *args, **kwargs):
        pass


# This will control a single file that contains file names
# It will also write out our file splits
class UEMManager(AudioDataManager):
    def __init__(self, entries: set[str] = None):
        self.entries: set[str] = set()
        if entries:
            self.entries = entries
        else:
            self.read_entries()

    def add_entry(self, entry: Path, file_duration: float) -> None:
        file_id: str = entry.relative_to(RTTM_ROOT).with_suffix("")
        _entry_to_append: str = f"{file_id} NA {0:.3f} {file_duration:.3f}"
        self.entries.add(_entry_to_append)

    # NOTE: might be better to have entires as a set instead to enforce uniqueness
    def read_entries(
        self, directory: Path = RTTM_ROOT, file_to_use: str = "all_items.uem"
    ) -> None:
        # TEST: file_to_use needs to exist inside of directory and be a UEM file
        # TEST: Need to test if all_uem_files returns nothing
        if not file_to_use:
            # Fetch all files in this directory
            all_uem_files: list[str] = glob((directory / "*.uem"))
        else:
            all_uem_files = [directory / file_to_use]
        for uem_file in all_uem_files:
            # TEST: Need to verify behavior when two files with different durations are found
            with open(uem_file, "r", encoding="utf-8") as _file:
                self.entries.update([x.strip() for x in _file.readlines()])

    def write_entries(self, output_file_path: Path = None):
        if not output_file_path:
            output_file_path = RTTM_ROOT / "all_items.uem"
        self.entries: list[str] = sorted(self.entries)
        with open(output_file_path, "w", encoding="utf-8") as _file:
            _file.write("\n".join(self.entries))

    def remove_entry(self, *args, **kwargs):
        pass

    def split_entries(self, *args, **kwargs):
        pass


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

    data: np.ndarray[np._Shape, np.dtype[np.float32]]
    sample_rate: int
    data, sample_rate = sf.read(file_path, dtype="float32")
    duration_in_seconds: float = len(data) / sample_rate  # expect up to 4 sig. figures
    return data, sample_rate, duration_in_seconds


def convert_to_mono_and_resample(
    data: np.ndarray,
    original_sample_rate: int,
    output_file: Optional[str] = None,
    sample_rate: int = SAMPLE_RATE,
) -> None:
    """Convert audio data from stereo output to mono output,
    resample to a new sample rate and write the result to a file.

    Args:
        data (np.ndarray): The WAV data as np.float32 values. Its expected shape
        is (num_samples, num_channels).
        original_sample_rate (int): The original sample rate of the audio data.
        output_file (str): The path to save the output, including its extension.
        sample_rate (int, optional): The sample rate to use for resampling.
        Defaults to SAMPLE_RATE.
    """

    # If it's already in mono and in the expected format then skip it.
    if data.ndim == 1 and original_sample_rate == sample_rate:
        return
    # Check if the audio is stereo (has more than 1 dimension and 2 channels)
    if data.ndim > 1 and data.shape[1] == 2:
        # Average the left and right channels to create a mono signal
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
    sf.write(
        output_file,
        resampled_data,
        samplerate=SAMPLE_RATE,
        format="wav",
        subtype="PCM_16",
    )


def convert_audacity_labels_to_rttm(
    file_path: str,
    output_file: str,
    delimiter: str = "\t",
    file_contains_header_row: bool = False,
) -> None:
    """Converts labels generated by Audacity into the Rich Transcription Time Marked
    (RTTM) format and writes the result to a file.

    Args:
        file_path (str): The path to the file, including its extension.
        output_file (str): The path to save the RTTM file, including its extension.
        delimiter (str, optional): The delimiter between values in the input file.
        Defaults to "\t".
        file_contains_header_row (bool, optional): A flag indicating if the input file
        contains a header row that needs to be skipped. Defaults to False.
    """

    with open(file_path, "r") as _file, open(output_file, "w") as _output:
        file_id: str = (
            f"{Path(file_path).parent.name.replace('_','')}_{Path(file_path).stem}"
        )
        lines: List[List[str]] = [
            line.split(delimiter) for line in map(str.strip, _file) if line
        ]
        if file_contains_header_row:
            lines = lines[1:]
        for line in lines:
            label = line[-1]
            if label in ("ignore", "non_speech"):
                continue
            start = float(line[0])
            end = float(line[1])
            duration = end - start

            rttm_line = (
                f"SPEAKER {file_id} 1 "
                f"{start:.6f} {duration:.6f} "
                f"<NA> <NA> {label} <NA> <NA>\n"
            )

            _output.write(rttm_line)


def process_wav_file(
    wav_file_path: Path,
    label_path: Path,
) -> None:
    # Step 1: Convert WAV to mono and resample audio (and get duration)
    data, sample_rate, file_duration = read_file_and_get_duration(wav_file_path)
    output_path = Path(AUDIO_ROOT, wav_file_path.parent.name, wav_file_path.name)
    convert_to_mono_and_resample(
        data=data,
        original_sample_rate=sample_rate,
        output_file=output_path,
        sample_rate=SAMPLE_RATE,
    )
    # Step 2: Convert Audacity label file (TSV) to RTTM
    rttm_output = Path(RTTM_ROOT, label_path.parent.name, f"{label_path.stem}.rttm")
    convert_audacity_labels_to_rttm(label_path, rttm_output)
    # Step 3: Update LST file to include new audio file
    lst_manager = LSTManager()
    lst_manager.add_entry(output_path)
    # Step 4: Update UEM file to include new RTTM file
    uem_manager = UEMManager()
    uem_manager.add_entry(rttm_output, file_duration)
    # Step 5: Write the things out
    uem_manager.write_entries()
    lst_manager.write_entries()
