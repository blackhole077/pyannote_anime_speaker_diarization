import abc
import threading
from pathlib import Path
from typing import ClassVar, List, Optional

import librosa
import numpy as np
import soundfile as sf
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from core.constants import (
    ALL_LST,
    ALL_RTTM,
    ALL_UEM,
    AUDIO_ROOT,
    RESERVED_LABELS,
    RTTM_ROOT,
    SAMPLE_RATE,
)


# ----- Pydantic entry schemas -----

class LSTEntry(BaseModel):
    """A single URI line in a ``.lst`` file."""

    model_config = ConfigDict(frozen=True)

    uri: str

    @field_validator("uri")
    @classmethod
    def _no_whitespace(cls, v: str) -> str:
        if not v or any(c.isspace() for c in v):
            raise ValueError(f"URI must be non-empty and whitespace-free: {v!r}")
        return v

    def to_line(self) -> str:
        return self.uri

    @classmethod
    def from_line(cls, line: str) -> "LSTEntry":
        return cls(uri=line.strip())


class UEMEntry(BaseModel):
    """A single line in a ``.uem`` file: ``<uri> <channel> <start> <end>``."""

    model_config = ConfigDict(frozen=True)

    uri: str
    channel: str = "NA"
    start: float
    end: float

    @model_validator(mode="after")
    def _valid_span(self) -> "UEMEntry":
        if self.end <= self.start:
            raise ValueError(f"end ({self.end}) must be > start ({self.start})")
        return self

    def to_line(self) -> str:
        return f"{self.uri} {self.channel} {self.start:.3f} {self.end:.3f}"

    @classmethod
    def from_line(cls, line: str) -> "UEMEntry":
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(f"expected 4 fields, got {len(parts)}: {line!r}")
        uri, channel, start, end = parts
        return cls(uri=uri, channel=channel, start=float(start), end=float(end))


class RTTMEntry(BaseModel):
    """A single SPEAKER row in a ``.rttm`` file.

    RTTM layout (10 space-separated fields):
    ``SPEAKER <uri> <channel> <start> <duration> <NA> <NA> <speaker> <NA> <NA>``
    """

    model_config = ConfigDict(frozen=True)

    uri: str
    channel: str = "1"
    start: float
    duration: float
    speaker: str

    @field_validator("duration")
    @classmethod
    def _positive_duration(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"duration must be > 0, got {v}")
        return v

    @field_validator("speaker")
    @classmethod
    def _non_empty_speaker(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("speaker label must be non-empty")
        return v

    def to_line(self) -> str:
        return (
            f"SPEAKER {self.uri} {self.channel} "
            f"{self.start:.6f} {self.duration:.6f} "
            f"<NA> <NA> {self.speaker} <NA> <NA>"
        )

    @classmethod
    def from_line(cls, line: str) -> "RTTMEntry":
        parts = line.split()
        if len(parts) != 10:
            raise ValueError(f"expected 10 fields, got {len(parts)}: {line!r}")
        if parts[0] != "SPEAKER":
            raise ValueError(f"first field must be 'SPEAKER', got {parts[0]!r}")
        for idx in (5, 6, 8, 9):
            if parts[idx] != "<NA>":
                raise ValueError(f"field {idx} must be '<NA>', got {parts[idx]!r}")
        return cls(
            uri=parts[1],
            channel=parts[2],
            start=float(parts[3]),
            duration=float(parts[4]),
            speaker=parts[7],
        )


# ----- Manager base -----

class AudioDataManager(metaclass=abc.ABCMeta):
    _instances: dict[type, "AudioDataManager"] = {}
    _lock = threading.Lock()
    _seed: ClassVar[int] = 42

    # Subclass overrides.
    suffix: ClassVar[str] = ""
    default_file: ClassVar[Path] = Path()
    entry_model: ClassVar[type[BaseModel]] = BaseModel

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in __class__._instances:
                if abc.ABCMeta.__abstractmethods__(cls):
                    raise TypeError(f"{cls} is abstract and cannot be instantiated.")
                __class__._instances[cls] = super().__new__(cls)
            return __class__._instances[cls]

    def _verify_basics(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist")
        if not path.is_file():
            raise ValueError(f"{path} is not a regular file")
        if path.stat().st_size == 0:
            raise ValueError(f"{path} is empty")
        if path.suffix != self.suffix:
            raise ValueError(
                f"{path} has suffix {path.suffix!r}, expected {self.suffix!r}"
            )

    def verify_file(self, path: Path) -> list[BaseModel]:
        """Verify a file and return its parsed entries.

        Raises on basic problems (missing, empty, wrong suffix) or any line
        that fails the entry schema.
        """
        self._verify_basics(path)
        parsed: list[BaseModel] = []
        with open(path, "r", encoding="utf-8") as f:
            for lineno, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    parsed.append(self.entry_model.from_line(line))
                except (ValueError, ValidationError) as e:
                    raise ValueError(f"{path}:{lineno}: {e}") from e
        return parsed

    def _discover_files(self, source: Path) -> list[Path]:
        """Glob ``source`` root + one level deep for files matching ``self.suffix``."""
        matches = set(source.glob(f"*{self.suffix}"))
        matches |= set(source.glob(f"*/*{self.suffix}"))
        return sorted(matches)

    def read_entries(self, source: Path | None = None) -> None:
        """Clear-then-load entries from a file, a 1-deep directory, or the default file."""
        if source is None:
            source = self.default_file

        if source.is_dir():
            files = self._discover_files(source)
            if not files:
                raise FileNotFoundError(
                    f"No {self.suffix} files found in {source} (root or 1 level deep)"
                )
        else:
            files = [source]

        self.entries.clear()
        for path in files:
            self.entries.update(self.verify_file(path))

    def write_entries(
        self,
        output_file_path: Path | None = None,
        entries: Optional[set[BaseModel]] = None,
    ) -> None:
        if output_file_path is None:
            output_file_path = self.default_file
        to_write = entries if entries is not None else self.entries
        sorted_entries = sorted(to_write, key=lambda e: e.to_line())
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(e.to_line() for e in sorted_entries))

    @abc.abstractmethod
    def add_entry(self, *args, **kwargs) -> None: ...

    @abc.abstractmethod
    def remove_entry(self, *args, **kwargs) -> None: ...

    @abc.abstractmethod
    def split_entries(self, *args, **kwargs) -> tuple[list, list, list]: ...


# ----- LST -----

class LSTManager(AudioDataManager):
    suffix = ".lst"
    default_file = ALL_LST
    entry_model = LSTEntry

    def __init__(self, entries: Optional[set[LSTEntry]] = None):
        self.entries: set[LSTEntry] = set(entries) if entries else set()
        if entries is None and self.default_file.exists():
            self.read_entries()

    def add_entry(self, entry: Path) -> None:
        uri = str(entry.relative_to(AUDIO_ROOT).with_suffix(""))
        self.entries.add(LSTEntry(uri=uri))

    def remove_entry(self, entry: Path) -> None:
        uri = str(entry.relative_to(AUDIO_ROOT).with_suffix(""))
        self.entries.discard(LSTEntry(uri=uri))

    def split_entries(
        self, train_split: float = 0.8, dev_split: float = 0.1
    ) -> tuple[list[LSTEntry], list[LSTEntry], list[LSTEntry]]:
        """Random URI partition into (train, development, test) with a fixed seed."""
        rng = np.random.default_rng(seed=self._seed)
        ordered = sorted(self.entries, key=lambda e: e.uri)
        shuffled = rng.permutation(ordered).tolist()
        train_end = int(len(shuffled) * train_split)
        dev_end = int(len(shuffled) * (train_split + dev_split))
        return shuffled[:train_end], shuffled[train_end:dev_end], shuffled[dev_end:]


# ----- UEM -----

class UEMManager(AudioDataManager):
    suffix = ".uem"
    default_file = ALL_UEM
    entry_model = UEMEntry

    def __init__(self, entries: Optional[set[UEMEntry]] = None):
        self.entries: set[UEMEntry] = set(entries) if entries else set()
        if entries is None and self.default_file.exists():
            self.read_entries()

    def add_entry(self, entry: Path, file_duration: float) -> None:
        uri = str(entry.relative_to(RTTM_ROOT).with_suffix(""))
        self.entries.add(UEMEntry(uri=uri, channel="NA", start=0.0, end=file_duration))

    def remove_entry(self, entry: Path) -> None:
        uri = str(entry.relative_to(RTTM_ROOT).with_suffix(""))
        self.entries = {e for e in self.entries if e.uri != uri}

    def split_entries(
        self, uri_subsets: dict[str, list[str]]
    ) -> tuple[list[UEMEntry], list[UEMEntry], list[UEMEntry]]:
        """Partition entries by URI assignment. Caller passes the LST split's URI lists."""
        by_uri: dict[str, UEMEntry] = {e.uri: e for e in self.entries}
        return (
            [by_uri[u] for u in uri_subsets["train"] if u in by_uri],
            [by_uri[u] for u in uri_subsets["development"] if u in by_uri],
            [by_uri[u] for u in uri_subsets["test"] if u in by_uri],
        )


# ----- RTTM -----

class RTTMManager(AudioDataManager):
    suffix = ".rttm"
    default_file = ALL_RTTM
    entry_model = RTTMEntry

    def __init__(self, entries: Optional[set[RTTMEntry]] = None):
        self.entries: set[RTTMEntry] = set(entries) if entries else set()
        if entries is None and self.default_file.exists():
            self.read_entries()

    def add_entry(self, entry: RTTMEntry) -> None:
        self.entries.add(entry)

    def ingest_file(self, path: Path) -> None:
        """Verify and add every row from a per-episode RTTM to the master set."""
        self.entries.update(self.verify_file(path))

    def remove_entry(self, uri: str) -> None:
        self.entries = {e for e in self.entries if e.uri != uri}

    def split_entries(
        self, uri_subsets: dict[str, list[str]]
    ) -> tuple[list[RTTMEntry], list[RTTMEntry], list[RTTMEntry]]:
        """Partition rows by URI assignment. Caller passes the LST split's URI lists."""
        from collections import defaultdict

        by_uri: dict[str, list[RTTMEntry]] = defaultdict(list)
        for e in self.entries:
            by_uri[e.uri].append(e)

        def _collect(uris: list[str]) -> list[RTTMEntry]:
            rows: list[RTTMEntry] = []
            for u in uris:
                rows.extend(sorted(by_uri.get(u, []), key=lambda e: (e.start, e.speaker)))
            return rows

        return (
            _collect(uri_subsets["train"]),
            _collect(uri_subsets["development"]),
            _collect(uri_subsets["test"]),
        )


# ----- Free helpers (unchanged behavior) -----

def read_file_and_get_duration(
    file_path: str,
) -> tuple[np.ndarray, int, float]:
    data, sample_rate = sf.read(file_path, dtype="float32")
    duration_in_seconds: float = len(data) / sample_rate
    return data, sample_rate, duration_in_seconds


def convert_to_mono_and_resample(
    data: np.ndarray,
    original_sample_rate: int,
    output_file: Optional[str] = None,
    sample_rate: int = SAMPLE_RATE,
) -> None:
    if data.ndim == 1 and original_sample_rate == sample_rate:
        return
    if data.ndim > 1 and data.shape[1] == 2:
        data = np.mean(data, axis=1)
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
    """Convert an Audacity TSV label file to RTTM via :class:`RTTMEntry`."""
    file_id: str = f"{Path(file_path).parent.name}/{Path(file_path).stem}"
    with open(file_path, "r") as _file:
        lines: List[List[str]] = [
            line.split(delimiter) for line in map(str.strip, _file) if line
        ]
    if file_contains_header_row:
        lines = lines[1:]

    rows: list[RTTMEntry] = []
    for line in lines:
        label = line[-1]
        if label in RESERVED_LABELS:
            continue
        start = float(line[0])
        end = float(line[1])
        rows.append(
            RTTMEntry(uri=file_id, start=start, duration=end - start, speaker=label)
        )

    with open(output_file, "w") as _output:
        _output.write("\n".join(r.to_line() for r in rows))


def process_wav_file(
    wav_file_path: Path, label_path: Path, register_only: bool = False
) -> None:
    # Step 1: Convert WAV to mono and resample audio (and get duration)
    data, sample_rate, file_duration = read_file_and_get_duration(wav_file_path)
    output_path = Path(AUDIO_ROOT, wav_file_path.parent.name, wav_file_path.name)
    if not register_only:
        convert_to_mono_and_resample(
            data=data,
            original_sample_rate=sample_rate,
            output_file=output_path,
            sample_rate=SAMPLE_RATE,
        )
    # Step 2: Convert Audacity label file (TSV) to RTTM
    rttm_output = Path(RTTM_ROOT, label_path.parent.name, f"{label_path.stem}.rttm")
    rttm_output.parent.mkdir(parents=True, exist_ok=True)
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
