from typing import Callable, Protocol

import numpy as np
from pyannote.core import Annotation

from core.constants import EPSILON


class QualityFilter(Protocol):
    name: str

    def __call__(
        self, annotation: Annotation, audio: np.ndarray, sample_rate: int
    ) -> tuple[Annotation, dict]: ...


def min_duration(min_seconds: float) -> QualityFilter:
    def _filter_func(annotation: Annotation, audio: np.ndarray, sample_rate: int):
        del audio, sample_rate
        kept = Annotation(uri=annotation.uri)
        for segment, track, label in annotation.itertracks(yield_label=True):
            if segment.duration >= min_seconds:
                kept[segment, track] = label
        return kept, {
            "name": "min_duration",
            "in": len(annotation),
            "out": len(kept),
            "param": min_seconds,
        }

    _filter_func.name = "min_duration"
    return _filter_func


def min_rms(min_rms_value: float) -> QualityFilter:
    def _filter_func(annotation: Annotation, audio: np.ndarray, sample_rate: int):
        kept = Annotation(uri=annotation.uri)
        for segment, track, label in annotation.itertracks(yield_label=True):
            sample = audio[
                int(segment.start * sample_rate) : int(segment.end * sample_rate)
            ]
            if sample.size == 0:
                continue
            rms_value = np.sqrt(np.mean(sample**2))
            if rms_value >= min_rms_value:
                kept[segment, track] = label
        return kept, {
            "name": "min_rms",
            "in": len(annotation),
            "out": len(kept),
            "param": min_rms_value,
        }

    _filter_func.name = "min_rms"
    return _filter_func


def min_snr(min_snr_db: float) -> QualityFilter:
    def _filter_func(annotation: Annotation, audio: np.ndarray, sample_rate: int):
        noise_floor = estimate_noise_floor(audio, sample_rate)
        # Avoid div by 0
        if noise_floor == 0:
            noise_floor = EPSILON
        kept = Annotation(uri=annotation.uri)
        for segment, track, label in annotation.itertracks(yield_label=True):
            sample = audio[
                int(segment.start * sample_rate) : int(segment.end * sample_rate)
            ]
            if sample.size == 0:
                continue
            rms_value = np.sqrt(np.mean(sample**2)) + EPSILON
            signal_to_noise_ratio = 20 * np.log10(rms_value / noise_floor)
            if signal_to_noise_ratio >= min_snr_db:
                kept[segment, track] = label
        return kept, {
            "name": "min_snr",
            "in": len(annotation),
            "out": len(kept),
            "param": min_snr_db,
            "noise_floor_rms": noise_floor,
        }

    _filter_func.name = "min_snr"
    return _filter_func


def apply_filters(
    annotation: Annotation,
    audio: np.ndarray,
    sr: int,
    filters: list[QualityFilter],
) -> tuple[Annotation, list[dict]]:
    stats: list[dict] = []
    current = annotation
    for f in filters:
        current, s = f(current, audio, sr)
        stats.append(s)
    return current, stats


_REGISTRY: dict[str, Callable[[dict], QualityFilter]] = {
    "min_duration": lambda c: min_duration(c["min_seconds"]),
    "min_rms": lambda c: min_rms(c["min_rms"]),
    "min_snr": lambda c: min_snr(c["min_snr_db"]),
}


def build_filters(spec: list[dict]) -> list[QualityFilter]:
    return [_REGISTRY[entry["kind"]](entry) for entry in spec]


### Helper functions


def estimate_noise_floor(
    audio: np.ndarray, sr: int, window_seconds: float = 0.5
) -> float:
    """RMS of the quietest non-overlapping window in the file."""
    win = int(window_seconds * sr)
    n_windows = len(audio) // win
    if n_windows == 0:
        return float(np.sqrt(np.mean(audio**2)))
    rms_per_window = [
        np.sqrt(np.mean(audio[i * win : (i + 1) * win] ** 2)) for i in range(n_windows)
    ]
    # Use the 10th percentile, not the absolute minimum, to avoid a single
    # silent window setting the floor unrealistically low.
    return float(np.percentile(rms_per_window, 10))
