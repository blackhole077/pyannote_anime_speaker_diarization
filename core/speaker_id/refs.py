"""Parse labeled episode files into per-character clip references.

This module is the entry point for enrollment data ingestion. It walks a directory
of `<episode>.wav` / `<episode>_labels.txt` pairs, drops reserved meta-labels
(``overlap``, ``unknown``, ``ignore``), and emits per-character ``ClipRef`` lists
that downstream selection and embedding stages consume.

Reserved labels are filtered here (not later) so they cannot accidentally
contribute to a character centroid.
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from pyannote.core.annotation import Segment

from core.postprocess.audacity import audacity_to_annotation_format

RESERVED_LABELS = frozenset({"overlap", "unknown", "ignore"})


@dataclass(frozen=True)
class ClipRef:
    """Immutable reference to a labeled span of audio in a specific WAV file.

    Used as the unit of enrollment data throughout ``speaker_id/``. We keep the
    file path alongside the segment so the audio can be re-cropped lazily by the
    embedding model without having to thread the WAV identity separately.
    """

    wav_path: Path
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_segment(self) -> Segment:
        """Bridge to pyannote: most pyannote APIs require a ``Segment``."""
        return Segment(start=self.start, end=self.end)

    @classmethod
    def from_segment(cls, wav_path: Path, segment: Segment) -> "ClipRef":
        """Build a ``ClipRef`` from a pyannote ``Segment`` plus its source WAV."""
        return ClipRef(wav_path=wav_path, start=segment.start, end=segment.end)


def collect_character_clips(
    pairs: list[tuple[Path, Path]],  # [(wav_path, labels_path), ...]
    *,
    min_clip_seconds: float = 2.0,
    reserved: frozenset[str] = RESERVED_LABELS,
) -> dict[str, list[ClipRef]]:
    """Group clips by character across a set of labeled episodes.

    Reserved meta-labels and segments shorter than ``min_clip_seconds`` are
    dropped at this stage so they cannot poison the downstream candidate pool.
    The minimum length matters because the embedding model has a non-trivial
    receptive field; very short clips produce unstable embeddings.

    Args:
        pairs: ``(wav_path, labels_path)`` pairs as returned by
            :func:`find_labeled_pairs`.
        min_clip_seconds: Drop segments shorter than this.
        reserved: Labels to ignore (overlap regions, unknown speakers, etc.).

    Returns:
        Mapping of character name to the list of clips belonging to that
        character across all input episodes.
    """
    by_char: dict[str, list[ClipRef]] = defaultdict(list)
    for wav_path, labels_path in pairs:
        ann = audacity_to_annotation_format(labels_path)
        for segment, _, label in ann.itertracks(yield_label=True):
            # Toss ones that we want to ignore or are simply too short
            if label in reserved or segment.duration < min_clip_seconds:
                continue
            by_char[label].append(ClipRef.from_segment(wav_path, segment))
    return dict(by_char)


def summarize(clips: dict[str, list[ClipRef]]) -> dict[str, dict]:
    """Compute per-character clip-count / duration stats.

    Used to populate the enrollment manifest and to flag characters that fall
    below the target duration budget before the embedding pass runs (so the
    user knows which centroids will be noisier).
    """
    return {
        char: {
            "n_clips": len(refs),
            "total_seconds": sum(r.duration for r in refs),
            "min_seconds": min(r.duration for r in refs),
            "max_seconds": max(r.duration for r in refs),
        }
        for char, refs in clips.items()
    }


def find_labeled_pairs(
    directory: Path,
    *,
    wav_suffix: str = ".wav",
    labels_suffix: str = "_labels.txt",
) -> list[tuple[Path, Path]]:
    """Find ``(wav, labels)`` pairs in ``directory`` by filename convention.

    Matches ``<stem>.wav`` ↔ ``<stem>_labels.txt``. Wavs without a matching
    labels file are silently skipped, which is the desired behaviour during
    incremental labeling: only labeled episodes contribute to enrollment.
    """
    pairs = []
    for labels_path in sorted(directory.glob(f"*{labels_suffix}")):
        stem = labels_path.name.removesuffix(labels_suffix)
        wav_path = directory / f"{stem}{wav_suffix}"
        if wav_path.exists():
            pairs.append((wav_path, labels_path))
    return pairs
