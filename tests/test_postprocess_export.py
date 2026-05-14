"""Tests for core.postprocess.export.export_per_speaker_audio.

Each test below has a docstring describing what to verify; bodies are stubs
to be filled in. Style follows tests/test_audio_io.py — class-grouped, data
generated in-test, no on-disk fixtures.
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from pyannote.core import Annotation, Segment
from pydub import AudioSegment

from core.postprocess.export import export_per_speaker_audio

# ---------- helpers (fill in) ----------


def _write_sine_wav(path: Path, duration_s: float, sr: int = 16000, freq: float = 440.0) -> Path:
    """Write a mono sine wav of the given duration. Returns the path."""
    raise NotImplementedError


def _annotation(*spans: tuple[float, float, str]) -> Annotation:
    """Build a pyannote Annotation from (start, end, speaker) tuples."""
    raise NotImplementedError


# ---------- tests ----------


class TestExportPerSpeakerAudio:

    def test_single_speaker_single_segment(self, tmp_path):
        """One speaker, one segment → one wav containing exactly that slice
        (length matches segment duration; no padding because nothing to pad between)."""
        raise NotImplementedError

    def test_multi_speaker_writes_one_file_each(self, tmp_path):
        """Two speakers, two segments each → two wav files, named
        `{stem}_{speaker}.wav`. Returned dict keys match speaker labels."""
        raise NotImplementedError

    def test_concatenation_order_follows_annotation_iteration(self, tmp_path):
        """Annotation contains segments in non-monotonic order for the same
        speaker → output concatenates in `annotation.itertracks` order
        (pyannote sorts by segment start). Verify by encoding distinguishable
        tones into each segment and checking sequence in the output."""
        raise NotImplementedError

    @pytest.mark.parametrize("pad_ms", [0, 200, 1000])
    def test_silence_padding_between_segments(self, tmp_path, pad_ms):
        """Two segments for one speaker → output length ==
        seg1_ms + pad_ms + seg2_ms (within ±1 frame tolerance).
        pad_ms=0 should produce contiguous concatenation."""
        raise NotImplementedError

    def test_no_padding_after_final_segment(self, tmp_path):
        """Trailing silence is between, not after — confirm last samples are
        non-silent (RMS above threshold) when source ends on speech."""
        raise NotImplementedError

    def test_creates_output_directory_if_missing(self, tmp_path):
        """Pass an output_path that does not yet exist → function creates it
        and writes successfully (no FileNotFoundError)."""
        raise NotImplementedError

    def test_returns_mapping_of_speaker_to_path(self, tmp_path):
        """Return value is dict[str, Path], keys are speaker labels present
        in the annotation, values are existing files on disk."""
        raise NotImplementedError

    def test_empty_annotation_writes_nothing(self, tmp_path):
        """Annotation with no segments → empty dict, no files written,
        output dir may or may not be created (document the chosen behavior)."""
        raise NotImplementedError

    def test_filename_uses_source_wav_stem(self, tmp_path):
        """Source wav `episode01.wav`, speaker `loid` → output file
        `episode01_loid.wav`. Confirms stem extraction (no `.name` bug)."""
        raise NotImplementedError

    def test_segment_outside_audio_bounds_is_handled(self, tmp_path):
        """Annotation segment extends past source wav duration → either
        clipped to available audio or raises a clear error (decide which;
        pydub silently clips, so this likely documents that behavior)."""
        raise NotImplementedError

    def test_overlapping_segments_same_speaker(self, tmp_path):
        """Two overlapping segments for one speaker → both are concatenated
        (the function does not dedupe). Documents current behavior so a
        future dedupe pass is a deliberate change, not a silent regression."""
        raise NotImplementedError
