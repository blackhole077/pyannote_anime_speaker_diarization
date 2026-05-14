"""Rank and pick the highest-quality clips per character for enrollment.

Given the candidate clips produced by :mod:`core.speaker_id.refs`, this module
scores each clip (currently by SNR) and greedily selects clips in descending
score order until a per-character duration budget is met. When a character has
less audio than the budget, all available clips are taken — the embedder is
free to produce a noisier centroid, and the manifest marks the character as
``below_target`` so downstream tooling knows.

Audio and noise-floor estimates are cached per file across all characters so
each episode WAV is decoded at most once per build.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.constants import EPSILON
from core.preprocess.audio_io import read_file_and_get_duration
from core.speaker_id.refs import ClipRef


@dataclass(frozen=True)
class ScoredClip:
    """A clip paired with its quality score for ranking purposes."""

    clip: ClipRef
    score: float


def select_enrollment_clips(
    clips_by_char: dict[str, list[ClipRef]],
    target_seconds_per_char: int,
    scoring_config: dict,
) -> dict[str, list[ClipRef]]:
    """Greedy top-by-score clip selection up to a duration budget per character.

    Clips are scored once (cached audio and noise-floor per file), sorted
    descending, and accumulated until the cumulative duration reaches
    ``target_seconds_per_char``. Characters below the budget receive all of
    their available clips and a stderr warning.

    Note: this mutates and returns the input dict for caller convenience.

    Args:
        clips_by_char: Candidate clips per character from
            :func:`core.speaker_id.refs.collect_character_clips`.
        target_seconds_per_char: Per-character duration budget in seconds.
        scoring_config: Reserved for future per-knob configuration; currently
            unused by ``score_clip`` (SNR-only after the duration term was
            removed).

    Returns:
        The same dict, with values replaced by the selected ``ClipRef`` lists.
    """
    # Load in the audio
    _audio_cache: dict[Path, tuple[np.ndarray, int]] = {}
    _floor_cache: dict[Path, float] = {}
    for character, character_clips in clips_by_char.items():
        scored_clips: list[ScoredClip] = []
        for clip in character_clips:
            if clip.wav_path not in _audio_cache:
                audio, sample_rate, _ = read_file_and_get_duration(clip.wav_path)
                _audio_cache[clip.wav_path] = (audio, sample_rate)
            audio, sample_rate = _audio_cache.get(clip.wav_path)
            noise_floor = _floor_cache.get(clip.wav_path)
            if noise_floor is None:
                noise_floor: float = max(
                    estimate_noise_floor(audio, sample_rate), EPSILON
                )
                _floor_cache[clip.wav_path] = noise_floor

            score = score_clip(clip, audio, sample_rate, noise_floor)
            scored_clips.append(ScoredClip(clip, score))
        if not scored_clips:
            print(f"No usable clips for {character}; skipping.")
            clips_by_char[character] = []
            continue
        sorted_clips = sorted(scored_clips, key=lambda x: x.score, reverse=True)

        duration_cumsum = np.cumsum([x.clip.duration for x in sorted_clips])
        if duration_cumsum[-1] < target_seconds_per_char:
            print(
                f"Not enough clips to satisfy duration for {character}, taking all clips available."
            )
            clips_to_enroll = sorted_clips
        else:
            num_clips_to_enroll = (
                int(np.argmax(duration_cumsum >= target_seconds_per_char)) + 1
            )
            clips_to_enroll = sorted_clips[:num_clips_to_enroll]
        clips_by_char[character] = [x.clip for x in clips_to_enroll]
    return clips_by_char


def score_clip(
    clip: ClipRef, audio: np.ndarray, sample_rate: int, noise_floor: float
) -> float:
    """Quality score for a clip in ``[0.0, 1.0]``.

    Currently SNR-only: the clip's RMS is compared against a precomputed
    file-level noise floor, the resulting dB ratio is normalized by a fixed
    20 dB knee and clipped to ``[0, 1]``. A previous duration term (Gaussian
    around 5 s) was removed after analysis showed it heavily down-weighted
    long clean monologues, which are exactly the clips that produce the most
    stable per-clip embeddings.

    Returns ``-inf`` for an empty sample window so it is always sorted last.
    """
    sample = audio[int(clip.start * sample_rate) : int(clip.end * sample_rate)]
    if sample.size == 0:
        return -np.inf
    # Calculate RMS using clips as floor
    _rms_score = np.sqrt(np.mean(sample**2)) + EPSILON
    signal_to_noise_ratio = 20 * np.log10(_rms_score / noise_floor)
    knee = 20.0
    snr_score = float(np.clip(signal_to_noise_ratio / knee, 0.0, 1.0))
    return snr_score


# TODO: Move this code to a separate utils.py so that it can be used by quality.py and this
def estimate_noise_floor(
    audio: np.ndarray, sr: int, window_seconds: float = 0.5
) -> float:
    """Estimate a per-file noise floor as the 10th-percentile windowed RMS.

    Using a percentile rather than the absolute minimum avoids a single very
    quiet window (e.g. a momentary silence between music cues) from setting an
    unrealistically low floor that would inflate every clip's SNR.

    Returns the overall RMS as a fallback when the audio is too short to form
    a single window.
    """
    win = int(window_seconds * sr)
    n_windows = len(audio) // win
    if n_windows == 0:
        return float(np.sqrt(np.mean(audio**2)))
    rms_per_window = [
        np.sqrt(np.mean(audio[i * win : (i + 1) * win] ** 2)) for i in range(n_windows)
    ]
    return float(np.percentile(rms_per_window, 10))
