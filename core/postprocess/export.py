from pathlib import Path

from pyannote.core.annotation import Annotation
from pydub import AudioSegment

from core.constants import SAMPLE_RATE


def export_per_speaker_audio(
    annotation: Annotation,
    source_wav: Path,
    output_path: Path,
    silence_pad_ms: int = 200,
    sample_rate: int = SAMPLE_RATE,
) -> dict[str, Path]:
    source_audio: AudioSegment = AudioSegment.from_wav(source_wav)
    source_audio.set_frame_rate(sample_rate)
    speaker_audio_dict: dict[str, AudioSegment] = {}
    between_sample_silence_ms = AudioSegment.silent(
        duration=silence_pad_ms, frame_rate=sample_rate
    )
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        speaker_audio = speaker_audio_dict.get(speaker)
        start_ms = int(segment.start * 1000)
        end_ms = int(segment.end * 1000)
        sample = source_audio[start_ms:end_ms]
        speaker_audio = (
            sample
            if not speaker_audio
            else speaker_audio + between_sample_silence_ms + sample
        )
        speaker_audio_dict[speaker] = speaker_audio
    exported: dict[str, Path] = {}
    for speaker, combined in speaker_audio_dict.items():
        path = (output_path / source_wav.stem / speaker).with_suffix(".wav")
        combined.export(path, format="wav")
        exported[speaker] = path
    return exported
