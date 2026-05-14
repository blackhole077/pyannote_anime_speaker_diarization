from pathlib import Path
from typing import Dict, List

import torch
# send pipeline to GPU (when available)
from omegaconf import DictConfig, OmegaConf
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_diarization import DiarizeOutput
from pyannote.core.segment import Segment
from pydub import AudioSegment

from core.constants import HF_AUTH_TOKEN
from core.postprocess.audacity import annotation_to_audacity_format


def load_diarization_pipeline(
    cfg: DictConfig,
    token: str | None = None,
) -> Pipeline:

    if cfg.accelerator == "gpu" and torch.cuda.is_available():
        device = "cuda"
    else:
        print(f"Unable to utilize GPU, falling back to CPU")
        device = "cpu"

    token = token or HF_AUTH_TOKEN
    pipeline = Pipeline.from_pretrained(cfg.pipeline.name, token=token)
    print(pipeline.parameters())
    overrides = {}
    seg = cfg.pipeline.get("segmentation", {})
    clust = cfg.pipeline.get("clustering", {})
    if seg:
        overrides["segmentation"] = OmegaConf.to_container(seg)
    if clust:
        overrides["clustering"] = OmegaConf.to_container(clust)
    if overrides:
        pipeline.instantiate(overrides)
    pipeline.to(torch.device(device))
    return pipeline


def main_speaker_diarization(wav_path: Path, pipeline: Pipeline):
    wav_path_stub: Path = Path(wav_path).with_suffix("")
    # apply pretrained pipeline
    diarization: DiarizeOutput = pipeline(wav_path)
    # print the result
    export_path = (
        wav_path_stub.parent / f"{wav_path_stub.name}_predicted_labels"
    ).with_suffix(".txt")
    annotation_to_audacity_format(
        diarization.exclusive_speaker_diarization, export_path
    )
