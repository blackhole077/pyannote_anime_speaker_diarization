"""Typer-based CLI entry point for the anime-diarization pipeline.

Each command wires a YAML config under ``configs/`` to a stage of the pipeline:

- ``add_file`` — ingest a labeled (wav, labels) pair into the dataset.
- ``run_diarization`` — run pyannote diarization, optionally resolving cluster
  IDs to enrolled character names.
- ``filter_annotations`` — drop low-quality segments via configured filters.
- ``build_enrollment`` — build a per-character speaker embedding store from
  already-labeled episodes.
"""

from pathlib import Path

import soundfile as sf
import typer
from omegaconf import OmegaConf

from core.config import get_cfg
from core.data.etl import process_wav_file
from core.postprocess.audacity import (
    annotation_to_audacity_format,
    audacity_to_annotation_format,
)
from core.postprocess.quality import apply_filters, build_filters
from core.speaker_diarization import load_diarization_pipeline

app = typer.Typer()


@app.command(
    name="add_file",
    help="Add a WAV file and associated label file to the audio dataset",
)
def add_file(wav_file: Path, label_file: Path, register_only: bool = False) -> None:
    """Ingest a single ``(wav, labels)`` pair into the dataset."""
    process_wav_file(wav_file, label_file, register_only)


@app.command(
    name="run_diarization",
    help="Run the Speaker Diarization pipeline on an audio clip.",
)
def run_speaker_diarization(
    wav_file: Path,
    token: str | None = None,
    output_label_path: Path | None = None,
    enrollment_dir: Path | None = None,
    similarity_threshold: float | None = None,
    min_margin: float | None = None,
):
    """Run pyannote diarization and (optionally) resolve clusters to characters.

    Writes an Audacity-format label file. When ``enrollment_dir`` is provided,
    anonymous cluster IDs are replaced by enrolled character names where the
    similarity and margin gates both pass (see
    :func:`core.speaker_id.assign.assign_speakers`); a confidences JSON is
    written alongside the labels for QC.

    Args:
        wav_file: Input audio file.
        token: HuggingFace auth token for pyannote models; falls back to the
            ``HF_AUTH_TOKEN`` environment constant.
        output_label_path: Destination labels file. Defaults to
            ``<wav_file_stem>_predicted_labels.txt`` beside the input.
        enrollment_dir: Path to an enrollment store (produced by
            ``build_enrollment``). If omitted, labels remain anonymous cluster
            IDs.
    """
    cfg = get_cfg("speaker_diarization")
    sd_pipeline = load_diarization_pipeline(cfg, token)
    diarization = sd_pipeline(str(wav_file))
    annotation = diarization.exclusive_speaker_diarization

    if enrollment_dir is not None:
        from core.speaker_id.assign import assign_speakers
        from core.speaker_id.embed import load_embedding_model_for_inference
        from core.speaker_id.store import load_centroids, load_manifest

        sid_cfg = get_cfg("speaker_id")
        if similarity_threshold is None:
            similarity_threshold = sid_cfg.gate.sim
        if min_margin is None:
            min_margin = sid_cfg.gate.margin
        centroids = load_centroids(enrollment_dir)
        manifest = load_manifest(enrollment_dir)
        if manifest["embedder_model"] != "pyannote/embedding":
            typer.echo(
                f"Enrollment was built with {manifest['embedder_model']}; mismatch will produce garbage.",
                err=True,
            )

        inference = load_embedding_model_for_inference(token=token)
        annotation, confidences = assign_speakers(
            annotation,
            wav_file,
            centroids,
            inference,
            similarity_threshold=similarity_threshold,
            min_margin=min_margin,
        )

        for cid, info in sorted(confidences.items()):
            if info["similarity"] is None:
                typer.echo(f"  {cid} → {info['assigned']:<20} (no usable segments)")
                continue
            typer.echo(
                f"  {cid} → {info['assigned']:<20} sim={info['similarity']:.3f}  margin={info['margin']:.3f}"
            )

    if output_label_path is None:
        output_label_path = wav_file.with_name(f"{wav_file.stem}_predicted_labels.txt")
    annotation_to_audacity_format(annotation, output_label_path)
    typer.echo(f"Wrote labels → {output_label_path}")

    if enrollment_dir is not None:
        import json

        confidences_path = output_label_path.with_suffix(".confidences.json")
        confidences_path.write_text(json.dumps(confidences, indent=2))
        typer.echo(f"Wrote confidences → {confidences_path}")


@app.command(
    name="filter_annotations",
    help="Run quality filters against the annotations to weed out low-value labels",
)
def run_quality_filter(
    wav_file: Path,
    annotation_file: Path,
):
    """Apply quality filters from ``configs/quality_filter.yaml`` to a labels file.

    Reads the WAV and existing annotations, runs each configured filter, prints
    per-filter stats, and writes ``<annotation_file_stem>_filtered.txt`` next to
    the original.
    """
    output_file = Path(
        annotation_file.parent / f"{annotation_file.stem}_filtered"
    ).with_suffix(".txt")
    cfg = get_cfg("quality_filter")
    filter_spec = OmegaConf.to_container(cfg.filters, resolve=True)
    quality_filters = build_filters(filter_spec)
    audio, sample_rate = sf.read(wav_file, dtype="float32")
    annotations = audacity_to_annotation_format(annotation_file)
    filtered_annotations, stats = apply_filters(
        annotations, audio, sample_rate, quality_filters
    )
    for stat in stats:
        for k, v in stat.items():
            print(f"Key: {k}\tValue:{v}")
        print("=" * 30)
    annotation_to_audacity_format(filtered_annotations, output_file)


@app.command(
    name="eval_der",
    help="Evaluate the diarization pipeline against a pyannote.database protocol.",
)
def eval_der(
    config_path: Path = Path("configs/eval.yaml"),
    token: str | None = None,
):
    """Run DER + per-character P/R against the configured protocol.

    Resolves ``configs/eval.yaml`` (with its include/overrides composition),
    runs the diarization pipeline over the configured subset, and writes a
    JSON report to ``eval.output_path``.
    """
    from core.eval.metrics import load_eval_cfg, run_eval

    cfg = load_eval_cfg(config_path)
    report = run_eval(cfg, token=token)
    der_value = report["gated"]["metrics"]["der"]["value"]
    typer.echo(f"DER (gated): {der_value:.4f}")
    if report["gated"]["metrics"].get("per_character"):
        typer.echo("Per-character (gated):")
        for char, vals in report["gated"]["metrics"]["per_character"].items():
            typer.echo(
                f"  {char:20s}  P={vals['precision']:.3f}  R={vals['recall']:.3f}  "
                f"support={vals['support_seconds']:.1f}s"
            )
    if report.get("gate_cost"):
        agg = report["gate_cost"]["aggregate"]
        typer.echo(
            f"Gate cost: DER Δ={agg['der_delta']:+.4f}  "
            f"gate_value={agg['gate_value']:+.2f}s"
        )


@app.command(
    name="build_enrollment",
    help="Build a per-character speaker embedding store from labeled episodes.",
)
def build_enrollment(
    labeled_dir: Path,
    output_dir: Path,
    show: str | None = None,
):
    """Build a per-character embedding store from labeled episodes.

    Pipeline: discover ``(wav, labels)`` pairs in ``labeled_dir`` → group clips
    by character → drop reserved labels and too-short clips → score and pick
    top clips up to the per-character duration budget → embed and average into
    centroids → persist to ``output_dir`` alongside a manifest.

    Args:
        labeled_dir: Directory containing matching ``<stem>.wav`` /
            ``<stem>_labels.txt`` pairs.
        output_dir: Destination for ``centroids.npz``, ``per_clip.npz``, and
            ``manifest.json``.
        show: Optional show name written to the manifest; defaults to
            ``labeled_dir.name`` if not provided.
    """
    from datetime import datetime, timezone

    from core.speaker_id.embed import (
        enroll_characters,
        load_embedding_model_for_inference,
    )
    from core.speaker_id.refs import (
        collect_character_clips,
        find_labeled_pairs,
        summarize,
    )
    from core.speaker_id.select import select_enrollment_clips
    from core.speaker_id.store import save_enrollment

    cfg = get_cfg("speaker_id")

    pairs = find_labeled_pairs(labeled_dir)
    if not pairs:
        typer.echo(f"No labeled episodes found in {labeled_dir}", err=True)
        raise typer.Exit(1)
    typer.echo(f"Found {len(pairs)} labeled episode(s):")
    for _, l in pairs:
        typer.echo(f"  {l.name}")

    candidates = collect_character_clips(
        pairs, min_clip_seconds=cfg.selection.min_clip_seconds
    )
    selected = select_enrollment_clips(
        candidates,
        target_seconds_per_char=cfg.selection.target_seconds_per_char,
        scoring_config=OmegaConf.to_container(cfg.scoring, resolve=True),
    )

    inference = load_embedding_model_for_inference()
    centroids, per_clip = enroll_characters(selected, inference=inference)

    selected_summary = summarize(selected)
    target = cfg.selection.target_seconds_per_char
    pairs_by_stem = {l.stem.removesuffix("_labels"): l for _, l in pairs}
    manifest = {
        "show": show or labeled_dir.name,
        "embedder_model": cfg.embedder.model,
        "embedding_dim": int(next(iter(centroids.values())).shape[0]),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "characters": {
            char: {
                **stats,
                "source_episodes": sorted({c.wav_path.stem for c in selected[char]}),
                "below_target": stats["total_seconds"] < target,
            }
            for char, stats in selected_summary.items()
        },
    }

    save_enrollment(output_dir, centroids, per_clip, manifest)
    typer.echo(f"Saved enrollment for {len(centroids)} characters → {output_dir}")
    below = [c for c, m in manifest["characters"].items() if m["below_target"]]
    if below:
        typer.echo(
            f"Below-target characters (used all available clips): {', '.join(below)}"
        )


if __name__ == "__main__":
    app()
