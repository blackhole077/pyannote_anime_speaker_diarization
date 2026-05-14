"""Persistence layer for enrollment artifacts.

Single source of truth for the on-disk layout of an enrollment store::

    <out_dir>/
        centroids.npz     # {character: (D,) float32}, the hot path
        per_clip.npz      # {character: (N_clips, D) float32}, cold path for QC
        manifest.json     # human-readable: embedder model, dim, source episodes, ...

``.npz`` is used for the embeddings because it is numpy-native, supports
named per-character access, and avoids pickle. The manifest is JSON so it can
be inspected by hand and diffed across enrollment rebuilds.
"""

import json
from pathlib import Path

import numpy as np


def save_enrollment(
    out_dir: Path,
    centroids: dict[str, np.ndarray],
    per_clip: dict[str, list[np.ndarray]],
    metadata: dict,  # the manifest dict
) -> None:
    """Write centroids, per-clip embeddings, and manifest to ``out_dir``.

    Per-character clip lists are stacked into ``(N, D)`` arrays before saving
    so each character is a single npz entry. Characters with no clips are
    omitted from ``per_clip.npz`` (centroids should already exclude them too).

    Args:
        out_dir: Destination directory; created if missing.
        centroids: Per-character unit-norm centroid vectors.
        per_clip: Per-character per-clip embedding lists.
        metadata: Manifest contents (embedder model, dim, source episodes,
            below-target flags, etc.).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "centroids.npz", **centroids)
    # stack per-character clips into (N, D) before saving
    stacked = {char: np.vstack(vecs) for char, vecs in per_clip.items() if vecs}
    np.savez(out_dir / "per_clip.npz", **stacked)
    (out_dir / "manifest.json").write_text(json.dumps(metadata, indent=2))


def load_centroids(in_dir: Path) -> dict[str, np.ndarray]:
    """Load the per-character centroid map (hot path used by ``assign.py``).

    Callers are expected to verify embedder-model compatibility against the
    manifest before using these; loading the centroids alone gives no clue
    which embedder produced them.
    """
    with np.load(in_dir / "centroids.npz") as f:
        return {k: f[k] for k in f.files}


def load_per_clip(in_dir: Path) -> dict[str, np.ndarray]:
    """Load the per-character per-clip embedding map (cold path, for QC).

    Returns ``(N_clips, D)`` arrays — useful for spotting outlier clips or
    measuring intra-character variance without re-running the embedder.
    """
    with np.load(in_dir / "per_clip.npz") as f:
        return {k: f[k] for k in f.files}


def load_manifest(in_dir: Path) -> dict:
    """Load the enrollment manifest (embedder model, characters, stats)."""
    return json.loads((in_dir / "manifest.json").read_text())
