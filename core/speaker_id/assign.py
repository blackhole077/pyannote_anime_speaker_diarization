"""Map anonymous pyannote cluster IDs to enrolled character names.

After pyannote diarization produces clusters like ``SPEAKER_00``, ``SPEAKER_01``,
..., this module embeds each cluster, compares it against the enrolled centroids
by cosine similarity, and renames the cluster either to the best-matching
character or to a generated ``unknown_*`` label.

Two gates protect against forced wrong assignments — they exist because
within-gender confusions in this domain are common and produce knife-edge
margins that a single absolute-similarity threshold cannot distinguish from a
confident match:

- ``similarity_threshold``: reject if the best match's similarity is too low
  in absolute terms (out-of-distribution speaker).
- ``min_margin``: reject if the gap between best and second-best is too small
  (two enrolled characters are equally close — assignment would be a coin flip).
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
from pyannote.audio import Inference
from pyannote.core.annotation import Annotation, Segment

from core.constants import EPSILON


def embed_cluster(
    inference: Inference,
    wav_path: Path,
    segments: list[Segment],
    *,
    min_segment_seconds: float = 1.0,
) -> np.ndarray | None:
    """Compute a single unit-norm embedding for an entire pyannote cluster.

    Each usable segment is embedded once, L2-normalized, then mean-pooled and
    re-normalized so the result is comparable against centroids via dot
    product. Segments shorter than ``min_segment_seconds`` are skipped because
    the embedding model has a minimum receptive field and crashes (or
    produces degenerate output) on inputs that are too short.

    Args:
        inference: A loaded pyannote ``Inference`` instance.
        wav_path: Source audio file the segments index into.
        segments: All pyannote segments assigned to one cluster.
        min_segment_seconds: Minimum segment length to use.

    Returns:
        The unit-norm cluster embedding, or ``None`` when no segment in the
        cluster is long enough — the caller should treat that as "unknown".
    """
    usable = [s for s in segments if s.duration >= min_segment_seconds]
    if not usable:
        return None
    per_clip = [np.asarray(inference.crop(wav_path, segment)) for segment in usable]
    per_clip = [v / (np.linalg.norm(v) + EPSILON) for v in per_clip]
    centroid = np.mean(per_clip, axis=0)
    return (centroid / (np.linalg.norm(centroid) + EPSILON)).astype(np.float32)


def assign_speakers(
    annotation: Annotation,
    wav_path: Path,
    centroids: dict[str, np.ndarray],
    inference: Inference,
    *,
    similarity_threshold: float = 0.40,
    min_margin: float = 0.05,
    unknown_label_prefix: str = "unknown_",
) -> tuple[Annotation, dict[str, dict]]:
    """Relabel pyannote cluster IDs in ``annotation`` with enrolled character names.

    Each cluster is embedded by :func:`embed_cluster`, then the best and
    second-best centroid matches are computed by cosine similarity. The
    cluster is renamed to the best-matching character only if both gates
    pass; otherwise it receives a generated ``unknown_<cluster_id>`` label.

    Args:
        annotation: pyannote ``Annotation`` whose labels are anonymous cluster
            IDs (``SPEAKER_00`` etc.).
        wav_path: Source audio for ``annotation``.
        centroids: Enrolled per-character unit-norm centroids.
        inference: A loaded pyannote embedding ``Inference`` instance. Must
            match the embedder used to build ``centroids``; mismatched
            embedders silently produce garbage similarities (verify upstream
            via the enrollment manifest).
        similarity_threshold: Reject as unknown if best-match cosine
            similarity falls below this.
        min_margin: Reject as unknown if (best - second-best) similarity
            falls below this. Set to ``0`` to disable the margin gate.
        unknown_label_prefix: Prefix used to build the fallback label.

    Returns:
        A tuple ``(relabeled_annotation, confidences)``. ``confidences`` maps
        each original cluster ID to a dict containing best/second-best matches,
        their similarities, the margin, the final assignment, and a ``reason``
        code (``ok``, ``below_similarity_threshold``, ``below_margin``, or
        ``no_usable_segments``).
    """
    by_cluster: dict[str, list[Segment]] = defaultdict(list)
    for segment, _, label in annotation.itertracks(yield_label=True):
        by_cluster[label].append(segment)

    char_names = list(centroids.keys())
    char_matrix = np.vstack([centroids[c] for c in char_names])  # (C, D)

    mapping: dict[str, str] = {}
    confidences: dict[str, dict] = {}
    for cluster_id, segments in by_cluster.items():
        emb = embed_cluster(inference, wav_path, segments)  # (D,) or None
        if emb is None:
            mapping[cluster_id] = f"{unknown_label_prefix}{cluster_id}"
            confidences[cluster_id] = {
                "best_match": None,
                "similarity": None,
                "second_best": None,
                "second_best_similarity": None,
                "assigned": mapping[cluster_id],
                "reason": "no_usable_segments",
            }
            continue
        sims = char_matrix @ emb  # cosine sim since both unit-norm
        best = int(np.argmax(sims))
        best_sim = float(sims[best])
        if len(sims) > 1:
            order = np.argsort(sims)
            second_best_name = char_names[int(order[-2])]
            second_best_sim = float(sims[order[-2]])
        else:
            second_best_name = None
            second_best_sim = None
        margin = best_sim - second_best_sim if second_best_sim is not None else float("inf")

        if best_sim < similarity_threshold:
            assigned = f"{unknown_label_prefix}{cluster_id}"
            reason = "below_similarity_threshold"
        elif margin < min_margin:
            assigned = f"{unknown_label_prefix}{cluster_id}"
            reason = "below_margin"
        else:
            assigned = char_names[best]
            reason = "ok"
        mapping[cluster_id] = assigned
        confidences[cluster_id] = {
            "best_match": char_names[best],
            "similarity": best_sim,
            "second_best": second_best_name,
            "second_best_similarity": second_best_sim,
            "margin": margin,
            "assigned": assigned,
            "reason": reason,
        }

    return annotation.rename_labels(mapping=mapping), confidences
