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

The pipeline is split into two pieces so the expensive embedding step can be
amortized across multiple gate configurations:

- :func:`score_clusters` runs the per-cluster embedding + scoring (GPU-bound,
  gate-independent).
- :func:`apply_gate` consumes those scores and produces a relabeled annotation
  for a specific ``(similarity_threshold, min_margin)`` pair (pure-Python, cheap).

:func:`assign_speakers` is a thin convenience wrapper that composes the two
for callers that only need one gated assignment.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
from pyannote.audio import Inference
from pyannote.core.annotation import Annotation, Segment

from core.constants import EPSILON

UNKNOWN_LABEL_PREFIX = "unknown_"

REASON_OK = "ok"
REASON_BELOW_SIM = "below_similarity_threshold"
REASON_BELOW_MARGIN = "below_margin"
REASON_NO_SEGMENTS = "no_usable_segments"


def embed_cluster(
    inference: Inference,
    wav_path: Path,
    segments: list[Segment],
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


def score_clusters(
    annotation: Annotation,
    wav_path: Path,
    centroids: dict[str, np.ndarray],
    inference: Inference,
) -> dict[str, dict]:
    """Embed each cluster and score it against every enrolled centroid.

    Gate-independent: the returned scores can be fed to :func:`apply_gate`
    multiple times with different thresholds without re-running inference.

    Clusters with no usable segments are emitted with all scoring fields set
    to ``None`` and ``reason=REASON_NO_SEGMENTS`` — :func:`apply_gate` honors
    that flag to produce an unknown label regardless of thresholds. All other
    clusters carry ``reason=None`` until a gate is applied.
    """
    by_cluster: dict[str, list[Segment]] = defaultdict(list)
    for segment, _, label in annotation.itertracks(yield_label=True):
        by_cluster[label].append(segment)

    char_names = list(centroids.keys())
    char_matrix = np.vstack([centroids[c] for c in char_names])

    scores: dict[str, dict] = {}
    for cluster_id, segments in by_cluster.items():
        emb = embed_cluster(inference, wav_path, segments)
        if emb is None:
            scores[cluster_id] = {
                "best_match": None,
                "similarity": None,
                "second_best": None,
                "second_best_similarity": None,
                "margin": None,
                "reason": REASON_NO_SEGMENTS,
            }
            continue
        # Both centroids and emb are unit-norm, so the dot product is cosine sim.
        sims = char_matrix @ emb
        best = int(np.argmax(sims))
        best_sim = float(sims[best])
        if len(sims) > 1:
            order = np.argsort(sims)
            second_best_name = char_names[int(order[-2])]
            second_best_sim = float(sims[order[-2]])
        else:
            second_best_name = None
            second_best_sim = None
        margin = (
            best_sim - second_best_sim if second_best_sim is not None else float("inf")
        )
        scores[cluster_id] = {
            "best_match": char_names[best],
            "similarity": best_sim,
            "second_best": second_best_name,
            "second_best_similarity": second_best_sim,
            "margin": margin,
            "reason": None,
        }
    return scores


def apply_gate(
    annotation: Annotation,
    scores: dict[str, dict],
    similarity_threshold: float,
    min_margin: float,
) -> tuple[Annotation, dict[str, dict]]:
    """Resolve scored clusters to character names by applying the two gates.

    Returns ``(relabeled_annotation, confidences)``. ``confidences`` maps each
    original cluster ID to a dict containing the scoring fields plus a final
    ``assigned`` label and ``reason`` code (one of ``REASON_*`` constants).
    """
    mapping: dict[str, str] = {}
    confidences: dict[str, dict] = {}
    for cluster_id, s in scores.items():
        unknown = f"{UNKNOWN_LABEL_PREFIX}{cluster_id}"
        if s["reason"] == REASON_NO_SEGMENTS:
            assigned, reason = unknown, REASON_NO_SEGMENTS
        elif s["similarity"] < similarity_threshold:
            assigned, reason = unknown, REASON_BELOW_SIM
        elif s["margin"] < min_margin:
            assigned, reason = unknown, REASON_BELOW_MARGIN
        else:
            assigned, reason = s["best_match"], REASON_OK
        mapping[cluster_id] = assigned
        confidences[cluster_id] = {
            "best_match": s["best_match"],
            "similarity": s["similarity"],
            "second_best": s["second_best"],
            "second_best_similarity": s["second_best_similarity"],
            "margin": s["margin"],
            "assigned": assigned,
            "reason": reason,
        }
    return annotation.rename_labels(mapping=mapping), confidences


def assign_speakers(
    annotation: Annotation,
    wav_path: Path,
    centroids: dict[str, np.ndarray],
    inference: Inference,
    similarity_threshold: float,
    min_margin: float,
) -> tuple[Annotation, dict[str, dict]]:
    """Score + gate in one call. Convenience wrapper around
    :func:`score_clusters` and :func:`apply_gate`. Use the two-step path
    directly when you need to apply different gates to the same scoring run
    (e.g., gated vs. ungated eval passes).
    """
    scores = score_clusters(annotation, wav_path, centroids, inference)
    return apply_gate(annotation, scores, similarity_threshold, min_margin)
