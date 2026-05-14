"""Speaker embedding and per-character centroid construction.

Wraps ``pyannote/embedding`` for the enrollment path. Centroids are mean-pooled
L2-normalized per-clip embeddings; the L2 normalization on both the inputs and
the final centroid lets downstream code use a dot product as cosine similarity
without an extra normalize step.

Per-clip embeddings are retained alongside centroids so offline QC tooling can
inspect intra-character variance (e.g. outlier clips) without re-running the
model.
"""

import numpy as np
from pyannote.audio import Inference, Model

from core.constants import EPSILON, HF_AUTH_TOKEN
from core.speaker_id.refs import ClipRef


def load_embedding_model_for_inference(
    token: str | None = None,
) -> Inference:
    """Load ``pyannote/embedding`` configured for whole-clip inference.

    ``window="whole"`` produces one embedding per input clip regardless of
    duration, which is what we want for both enrollment (one vector per
    labeled clip) and assignment (one vector per concatenated cluster).

    Args:
        token: HuggingFace auth token. Falls back to the ``HF_AUTH_TOKEN``
            constant when omitted.
    """
    token = token or HF_AUTH_TOKEN
    model = Model.from_pretrained("pyannote/embedding", token=token)
    inference = Inference(model, window="whole")
    return inference


def enroll_characters(
    clips_by_character: dict[str, list[ClipRef]],
    inference: Inference | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, list[np.ndarray]]]:
    """Build per-character centroids from labeled clips.

    Each clip is embedded once, L2-normalized, then averaged into a centroid
    that is itself L2-normalized. Both the centroid and the per-clip vectors
    are returned: the centroid is the hot-path lookup target used during
    assignment, while per-clip vectors are kept for offline QC.

    Characters with no clips after selection are skipped (no centroid emitted).

    Args:
        clips_by_character: Per-character clip references to enroll.
        inference: Optional preloaded ``Inference`` instance. If omitted, a
            fresh one is loaded (incurs a model download / GPU init).

    Returns:
        A tuple ``(centroids, per_clip_embeddings)``. ``centroids`` maps each
        character to a unit-norm ``(D,)`` ``float32`` vector. ``per_clip_embeddings``
        maps each character to the list of unit-norm per-clip vectors that
        produced that centroid.
    """
    inference = inference or load_embedding_model_for_inference()
    centroids: dict[str, np.ndarray] = {}
    per_clip_store: dict[str, list[np.ndarray]] = {}
    for character, clips in clips_by_character.items():
        if not clips:
            continue
        # one forward pass per clip — required for QC to reuse later
        per_clip = [
            np.asarray(inference.crop(c.wav_path, c.to_segment())) for c in clips
        ]
        per_clip = [v / (np.linalg.norm(v) + EPSILON) for v in per_clip]
        centroid = np.mean(per_clip, axis=0)
        centroids[character] = (centroid / (np.linalg.norm(centroid) + EPSILON)).astype(
            np.float32
        )
        per_clip_store[character] = per_clip
    return centroids, per_clip_store
