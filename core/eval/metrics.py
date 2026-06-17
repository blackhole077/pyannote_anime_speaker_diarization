"""Eval metrics: DER + per-character identification precision/recall.

Thin wrapper over `pyannote.metrics`. Stops where custom anime-specific logic
begins: `unknown_*` rejection handling, confusion matrices, and any
speaker-ID-aware post-processing are intentionally not implemented here.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional, TypedDict, cast

from omegaconf import DictConfig, OmegaConf
from pyannote.core import Annotation, Timeline
from pyannote.database import FileFinder, registry
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.metrics.identification import (
    IER_CONFUSION,
    IER_FALSE_ALARM,
    IER_MISS,
    IER_TOTAL,
    IdentificationPrecision,
    IdentificationRecall,
)

from core.config import load_cfg
from core.constants import DATASET_ROOT, RESERVED_LABELS
from core.speaker_diarization import load_diarization_pipeline
from core.speaker_id.assign import UNKNOWN_LABEL_PREFIX, apply_gate, score_clusters
from core.speaker_id.embed import load_embedding_model_for_inference
from core.speaker_id.store import load_centroids


class FileRecord(TypedDict):
    uri: str
    audio: str
    annotation: Annotation
    annotated: Timeline


class DERReport(TypedDict):
    value: float
    false_alarm: float
    missed_detection: float
    confusion: float
    total: float


class PerCharacterRawValues(TypedDict):
    correct: float
    confusion: float
    rejection: float
    miss: float
    total: float


class PerCharacterPR(TypedDict):
    precision: float
    recall: float
    support_seconds: float


class EvaluatorReport(TypedDict):
    der: DERReport
    per_character: dict[str, PerCharacterPR]


def _zero_raw() -> PerCharacterRawValues:
    return {"correct": 0.0, "confusion": 0.0, "rejection": 0.0, "miss": 0.0, "total": 0.0}


class Evaluator:
    """Accumulating evaluator over a protocol.

    Call :meth:`add` once per file; read :meth:`report` at the end. The
    underlying pyannote metrics accumulate components internally, so the
    aggregate numbers in the report are time-weighted across all files added.
    """

    def __init__(self, collar: float = 0.25, skip_overlap: bool = False) -> None:
        self._collar = collar
        self._skip_overlap = skip_overlap
        self.der = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
        self.precision: dict[str, IdentificationPrecision] = {}
        self.recall: dict[str, IdentificationRecall] = {}

    def add(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Optional[Timeline] = None,
        uri: Optional[str] = None,
    ) -> None:
        """Score a single file against its reference."""
        self.der(reference, hypothesis, uem=uem, uri=uri)

        for char in reference.labels():
            ref_c = reference.subset([char])
            hyp_c = hypothesis.subset([char])
            p = self.precision.setdefault(
                char,
                IdentificationPrecision(
                    collar=self._collar, skip_overlap=self._skip_overlap
                ),
            )
            r = self.recall.setdefault(
                char,
                IdentificationRecall(
                    collar=self._collar, skip_overlap=self._skip_overlap
                ),
            )
            p(ref_c, hyp_c, uem=uem, uri=uri)
            r(ref_c, hyp_c, uem=uem, uri=uri)

    def report(self) -> EvaluatorReport:
        """Return aggregate metrics across all files added so far."""
        return {
            "der": {
                "value": abs(self.der),
                "false_alarm": self.der.accumulated_[IER_FALSE_ALARM],
                "missed_detection": self.der.accumulated_[IER_MISS],
                "confusion": self.der.accumulated_[IER_CONFUSION],
                "total": self.der.accumulated_[IER_TOTAL],
            },
            "per_character": {
                char: {
                    "precision": abs(self.precision[char]),
                    "recall": abs(self.recall[char]),
                    "support_seconds": self.recall[char].accumulated_["# relevant"],
                }
                for char in sorted(self.precision)
            },
        }

    def per_file(self) -> list[dict]:
        """Per-file DER results, in the order files were added."""
        return [
            {
                "uri": uri,
                "der": components[self.der.metric_name_],
                "false_alarm": components[IER_FALSE_ALARM],
                "missed_detection": components[IER_MISS],
                "confusion": components[IER_CONFUSION],
                "total": components[IER_TOTAL],
            }
            for uri, components in self.der.results_
        ]


def decompose_recall(
    reference: Annotation,
    hypothesis: Annotation,
    collar: float = 0.25,
    skip_overlap: bool = False,
    uem: Timeline | None = None,
    unknown_prefix: str = UNKNOWN_LABEL_PREFIX,
) -> dict[str, PerCharacterRawValues]:
    """Per-character recall split into correct / confusion / rejection / miss seconds.

    Returns {char: {"correct", "confusion", "rejection", "miss", "total"}}.
    """
    metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
    reference, hypothesis = metric.uemify(
        reference, hypothesis, uem, collar, skip_overlap
    )
    metrics_per_character: dict[str, PerCharacterRawValues] = defaultdict(_zero_raw)
    for ref_segment, _, ref_label in reference.itertracks(yield_label=True):
        metrics_per_character[ref_label]["total"] += ref_segment.duration
        sub = hypothesis.crop(ref_segment, mode="intersection")
        for sub_segment, _, hyp_label in sub.itertracks(yield_label=True):
            if hyp_label == ref_label:
                metrics_per_character[ref_label]["correct"] += sub_segment.duration
            elif hyp_label.startswith(unknown_prefix):
                metrics_per_character[ref_label]["rejection"] += sub_segment.duration
            else:
                metrics_per_character[ref_label]["confusion"] += sub_segment.duration
    for metrics in metrics_per_character.values():
        metrics["miss"] = (
            metrics["total"]
            - metrics["correct"]
            - metrics["rejection"]
            - metrics["confusion"]
        )
    return dict(metrics_per_character)


def aggregate_recall_decomposition(
    decomposed_recalls: list[tuple[str, dict[str, PerCharacterRawValues]]],
) -> tuple[
    dict[str, PerCharacterRawValues],
    dict[str, dict[str, PerCharacterRawValues]],
]:
    per_character_aggregate: dict[str, PerCharacterRawValues] = defaultdict(_zero_raw)
    per_episode_per_character_aggregate: dict[str, dict[str, PerCharacterRawValues]] = {}
    for uri, per_char_values in decomposed_recalls:
        per_episode_per_character_aggregate[uri] = dict(per_char_values)
        for character, values in per_char_values.items():
            for key, value in values.items():
                per_character_aggregate[character][key] += value
    return dict(per_character_aggregate), per_episode_per_character_aggregate


def build_report(
    cfg: DictConfig,
    gated: Evaluator,
    ungated: Optional[Evaluator],
    gated_decomp: list[tuple[str, dict[str, PerCharacterRawValues]]],
    ungated_decomp: list[tuple[str, dict[str, PerCharacterRawValues]]],
) -> dict:
    config_block = OmegaConf.to_container(cfg, resolve=True)
    gated_per_char_agg, gated_per_ep_per_char_agg = aggregate_recall_decomposition(
        gated_decomp
    )
    gated_block = {
        "metrics": gated.report(),
        "decompose": gated_per_char_agg,
        "per_episode_decompose": gated_per_ep_per_char_agg,
    }

    if ungated is None:
        return {
            "config": config_block,
            "gated": gated_block,
            "ungated": None,
            "gate_cost": None,
            "per_file_der": {"gated": gated.per_file(), "ungated": None},
        }

    ungated_per_char_agg, ungated_per_ep_per_char_agg = aggregate_recall_decomposition(
        ungated_decomp
    )

    per_char_cost: dict[str, dict[str, float]] = {}
    aggregate_cost = {"correct_delta": 0.0, "confusion_delta": 0.0, "rejection_delta": 0.0}
    for char in sorted(set(gated_per_char_agg) | set(ungated_per_char_agg)):
        g = gated_per_char_agg.get(char, _zero_raw())
        u = ungated_per_char_agg.get(char, _zero_raw())
        # Invariant: total is reference-side, must match across passes.
        assert abs(g["total"] - u["total"]) < 1e-6, (
            f"total mismatch for {char}: gated={g['total']} ungated={u['total']}"
        )
        deltas = {
            "correct_delta": g["correct"] - u["correct"],
            "confusion_delta": g["confusion"] - u["confusion"],
            "rejection_delta": g["rejection"] - u["rejection"],
        }
        # gate_value > 0 means the gate hid more wrong calls than correct ones.
        per_char_cost[char] = {
            **deltas,
            "gate_value": deltas["correct_delta"] - deltas["confusion_delta"],
        }
        for k, v in deltas.items():
            aggregate_cost[k] += v
    aggregate_cost["gate_value"] = aggregate_cost["correct_delta"] - aggregate_cost["confusion_delta"]
    aggregate_cost["der_delta"] = abs(gated.der) - abs(ungated.der)

    return {
        "config": config_block,
        "gated": gated_block,
        "ungated": {
            "metrics": ungated.report(),
            "decompose": ungated_per_char_agg,
            "per_episode_decompose": ungated_per_ep_per_char_agg,
        },
        "gate_cost": {"per_character": per_char_cost, "aggregate": aggregate_cost},
        "per_file_der": {
            "gated": gated.per_file(),
            "ungated": ungated.per_file(),
        },
    }


def normalize_reference(annotation: Annotation) -> tuple[Annotation, Timeline]:
    relabel = {l: l.rstrip("?") for l in annotation.labels() if l.endswith("?")}
    annotation = annotation.rename_labels(mapping=relabel)
    exclude = (
        annotation.subset(RESERVED_LABELS & set(annotation.labels()))
        .get_timeline()
        .support()
    )
    annotation = annotation.subset(set(annotation.labels()) - RESERVED_LABELS)
    return annotation, exclude


def load_eval_cfg(path: Path) -> DictConfig:
    """Resolve eval.yaml into a flat cfg with three sections.

    Reads the ``include`` / ``overrides`` / ``eval`` shape, loads the
    referenced config files, merges any overrides, and returns a single
    DictConfig with ``cfg.eval``, ``cfg.diarization``, ``cfg.speaker_id``.
    """
    raw = load_cfg(path)
    diar_cfg = OmegaConf.load(raw.include.diarization)
    sid_cfg = OmegaConf.load(raw.include.speaker_id)
    sid_overrides = OmegaConf.select(raw, "overrides.speaker_id", default={})
    if sid_overrides:
        sid_cfg = OmegaConf.merge(sid_cfg, sid_overrides)
    return OmegaConf.create({
        "eval": raw.eval,
        "diarization": diar_cfg,
        "speaker_id": sid_cfg,
    })


def _register_protocol(template_path: Path) -> None:
    """Render ``${ANIME_DATA_ROOT}`` in a protocol YAML and register it.

    pyannote.database does not expand environment variables in protocol paths and
    resolves relative paths against the database YAML's own location, so we
    substitute the dataset root into a temp copy and load that.
    """
    import tempfile

    rendered = template_path.read_text().replace("${ANIME_DATA_ROOT}", str(DATASET_ROOT))
    tmp = Path(tempfile.gettempdir()) / "anime_database.yml"
    tmp.write_text(rendered)
    registry.load_database(str(tmp))


def run_eval(cfg: DictConfig, token: Optional[str] = None) -> dict:
    eval_cfg = cfg.eval
    protocol_config = eval_cfg.get("protocol_config")
    if protocol_config:
        _register_protocol(Path(protocol_config))
    protocol = registry.get_protocol(eval_cfg.protocol, preprocessors={"audio": FileFinder()})
    pipeline = load_diarization_pipeline(cfg.diarization, token)

    enrollment_dir = eval_cfg.get("enrollment_dir")
    use_enrollment = bool(enrollment_dir)
    collar = eval_cfg.metrics.collar
    skip_overlap = eval_cfg.metrics.skip_overlap

    gated = Evaluator(collar=collar, skip_overlap=skip_overlap)
    ungated = Evaluator(collar=collar, skip_overlap=skip_overlap) if use_enrollment else None
    decomp_gated: list[tuple[str, dict[str, PerCharacterRawValues]]] = []
    decomp_ungated: list[tuple[str, dict[str, PerCharacterRawValues]]] = []

    centroids = load_centroids(Path(enrollment_dir)) if use_enrollment else None
    inference = load_embedding_model_for_inference(token=token) if use_enrollment else None

    def score(evaluator: Evaluator, decomp_list, hyp: Annotation, ref, uem, uri):
        evaluator.add(ref, hyp, uem=uem, uri=uri)
        decomp_list.append(
            (uri, decompose_recall(ref, hyp, uem=uem, collar=collar, skip_overlap=skip_overlap))
        )

    for proto_file in getattr(protocol, eval_cfg.subset)():
        record = cast(FileRecord, proto_file)
        uri = record["uri"]
        ref, exclude = normalize_reference(record["annotation"])
        uem = record["annotated"].extrude(exclude)
        wav_path: Path = Path(record["audio"])
        clusters: Annotation = pipeline(record["audio"]).exclusive_speaker_diarization

        if use_enrollment:
            cluster_scores = score_clusters(clusters, wav_path, centroids, inference)
            gated_hyp, _ = apply_gate(
                clusters, cluster_scores,
                similarity_threshold=cfg.speaker_id.gate.sim,
                min_margin=cfg.speaker_id.gate.margin,
            )
            ungated_hyp, _ = apply_gate(
                clusters, cluster_scores,
                similarity_threshold=0.0,
                min_margin=0.0,
            )
            score(gated, decomp_gated, gated_hyp, ref, uem, uri)
            score(ungated, decomp_ungated, ungated_hyp, ref, uem, uri)
        else:
            gated.add(ref, clusters, uem=uem, uri=uri)

    report = build_report(cfg, gated, ungated, decomp_gated, decomp_ungated)

    if eval_cfg.get("output_path"):
        output_path = Path(eval_cfg.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, default=str))

    return report
