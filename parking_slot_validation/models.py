"""Prediction normalization and stable evidence identities."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


STATE_MAP = {
    "box_vehicle_core_supported": "occupied",
    "box_low_height_residual": "free",
    "box_boundary_conflict": "unknown",
}
PREDICTIONS = frozenset({"occupied", "free", "unknown"})


@dataclass(frozen=True)
class Prediction:
    run_id: str
    dataset_id: str
    slot_id: str
    prediction: str
    score: float
    anchor_frame: int
    audit: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "dataset_id": self.dataset_id,
            "slot_id": self.slot_id,
            "prediction": self.prediction,
            "score": self.score,
            "anchor_frame": self.anchor_frame,
            "audit": dict(self.audit),
        }


def normalize_prediction_row(
    row: Mapping[str, Any],
    run_id: str,
    dataset_id: str,
) -> Prediction:
    slot_id = str(row.get("slot_id", "")).strip()
    if not slot_id:
        raise ValueError("prediction row is missing slot_id")
    raw_prediction = str(row.get("prediction", "")).strip()
    raw_state = str(row.get("state", "")).strip()
    if raw_prediction:
        prediction = raw_prediction
    else:
        prediction = STATE_MAP.get(raw_state, "")
    if prediction not in PREDICTIONS:
        raw = raw_prediction or raw_state or "<empty>"
        raise ValueError(f"unmapped prediction state: {raw}")
    try:
        score = float(row.get("score", 0.0))
        anchor_frame = int(float(row.get("anchor_frame", 0)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid numeric prediction fields for {slot_id}") from exc
    return Prediction(
        run_id=str(run_id),
        dataset_id=str(dataset_id),
        slot_id=slot_id,
        prediction=prediction,
        score=score,
        anchor_frame=anchor_frame,
        audit={str(key): str(value) for key, value in row.items()},
    )


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fingerprint_files(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    resolved = sorted(Path(path).resolve() for path in paths)
    if not resolved:
        raise ValueError("at least one dataset identity file is required")
    for path in resolved:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def make_manifest_id(dataset_id: str, settings: Mapping[str, Any]) -> str:
    return _canonical_hash({"dataset_id": str(dataset_id), "settings": dict(settings)})


def make_sample_id(dataset_id: str, slot_id: str, encounter_timestamp: float) -> str:
    return _canonical_hash(
        {
            "dataset_id": str(dataset_id),
            "slot_id": str(slot_id),
            "encounter_timestamp": format(float(encounter_timestamp), ".9f"),
        }
    )
