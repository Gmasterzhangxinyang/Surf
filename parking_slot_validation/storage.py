"""Durable human-label storage with dataset and evidence guards."""

from __future__ import annotations

import copy
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import shutil
import threading
from typing import Any, Mapping


HUMAN_LABELS = frozenset({"occupied", "free", "unobservable"})
UNOBSERVABLE_REASONS = frozenset(
    {
        "outside_fov",
        "occluded",
        "bad_projection",
        "temporal_conflict",
        "missing_image",
        "unclear",
    }
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class LabelRepository:
    def __init__(
        self,
        path: Path,
        dataset_id: str,
        manifest_id: str,
        *,
        reviewer: str = "local-reviewer",
    ) -> None:
        self.path = Path(path)
        self.backup_path = Path(str(self.path) + ".bak")
        self.dataset_id = str(dataset_id)
        self.manifest_id = str(manifest_id)
        self.reviewer = str(reviewer)
        self._lock = threading.RLock()
        self._payload = self._load_or_empty()

    def _empty(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "dataset_id": self.dataset_id,
            "manifest_id": self.manifest_id,
            "labels": {},
        }

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("labels"), dict):
            raise ValueError("invalid human label file structure")
        return payload

    def _validate_identity(self, payload: Mapping[str, Any]) -> None:
        if str(payload.get("dataset_id", "")) != self.dataset_id:
            raise ValueError("human label dataset does not match requested dataset")
        if str(payload.get("manifest_id", "")) != self.manifest_id:
            raise ValueError("human label manifest does not match requested manifest")

    def _load_or_empty(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty()
        try:
            payload = self._read(self.path)
        except (OSError, ValueError, json.JSONDecodeError) as primary_error:
            if not self.backup_path.exists():
                raise ValueError(f"human label file is corrupt: {primary_error}") from primary_error
            try:
                payload = self._read(self.backup_path)
            except (OSError, ValueError, json.JSONDecodeError) as backup_error:
                raise ValueError("human label file and backup are both corrupt") from backup_error
            self._validate_identity(payload)
            self._write_payload(payload, create_backup=False)
            return payload
        self._validate_identity(payload)
        return payload

    def _write_payload(self, payload: Mapping[str, Any], *, create_backup: bool = True) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if create_backup and self.path.exists():
            shutil.copy2(self.path, self.backup_path)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)

    def list_labels(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self._payload["labels"])

    def upsert(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        sample_id = str(payload.get("sample_id", "")).strip()
        slot_id = str(payload.get("slot_id", "")).strip()
        human_label = str(payload.get("human_label", "")).strip()
        reason = str(payload.get("reason", "")).strip()
        if not sample_id:
            raise ValueError("sample_id is required")
        if not slot_id:
            raise ValueError("slot_id is required")
        if human_label not in HUMAN_LABELS:
            raise ValueError(f"invalid human_label: {human_label or '<empty>'}")
        if human_label == "unobservable" and reason not in UNOBSERVABLE_REASONS:
            raise ValueError("unobservable label requires an allowed reason")
        if human_label != "unobservable":
            reason = ""
        refs = payload.get("evidence_frame_refs", [])
        if not isinstance(refs, list):
            raise ValueError("evidence_frame_refs must be a list")
        with self._lock:
            existing = self._payload["labels"].get(sample_id, {})
            now = _now()
            record = {
                "sample_id": sample_id,
                "dataset_id": self.dataset_id,
                "slot_id": slot_id,
                "human_label": human_label,
                "reason": reason,
                "evidence_manifest_id": self.manifest_id,
                "evidence_frame_refs": [str(ref) for ref in refs],
                "reviewer": str(payload.get("reviewer") or self.reviewer),
                "created_at": str(existing.get("created_at") or now),
                "updated_at": now,
                "annotation_schema_version": 1,
            }
            self._payload["labels"][sample_id] = record
            self._write_payload(self._payload)
            return copy.deepcopy(record)

    def remove(self, sample_id: str) -> bool:
        with self._lock:
            if sample_id not in self._payload["labels"]:
                return False
            del self._payload["labels"][sample_id]
            self._write_payload(self._payload)
            return True

    def export_json(self) -> str:
        with self._lock:
            return json.dumps(self._payload, ensure_ascii=False, sort_keys=True, indent=2)

    def export_csv(self) -> str:
        fieldnames = [
            "sample_id",
            "dataset_id",
            "slot_id",
            "human_label",
            "reason",
            "evidence_manifest_id",
            "evidence_frame_refs",
            "reviewer",
            "created_at",
            "updated_at",
            "annotation_schema_version",
        ]
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        with self._lock:
            for record in self._payload["labels"].values():
                row = dict(record)
                row["evidence_frame_refs"] = json.dumps(row.get("evidence_frame_refs", []), ensure_ascii=False)
                writer.writerow(row)
        return stream.getvalue()
