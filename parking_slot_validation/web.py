"""Local HTTP service for persistent parking-slot annotation."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import unquote, urlparse

from .metrics import evaluate
from .models import Prediction
from .storage import LabelRepository


@dataclass
class ValidationState:
    manifest: dict[str, Any]
    predictions: dict[str, Prediction]
    repository: LabelRepository
    output_dir: Path
    index_html: Path

    @property
    def cases_by_sample(self) -> dict[str, dict[str, Any]]:
        return {str(case["sample_id"]): case for case in self.manifest["cases"]}

    def prediction_values(self) -> dict[str, str]:
        return {slot_id: prediction.prediction for slot_id, prediction in self.predictions.items()}

    def summary(self) -> dict[str, Any]:
        labels = self.repository.list_labels()
        by_slot = {record["slot_id"]: record for record in labels.values()}
        return evaluate(self.prediction_values(), by_slot, len(self.manifest["cases"]))

    @staticmethod
    def _comparison(prediction: str, human_label: str) -> str:
        if human_label == "unobservable":
            return "not_evaluated"
        if prediction == "unknown":
            return "abstention"
        if prediction == human_label:
            return "correct"
        if prediction == "occupied":
            return "false_occupied"
        return "false_free"

    def public_cases(self) -> list[dict[str, Any]]:
        labels = self.repository.list_labels()
        result: list[dict[str, Any]] = []
        for raw_case in self.manifest["cases"]:
            case = copy.deepcopy(raw_case)
            for frame in case.get("evidence_frames", []):
                frame.pop("camera_image_path", None)
                frame["asset_url"] = "/assets/" + str(frame["asset_path"])
            if case.get("map_asset"):
                case["map_url"] = "/assets/" + str(case["map_asset"])
            label = labels.get(str(case["sample_id"]))
            case["human_label"] = label
            if label is not None:
                prediction = self.predictions.get(str(case["slot_id"]))
                if prediction is not None:
                    case["algorithm"] = prediction.to_dict()
                    case["comparison"] = self._comparison(
                        prediction.prediction,
                        str(label["human_label"]),
                    )
            result.append(case)
        return result


def create_server(host: str, port: int, state: ValidationState) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        server_version = "ParkingSlotValidation/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: Mapping[str, Any]) -> None:
            self._send(
                status,
                json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"),
                "application/json; charset=utf-8",
            )

        def _asset(self, requested: str) -> None:
            relative = Path(unquote(requested).lstrip("/"))
            try:
                target = (state.output_dir / relative).resolve()
                target.relative_to(state.output_dir.resolve())
            except (ValueError, OSError):
                self._json(HTTPStatus.NOT_FOUND, {"error": "asset not found"})
                return
            if not target.is_file():
                self._json(HTTPStatus.NOT_FOUND, {"error": "asset not found"})
                return
            content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            self._send(HTTPStatus.OK, target.read_bytes(), content_type)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                self._send(HTTPStatus.OK, state.index_html.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/cases":
                self._json(HTTPStatus.OK, {"cases": state.public_cases()})
            elif path == "/api/summary":
                self._json(HTTPStatus.OK, state.summary())
            elif path == "/api/export.json":
                self._send(HTTPStatus.OK, state.repository.export_json().encode("utf-8"), "application/json; charset=utf-8")
            elif path == "/api/export.csv":
                self._send(HTTPStatus.OK, state.repository.export_csv().encode("utf-8"), "text/csv; charset=utf-8")
            elif path.startswith("/assets/"):
                self._asset(path[len("/assets/"):])
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/labels":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size <= 0 or size > 65536:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(size))
                sample_id = str(payload.get("sample_id", ""))
                case = state.cases_by_sample.get(sample_id)
                if case is None:
                    raise ValueError("sample_id is not in the evidence manifest")
                if str(payload.get("slot_id", "")) != str(case["slot_id"]):
                    raise ValueError("slot_id does not match sample_id")
                payload["evidence_frame_refs"] = [
                    frame["asset_path"] for frame in case.get("evidence_frames", [])
                ]
                record = state.repository.upsert(payload)
                prediction = state.predictions.get(str(case["slot_id"]))
                response: dict[str, Any] = {"label": record, "summary": state.summary()}
                if prediction is not None:
                    response["algorithm"] = prediction.to_dict()
                    response["comparison"] = state._comparison(prediction.prediction, record["human_label"])
                self._json(HTTPStatus.OK, response)
            except (ValueError, json.JSONDecodeError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    return ThreadingHTTPServer((host, int(port)), Handler)
