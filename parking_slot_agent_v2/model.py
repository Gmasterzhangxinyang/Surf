"""Strict model actions and local/replay adapters for ParkingAgent v2."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import ipaddress
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Mapping, Protocol, Sequence
from urllib.parse import urlsplit

from .prompts import SYSTEM_PROMPT


TOOL_NAMES = (
    "camera_context",
    "camera_sequence",
    "camera_crop",
    "lidar_detail",
)
ENHANCEMENTS = ("none", "contrast", "sharpen")
FINAL_STATES = ("free", "occupied", "unknown")
LOCALIZATION_STAGES = (
    "not_attempted",
    "hypothesis",
    "supported",
    "refuted",
    "ambiguous",
    "not_visible",
)
TARGET_SIDES = ("left", "center", "right", "unknown")
DEPTH_BANDS = ("near", "middle", "far", "unknown")
_CODE = re.compile(r"^[a-z][a-z0-9_]{0,127}$")


class ActionError(ValueError):
    """Raised when a model action does not match the v2 wire contract."""


class ModelProviderError(RuntimeError):
    """The model provider failed before returning a semantic action."""


class ModelAdapter(Protocol):
    def next_action(
        self,
        request: Mapping[str, Any],
        *,
        image_paths: Sequence[Path] = (),
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class BeliefEstimate:
    state: str
    free_confidence: float
    occupied_confidence: float
    unknown_confidence: float
    resolved_unknown_reasons: tuple[str, ...]
    remaining_unknown_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LocalizationEstimate:
    stage: str
    hypothesis_id: str | None
    target_side: str
    depth_band: str
    target_row: str | None
    target_order_in_row: int | None
    bbox_norm: tuple[float, float, float, float] | None
    matched_landmarks: tuple[str, ...]
    missing_landmarks: tuple[str, ...]
    confidence_before: float
    confidence_after: float
    ambiguity_reasons: tuple[str, ...]

    @classmethod
    def not_attempted(cls) -> "LocalizationEstimate":
        return cls(
            stage="not_attempted",
            hypothesis_id=None,
            target_side="unknown",
            depth_band="unknown",
            target_row=None,
            target_order_in_row=None,
            bbox_norm=None,
            matched_landmarks=(),
            missing_landmarks=(),
            confidence_before=0.0,
            confidence_after=0.0,
            ambiguity_reasons=(),
        )


@dataclass(frozen=True, slots=True)
class ToolAction:
    tool: str
    rationale: str
    arguments: Mapping[str, Any]
    belief: BeliefEstimate
    localization: LocalizationEstimate = field(default_factory=LocalizationEstimate.not_attempted)


@dataclass(frozen=True, slots=True)
class FinalAction:
    state: str
    free_confidence: float
    occupied_confidence: float
    localization_confidence: float | None
    occupancy_confidence: float | None
    evidence_ids: tuple[str, ...]
    reason: str
    reason_codes: tuple[str, ...]
    localization: LocalizationEstimate = field(default_factory=LocalizationEstimate.not_attempted)


AgentAction = ToolAction | FinalAction


def _finite_confidence(value: Any, field: str, *, nullable: bool = False) -> float | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ActionError(f"{field} must be a number in [0,1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ActionError(f"{field} must be a number in [0,1]")
    return result


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ActionError(f"{field} must be a non-empty string")
    return value.strip()


def _string_array(value: Any, field: str, *, codes: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ActionError(f"{field} must be an array")
    result: list[str] = []
    for item in value:
        item = _text(item, field)
        if codes and _CODE.fullmatch(item) is None:
            raise ActionError(f"{field} must contain lower_snake_case codes")
        result.append(item)
    if len(set(result)) != len(result):
        raise ActionError(f"{field} must not contain duplicates")
    return tuple(result)


def _belief(value: Any) -> BeliefEstimate:
    if not isinstance(value, Mapping):
        raise ActionError("belief must be an object")
    expected = {
        "state",
        "free_confidence",
        "occupied_confidence",
        "unknown_confidence",
        "resolved_unknown_reasons",
        "remaining_unknown_reasons",
    }
    if set(value) != expected:
        raise ActionError("belief fields do not match the v2 schema")
    state = _text(value["state"], "belief.state")
    if state not in FINAL_STATES:
        raise ActionError("belief.state is invalid")
    remaining = _string_array(
        value["remaining_unknown_reasons"],
        "belief.remaining_unknown_reasons",
        codes=True,
    )
    if state == "unknown" and not remaining:
        raise ActionError("unknown belief requires a remaining uncertainty reason")
    return BeliefEstimate(
        state=state,
        free_confidence=float(
            _finite_confidence(value["free_confidence"], "belief.free_confidence")
        ),
        occupied_confidence=float(
            _finite_confidence(
                value["occupied_confidence"], "belief.occupied_confidence"
            )
        ),
        unknown_confidence=float(
            _finite_confidence(value["unknown_confidence"], "belief.unknown_confidence")
        ),
        resolved_unknown_reasons=_string_array(
            value["resolved_unknown_reasons"],
            "belief.resolved_unknown_reasons",
            codes=True,
        ),
        remaining_unknown_reasons=remaining,
    )


def _optional_text(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _text(value, field)


def _bbox_or_none(value: Any, field: str) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 4:
        raise ActionError(f"{field} must be null or contain four coordinates")
    coordinates = tuple(float(_finite_confidence(item, field)) for item in value)
    x1, y1, x2, y2 = coordinates
    if not (x1 < x2 and y1 < y2):
        raise ActionError(f"{field} must have positive area")
    return coordinates  # type: ignore[return-value]


def _localization(value: Any) -> LocalizationEstimate:
    if value is None:
        return LocalizationEstimate.not_attempted()
    if not isinstance(value, Mapping):
        raise ActionError("localization must be an object")
    expected = {
        "stage",
        "hypothesis_id",
        "target_side",
        "depth_band",
        "target_row",
        "target_order_in_row",
        "bbox_norm",
        "matched_landmarks",
        "missing_landmarks",
        "confidence_before",
        "confidence_after",
        "ambiguity_reasons",
    }
    if set(value) != expected:
        raise ActionError("localization fields do not match the v2 schema")
    stage = _text(value["stage"], "localization.stage")
    if stage not in LOCALIZATION_STAGES:
        raise ActionError("localization.stage is invalid")
    target_side = _text(value["target_side"], "localization.target_side")
    if target_side not in TARGET_SIDES:
        raise ActionError("localization.target_side is invalid")
    depth_band = _text(value["depth_band"], "localization.depth_band")
    if depth_band not in DEPTH_BANDS:
        raise ActionError("localization.depth_band is invalid")
    order = value["target_order_in_row"]
    if order is not None and (
        isinstance(order, bool) or not isinstance(order, int) or order < 1
    ):
        raise ActionError("localization.target_order_in_row must be null or a positive integer")
    return LocalizationEstimate(
        stage=stage,
        hypothesis_id=_optional_text(value["hypothesis_id"], "localization.hypothesis_id"),
        target_side=target_side,
        depth_band=depth_band,
        target_row=_optional_text(value["target_row"], "localization.target_row"),
        target_order_in_row=order,
        bbox_norm=_bbox_or_none(value["bbox_norm"], "localization.bbox_norm"),
        matched_landmarks=_string_array(value["matched_landmarks"], "localization.matched_landmarks"),
        missing_landmarks=_string_array(value["missing_landmarks"], "localization.missing_landmarks"),
        confidence_before=float(_finite_confidence(value["confidence_before"], "localization.confidence_before")),
        confidence_after=float(_finite_confidence(value["confidence_after"], "localization.confidence_after")),
        ambiguity_reasons=_string_array(value["ambiguity_reasons"], "localization.ambiguity_reasons", codes=True),
    )
def parse_action(payload: Any) -> AgentAction:
    """Parse one strict JSON action returned by an agent model."""

    if not isinstance(payload, Mapping):
        raise ActionError("model action must be a JSON object")
    action_type = payload.get("type")
    if action_type == "tool":
        expected = {"type", "tool", "rationale", "arguments", "belief", "localization"}
        legacy_expected = expected - {"localization"}
        if frozenset(payload) not in {frozenset(expected), frozenset(legacy_expected)}:
            raise ActionError("tool action fields do not match the v2 schema")
        tool = _text(payload["tool"], "tool")
        if tool not in TOOL_NAMES:
            raise ActionError("tool is not allowlisted")
        arguments = payload["arguments"]
        if not isinstance(arguments, Mapping):
            raise ActionError("arguments must be an object")
        if tool == "camera_crop":
            if set(arguments) != {"bbox_norm", "enhancement"}:
                raise ActionError("camera_crop arguments are invalid")
            bbox = arguments["bbox_norm"]
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ActionError("bbox_norm must contain four coordinates")
            coordinates = tuple(
                _finite_confidence(value, "bbox_norm") for value in bbox
            )
            x1, y1, x2, y2 = coordinates
            if not (x1 < x2 and y1 < y2):
                raise ActionError("bbox_norm must have positive area")
            enhancement = _text(arguments["enhancement"], "enhancement")
            if enhancement not in ENHANCEMENTS:
                raise ActionError("camera_crop enhancement is invalid")
            normalized_arguments: Mapping[str, Any] = {
                "bbox_norm": coordinates,
                "enhancement": enhancement,
            }
        else:
            if arguments:
                raise ActionError(f"{tool} does not accept arguments")
            normalized_arguments = {}
        return ToolAction(
            tool=tool,
            rationale=_text(payload["rationale"], "rationale"),
            arguments=normalized_arguments,
            belief=_belief(payload["belief"]),
            localization=_localization(payload.get("localization")),
        )

    if action_type == "final":
        expected = {
            "type",
            "state",
            "free_confidence",
            "occupied_confidence",
            "localization_confidence",
            "occupancy_confidence",
            "evidence_ids",
            "reason",
            "reason_codes",
            "localization",
        }
        legacy_expected = expected - {"localization"}
        if frozenset(payload) not in {frozenset(expected), frozenset(legacy_expected)}:
            raise ActionError("final action fields do not match the v2 schema")
        state = _text(payload["state"], "state")
        if state not in FINAL_STATES:
            raise ActionError("final state is invalid")
        return FinalAction(
            state=state,
            free_confidence=float(_finite_confidence(payload["free_confidence"], "free_confidence")),
            occupied_confidence=float(
                _finite_confidence(payload["occupied_confidence"], "occupied_confidence")
            ),
            localization_confidence=_finite_confidence(
                payload["localization_confidence"],
                "localization_confidence",
                nullable=True,
            ),
            occupancy_confidence=_finite_confidence(
                payload["occupancy_confidence"],
                "occupancy_confidence",
                nullable=True,
            ),
            evidence_ids=_string_array(payload["evidence_ids"], "evidence_ids"),
            reason=_text(payload["reason"], "reason"),
            reason_codes=_string_array(payload["reason_codes"], "reason_codes", codes=True),
            localization=_localization(payload.get("localization")),
        )
    raise ActionError("action type must be tool or final")


class ReplayModelAdapter:
    """Deterministic per-case actions for tests and offline reproducibility."""

    def __init__(
        self,
        actions: Mapping[str, Sequence[Mapping[str, Any]]],
        evidence_bindings: Mapping[str, Sequence[Sequence[Mapping[str, Any]]]] | None = None,
    ) -> None:
        self._actions = {str(key): list(value) for key, value in actions.items()}
        self._evidence_bindings = {
            str(key): [list(refs) for refs in value]
            for key, value in (evidence_bindings or {}).items()
        }
        self._offsets: dict[str, int] = {}
        self.requests: list[dict[str, Any]] = []

    @classmethod
    def from_path(cls, path: str | Path) -> "ReplayModelAdapter":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping) or payload.get("schema_version") != "parking-slot-agent-v2-replay/1.0":
            raise ValueError("replay file has an unsupported schema")
        actions = payload.get("actions")
        if not isinstance(actions, Mapping):
            raise ValueError("replay actions must be an object keyed by case_id")
        bindings = payload.get("evidence_bindings")
        if bindings is not None and not isinstance(bindings, Mapping):
            raise ValueError("replay evidence_bindings must be an object keyed by case_id")
        return cls(actions, bindings)  # type: ignore[arg-type]

    def next_action(
        self,
        request: Mapping[str, Any],
        *,
        image_paths: Sequence[Path] = (),
    ) -> Mapping[str, Any]:
        case_id = request.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("model request requires case_id")
        self.requests.append(json.loads(json.dumps(request)))
        offset = self._offsets.get(case_id, 0)
        sequence = self._actions.get(case_id)
        if sequence is None or offset >= len(sequence):
            raise RuntimeError(f"replay exhausted for {case_id}")
        self._offsets[case_id] = offset + 1
        action = dict(sequence[offset])
        if action.get("type") == "replay_error":
            error_type = action.get("error_type", "ValueError")
            message = action.get("message", "recorded invalid structured action")
            if error_type == "ValueError":
                raise ValueError(str(message))
            raise RuntimeError(str(message))
        # Rendered evidence ids contain an output-path-dependent digest. Replay
        # stores semantic bindings and maps only evidence cited by the live Agent.
        if action.get("type") == "final":
            observations = request.get("observations")
            if isinstance(observations, list):
                refs_by_case = self._evidence_bindings.get(case_id, [])
                refs = refs_by_case[offset] if offset < len(refs_by_case) else []
                rebound_ids: list[str] = []
                for ref in refs:
                    if not isinstance(ref, Mapping):
                        continue
                    for observation in observations:
                        if not isinstance(observation, Mapping):
                            continue
                        if all(
                            observation.get(key) == ref.get(key)
                            for key in ("tool_name", "round_index", "modality")
                        ):
                            evidence_id = observation.get("evidence_id")
                            if (
                                isinstance(evidence_id, str)
                                and evidence_id
                                and evidence_id not in rebound_ids
                            ):
                                rebound_ids.append(evidence_id)
                            break
                if refs:
                    action["evidence_ids"] = rebound_ids
                elif case_id not in self._evidence_bindings:
                    # Backward compatibility for legacy replay files.
                    action["evidence_ids"] = [
                        observation["evidence_id"]
                        for observation in observations
                        if isinstance(observation, Mapping)
                        and isinstance(observation.get("evidence_id"), str)
                    ]
        return action


def _completion_url(base_url: str) -> str:
    value = str(base_url).strip().rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("base_url must be a valid loopback HTTP URL")
    host = parsed.hostname.rstrip(".").lower()
    if host != "localhost":
        try:
            if not ipaddress.ip_address(host).is_loopback:
                raise ValueError("base_url must use a loopback host")
        except ValueError as exc:
            raise ValueError("base_url must use a loopback host") from exc
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("base_url must not contain credentials, query, or fragment")
    return value + "/chat/completions"


class LocalVLMAdapter:
    """Small loopback-only OpenAI-compatible multimodal adapter."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float = 90.0,
        max_image_bytes: int = 20 * 1024 * 1024,
        max_output_tokens: int = 1024,
        system_prompt: str | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be non-empty")
        if (
            isinstance(max_output_tokens, bool)
            or not isinstance(max_output_tokens, int)
            or not 1 <= max_output_tokens <= 4096
        ):
            raise ValueError("max_output_tokens must be an integer in [1,4096]")
        self.endpoint = _completion_url(base_url)
        self.model = model.strip()
        self.timeout_seconds = float(timeout_seconds)
        self.max_image_bytes = int(max_image_bytes)
        self.max_output_tokens = max_output_tokens
        self.system_prompt = SYSTEM_PROMPT if system_prompt is None else str(system_prompt).strip()
        if not self.system_prompt:
            raise ValueError("system_prompt must be non-empty")
        self.transcript: dict[str, list[Mapping[str, Any]]] = {}
        self.calls: list[dict[str, Any]] = []
        self.invalid_calls: list[dict[str, Any]] = []

    @staticmethod
    def _media_type(path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".png":
            return "image/png"
        if suffix in {".jpg", ".jpeg"}:
            return "image/jpeg"
        if suffix == ".webp":
            return "image/webp"
        raise ValueError(f"unsupported model image type: {suffix}")

    def next_action(
        self,
        request: Mapping[str, Any],
        *,
        image_paths: Sequence[Path] = (),
    ) -> Mapping[str, Any]:
        # Lazy import preserves the replay/contract path when httpx is absent.
        import httpx
        # Keep the local OpenAI-compatible endpoint on exactly the same strict
        # root schema and type-specific normalization used by the API adapter.
        from .openai_adapter import (
            _normalize_structured_action,
            openai_action_schema,
        )

        user_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": "Current single-slot request:\n" + json.dumps(
                    request,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
            }
        ]
        total = 0
        for path in image_paths:
            data = Path(path).read_bytes()
            total += len(data)
            if total > self.max_image_bytes:
                raise ValueError("model image byte budget exceeded")
            media_type = self._media_type(Path(path))
            encoded = base64.b64encode(data).decode("ascii")
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{encoded}"},
                }
            )
        wire = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.max_output_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "parking_slot_agent_action",
                    "strict": True,
                    "schema": openai_action_schema(),
                },
            },
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
            ],
        }
        started = time.perf_counter()
        with httpx.Client(trust_env=False, timeout=self.timeout_seconds) as client:
            response = client.post(self.endpoint, json=wire)
            response.raise_for_status()
            payload = response.json()
        elapsed_seconds = time.perf_counter() - started
        case_id = request.get("case_id")
        base_audit = {
            "case_id": case_id,
            "turn": request.get("turn"),
            "elapsed_seconds": elapsed_seconds,
            "image_count": len(image_paths),
            "input_image_bytes": total,
            "usage": payload.get("usage", {}),
            "model": self.model,
        }
        try:
            content = payload["choices"][0]["message"]["content"]
            raw_action = json.loads(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            self.invalid_calls.append(
                {
                    **base_audit,
                    "error": "local VLM returned an invalid JSON completion",
                    "raw_content": payload.get("choices"),
                }
            )
            raise ValueError("local VLM returned an invalid JSON completion") from exc
        if not isinstance(raw_action, Mapping):
            self.invalid_calls.append(
                {
                    **base_audit,
                    "error": "local VLM action must be a JSON object",
                    "action": raw_action,
                }
            )
            raise ValueError("local VLM action must be a JSON object")
        # Normalize the universal strict schema into the same v2 action used by
        # the OpenAI Responses adapter, then validate before retaining replay.
        try:
            action = _normalize_structured_action(raw_action)
        except ValueError as exc:
            self.invalid_calls.append(
                {**base_audit, "error": str(exc), "action": dict(raw_action)}
            )
            raise
        if isinstance(case_id, str):
            self.transcript.setdefault(case_id, []).append(dict(action))
        self.calls.append(
            {
                **base_audit,
                "action": dict(action),
            }
        )
        return dict(action)

    def replay_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "parking-slot-agent-v2-replay/1.0",
            "actions": self.transcript,
        }


__all__ = [
    "ActionError",
    "AgentAction",
    "FinalAction",
    "LocalizationEstimate",
    "LocalVLMAdapter",
    "ModelAdapter",
    "ModelProviderError",
    "ReplayModelAdapter",
    "ToolAction",
    "parse_action",
]
