"""Official OpenAI Responses API adapter for the bounded v2 agent."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from .io import save_json_atomic
from .model import ModelAdapter, ModelProviderError, parse_action
from .prompts import SYSTEM_PROMPT


DEFAULT_OPENAI_MODEL = "gpt-5.6-terra"
_DETAIL_LEVELS = frozenset({"low", "high", "original", "auto"})
_REASONING_EFFORTS = frozenset({"none", "low", "medium", "high", "xhigh", "max"})


def _nullable(schema: Mapping[str, Any]) -> dict[str, Any]:
    return {"anyOf": [dict(schema), {"type": "null"}]}


def _confidence_schema() -> dict[str, Any]:
    return {"type": "number", "minimum": 0.0, "maximum": 1.0}


def openai_action_schema() -> dict[str, Any]:
    """Return one strict root-object schema, normalized later by action type."""

    code_array = {
        "type": "array",
        "items": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,127}$"},
    }
    belief = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "state": {"type": "string", "enum": ["free", "occupied", "unknown"]},
            "free_confidence": _confidence_schema(),
            "occupied_confidence": _confidence_schema(),
            "unknown_confidence": _confidence_schema(),
            "resolved_unknown_reasons": code_array,
            "remaining_unknown_reasons": code_array,
        },
        "required": [
            "state",
            "free_confidence",
            "occupied_confidence",
            "unknown_confidence",
            "resolved_unknown_reasons",
            "remaining_unknown_reasons",
        ],
    }
    localization = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "stage": {
                "type": "string",
                "enum": [
                    "not_attempted",
                    "hypothesis",
                    "supported",
                    "refuted",
                    "ambiguous",
                    "not_visible",
                ],
            },
            "hypothesis_id": _nullable({"type": "string", "minLength": 1}),
            "target_side": {
                "type": "string",
                "enum": ["left", "center", "right", "unknown"],
            },
            "depth_band": {
                "type": "string",
                "enum": ["near", "middle", "far", "unknown"],
            },
            "target_row": _nullable({"type": "string", "minLength": 1}),
            "target_order_in_row": _nullable({"type": "integer", "minimum": 1}),
            "bbox_norm": _nullable(
                {
                    "type": "array",
                    "items": _confidence_schema(),
                    "minItems": 4,
                    "maxItems": 4,
                }
            ),
            "matched_landmarks": {"type": "array", "items": {"type": "string"}},
            "missing_landmarks": {"type": "array", "items": {"type": "string"}},
            "confidence_before": _confidence_schema(),
            "confidence_after": _confidence_schema(),
            "ambiguity_reasons": code_array,
        },
        "required": [
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
        ],
    }
    properties: dict[str, Any] = {
        "type": {"type": "string", "enum": ["tool", "final"]},
        "tool": _nullable(
            {
                "type": "string",
                "enum": [
                    "camera_context",
                    "camera_sequence",
                    "camera_crop",
                    "lidar_detail",
                ],
            }
        ),
        "rationale": _nullable({"type": "string", "minLength": 1}),
        "arguments": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "bbox_norm": _nullable(
                    {
                        "type": "array",
                        "items": _confidence_schema(),
                        "minItems": 4,
                        "maxItems": 4,
                    }
                ),
                "enhancement": _nullable(
                    {"type": "string", "enum": ["none", "contrast", "sharpen"]}
                ),
            },
            "required": ["bbox_norm", "enhancement"],
        },
        "belief": _nullable(belief),
        "localization": localization,
        "state": _nullable(
            {"type": "string", "enum": ["free", "occupied", "unknown"]}
        ),
        "free_confidence": _nullable(_confidence_schema()),
        "occupied_confidence": _nullable(_confidence_schema()),
        "localization_confidence": _nullable(_confidence_schema()),
        "occupancy_confidence": _nullable(_confidence_schema()),
        "evidence_ids": _nullable({"type": "array", "items": {"type": "string"}}),
        "reason": _nullable({"type": "string", "minLength": 1}),
        "reason_codes": _nullable(code_array),
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


def _normalize_structured_action(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("OpenAI action must be a JSON object")
    action_type = value.get("type")
    if action_type == "tool":
        tool = value.get("tool")
        raw_arguments = value.get("arguments")
        if not isinstance(raw_arguments, Mapping):
            raise ValueError("OpenAI tool action arguments must be an object")
        if tool == "camera_crop":
            arguments = {
                "bbox_norm": raw_arguments.get("bbox_norm"),
                "enhancement": raw_arguments.get("enhancement"),
            }
        else:
            arguments = {}
        action = {
            "type": "tool",
            "tool": tool,
            "rationale": value.get("rationale"),
            "arguments": arguments,
            "belief": value.get("belief"),
        }
        if "localization" in value:
            action["localization"] = value.get("localization")
    elif action_type == "final":
        action = {
            "type": "final",
            "state": value.get("state"),
            "free_confidence": value.get("free_confidence"),
            "occupied_confidence": value.get("occupied_confidence"),
            "localization_confidence": value.get("localization_confidence"),
            "occupancy_confidence": value.get("occupancy_confidence"),
            "evidence_ids": value.get("evidence_ids"),
            "reason": value.get("reason"),
            "reason_codes": value.get("reason_codes"),
        }
        if "localization" in value:
            action["localization"] = value.get("localization")
    else:
        raise ValueError("OpenAI action type must be tool or final")
    parse_action(action)
    return action


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


class OpenAIResponsesAdapter(ModelAdapter):
    """Multimodal adapter with strict output, bounded media, and audit records."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_OPENAI_MODEL,
        reasoning_effort: str = "medium",
        image_detail: str = "original",
        timeout_seconds: float = 120.0,
        max_retries: int = 2,
        max_image_bytes: int = 24 * 1024 * 1024,
        max_output_tokens: int = 2048,
        audit_dir: str | Path | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        api_key_file: str | Path | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be non-empty")
        if reasoning_effort not in _REASONING_EFFORTS:
            raise ValueError("unsupported OpenAI reasoning effort")
        if image_detail not in _DETAIL_LEVELS:
            raise ValueError("unsupported OpenAI image detail")
        if not isinstance(api_key_env, str) or not api_key_env.strip():
            raise ValueError("api_key_env must be non-empty")
        self.model = model.strip()
        self.reasoning_effort = reasoning_effort
        self.image_detail = image_detail
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.max_image_bytes = int(max_image_bytes)
        self.max_output_tokens = int(max_output_tokens)
        self.audit_dir = (
            None if audit_dir is None else Path(audit_dir).resolve(strict=False)
        )
        self.api_key_env = api_key_env.strip()
        self.api_key_file = (
            None if api_key_file is None else Path(api_key_file).resolve(strict=False)
        )
        if self.timeout_seconds <= 0.0 or self.max_image_bytes <= 0:
            raise ValueError("OpenAI timeout and image budget must be positive")
        if not 0 <= self.max_retries <= 5:
            raise ValueError("OpenAI max_retries must be within [0,5]")
        if not 1 <= self.max_output_tokens <= 8192:
            raise ValueError("max_output_tokens must be within [1,8192]")
        self.transcript: dict[str, list[Mapping[str, Any]]] = {}
        self.evidence_bindings: dict[str, list[list[dict[str, Any]]]] = {}
        self.calls: list[dict[str, Any]] = []

    def _load_api_key(self) -> tuple[str, str]:
        value = os.environ.get(self.api_key_env)
        if value:
            return value, f"env:{self.api_key_env}"
        if self.api_key_file is None:
            raise RuntimeError(
                f"{self.api_key_env} is not set and no API key file was configured"
            )
        path = self.api_key_file
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("OpenAI API key file must be a regular non-symlink file")
        if path.stat().st_mode & 0o077:
            raise RuntimeError("OpenAI API key file permissions must be 600 or stricter")
        value = path.read_text(encoding="utf-8").strip()
        if not value:
            raise RuntimeError("OpenAI API key file is empty")
        return value, "file:permission_checked"

    @staticmethod
    def _media_type(path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".png":
            return "image/png"
        if suffix in {".jpg", ".jpeg"}:
            return "image/jpeg"
        if suffix == ".webp":
            return "image/webp"
        raise ValueError(f"unsupported OpenAI image type: {suffix}")

    def _image_content(
        self, image_paths: Sequence[Path]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        content: list[dict[str, Any]] = []
        audit: list[dict[str, Any]] = []
        total = 0
        for index, raw_path in enumerate(image_paths):
            path = Path(raw_path)
            if path.is_symlink() or not path.is_file():
                raise ValueError("OpenAI image inputs must be regular non-symlink files")
            data = path.read_bytes()
            total += len(data)
            if total > self.max_image_bytes:
                raise ValueError("OpenAI image byte budget exceeded")
            media_type = self._media_type(path)
            encoded = base64.b64encode(data).decode("ascii")
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{media_type};base64,{encoded}",
                    "detail": self.image_detail,
                }
            )
            audit.append(
                {
                    "index": index,
                    "path": str(path.resolve()),
                    "bytes": len(data),
                    "sha256": _sha256_bytes(data),
                    "media_type": media_type,
                    "detail": self.image_detail,
                }
            )
        return content, audit

    def next_action(
        self,
        request: Mapping[str, Any],
        *,
        image_paths: Sequence[Path] = (),
    ) -> Mapping[str, Any]:
        api_key, api_key_source = self._load_api_key()
        case_id = request.get("case_id")
        turn = request.get("turn")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("OpenAI request requires case_id")
        if not isinstance(turn, int) or turn < 1:
            raise ValueError("OpenAI request requires a positive turn")

        from openai import OpenAI

        serialized = json.dumps(
            request,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        image_content, image_audit = self._image_content(image_paths)
        content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": "Current single-slot request:\n" + serialized,
            },
            *image_content,
        ]
        started = time.monotonic()
        client = OpenAI(
            api_key=api_key,
            timeout=self.timeout_seconds,
            max_retries=self.max_retries,
        )
        response = None
        provider_attempts = 4
        for provider_attempt in range(provider_attempts):
            try:
                response = client.responses.create(
                    model=self.model,
                    instructions=SYSTEM_PROMPT,
                    input=[{"role": "user", "content": content}],
                    reasoning={"effort": self.reasoning_effort},
                    max_output_tokens=self.max_output_tokens,
                    store=False,
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "parking_slot_agent_action",
                            "strict": True,
                            "schema": openai_action_schema(),
                        }
                    },
                )
                break
            except Exception as exc:
                # The current project occasionally returns a short-lived 403
                # after a multimodal request and accepts the next request a few
                # seconds later. Retry it with bounded backoff, then promote it to a run-level
                # provider failure. A provider outage must never be serialized
                # as semantic parking-slot Unknown.
                if (
                    type(exc).__name__ == "PermissionDeniedError"
                    and provider_attempt + 1 < provider_attempts
                ):
                    time.sleep(5.0 * (provider_attempt + 1))
                    continue
                raise ModelProviderError(
                    f"openai_provider_error:{type(exc).__name__}"
                ) from exc
        if response is None:
            raise ModelProviderError("openai_provider_error:no_response")
        elapsed = time.monotonic() - started
        try:
            raw_action = json.loads(response.output_text)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("OpenAI returned invalid structured output") from exc
        try:
            action = _normalize_structured_action(raw_action)
        except ValueError as exc:
            # Preserve invalid semantic actions in their exact sequence so an
            # offline replay reproduces Agent self-correction and turn limits.
            self.transcript.setdefault(case_id, []).append(
                {
                    "type": "replay_error",
                    "error_type": "ValueError",
                    "message": str(exc),
                }
            )
            self.evidence_bindings.setdefault(case_id, []).append([])
            raise
        self.transcript.setdefault(case_id, []).append(dict(action))
        binding: list[dict[str, Any]] = []
        if action.get("type") == "final":
            cited = action.get("evidence_ids")
            observations = request.get("observations")
            if isinstance(cited, list) and isinstance(observations, list):
                by_id = {
                    row.get("evidence_id"): row
                    for row in observations
                    if isinstance(row, Mapping)
                    and isinstance(row.get("evidence_id"), str)
                }
                for evidence_id in cited:
                    row = by_id.get(evidence_id)
                    if not isinstance(row, Mapping):
                        continue
                    binding.append(
                        {
                            "tool_name": row.get("tool_name"),
                            "round_index": row.get("round_index"),
                            "modality": row.get("modality"),
                        }
                    )
        self.evidence_bindings.setdefault(case_id, []).append(binding)
        usage = getattr(response, "usage", None)
        usage_payload = None
        if usage is not None:
            usage_payload = (
                usage.model_dump(mode="json")
                if hasattr(usage, "model_dump")
                else str(usage)
            )
        call = {
            "schema_version": "parking-slot-agent-v2-openai-call/1.0",
            "case_id": case_id,
            "turn": turn,
            "model": self.model,
            "resolved_model": getattr(response, "model", None),
            "reasoning_effort": self.reasoning_effort,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "response_id": getattr(response, "id", None),
            "elapsed_seconds": round(elapsed, 6),
            "request_sha256": _sha256_bytes(serialized.encode("utf-8")),
            "system_prompt_sha256": _sha256_bytes(SYSTEM_PROMPT.encode("utf-8")),
            "schema_sha256": _sha256_bytes(
                json.dumps(
                    openai_action_schema(), sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ),
            "images": image_audit,
            "usage": usage_payload,
            "action": action,
            "api_key_source": api_key_source,
            "api_key_recorded": False,
        }
        self.calls.append(call)
        if self.audit_dir is not None:
            save_json_atomic(
                self.audit_dir
                / case_id.replace("/", "_")
                / f"turn_{turn:02d}.json",
                call,
            )
        return action

    def replay_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "parking-slot-agent-v2-replay/1.0",
            "actions": self.transcript,
            "evidence_bindings": self.evidence_bindings,
        }


__all__ = [
    "DEFAULT_OPENAI_MODEL",
    "OpenAIResponsesAdapter",
    "openai_action_schema",
]
