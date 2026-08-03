"""Bounded OpenAI-compatible HTTP adapter for a caller-managed local VLM.

The adapter deliberately owns no model discovery, API credentials, media
paths, or provider transcript.  A caller supplies an explicit ``base_url`` and
``model`` plus an optional path-private media resolver.  Only strict Part2 JSON
actions cross back into the orchestrator.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import ipaddress
import json
import math
import re
import threading
from typing import Any, Mapping, Protocol
from urllib.parse import urlsplit

import httpx

from .model import (
    FINAL_STATES,
    MODEL_TURN_SCHEMA_VERSION,
    REPLAY_SCHEMA_VERSION,
)
from .queueing import canonical_json_bytes


DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_RESPONSE_BYTES = 1024 * 1024
DEFAULT_MAX_DATA_URL_BYTES = 32 * 1024 * 1024
DEFAULT_MAX_TOKENS = 4096
LOCAL_VLM_ADAPTER_VERSION = "part2-local-openai-compatible-vlm/1.0"
SUPPORTED_IMAGE_MEDIA_TYPES = frozenset(
    {"image/jpeg", "image/png", "image/webp"}
)
VISUAL_TOOL_NAMES = frozenset(
    {"inspect_lidar_map", "inspect_rgb_frame", "inspect_rgb_sequence"}
)

_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "group_id",
        "turn",
        "decision_tasks",
        "basic_context",
        "observations",
        "budget",
    }
)
_TASK_FIELDS = frozenset(
    {
        "task_id",
        "slot_id",
        "unknown_reasons",
        "allowed_final_states",
        "occupied_evidence",
        "free_evidence",
    }
)
_FORBIDDEN_ACTION_KEYS = frozenset(
    {
        "apikey",
        "accesstoken",
        "authorization",
        "baseurl",
        "bearertoken",
        "completion",
        "endpoint",
        "message",
        "messages",
        "model",
        "modelid",
        "modelname",
        "prompt",
        "provider",
        "providerid",
        "providername",
        "response",
        "responseid",
        "secret",
        "token",
    }
)

SYSTEM_PROMPT = """You are the bounded ParkingAgent Part2 decision model.
Return exactly one JSON object and no prose, markdown, code fence, or provider metadata.

The only allowed actions are:
1. {"type":"tool_request","tool_name":"inspect_lidar_map|inspect_rgb_frame|inspect_rgb_sequence","arguments":{"evidence_id":"ev_<64 lowercase hex>"}}
2. {"type":"final_proposal","assessments":[...]}

Every final assessment must contain exactly these fields:
task_id, slot_id, proposed_state, target_visibility, target_ownership,
semantic_finding, resolved_unknown_reasons, unresolved_blockers,
evidence_refs, reason_codes.

Assess every decision task exactly once. Use only task IDs, slot IDs, allowed final
states, and opaque evidence IDs present in the request. Never invent evidence.
If evidence is insufficient, return unknown with explicit blockers. The runtime
will strictly validate the returned object and will reject unsupported fields.

Rendered RGB evidence uses a bright magenta polygon/fill and white vertices for
the TARGET slot; yellow outlines are adjacent slots. Judge occupancy only inside
the labeled target and report uncertain ownership when an object crosses a
boundary. An RGB sequence is temporal context only and cannot by itself justify
a terminal occupied/free state. LiDAR evidence is a labeled diagnostic
triptych, not an RGB photograph."""


class LocalVLMError(RuntimeError):
    """Base error for the local HTTP boundary."""


class LocalVLMRequestError(LocalVLMError):
    """The provider-neutral Part2 turn request is malformed."""


class LocalVLMHTTPError(LocalVLMError):
    """The local endpoint could not return a successful bounded response."""


class LocalVLMTimeoutError(LocalVLMHTTPError):
    """The configured local endpoint deadline expired."""


class LocalVLMResponseTooLarge(LocalVLMHTTPError):
    """The local endpoint response exceeded its byte budget."""


class LocalVLMMediaError(LocalVLMError):
    """Caller-provided visual media is unavailable or exceeds its byte budget."""


class LocalVLMProtocolError(LocalVLMError):
    """The endpoint response is not an OpenAI-compatible JSON completion."""


@dataclass(frozen=True, slots=True)
class LocalImage:
    """Path-free image bytes returned by the caller's private media resolver."""

    media_type: str
    content: bytes

    def __post_init__(self) -> None:
        if self.media_type not in SUPPORTED_IMAGE_MEDIA_TYPES:
            raise ValueError("LocalImage media_type is not supported")
        if not isinstance(self.content, bytes) or not self.content:
            raise ValueError("LocalImage content must be non-empty bytes")


class MediaResolver(Protocol):
    """Resolve one successful visual evidence ID without exposing a local path."""

    def __call__(self, evidence_id: str) -> LocalImage | None: ...


def _strict_json_loads(content: bytes | str, *, subject: str) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number is not allowed: {value}")

    try:
        if isinstance(content, bytes):
            content = content.decode("utf-8")
        return json.loads(
            content,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise LocalVLMProtocolError(f"{subject} is not strict UTF-8 JSON") from exc


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _completion_url(base_url: str) -> str:
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("base_url must be a non-empty HTTP URL")
    value = base_url.strip().rstrip("/")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("base_url must be an HTTP(S) URL without credentials or query")
    try:
        hostname = parsed.hostname
    except ValueError as exc:
        raise ValueError("base_url must use a valid loopback host") from exc
    if hostname is None:
        raise ValueError("base_url must use a valid loopback host")
    normalized = hostname.rstrip(".").lower()
    is_loopback = normalized == "localhost"
    if not is_loopback:
        try:
            is_loopback = ipaddress.ip_address(normalized).is_loopback
        except ValueError:
            is_loopback = False
    if not is_loopback:
        raise ValueError("LocalVLMAdapter only accepts localhost or a loopback IP address")
    return value + "/chat/completions"


def _string_array(value: Any, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise LocalVLMRequestError(f"{name} must be a JSON string array")
    if len(set(value)) != len(value):
        raise LocalVLMRequestError(f"{name} must not contain duplicates")
    return tuple(value)


def _validate_request(
    request: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    if not isinstance(request, Mapping):
        raise LocalVLMRequestError("model turn request must be a JSON object")
    try:
        normalized = _strict_json_loads(
            canonical_json_bytes(request), subject="model turn request"
        )
    except (TypeError, ValueError) as exc:
        raise LocalVLMRequestError("model turn request must contain finite JSON") from exc
    if not isinstance(normalized, dict) or set(normalized) != _REQUEST_FIELDS:
        raise LocalVLMRequestError("model turn request fields do not match Part2 v1")
    if normalized["schema_version"] != MODEL_TURN_SCHEMA_VERSION:
        raise LocalVLMRequestError("model turn request schema_version is invalid")
    group_id = normalized["group_id"]
    if not isinstance(group_id, str) or not group_id:
        raise LocalVLMRequestError("model turn request requires a non-empty group_id")
    turn = normalized["turn"]
    if isinstance(turn, bool) or not isinstance(turn, int) or turn <= 0:
        raise LocalVLMRequestError("model turn request turn must be positive")
    if not isinstance(normalized["basic_context"], dict):
        raise LocalVLMRequestError("basic_context must be a JSON object")
    if not isinstance(normalized["budget"], dict):
        raise LocalVLMRequestError("budget must be a JSON object")
    observations = normalized["observations"]
    if not isinstance(observations, list) or any(
        not isinstance(observation, dict) for observation in observations
    ):
        raise LocalVLMRequestError("observations must be an array of JSON objects")

    tasks = normalized["decision_tasks"]
    if not isinstance(tasks, list) or not tasks:
        raise LocalVLMRequestError("decision_tasks must be a non-empty array")
    task_ids: set[str] = set()
    for index, task in enumerate(tasks):
        if not isinstance(task, dict) or set(task) != _TASK_FIELDS:
            raise LocalVLMRequestError(
                f"decision_tasks[{index}] fields do not match Part2 v1"
            )
        task_id = task["task_id"]
        slot_id = task["slot_id"]
        if not isinstance(task_id, str) or not task_id:
            raise LocalVLMRequestError(f"decision_tasks[{index}] task_id is invalid")
        if task_id in task_ids:
            raise LocalVLMRequestError("decision task IDs must be unique")
        task_ids.add(task_id)
        if not isinstance(slot_id, str) or not slot_id:
            raise LocalVLMRequestError(f"decision_tasks[{index}] slot_id is invalid")
        _string_array(task["unknown_reasons"], name="unknown_reasons")
        allowed = _string_array(
            task["allowed_final_states"], name="allowed_final_states"
        )
        if not allowed or any(state not in FINAL_STATES for state in allowed):
            raise LocalVLMRequestError("allowed_final_states contains an invalid state")
        if not isinstance(task["occupied_evidence"], dict) or not isinstance(
            task["free_evidence"], dict
        ):
            raise LocalVLMRequestError("task evidence summaries must be JSON objects")
    return normalized, group_id


def _compact_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _reject_action_leakage(value: Any, location: str = "action") -> None:
    """Reject provider transcript, credentials, and local-location fields.

    Schema and enum mistakes intentionally remain untouched: the orchestrator's
    existing validator must observe them and issue repair feedback on the next
    bounded turn.
    """

    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise LocalVLMProtocolError(f"{location} keys must be strings")
            compact = _compact_key(key)
            forbidden = (
                compact in _FORBIDDEN_ACTION_KEYS
                or "provider" in compact
                or compact.startswith(("model", "prompt", "message", "completion"))
                or "path" in compact
                or compact.endswith(("uri", "url", "file", "filename"))
            )
            if forbidden:
                raise LocalVLMProtocolError(
                    f"completion action contains forbidden field {key!r}"
                )
            _reject_action_leakage(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_action_leakage(child, f"{location}[{index}]")


def _extract_completion_content(response_payload: Any) -> str:
    if not isinstance(response_payload, dict):
        raise LocalVLMProtocolError("completion response must be a JSON object")
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LocalVLMProtocolError("completion response requires a non-empty choices array")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise LocalVLMProtocolError("completion choice must be a JSON object")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise LocalVLMProtocolError("completion choice requires a message object")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise LocalVLMProtocolError("completion message content must be a JSON string")
    return content


class LocalOpenAICompatibleVLMAdapter:
    """Call an explicitly configured OpenAI-compatible local chat endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        media_resolver: MediaResolver | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        max_data_url_bytes: int = DEFAULT_MAX_DATA_URL_BYTES,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if media_resolver is not None and not callable(media_resolver):
            raise TypeError("media_resolver must be callable")
        self.endpoint = _completion_url(base_url)
        self.model = model.strip()
        self.media_resolver = media_resolver
        self.timeout_seconds = _positive_number(timeout_seconds, "timeout_seconds")
        self.max_response_bytes = _positive_int(
            max_response_bytes, "max_response_bytes"
        )
        self.max_data_url_bytes = _positive_int(
            max_data_url_bytes, "max_data_url_bytes"
        )
        self.max_tokens = _positive_int(max_tokens, "max_tokens")
        self._client = httpx.Client(
            transport=transport,
            # This adapter is intentionally a local-only boundary.  Ignoring
            # HTTP(S)_PROXY/ALL_PROXY prevents image data URLs from being
            # forwarded to a process selected by the caller's environment.
            trust_env=False,
            timeout=httpx.Timeout(self.timeout_seconds),
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )
        self._transcript: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "LocalOpenAICompatibleVLMAdapter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _image_parts(self, observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.media_resolver is None:
            return []
        parts: list[dict[str, Any]] = []
        seen: set[str] = set()
        total_data_url_bytes = 0
        for observation in observations:
            if (
                observation.get("kind") != "tool_result"
                or observation.get("status") != "ok"
                or observation.get("tool_name") not in VISUAL_TOOL_NAMES
            ):
                continue
            evidence_id = observation.get("evidence_id")
            if not isinstance(evidence_id, str) or not evidence_id or evidence_id in seen:
                continue
            seen.add(evidence_id)
            try:
                media = self.media_resolver(evidence_id)
            except Exception:
                raise LocalVLMMediaError(
                    f"media resolver failed for evidence {evidence_id}"
                ) from None
            if media is None:
                raise LocalVLMMediaError(
                    f"successful visual evidence has no published media for {evidence_id}"
                )
            if not isinstance(media, LocalImage):
                raise LocalVLMMediaError(
                    f"media resolver returned an invalid image for evidence {evidence_id}"
                )
            prefix = f"data:{media.media_type};base64,"
            encoded_size = 4 * ((len(media.content) + 2) // 3)
            data_url_size = len(prefix) + encoded_size
            if data_url_size > self.max_data_url_bytes or (
                total_data_url_bytes + data_url_size > self.max_data_url_bytes
            ):
                raise LocalVLMMediaError("resolved image data exceeds the data URL budget")
            encoded = base64.b64encode(media.content).decode("ascii")
            data_url = prefix + encoded
            total_data_url_bytes += data_url_size
            parts.append(
                {
                    "type": "text",
                    "text": f"Successful visual tool evidence: {evidence_id}",
                }
            )
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                }
            )
        return parts

    def _read_response(self, request_payload: Mapping[str, Any]) -> bytes:
        try:
            with self._client.stream(
                "POST",
                self.endpoint,
                json=request_payload,
                timeout=self.timeout_seconds,
            ) as response:
                content_length = response.headers.get("content-length")
                if content_length is not None:
                    try:
                        if int(content_length) > self.max_response_bytes:
                            raise LocalVLMResponseTooLarge(
                                "local completion response exceeds the byte budget"
                            )
                    except ValueError:
                        pass
                if response.status_code < 200 or response.status_code >= 300:
                    raise LocalVLMHTTPError(
                        f"local completion endpoint returned HTTP {response.status_code}"
                    )
                chunks: list[bytes] = []
                size = 0
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > self.max_response_bytes:
                        raise LocalVLMResponseTooLarge(
                            "local completion response exceeds the byte budget"
                        )
                    chunks.append(chunk)
                return b"".join(chunks)
        except LocalVLMError:
            raise
        except httpx.TimeoutException:
            raise LocalVLMTimeoutError("local completion endpoint timed out") from None
        except httpx.HTTPError as exc:
            raise LocalVLMHTTPError(
                f"local completion request failed with {type(exc).__name__}"
            ) from None

    def next_action(self, request: Mapping[str, Any]) -> Any:
        normalized, group_id = _validate_request(request)
        canonical_request = canonical_json_bytes(normalized).decode("utf-8")
        user_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Part2 model-turn request (canonical JSON):\n" + canonical_request
                ),
            }
        ]
        user_content.extend(self._image_parts(normalized["observations"]))
        wire_request = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        }
        response_bytes = self._read_response(wire_request)
        response_payload = _strict_json_loads(
            response_bytes, subject="completion response"
        )
        content = _extract_completion_content(response_payload)
        action = _strict_json_loads(content, subject="completion action")
        if not isinstance(action, dict):
            raise LocalVLMProtocolError("completion action must be one JSON object")
        _reject_action_leakage(action)
        # The transcript contains only validated Part2 actions.  It never
        # retains provider messages, request prompts, data URLs, or paths.
        plain_action = _strict_json_loads(
            canonical_json_bytes(action), subject="completion action"
        )
        with self._lock:
            self._transcript.setdefault(group_id, []).append(plain_action)
        return plain_action

    def replay_payload(self) -> dict[str, Any]:
        """Return a path/provider-free replay envelope for deterministic reports."""

        with self._lock:
            actions = {
                group_id: [
                    _strict_json_loads(
                        canonical_json_bytes(action), subject="recorded action"
                    )
                    for action in group_actions
                ]
                for group_id, group_actions in sorted(self._transcript.items())
            }
        return {"schema_version": REPLAY_SCHEMA_VERSION, "actions": actions}


# Short public name for CLI integrations; the descriptive class name remains
# available for callers that want to make the wire protocol explicit.
LocalVLMAdapter = LocalOpenAICompatibleVLMAdapter


__all__ = [
    "DEFAULT_MAX_DATA_URL_BYTES",
    "DEFAULT_MAX_RESPONSE_BYTES",
    "DEFAULT_TIMEOUT_SECONDS",
    "LOCAL_VLM_ADAPTER_VERSION",
    "LocalImage",
    "LocalOpenAICompatibleVLMAdapter",
    "LocalVLMAdapter",
    "LocalVLMError",
    "LocalVLMHTTPError",
    "LocalVLMMediaError",
    "LocalVLMProtocolError",
    "LocalVLMRequestError",
    "LocalVLMResponseTooLarge",
    "LocalVLMTimeoutError",
    "MediaResolver",
]
