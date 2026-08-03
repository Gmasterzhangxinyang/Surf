#!/usr/bin/env python3
"""Validate or deterministically replay the Part2 unknown-agent v1 pipeline."""

from __future__ import annotations

import argparse
from collections import Counter
import ipaddress
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
from urllib.parse import unquote, urlsplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parking_slot_part2.decision import (  # noqa: E402
    consolidate_group_assessments,
    merge_final_route_states,
)
from parking_slot_part2.grouping import TaskGroup, build_groups  # noqa: E402
from parking_slot_part2.model import ReplayModelAdapter  # noqa: E402
from parking_slot_part2.media import EvidenceMediaStore  # noqa: E402
from parking_slot_part2.orchestrator import run_group  # noqa: E402
from parking_slot_part2.preflight import EvidenceCatalog  # noqa: E402
from parking_slot_part2.queueing import (  # noqa: E402
    canonical_json_bytes,
    canonical_sha256,
    load_queue,
)
from parking_slot_part2.reporting import (  # noqa: E402
    derive_part2_run_id,
    load_base_slot_decisions,
    write_run_outputs,
)
from parking_slot_part2.shadow import build_shadow_subset  # noqa: E402
from parking_slot_part2.shadow_replay import compile_shadow_replay  # noqa: E402
from parking_slot_part2.tools import ToolRegistry  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("outputs/slot_part2_agent_v1")
DEFAULT_LOCAL_OUTPUT_DIR = Path("outputs/slot_part2_local_vlm_v1")
DEFAULT_LOCAL_MEDIA_DIR = Path("outputs/slot_part2_local_vlm_media_v1")
BASIC_CONTEXT_SCHEMA_VERSION = "part2-basic-context/1.0"
REPLAY_EQUIVALENCE_SCHEMA_VERSION = "part2-live-replay-equivalence/1.0"


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _load_json(path: str | Path) -> Mapping[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(
            handle,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    if not isinstance(payload, Mapping):
        raise ValueError("JSON input must be an object")
    return payload


def _basic_context(
    group: TaskGroup,
    queue,
    catalog: EvidenceCatalog,
) -> dict[str, Any]:
    task_ids = frozenset(group.task_ids)
    item_by_task = {item.task_id: item for item in queue.items}
    task_context = []
    for task_id in group.task_ids:
        item = item_by_task[task_id]
        preflight = catalog.camera_preflight(task_id)
        task_context.append(
            {
                "task_id": item.task_id,
                "slot_id": item.slot_id,
                "scope_status": item.scope_status,
                "encounter_id": item.encounter.encounter_id,
                "available_modalities": list(item.available_modalities),
                "relationships": {
                    "adjacent_slot_ids": list(item.relationships.adjacent_slot_ids),
                    "conflict_slot_ids": list(item.relationships.conflict_slot_ids),
                    "shared_evidence_ids": list(item.relationships.shared_evidence_ids),
                },
                "camera_preflight": {
                    "status": preflight.status,
                    "capabilities": list(preflight.capabilities),
                    "frame_evidence_ids": list(preflight.frame_evidence_ids),
                    "sequence_evidence_id": preflight.sequence_evidence_id,
                    "rejections": {
                        key: list(value) for key, value in preflight.rejections.items()
                    },
                },
            }
        )
    evidence = []
    for evidence_id, record in sorted(catalog.entries.items()):
        if record.task_id not in task_ids:
            continue
        evidence.append(
            {
                "evidence_id": evidence_id,
                "tool_name": record.tool_name,
                "task_id": record.task_id,
                "slot_id": record.slot_id,
                "encounter_id": record.encounter_id,
                "visual_frame_ids": list(record.visual_frame_ids),
                "capabilities": list(record.capabilities),
                "available": record.available,
                "unavailable_reason": record.unavailable_reason,
            }
        )
    return {
        "schema_version": BASIC_CONTEXT_SCHEMA_VERSION,
        "encounter_id": group.encounter_id,
        "task_context": task_context,
        "evidence_catalog": evidence,
    }


def _validate_summary(queue) -> dict[str, Any]:
    groups = build_groups(queue)
    scopes = Counter(item.scope_status for item in queue.items)
    return {
        "schema_version": "part2-queue-validation-summary/1.0",
        "queue_id": queue.queue_id,
        "item_count": len(queue.items),
        "resource_count": len(queue.resources),
        "group_count": len(groups),
        "scope_counts": {
            scope: scopes.get(scope, 0)
            for scope in ("in_route_scope", "partial_route_scope")
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate-queue",
        help="validate a Part1 unknown-agent queue without writing files",
    )
    validate_parser.add_argument("--queue", required=True, type=Path)

    run_parser = subparsers.add_parser(
        "run",
        help="run deterministic replay, decision gate, merge, and reporting",
    )
    run_parser.add_argument("--queue", required=True, type=Path)
    run_parser.add_argument("--replay-actions", required=True, type=Path)
    run_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    local_parser = subparsers.add_parser(
        "run-local-vlm",
        help="run Part2 against an explicit loopback OpenAI-compatible vision endpoint",
    )
    local_parser.add_argument("--queue", required=True, type=Path)
    local_parser.add_argument("--base-url", required=True)
    local_parser.add_argument("--model", required=True)
    local_parser.add_argument("--output-dir", type=Path, default=DEFAULT_LOCAL_OUTPUT_DIR)
    local_parser.add_argument("--media-dir", type=Path, default=DEFAULT_LOCAL_MEDIA_DIR)
    local_parser.add_argument("--replay-output", type=Path)
    local_parser.add_argument("--timeout-seconds", type=float, default=60.0)
    local_parser.add_argument("--max-response-bytes", type=int, default=1024 * 1024)
    local_parser.add_argument("--max-data-url-bytes", type=int, default=32 * 1024 * 1024)
    local_parser.add_argument("--max-tokens", type=int, default=4096)

    shadow_parser = subparsers.add_parser(
        "prepare-shadow",
        help="build the deterministic 5+20 shadow subset without reading labels or GT",
    )
    shadow_parser.add_argument("--queue", required=True, type=Path)
    shadow_parser.add_argument("--output-dir", required=True, type=Path)

    media_parser = subparsers.add_parser(
        "render-shadow-media",
        help="execute each LiDAR evidence tool once and build private content-addressed media",
    )
    media_parser.add_argument("--queue", required=True, type=Path)
    media_parser.add_argument("--media-dir", required=True, type=Path)
    media_parser.add_argument("--manifest", type=Path)

    compile_parser = subparsers.add_parser(
        "compile-shadow-replay",
        help="compile one strict blind judgement per slot into canonical Replay actions",
    )
    compile_parser.add_argument("--queue", required=True, type=Path)
    compile_parser.add_argument("--blind-records", required=True, type=Path)
    compile_parser.add_argument("--output", required=True, type=Path)
    return parser


def _write_canonical(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(canonical_json_bytes(payload) + b"\n")
    temporary.replace(path)


def _prepare_shadow(queue_path: Path, output_dir: Path) -> dict[str, Any]:
    queue = load_queue(queue_path)
    subset = build_shadow_subset(queue, source_base_dir=queue_path.parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    allowed = {"unknown_agent_queue.json", "selection_manifest.json"}
    unsupported = sorted(path.name for path in output_dir.iterdir() if path.name not in allowed)
    if unsupported:
        raise ValueError(
            "shadow output directory contains unsupported entries: " + ", ".join(unsupported)
        )
    _write_canonical(output_dir / "unknown_agent_queue.json", subset.queue_payload)
    _write_canonical(output_dir / "selection_manifest.json", subset.selection_manifest)
    counts = subset.selection_manifest["counts"]
    return {
        "schema_version": "part2-shadow-prepare-result/1.0",
        "source_queue_id": queue.queue_id,
        "subset_queue_id": subset.queue.queue_id,
        "seed_task_count": counts["seed_tasks"],
        "selected_task_count": counts["selected_tasks"],
        "closure_added_task_count": counts["closure_added_tasks"],
        "output_dir": str(output_dir),
    }


def _render_shadow_media(
    queue_path: Path,
    media_dir: Path,
    manifest_path: Path | None,
) -> dict[str, Any]:
    queue = load_queue(queue_path)
    catalog = EvidenceCatalog(queue, base_dir=queue_path.parent)
    store = EvidenceMediaStore(media_dir)
    registry = ToolRegistry(catalog, media_store=store)
    attempts: list[dict[str, Any]] = []
    for evidence_id, record in sorted(catalog.entries.items()):
        if record.tool_name != "inspect_lidar_map":
            continue
        result = registry.execute(
            "inspect_lidar_map",
            {"evidence_id": evidence_id},
        )
        attempts.append(
            {
                "evidence_id": evidence_id,
                "task_id": record.task_id,
                "slot_id": record.slot_id,
                "status": result.status,
                "error_code": result.error_code,
                "media_published": store.resolve(evidence_id) is not None,
            }
        )
    media_records: list[dict[str, Any]] = []
    for record in store.manifest_records():
        evidence = catalog.get(record["evidence_id"])
        media_records.append(
            {
                **record,
                "task_id": evidence.task_id,
                "slot_id": evidence.slot_id,
                "encounter_id": evidence.encounter_id,
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": "part2-shadow-media-manifest/1.0",
        "queue_id": queue.queue_id,
        "attempts": attempts,
        "media": media_records,
    }
    manifest["manifest_id"] = canonical_sha256(manifest)
    destination = manifest_path or media_dir / "generation_manifest.json"
    _write_canonical(destination, manifest)
    return {
        "schema_version": "part2-shadow-media-result/1.0",
        "queue_id": queue.queue_id,
        "lidar_attempt_count": len(attempts),
        "media_count": len(media_records),
        "failed_count": sum(row["status"] != "ok" for row in attempts),
        "manifest": str(destination),
    }


def _compile_shadow_replay(
    queue_path: Path,
    blind_records_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    queue = load_queue(queue_path)
    blind_records = _load_json(blind_records_path)
    replay = compile_shadow_replay(
        queue,
        blind_records,
        base_dir=queue_path.parent,
    )
    _write_canonical(output_path, replay)
    action_count = sum(len(actions) for actions in replay["actions"].values())
    tool_count = sum(
        action.get("type") == "tool_request"
        for actions in replay["actions"].values()
        for action in actions
    )
    return {
        "schema_version": "part2-shadow-replay-compile-result/1.0",
        "queue_id": queue.queue_id,
        "group_count": len(replay["actions"]),
        "action_count": action_count,
        "tool_request_count": tool_count,
        "output": str(output_path),
    }


def _run(queue_path: Path, replay_path: Path, output_dir: Path) -> dict[str, Any]:
    queue = load_queue(queue_path)
    replay_payload = _load_json(replay_path)
    replay = ReplayModelAdapter(replay_payload)
    groups = build_groups(queue)
    catalog = EvidenceCatalog(queue, base_dir=queue_path.parent)
    registry = ToolRegistry(catalog)
    group_results = tuple(
        run_group(
            group,
            queue,
            replay,
            registry,
            basic_context=_basic_context(group, queue, catalog),
        )
        for group in groups
    )
    resolutions = consolidate_group_assessments(queue, group_results, catalog)
    part2_run_id = derive_part2_run_id(queue, replay_payload)

    # Keep this load/rehash directly adjacent to the merge boundary.
    base_payload = load_base_slot_decisions(queue, base_dir=queue_path.parent)
    final_route_states = merge_final_route_states(
        base_payload,
        queue,
        resolutions,
        part2_run_id=part2_run_id,
    )
    replay_identity = canonical_sha256(replay_payload)
    write_run_outputs(
        output_dir,
        part2_run_id=part2_run_id,
        queue=queue,
        catalog=catalog,
        base_dir=queue_path.parent,
        replay_identity=replay_identity,
        group_results=group_results,
        resolutions=resolutions,
        final_route_states=final_route_states,
    )
    states = Counter(row.state for row in resolutions)
    return {
        "schema_version": "part2-run-result/1.0",
        "part2_run_id": part2_run_id,
        "queue_id": queue.queue_id,
        "resolution_count": len(resolutions),
        "state_counts": {
            state: states.get(state, 0) for state in ("occupied", "free", "unknown")
        },
    }


def _require_loopback_base_url(base_url: str) -> None:
    """Keep the credential-free CLI boundary on an explicit local endpoint."""

    try:
        parsed = urlsplit(base_url)
        hostname = parsed.hostname
    except ValueError as exc:
        raise ValueError("base_url must be a valid loopback HTTP(S) URL") from exc
    if hostname is None:
        raise ValueError("base_url must be a valid loopback HTTP(S) URL")
    normalized = hostname.rstrip(".").lower()
    is_loopback = normalized == "localhost"
    if not is_loopback:
        try:
            is_loopback = ipaddress.ip_address(normalized).is_loopback
        except ValueError:
            is_loopback = False
    if not is_loopback:
        raise ValueError("run-local-vlm only accepts localhost or a loopback IP address")


def _path_is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def _local_resource_path(uri: str, base_dir: Path) -> Path | None:
    parsed = urlsplit(uri)
    if parsed.scheme not in {"", "file"}:
        return None
    if parsed.scheme == "file" and parsed.netloc not in {"", "localhost"}:
        return None
    raw_path = unquote(parsed.path)
    path = Path(raw_path)
    return path if path.is_absolute() else base_dir / path


def _group_results_equivalence_payload(group_results) -> dict[str, Any]:
    """Canonicalize every decision-bearing live/replay group outcome."""

    return {
        "schema_version": REPLAY_EQUIVALENCE_SCHEMA_VERSION,
        "groups": [
            {
                "group_id": result.group_id,
                "proposal": result.proposal.to_action(),
                "model_turns": result.model_turns,
                "tool_attempts": result.tool_attempts,
                "stop_reason": result.stop_reason,
                "trace_span_id": result.trace_span_id,
                "trace_events": list(result.trace_events),
                "tool_attempt_records": [
                    attempt.to_dict() for attempt in result.tool_attempt_records
                ],
                "tool_results": [row.to_dict() for row in result.tool_results],
            }
            for result in sorted(group_results, key=lambda row: row.group_id)
        ],
    }


def _run_local_vlm(
    queue_path: Path,
    output_dir: Path,
    media_dir: Path,
    *,
    base_url: str,
    model: str,
    replay_output: Path | None,
    timeout_seconds: float,
    max_response_bytes: int,
    max_data_url_bytes: int,
    max_tokens: int,
) -> dict[str, Any]:
    # Keep the optional HTTP client out of replay/validation-only commands.
    # Importing here also makes the dependency boundary explicit to callers.
    from parking_slot_part2.local_vlm import LocalImage, LocalVLMAdapter

    _require_loopback_base_url(base_url)
    if _path_is_within(output_dir, media_dir) or _path_is_within(media_dir, output_dir):
        raise ValueError("media_dir and output_dir must be disjoint directory trees")
    if _path_is_within(queue_path, output_dir) or _path_is_within(queue_path, media_dir):
        raise ValueError("queue input must be outside output_dir and media_dir")
    if replay_output is not None:
        if _path_is_within(replay_output, output_dir) or _path_is_within(
            replay_output, media_dir
        ):
            raise ValueError("replay_output must be outside output_dir and media_dir")
        if replay_output.resolve() == queue_path.resolve():
            raise ValueError("replay_output must not overwrite the queue input")

    queue = load_queue(queue_path)
    if replay_output is not None:
        replay_path = replay_output.resolve()
        conflicting_resources = sorted(
            resource_id
            for resource_id, resource in queue.resources.items()
            if (
                (path := _local_resource_path(resource.uri, queue_path.parent))
                is not None
                and path.resolve() == replay_path
            )
        )
        if conflicting_resources:
            raise ValueError(
                "replay_output must not overwrite queue resource(s): "
                + ", ".join(conflicting_resources)
            )
    groups = build_groups(queue)
    catalog = EvidenceCatalog(queue, base_dir=queue_path.parent)
    store = EvidenceMediaStore(media_dir)
    registry = ToolRegistry(catalog, media_store=store)

    def resolve_media(evidence_id: str) -> LocalImage | None:
        resolved = store.read_image(evidence_id)
        if resolved is None:
            return None
        media_type, content = resolved
        return LocalImage(media_type, content)

    group_results_list = []
    with LocalVLMAdapter(
        base_url=base_url,
        model=model,
        media_resolver=resolve_media,
        timeout_seconds=timeout_seconds,
        max_response_bytes=max_response_bytes,
        max_data_url_bytes=max_data_url_bytes,
        max_tokens=max_tokens,
    ) as adapter:
        for group in groups:
            result = run_group(
                group,
                queue,
                adapter,
                registry,
                basic_context=_basic_context(group, queue, catalog),
            )
            group_results_list.append(result)
            if result.stop_reason == "model_error":
                raise RuntimeError(
                    "local model failed before producing a replayable action for group: "
                    + result.group_id
                )
        replay_payload = adapter.replay_payload()
    group_results = tuple(group_results_list)

    # The saved transcript is the canonical run identity.  A successful live
    # run and its deterministic replay must therefore produce the same run ID.
    replay_adapter = ReplayModelAdapter(replay_payload)
    replay_registry = ToolRegistry(catalog)
    replayed_group_results = tuple(
        run_group(
            group,
            queue,
            replay_adapter,
            replay_registry,
            basic_context=_basic_context(group, queue, catalog),
        )
        for group in groups
    )
    if canonical_json_bytes(_group_results_equivalence_payload(group_results)) != canonical_json_bytes(
        _group_results_equivalence_payload(replayed_group_results)
    ):
        raise RuntimeError(
            "local model transcript is not exactly reproducible by deterministic replay"
        )
    group_results = replayed_group_results
    replay_identity = canonical_sha256(replay_payload)
    part2_run_id = derive_part2_run_id(queue, replay_payload)
    resolutions = consolidate_group_assessments(queue, group_results, catalog)

    # Keep this load/rehash directly adjacent to the merge boundary.
    base_payload = load_base_slot_decisions(queue, base_dir=queue_path.parent)
    final_route_states = merge_final_route_states(
        base_payload,
        queue,
        resolutions,
        part2_run_id=part2_run_id,
    )
    write_run_outputs(
        output_dir,
        part2_run_id=part2_run_id,
        queue=queue,
        catalog=catalog,
        base_dir=queue_path.parent,
        replay_identity=replay_identity,
        group_results=group_results,
        resolutions=resolutions,
        final_route_states=final_route_states,
    )
    media_records: list[dict[str, Any]] = []
    for record in store.manifest_records():
        evidence = catalog.get(record["evidence_id"])
        media_records.append(
            {
                **record,
                "task_id": evidence.task_id,
                "slot_id": evidence.slot_id,
                "encounter_id": evidence.encounter_id,
            }
        )
    media_manifest: dict[str, Any] = {
        "schema_version": "part2-local-vlm-media-manifest/1.0",
        "part2_run_id": part2_run_id,
        "queue_id": queue.queue_id,
        "media": media_records,
    }
    media_manifest["manifest_id"] = canonical_sha256(media_manifest)
    media_manifest_path = media_dir / "generation_manifest.json"
    _write_canonical(media_manifest_path, media_manifest)
    if replay_output is not None:
        _write_canonical(replay_output, replay_payload)

    states = Counter(row.state for row in resolutions)
    return {
        "schema_version": "part2-local-vlm-run-result/1.0",
        "part2_run_id": part2_run_id,
        "queue_id": queue.queue_id,
        "resolution_count": len(resolutions),
        "state_counts": {
            state: states.get(state, 0) for state in ("occupied", "free", "unknown")
        },
        "model_turn_count": sum(result.model_turns for result in group_results),
        "tool_attempt_count": sum(result.tool_attempt_count for result in group_results),
        "media_count": len(store.manifest_records()),
        "media_manifest": str(media_manifest_path),
        "replay_output": None if replay_output is None else str(replay_output),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "validate-queue":
            payload = _validate_summary(load_queue(arguments.queue))
        elif arguments.command == "prepare-shadow":
            payload = _prepare_shadow(arguments.queue, arguments.output_dir)
        elif arguments.command == "render-shadow-media":
            payload = _render_shadow_media(
                arguments.queue,
                arguments.media_dir,
                arguments.manifest,
            )
        elif arguments.command == "compile-shadow-replay":
            payload = _compile_shadow_replay(
                arguments.queue,
                arguments.blind_records,
                arguments.output,
            )
        elif arguments.command == "run-local-vlm":
            payload = _run_local_vlm(
                arguments.queue,
                arguments.output_dir,
                arguments.media_dir,
                base_url=arguments.base_url,
                model=arguments.model,
                replay_output=arguments.replay_output,
                timeout_seconds=arguments.timeout_seconds,
                max_response_bytes=arguments.max_response_bytes,
                max_data_url_bytes=arguments.max_data_url_bytes,
                max_tokens=arguments.max_tokens,
            )
        else:
            payload = _run(
                arguments.queue,
                arguments.replay_actions,
                arguments.output_dir,
            )
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    sys.stdout.buffer.write(canonical_json_bytes(payload) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
