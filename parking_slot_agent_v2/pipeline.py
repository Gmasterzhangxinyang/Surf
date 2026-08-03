"""Free-first linear candidate scheduling with trustworthy-Free early stop."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from .agent import SingleSlotAgent, SlotRunResult, TERMINAL_CONFIDENCE
from .contracts import CaseStatus, Part1Output, SlotCase, SlotState


@dataclass(frozen=True, slots=True)
class Part2RunResult:
    snapshot_id: str
    queue_order: tuple[str, ...]
    processed_case_ids: tuple[str, ...]
    selected_slot_id: str | None
    stop_reason: str
    slot_results: tuple[SlotRunResult, ...]
    slot_cases: tuple[SlotCase, ...]
    evaluation_exhaustive: bool = False

    @property
    def found_trustworthy_free(self) -> bool:
        return self.selected_slot_id is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "parking-slot-agent-v2-part2-result/1.0",
            "snapshot_id": self.snapshot_id,
            "queue_order": list(self.queue_order),
            "processed_case_ids": list(self.processed_case_ids),
            "selected_slot_id": self.selected_slot_id,
            "found_trustworthy_free": self.found_trustworthy_free,
            "stop_reason": self.stop_reason,
            "evaluation_exhaustive": self.evaluation_exhaustive,
            "terminal_free_slot_ids": [
                result.case.slot_id
                for result in self.slot_results
                if result.case.final_state is SlotState.FREE
            ],
            "slot_results": [result.to_dict() for result in self.slot_results],
            "slot_cases": [case.to_dict() for case in self.slot_cases],
        }


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            temporary = Path(temporary_name)
            if temporary.exists():
                temporary.unlink()


def _checkpoint(
    output_dir: Path,
    part1: Part1Output,
    results: list[SlotRunResult],
    selected_slot_id: str | None,
    stop_reason: str,
    evaluation_exhaustive: bool = False,
) -> None:
    processed = tuple(result.case.case_id for result in results)
    queue_order = tuple(case.case_id for case in part1.slot_cases)
    run = Part2RunResult(
        snapshot_id=part1.scene.snapshot_id,
        queue_order=queue_order,
        processed_case_ids=processed,
        selected_slot_id=selected_slot_id,
        stop_reason=stop_reason,
        slot_results=tuple(results),
        slot_cases=tuple(part1.slot_cases),
        evaluation_exhaustive=evaluation_exhaustive,
    )
    _write_json_atomic(output_dir / "part2_result.json", run.to_dict())
    for result in results:
        _write_json_atomic(
            output_dir / "slot_cases" / f"{result.case.slot_id}.json",
            result.case.to_dict(),
        )


def run_candidate_queue(
    part1: Part1Output,
    agent: SingleSlotAgent,
    output_dir: str | Path,
    *,
    stop_at_first_free: bool = True,
    resume: bool = False,
) -> Part2RunResult:
    """Process one SlotCase at a time and stop at the first >=0.50 Free."""

    destination = Path(output_dir).resolve(strict=False)
    destination.mkdir(parents=True, exist_ok=True)
    queue_order = tuple(case.case_id for case in part1.slot_cases)
    results: list[SlotRunResult] = []
    if resume:
        checkpoint_path = destination / "part2_result.json"
        if not checkpoint_path.is_file():
            raise ValueError("resume requested but no Part2 checkpoint exists")
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("snapshot_id") != part1.scene.snapshot_id:
            raise ValueError("Part2 checkpoint snapshot does not match input")
        if tuple(checkpoint.get("queue_order", ())) != queue_order:
            raise ValueError("Part2 checkpoint queue order does not match input")
        if bool(checkpoint.get("evaluation_exhaustive", False)) != (
            not stop_at_first_free
        ):
            raise ValueError("Part2 checkpoint mode does not match requested mode")
        resumed_by_id: dict[str, SlotCase] = {}
        for row in checkpoint.get("slot_results", ()):
            case = SlotCase.from_dict(row["case"])
            if not case.terminal:
                raise ValueError("Part2 checkpoint contains a non-terminal processed case")
            resumed_by_id[case.case_id] = case
            results.append(
                SlotRunResult(
                    case=case,
                    stop_reason=str(row["stop_reason"]),
                    model_turns=int(row["model_turns"]),
                    tool_rounds=int(row["tool_rounds"]),
                    validation_errors=tuple(
                        str(item) for item in row.get("validation_errors", ())
                    ),
                )
            )
        processed_ids = tuple(checkpoint.get("processed_case_ids", ()))
        if processed_ids != tuple(result.case.case_id for result in results):
            raise ValueError("Part2 checkpoint processed IDs are inconsistent")
        original_by_id = {case.case_id: case for case in part1.slot_cases}
        if not set(resumed_by_id).issubset(original_by_id):
            raise ValueError("Part2 checkpoint contains cases outside the input queue")
        part1.slot_cases = [
            resumed_by_id.get(case.case_id, case) for case in part1.slot_cases
        ]
        if checkpoint.get("stop_reason") != "in_progress":
            return Part2RunResult(
                snapshot_id=part1.scene.snapshot_id,
                queue_order=queue_order,
                processed_case_ids=processed_ids,
                selected_slot_id=checkpoint.get("selected_slot_id"),
                stop_reason=str(checkpoint["stop_reason"]),
                slot_results=tuple(results),
                slot_cases=tuple(part1.slot_cases),
                evaluation_exhaustive=not stop_at_first_free,
            )
    selected_slot_id: str | None = None
    stop_reason = (
        "candidate_queue_exhausted"
        if stop_at_first_free
        else "evaluation_queue_exhausted"
    )

    resumed_ids = {item.case.case_id for item in results}
    for case in part1.slot_cases:
        if case.case_id in resumed_ids:
            continue
        if case.terminal:
            raise ValueError(
                f"candidate queue requires fresh non-terminal SlotCase: {case.case_id}"
            )
        result = agent.run(
            part1.scene,
            case,
            destination / "media" / case.slot_id,
        )
        results.append(result)
        if (
            stop_at_first_free
            and
            case.status is CaseStatus.RESOLVED
            and case.final_state is SlotState.FREE
            and case.final_scores is not None
            and case.final_scores.free_confidence >= TERMINAL_CONFIDENCE
            and case.final_scores.occupied_confidence < TERMINAL_CONFIDENCE
        ):
            case.mark_selected()
            selected_slot_id = case.slot_id
            stop_reason = "trustworthy_free_found"
            _checkpoint(
                destination,
                part1,
                results,
                selected_slot_id,
                stop_reason,
                False,
            )
            break
        _checkpoint(
            destination,
            part1,
            results,
            None,
            "in_progress",
            not stop_at_first_free,
        )

    final = Part2RunResult(
        snapshot_id=part1.scene.snapshot_id,
        queue_order=queue_order,
        processed_case_ids=tuple(result.case.case_id for result in results),
        selected_slot_id=selected_slot_id,
        stop_reason=stop_reason,
        slot_results=tuple(results),
        slot_cases=tuple(part1.slot_cases),
        evaluation_exhaustive=not stop_at_first_free,
    )
    _write_json_atomic(destination / "part2_result.json", final.to_dict())
    return final


__all__ = ["Part2RunResult", "run_candidate_queue"]
