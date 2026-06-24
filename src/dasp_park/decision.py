from __future__ import annotations

from dataclasses import dataclass
from typing import Any


COMMIT_SLOT = "COMMIT_SLOT"
CONTINUE_PERCEPTION = "CONTINUE_PERCEPTION"
DRIVE_FORWARD_EXPLORE = "DRIVE_FORWARD_EXPLORE"


@dataclass
class ParkingDecision:
    """Final parking decision produced by deterministic safety rules."""

    action: str
    slot_id: str | None
    reason: str
    checks: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "slot_id": self.slot_id,
            "reason": self.reason,
            "checks": self.checks,
        }


def _top1_stable_rounds(top1_history: list[str]) -> int:
    if not top1_history:
        return 0
    last = top1_history[-1]
    stable = 0
    for slot_id in reversed(top1_history):
        if slot_id != last:
            break
        stable += 1
    return stable


def evaluate_parking_decision(
    slot_scores: list[dict],
    top1_history: list[str],
    rounds_completed: int,
    max_rounds: int,
    competitor_explorations: int,
    max_competitor_explorations: int,
    cfg: dict,
) -> ParkingDecision:
    """
    Decide whether to commit to a slot, continue perception, or drive forward.

    The LLM is intentionally not part of this decision. It may choose a bounded
    perception query earlier in the loop, but final parking commitment is made
    by deterministic thresholds over the updated belief and ranking stability.
    """
    decision_cfg = cfg["agent"].get("final_decision", {})
    min_stable_rounds = int(decision_cfg.get("min_top1_stable_rounds", 2))
    min_score_margin = float(decision_cfg.get("min_score_margin", 0.08))
    max_occupied_ratio = float(decision_cfg.get("max_occupied_ratio", 0.05))
    max_unknown_ratio = float(decision_cfg.get("max_unknown_ratio", 0.30))
    max_occlusion_mean = float(decision_cfg.get("max_occlusion_mean", 0.25))
    max_entrance_unknown = float(decision_cfg.get("max_entrance_unknown", 0.30))
    max_entrance_occupied = float(decision_cfg.get("max_entrance_occupied", 0.02))

    if not slot_scores:
        return ParkingDecision(
            action=DRIVE_FORWARD_EXPLORE,
            slot_id=None,
            reason="no candidate slot score is available",
            checks={"rounds_completed": rounds_completed},
        )

    top1 = slot_scores[0]
    top2 = slot_scores[1] if len(slot_scores) > 1 else None
    margin = float(top1["score"] - top2["score"]) if top2 else float("inf")
    stable_rounds = _top1_stable_rounds(top1_history)

    checks = {
        "top1_slot": top1["slot_id"],
        "top1_stable_rounds": stable_rounds,
        "required_top1_stable_rounds": min_stable_rounds,
        "score_margin": margin,
        "required_score_margin": min_score_margin,
        "occupied_ratio": float(top1.get("occupied_ratio", 1.0)),
        "max_occupied_ratio": max_occupied_ratio,
        "unknown_ratio": float(top1.get("unknown_ratio", 1.0)),
        "max_unknown_ratio": max_unknown_ratio,
        "occlusion_mean": float(top1.get("occlusion_mean", 1.0)),
        "max_occlusion_mean": max_occlusion_mean,
        "entrance_unknown": float(top1.get("entrance_unknown", 1.0)),
        "max_entrance_unknown": max_entrance_unknown,
        "entrance_occupied": float(top1.get("entrance_occupied", 1.0)),
        "max_entrance_occupied": max_entrance_occupied,
        "rounds_completed": rounds_completed,
        "max_rounds": max_rounds,
        "competitor_explorations": competitor_explorations,
        "max_competitor_explorations": max_competitor_explorations,
    }

    failed = []
    if stable_rounds < min_stable_rounds:
        failed.append("top1 is not stable enough")
    if margin < min_score_margin:
        failed.append("top1-top2 score margin is too small")
    if checks["occupied_ratio"] > max_occupied_ratio:
        failed.append("top1 still has occupied evidence")
    if checks["unknown_ratio"] > max_unknown_ratio:
        failed.append("top1 still has too much unknown area")
    if checks["occlusion_mean"] > max_occlusion_mean:
        failed.append("top1 still has high occlusion risk")
    if checks["entrance_unknown"] > max_entrance_unknown:
        failed.append("slot entrance is not sufficiently observed")
    if checks["entrance_occupied"] > max_entrance_occupied:
        failed.append("slot entrance has occupied evidence")

    if not failed:
        return ParkingDecision(
            action=COMMIT_SLOT,
            slot_id=top1["slot_id"],
            reason="top ranked slot is stable and passes deterministic safety checks",
            checks={**checks, "failed_checks": []},
        )

    exhausted = (
        rounds_completed >= max_rounds
        or competitor_explorations >= max_competitor_explorations
    )
    if exhausted:
        return ParkingDecision(
            action=DRIVE_FORWARD_EXPLORE,
            slot_id=None,
            reason="current viewpoint is insufficient after bounded perception; drive forward for a new view",
            checks={**checks, "failed_checks": failed},
        )

    return ParkingDecision(
        action=CONTINUE_PERCEPTION,
        slot_id=None,
        reason="more bounded perception is allowed before committing or driving forward",
        checks={**checks, "failed_checks": failed},
    )
