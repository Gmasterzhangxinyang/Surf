import json
import os
import urllib.error
import urllib.request
from typing import Any


def build_llm_agent_context(
    agent_reasoning: dict,
    target_slot_history: list[dict],
    metrics: dict,
    final_decision: dict | None = None,
) -> dict:
    """Build compact context for the optional LLM briefing layer."""
    rounds = agent_reasoning.get("rounds", [])
    return {
        "agent_role": "LLM briefing layer. It explains and critiques the deterministic active perception agent; it does not control occupancy updates.",
        "safety_boundary": "The LLM must not claim it used ground truth, controlled the vehicle, or directly modified occupancy.",
        "final_parking_decision": final_decision or {},
        "target_slot_history": target_slot_history,
        "rounds": [
            {
                "round": item.get("round"),
                "target_before": item.get("observation", {}).get("target_slot_before"),
                "target_after": item.get("target_slot_after"),
                "switched_target": item.get("switched_target"),
                "selected_regions": item.get("selected_regions", []),
                "active_questions": item.get("active_questions", []),
                "tool_calls": item.get("tool_calls", []),
                "target_unknown_before": item.get("observation", {}).get("target_slot_unknown_ratio_before"),
                "target_unknown_after": item.get("target_slot_unknown_ratio_after"),
            }
            for item in rounds
        ],
        "metrics": {
            "false_free_before": metrics.get("false_free_before"),
            "false_free_after": metrics.get("false_free_after"),
            "occupied_iou_before": metrics.get("occupied_iou_before"),
            "occupied_iou_after": metrics.get("occupied_iou_after"),
            "target_slot_unknown_ratio_before": metrics.get("target_slot_unknown_ratio_before"),
            "target_slot_unknown_ratio_after": metrics.get("target_slot_unknown_ratio_after"),
            "num_tool_calls": metrics.get("num_tool_calls"),
        },
    }


def build_ai_policy_context(
    round_index: int,
    current_target: str,
    candidate_actions: list[dict],
    target_unknown_ratio: float,
    slot_scores: list[dict],
    previous_rounds: list[dict],
    phase: str = "verify_top1",
    ranking_context: dict | None = None,
) -> dict:
    """Build compact context for the optional in-loop AI policy advisor."""
    return {
        "agent_role": "AI policy advisor inside a rule-dominant active perception loop.",
        "safety_boundary": (
            "Choose only from candidate_actions. Do not invent regions, tools, slots, or occupancy updates. "
            "Rules and validators remain final authority."
        ),
        "round": round_index,
        "phase": phase,
        "current_target": current_target,
        "target_unknown_ratio": target_unknown_ratio,
        "top_slot_scores": slot_scores[:5],
        "ranking_context": ranking_context or {},
        "candidate_actions": candidate_actions,
        "previous_rounds": previous_rounds,
    }


def deterministic_policy_advice(context: dict, max_actions: int, reason: str = "OPENAI_API_KEY not set") -> dict:
    """Stable advisor that follows the staged rule ordering but emits the same schema as an AI policy."""
    candidates = context.get("candidate_actions", [])
    selected = [item["action_id"] for item in candidates[:max_actions]]
    selected_plan = []
    by_id = {item["action_id"]: item for item in candidates}
    for action_id in selected:
        action = by_id[action_id]
        allowed_tools = action.get("allowed_tools", [])
        if action.get("issue_type") == "occluded_unknown" and "occlusion_reasoning_tool" in allowed_tools:
            tool_ids = ["occlusion_reasoning_tool"]
        elif "lidar_geometry_checker" in allowed_tools:
            tool_ids = ["lidar_geometry_checker"]
            if (
                action.get("is_target_verification") or action.get("is_competitor_slot_exploration")
            ) and "image_crop_reinspect_placeholder" in allowed_tools:
                tool_ids.append("image_crop_reinspect_placeholder")
        else:
            tool_ids = allowed_tools[:1]
        selected_plan.append(
            {
                "action_id": action_id,
                "tool_ids": tool_ids,
                "reason": "Stable fallback chooses the first staged candidate and its safest allowed tool set.",
            }
        )

    phase = context.get("phase", "unknown")
    if phase == "verify_top1":
        rationale = "Verify the current top-ranked slot first before exploring competitors."
    elif phase == "explore_competitor":
        rationale = "Ranking changed, so explore the best competing slot under the limited sensing budget."
    else:
        rationale = "Prioritize the first staged candidate under the constrained sensing budget."
    return {
        "mode": "deterministic_policy_fallback",
        "fallback_reason": reason,
        "selected_action_ids": selected,
        "selected_plan": selected_plan,
        "rationale": rationale,
        "risk_assessment": (
            f"Current target {context.get('current_target')} has target_unknown_ratio="
            f"{context.get('target_unknown_ratio', 0.0):.3f}; action set is constrained by validator."
        ),
    }


def validate_policy_advice(advice: dict, candidate_actions: list[dict], max_actions: int) -> list[str]:
    """Validate AI-selected action ids against rule-generated candidates."""
    valid_ids = [item["action_id"] for item in candidate_actions]
    seen = set()
    selected = []
    for action_id in advice.get("selected_action_ids", []):
        if action_id in valid_ids and action_id not in seen:
            selected.append(action_id)
            seen.add(action_id)
        if len(selected) >= max_actions:
            break
    for action_id in valid_ids:
        if len(selected) >= max_actions:
            break
        if action_id not in seen:
            selected.append(action_id)
            seen.add(action_id)
    return selected


def validate_policy_plan(advice: dict, candidate_actions: list[dict], max_actions: int) -> list[dict]:
    """Validate AI-selected actions and tools against rule-generated candidates."""
    valid_by_id = {item["action_id"]: item for item in candidate_actions}
    raw_plan = advice.get("selected_plan")
    if not isinstance(raw_plan, list):
        raw_plan = [{"action_id": action_id} for action_id in advice.get("selected_action_ids", [])]

    plan = []
    seen = set()
    for item in raw_plan:
        if not isinstance(item, dict):
            continue
        action_id = item.get("action_id")
        if action_id not in valid_by_id or action_id in seen:
            continue
        action = valid_by_id[action_id]
        allowed_tools = action.get("allowed_tools", [])
        requested_tools = item.get("tool_ids", [])
        if not isinstance(requested_tools, list):
            requested_tools = []
        tool_ids = [tool for tool in requested_tools if tool in allowed_tools]
        if not tool_ids and allowed_tools:
            tool_ids = [allowed_tools[0]]
        plan.append(
            {
                "action_id": action_id,
                "tool_ids": tool_ids,
                "reason": item.get("reason", ""),
            }
        )
        seen.add(action_id)
        if len(plan) >= max_actions:
            break

    for action in candidate_actions:
        if len(plan) >= max_actions:
            break
        action_id = action["action_id"]
        if action_id in seen:
            continue
        allowed_tools = action.get("allowed_tools", [])
        plan.append(
            {
                "action_id": action_id,
                "tool_ids": allowed_tools[:1],
                "reason": "Validator fallback filled an unselected legal action.",
            }
        )
        seen.add(action_id)
    return plan


def call_openai_policy_advice(context: dict, cfg: dict, max_actions: int) -> dict:
    """Optionally call OpenAI for in-loop action selection among validated candidates."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return deterministic_policy_advice(context, max_actions)

    system_prompt = (
        "You are an AI policy advisor for a rule-dominant parking active perception loop. "
        "You may only choose action_id values from candidate_actions and tool_ids from each action's allowed_tools. "
        f"Choose at most {max_actions} action_id value(s), because the sensing budget is limited. "
        "Do not invent coordinates, tools, slots, or belief updates. "
        "Return strict JSON with keys: selected_plan, selected_action_ids, rationale, risk_assessment. "
        "selected_plan must be a list of objects with action_id, tool_ids, and reason. "
        "Keep each reason, rationale, and risk_assessment concise."
    )
    model_context = {
        **context,
        "selection_budget": max_actions,
        "selection_goal": (
            "Pick the legal inspection action(s) that most reduce parking-decision risk under the budget. "
            "Do not simply echo every candidate."
        ),
    }
    body = {
        "model": cfg.get("model", "gpt-5.5"),
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(model_context, ensure_ascii=False, indent=2)},
        ],
        "max_output_tokens": 900,
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
        text = _extract_response_text(payload)
        advice = json.loads(text)
        advice["mode"] = "openai_policy_advisor"
        advice["model"] = cfg.get("model", "gpt-5.5")
        return advice
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
        return deterministic_policy_advice(context, max_actions, reason=f"OpenAI policy call failed: {exc}")


def deterministic_briefing(context: dict, reason: str = "OPENAI_API_KEY not set") -> dict:
    """Create a stable briefing with the same schema as the LLM output."""
    history = context.get("target_slot_history", [])
    initial_target = history[0]["slot_id"] if history else "unknown"
    final_target = history[-1]["slot_id"] if history else initial_target
    switched = initial_target != final_target
    metrics = context.get("metrics", {})
    rounds = context.get("rounds", [])

    if switched:
        decision = f"Reject initial target {initial_target} and continue with {final_target}."
        key_finding = (
            f"Active reinspection changed the belief enough to switch from {initial_target} "
            f"to {final_target}."
        )
    else:
        decision = f"Continue validating target {final_target}."
        key_finding = f"The target {final_target} remained the best candidate after active checks."

    evidence_lines = []
    for item in rounds:
        switch_text = " switched target" if item.get("switched_target") else " kept target"
        evidence_lines.append(
            f"Round {item['round']}: {item['target_before']} -> {item['target_after']}{switch_text}; "
            f"target_unknown {item['target_unknown_before']:.3f} -> {item['target_unknown_after']:.3f}; "
            f"tools={', '.join(item.get('tool_calls', []))}."
        )

    return {
        "mode": "deterministic_fallback",
        "fallback_reason": reason,
        "headline": "AI Agent Briefing: active perception found risk and updated the parking target.",
        "initial_target": initial_target,
        "final_target": final_target,
        "recommended_decision": decision,
        "key_finding": key_finding,
        "round_by_round": evidence_lines,
        "metric_summary": [
            f"False-free errors: {metrics.get('false_free_before')} -> {metrics.get('false_free_after')}.",
            f"Occupied IoU: {metrics.get('occupied_iou_before'):.3f} -> {metrics.get('occupied_iou_after'):.3f}.",
            "Target-slot unknown ratio: "
            f"{metrics.get('target_slot_unknown_ratio_before'):.3f} -> "
            f"{metrics.get('target_slot_unknown_ratio_after'):.3f}.",
            f"Tool calls: {metrics.get('num_tool_calls')}.",
        ],
        "demo_takeaway": (
            "The useful agent behavior is not the initial slot choice; it is the loop that inspects "
            "a risky target, updates belief from local evidence, and revises the target when needed."
        ),
        "limitations": [
            "The LLM layer is explanatory and advisory; deterministic code performs the actual belief update.",
            "Camera reinspection is still a placeholder.",
            "The scene is synthetic and controlled, not a real-world parking benchmark.",
        ],
    }


def _extract_response_text(payload: dict) -> str:
    """Extract text from a Responses API payload without depending on the SDK."""
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and isinstance(content.get("text"), str):
                return content["text"]
    raise ValueError("No text found in LLM response")


def normalize_briefing(briefing: dict) -> dict:
    """Normalize live or fallback briefing into the panel/markdown schema."""
    normalized = dict(briefing)

    rounds = normalized.get("round_by_round", [])
    round_lines = []
    for item in rounds:
        if isinstance(item, str):
            round_lines.append(item)
        elif isinstance(item, dict):
            tools = ", ".join(item.get("tools_referenced", []))
            round_lines.append(
                f"Round {item.get('round')}: {item.get('target_before')} -> {item.get('target_after')}; "
                f"{item.get('summary', '')} {item.get('uncertainty_change', '')} Tools: {tools}."
            )
        else:
            round_lines.append(str(item))
    normalized["round_by_round"] = round_lines

    metrics = normalized.get("metric_summary", [])
    if isinstance(metrics, dict):
        metric_lines = []
        if "false_free_cells" in metrics:
            item = metrics["false_free_cells"]
            metric_lines.append(f"False-free errors: {item.get('before')} -> {item.get('after')}.")
        if "occupied_iou" in metrics:
            item = metrics["occupied_iou"]
            metric_lines.append(f"Occupied IoU: {item.get('before'):.3f} -> {item.get('after'):.3f}.")
        if "target_slot_unknown_ratio" in metrics:
            item = metrics["target_slot_unknown_ratio"]
            metric_lines.append(f"Target-slot unknown ratio: {item.get('before'):.3f} -> {item.get('after'):.3f}.")
        if "num_tool_calls" in metrics:
            metric_lines.append(f"Tool calls: {metrics.get('num_tool_calls')}.")
        normalized["metric_summary"] = metric_lines
    elif not isinstance(metrics, list):
        normalized["metric_summary"] = [str(metrics)]

    limitations = normalized.get("limitations", [])
    if isinstance(limitations, str):
        normalized["limitations"] = [limitations]
    elif not isinstance(limitations, list):
        normalized["limitations"] = [str(limitations)]

    for key in ["initial_target", "final_target", "recommended_decision", "key_finding", "demo_takeaway"]:
        if not isinstance(normalized.get(key), str):
            normalized[key] = json.dumps(normalized.get(key), ensure_ascii=False)
    return normalized


def call_openai_briefing(context: dict, cfg: dict) -> dict:
    """Optionally call OpenAI Responses API using stdlib only."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return deterministic_briefing(context)

    system_prompt = (
        "You are an AI parking active-perception briefing agent. "
        "Explain the deterministic agent trace in clear demo-ready language. "
        "Do not claim you directly controlled the vehicle, used ground truth for updates, "
        "or modified occupancy. Return strict JSON with keys: headline, initial_target, "
        "final_target, recommended_decision, key_finding, round_by_round, metric_summary, "
        "demo_takeaway, limitations."
    )
    user_prompt = json.dumps(context, ensure_ascii=False, indent=2)
    body = {
        "model": cfg.get("model", "gpt-5.5"),
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        text = _extract_response_text(payload)
        result = json.loads(text)
        result["mode"] = "openai_responses_api"
        result["model"] = cfg.get("model", "gpt-5.5")
        return normalize_briefing(result)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
        return normalize_briefing(deterministic_briefing(context, reason=f"OpenAI call failed: {exc}"))


def briefing_to_markdown(briefing: dict) -> str:
    """Render the briefing JSON as Markdown."""
    lines = [
        "# AI Agent Briefing",
        "",
        f"**Mode:** {briefing.get('mode', 'unknown')}",
        "",
        f"## {briefing.get('headline', 'Agent briefing')}",
        "",
        f"- Initial target: `{briefing.get('initial_target')}`",
        f"- Final target: `{briefing.get('final_target')}`",
        f"- Recommended decision: {briefing.get('recommended_decision')}",
        "",
        "## Key Finding",
        "",
        briefing.get("key_finding", ""),
        "",
        "## Round By Round",
        "",
    ]
    lines.extend(f"- {item}" for item in briefing.get("round_by_round", []))
    lines.extend(["", "## Metrics", ""])
    lines.extend(f"- {item}" for item in briefing.get("metric_summary", []))
    lines.extend(["", "## Demo Takeaway", "", briefing.get("demo_takeaway", ""), "", "## Limitations", ""])
    lines.extend(f"- {item}" for item in briefing.get("limitations", []))
    lines.append("")
    return "\n".join(lines)
