#!/usr/bin/env python3
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "vibe_demo"


def font(size: int, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
    ]
    for item in candidates:
        if item and Path(item).exists():
            return ImageFont.truetype(item, size)
    return ImageFont.load_default()


TITLE = font(44, True)
H = font(30, True)
BODY = font(24)
SMALL = font(19)
MONO = font(20)


def wrap(draw, text, xy, fnt, fill, width, gap=7):
    words = str(text).split()
    lines = []
    line = ""
    for word in words:
        candidate = word if not line else f"{line} {word}"
        if draw.textbbox((0, 0), candidate, font=fnt)[2] <= width:
            line = candidate
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=fnt, fill=fill)
        y += fnt.size + gap
    return y


def fit(path, box):
    img = Image.open(path).convert("RGB")
    img.thumbnail(box, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", box, "white")
    canvas.paste(img, ((box[0] - img.width) // 2, (box[1] - img.height) // 2))
    return canvas


def panel(draw, xywh, title, fill="#f8fafc"):
    x, y, w, h = xywh
    draw.rounded_rectangle((x, y, x + w, y + h), radius=18, fill=fill, outline="#94a3b8", width=2)
    draw.text((x + 24, y + 18), title, font=H, fill="#0f172a")


def candidate_lines(round_info):
    out = []
    for action in round_info.get("candidate_actions", []):
        out.append(
            f"{action['action_id']} | {action['region_id']} | {action['issue_type']} | "
            f"priority={action['priority']:.3f} | target_check={action['is_target_verification']}"
        )
    return out


def make_frame(step_no, title, left_title, left_lines, right_title, right_lines, image_path=None):
    frame = Image.new("RGB", (1280, 720), "#ffffff")
    draw = ImageDraw.Draw(frame)
    draw.rectangle((0, 0, 1280, 88), fill="#111827")
    draw.text((36, 23), f"{step_no}. {title}", font=TITLE, fill="white")

    panel(draw, (40, 120, 570, 530), left_title)
    y = 178
    for line in left_lines:
        y = wrap(draw, line, (70, y), BODY if not line.startswith("  ") else MONO, "#1f2937", 510, gap=7) + 12

    panel(draw, (670, 120, 570, 530), right_title)
    if image_path is not None:
        img = fit(image_path, (500, 330))
        frame.paste(img, (705, 178))
        y = 525
    else:
        y = 178
    for line in right_lines:
        y = wrap(draw, line, (700, y), SMALL if len(line) > 85 else BODY, "#1f2937", 510, gap=6) + 8

    draw.rounded_rectangle((40, 666, 1240, 704), radius=12, fill="#eff6ff", outline="#60a5fa", width=2)
    draw.text((60, 674), "AI is constrained: it chooses only from legal candidate actions; validator and deterministic tools do the execution.", font=SMALL, fill="#1e3a8a")
    return frame


def main():
    trace = json.loads((OUT / "agent_trace.json").read_text())
    metrics = json.loads((OUT / "metrics_before_after.json").read_text())["metrics"]
    rounds = trace["agent_reasoning"]["rounds"]
    history = trace["target_slot_history"]
    initial = history[0]["slot_id"]
    final = history[-1]["slot_id"]

    frames = []
    frames.append(
        make_frame(
            1,
            "AI Receives Structured State",
            "Input To AI Advisor",
            [
                f"Current target: {initial}",
                f"Target unknown ratio: {rounds[0]['observation']['target_slot_unknown_ratio_before']:.3f}",
                "Known slot map and sparse LiDAR belief are already summarized.",
                "AI does not see ground truth and cannot modify occupancy.",
            ],
            "Scene Context",
            [
                "Sparse LiDAR makes the initial target look plausible.",
                "The hidden obstacle is only discoverable through local active reinspection.",
            ],
            OUT / "02_initial_observed_lidar.png",
        )
    )

    first_round = rounds[0]
    frames.append(
        make_frame(
            2,
            "Rules Generate Legal Candidate Actions",
            "Candidate Actions",
            candidate_lines(first_round),
            "What AI Is Allowed To Do",
            [
                "AI can rank/select these action_id values only.",
                "It cannot invent new regions, tools, or coordinates.",
                "This keeps the agent useful but bounded.",
            ],
        )
    )

    advice = first_round["ai_policy_advice"]
    selected_plan = first_round.get("validated_plan", [])
    plan_lines = []
    for item in selected_plan:
        tools = ", ".join(item.get("tool_ids", []))
        plan_lines.append(f"{item.get('action_id')}: tools={tools}")
    frames.append(
        make_frame(
            3,
            "OpenAI Advisor Chooses Actions",
            "AI Policy Output",
            [
                f"Mode: {advice.get('mode')}",
                f"Selected action ids: {', '.join(advice.get('selected_action_ids', []))}",
                *plan_lines,
                f"Rationale: {advice.get('rationale')}",
            ],
            "Validated Actions",
            [
                f"Validator accepted: {', '.join(first_round.get('validated_action_ids', []))}",
                "Next, deterministic tools execute these actions.",
            ],
        )
    )

    frames.append(
        make_frame(
            4,
            "Tools Inspect The Selected Regions",
            "Tool Execution",
            [
                f"Selected regions: {', '.join(first_round['selected_regions'])}",
                f"Tools: {', '.join(first_round['tool_calls'])}",
                "Local LiDAR evidence is queried only inside selected regions.",
            ],
            "Belief Update",
            [
                "The initial target gets new local evidence.",
                f"Target changes: {first_round['observation']['target_slot_before']} -> {first_round['target_slot_after']}",
                "This is the key agent behavior: inspect, update, switch.",
            ],
            OUT / "13_occupancy_after.png",
        )
    )

    for idx, round_info in enumerate(rounds[1:], start=5):
        advice = round_info["ai_policy_advice"]
        frames.append(
            make_frame(
                idx,
                f"Round {round_info['round']}: Continue Validation",
                "Updated State",
                [
                    f"Current target: {round_info['observation']['target_slot_before']}",
                    f"Target unknown: {round_info['observation']['target_slot_unknown_ratio_before']:.3f}",
                    f"AI selected: {', '.join(advice.get('selected_action_ids', []))}",
                    f"Validator accepted: {', '.join(round_info.get('validated_action_ids', []))}",
                ],
                "Round Result",
                [
                    f"Target after round: {round_info['target_slot_after']}",
                    f"Target unknown: {round_info['observation']['target_slot_unknown_ratio_before']:.3f} -> {round_info['target_slot_unknown_ratio_after']:.3f}",
                    f"Tools: {', '.join(round_info['tool_calls'])}",
                ],
            )
        )

    frames.append(
        make_frame(
            len(frames) + 1,
            "Final Outcome",
            "Measured Result",
            [
                f"Initial target: {initial}",
                f"Final target: {final}",
                f"False-free: {metrics['false_free_before']} -> {metrics['false_free_after']}",
                f"Occupied IoU: {metrics['occupied_iou_before']:.3f} -> {metrics['occupied_iou_after']:.3f}",
                f"Target unknown: {metrics['target_slot_unknown_ratio_before']:.3f} -> {metrics['target_slot_unknown_ratio_after']:.3f}",
            ],
            "Takeaway",
            [
                "AI participates slowly and explicitly in action selection.",
                "Rules enforce safety and execute the actual perception update.",
                "The demo shows why the target changed and what evidence caused it.",
            ],
            OUT / "15_effectiveness_panel.png",
        )
    )

    gif_path = OUT / "20_ai_thinking_slow.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=[3200, 3800, 4200, 4200, 3600, 3600, 4200],
        loop=0,
        optimize=True,
    )

    # Also save individual PNG frames for manual clicking during a presentation.
    frame_dir = OUT / "ai_thinking_frames"
    frame_dir.mkdir(exist_ok=True)
    for idx, frame in enumerate(frames, start=1):
        frame.save(frame_dir / f"frame_{idx:02d}.png")
    print(f"Saved {gif_path}")
    print(f"Saved frames to {frame_dir}")


if __name__ == "__main__":
    main()
