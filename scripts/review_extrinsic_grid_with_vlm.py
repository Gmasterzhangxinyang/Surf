#!/usr/bin/env python3
"""Ask a vision model to rank a pre-rendered camera-extrinsic overlay grid."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from openai import OpenAI

    image_bytes = args.image.read_bytes()
    encoded = base64.b64encode(image_bytes).decode("ascii")
    prompt = """You are auditing LiDAR-to-camera extrinsics. The image is a grid:
rows vary the camera's left lateral offset and columns vary its left yaw.
Colored dots are projected LiDAR, overlaid on the same synchronized RGB frame.
Rank panels by geometric alignment of LiDAR surfaces/discontinuities with visible
vehicles, columns, walls, curbs, and ground boundaries. Ignore mere point count.
Return JSON only with keys best_y_m, best_yaw_deg, runner_up, confidence_0_to_1,
visual_evidence, and failure_warning. If no panel is reliable, say so explicitly."""
    response = OpenAI(
        api_key=args.key_file.read_text(encoding="utf-8").strip(),
        timeout=180.0,
        max_retries=2,
    ).responses.create(
        model=args.model,
        instructions="Be a conservative geometric calibration auditor.",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{encoded}",
                        "detail": "original",
                    },
                ],
            }
        ],
        reasoning={"effort": "medium"},
        max_output_tokens=1200,
        store=False,
    )
    raw = response.output_text
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = {"raw": raw, "parse_error": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
