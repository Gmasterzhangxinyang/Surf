#!/usr/bin/env python3
"""运行 OpenAI Part2 前的 fail-closed 稳定性检查。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parking_slot_agent_v2.preflight import assert_openai_preflight, run_openai_preflight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--allow-existing-output", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kwargs = {
        "part1_path": args.input,
        "output_dir": args.output_dir,
        "key_file": args.key_file,
        "allow_existing_output": args.allow_existing_output,
    }
    result = run_openai_preflight(**kwargs)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output is not None:
        args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output.resolve().write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    assert_openai_preflight(**kwargs)


if __name__ == "__main__":
    main()
