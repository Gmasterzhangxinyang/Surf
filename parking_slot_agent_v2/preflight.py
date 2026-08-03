"""Fail-closed stability checks for an OpenAI-backed Part2 run."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.metadata
import json
from pathlib import Path
import shutil
import stat
import sys
from typing import Any

from .contracts import Part1Output


EXPECTED_OPENAI_SDK = "2.46.0"
MIN_FREE_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class StabilityCheck:
    name: str
    ok: bool
    detail: str


def _check(name: str, condition: bool, detail: str) -> StabilityCheck:
    return StabilityCheck(name=name, ok=bool(condition), detail=detail)


def run_openai_preflight(
    *,
    part1_path: str | Path,
    output_dir: str | Path,
    key_file: str | Path,
    allow_existing_output: bool = False,
) -> dict[str, Any]:
    """Validate secrets, runtime, input media, disk, and output isolation."""

    input_path = Path(part1_path).resolve()
    destination = Path(output_dir).resolve()
    secret_path = Path(key_file).resolve(strict=False)
    checks: list[StabilityCheck] = []

    checks.append(_check("python_version", sys.version_info >= (3, 10), sys.version.split()[0]))
    try:
        sdk_version = importlib.metadata.version("openai")
    except importlib.metadata.PackageNotFoundError:
        sdk_version = "missing"
    checks.append(
        _check(
            "openai_sdk_version",
            sdk_version == EXPECTED_OPENAI_SDK,
            f"installed={sdk_version}, expected={EXPECTED_OPENAI_SDK}",
        )
    )

    key_ok = secret_path.is_file() and not secret_path.is_symlink()
    key_bytes = secret_path.stat().st_size if key_ok else 0
    key_mode = stat.S_IMODE(secret_path.stat().st_mode) if key_ok else 0
    checks.append(
        _check(
            "api_key_file",
            key_ok and key_bytes > 0 and key_mode == 0o600,
            f"regular_non_symlink={key_ok}, nonempty={key_bytes > 0}, mode={key_mode:o}",
        )
    )

    part1: Part1Output | None = None
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        part1 = Part1Output.from_dict(payload)
        input_detail = (
            f"snapshot={part1.scene.snapshot_id}, candidates={len(part1.slot_cases)}, "
            f"frames={len(part1.scene.frames)}"
        )
        input_ok = bool(part1.slot_cases) and all(not case.terminal for case in part1.slot_cases)
    except Exception as exc:
        input_ok = False
        input_detail = f"{type(exc).__name__}: {exc}"
    checks.append(_check("part1_contract", input_ok, input_detail))

    missing_media: list[str] = []
    invalid_camera_matches: list[int] = []
    if part1 is not None:
        for frame in part1.scene.frames:
            for raw_path in (
                frame.camera_image_path,
                frame.lidar_path,
                frame.map_points_path,
            ):
                path = Path(raw_path)
                if path.is_symlink() or not path.is_file():
                    missing_media.append(str(path))
            if not frame.camera_match_valid:
                invalid_camera_matches.append(frame.frame_id)
    checks.append(
        _check(
            "sensor_media",
            part1 is not None and not missing_media and not invalid_camera_matches,
            f"missing={len(missing_media)}, invalid_camera_matches={invalid_camera_matches}",
        )
    )

    existing_result = destination / "part2_result.json"
    output_ok = allow_existing_output or not existing_result.exists()
    checks.append(
        _check(
            "output_isolation",
            output_ok,
            (
                "explicit overwrite enabled"
                if allow_existing_output
                else f"existing_result={existing_result.exists()}"
            ),
        )
    )
    disk_probe = destination if destination.exists() else destination.parent
    while not disk_probe.exists() and disk_probe != disk_probe.parent:
        disk_probe = disk_probe.parent
    free_bytes = shutil.disk_usage(disk_probe).free
    checks.append(
        _check(
            "free_disk",
            free_bytes >= MIN_FREE_BYTES,
            f"free_bytes={free_bytes}, required={MIN_FREE_BYTES}",
        )
    )

    return {
        "schema_version": "parking-slot-agent-v2-openai-preflight/1.0",
        "ok": all(item.ok for item in checks),
        "checks": [asdict(item) for item in checks],
        "secret_recorded": False,
    }


def assert_openai_preflight(**kwargs: Any) -> dict[str, Any]:
    result = run_openai_preflight(**kwargs)
    if not result["ok"]:
        failures = [
            f"{item['name']}: {item['detail']}"
            for item in result["checks"]
            if not item["ok"]
        ]
        raise RuntimeError("OpenAI Part2 preflight failed: " + "; ".join(failures))
    return result


__all__ = [
    "EXPECTED_OPENAI_SDK",
    "StabilityCheck",
    "assert_openai_preflight",
    "run_openai_preflight",
]
