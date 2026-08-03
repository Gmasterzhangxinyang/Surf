"""Atomic JSON persistence for Parking Slot Agent v2 contracts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Protocol

from .contracts import Part1Output, SlotCase


class _JsonContract(Protocol):
    def to_dict(self) -> dict[str, Any]: ...


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _payload(value: Any) -> Any:
    method = getattr(value, "to_dict", None)
    result = method() if callable(method) else value
    # A serialization pass validates the complete tree, including finite
    # numbers and string object keys, before a temporary file is created.
    encoded = json.dumps(
        result,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return json.loads(encoded, parse_constant=_reject_constant)


def load_json(path: str | Path) -> Any:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        return json.load(handle, parse_constant=_reject_constant)


def load_json_object(path: str | Path) -> Mapping[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {Path(path)}")
    return payload


def save_json_atomic(
    path: str | Path,
    value: Any,
    *,
    indent: int | None = 2,
    overwrite: bool = True,
) -> Path:
    """Atomically serialize a JSON value or any object with ``to_dict``.

    The temporary file is created beside the destination, flushed and fsynced,
    then moved into place with ``os.replace``.  A failed write never exposes a
    partially written destination.
    """

    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized = _payload(value)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(
                normalized,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=indent,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        return destination
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


atomic_save_json = save_json_atomic


def load_part1_output(path: str | Path) -> Part1Output:
    return Part1Output.from_dict(load_json_object(path))


def save_part1_output(
    path: str | Path,
    output: Part1Output,
    *,
    indent: int | None = 2,
    overwrite: bool = True,
) -> Path:
    if not isinstance(output, Part1Output):
        raise TypeError("output must be a Part1Output")
    return save_json_atomic(path, output, indent=indent, overwrite=overwrite)


def load_slot_case(path: str | Path) -> SlotCase:
    return SlotCase.from_dict(load_json_object(path))


def save_slot_case(
    path: str | Path,
    case: SlotCase,
    *,
    indent: int | None = 2,
    overwrite: bool = True,
) -> Path:
    if not isinstance(case, SlotCase):
        raise TypeError("case must be a SlotCase")
    return save_json_atomic(path, case, indent=indent, overwrite=overwrite)


__all__ = [
    "atomic_save_json",
    "load_json",
    "load_json_object",
    "load_part1_output",
    "load_slot_case",
    "save_json_atomic",
    "save_part1_output",
    "save_slot_case",
]
