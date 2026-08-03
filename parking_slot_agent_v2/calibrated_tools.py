"""Agent tools that expose audited target-projected Camera sheets and crops."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from .contracts import EvidenceRecord, SceneSnapshot, SlotCase
from .tools import V2ToolSuite, _case_id, _get, _record, _slot_id


def _manifest_frames(
    sheet: Path,
    resources: Mapping[str, Any],
    slot_id: str,
) -> list[int]:
    value = resources.get("calibrated_camera_manifest_path")
    manifest = (
        Path(str(value)).resolve(strict=False)
        if value
        else sheet.parent / "manifest.json"
    )
    if not manifest.is_file():
        return []
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        row = next(
            item
            for item in payload.get("rows", [])
            if isinstance(item, Mapping) and item.get("slot_id") == slot_id
        )
        return [int(value) for value in row.get("frames", [])]
    except (OSError, TypeError, ValueError, StopIteration, json.JSONDecodeError):
        return []


def _sheet_layout(sheet: Path, frames: Sequence[int]) -> dict[str, Any]:
    with Image.open(sheet) as image:
        width, height = image.size
    columns = 2
    header_px = 52 if frames and height > 52 else 0
    row_count = max(1, math.ceil(max(1, len(frames)) / columns))
    tile_width = width / columns
    tile_height = (height - header_px) / row_count
    tiles: list[dict[str, Any]] = []
    for index, frame_id in enumerate(frames):
        column = index % columns
        row = index // columns
        tiles.append(
            {
                "tile_index": index + 1,
                "lidar_frame_id": int(frame_id),
                "bbox_norm": [
                    column * tile_width / width,
                    (header_px + row * tile_height) / height,
                    (column + 1) * tile_width / width,
                    (header_px + (row + 1) * tile_height) / height,
                ],
            }
        )
    return {
        "coordinate_space": "calibrated_contact_sheet",
        "sheet_size_px": [width, height],
        "columns": columns,
        "header_px": header_px,
        "tiles": tiles,
    }


def _selected_tiles(
    bbox: Sequence[float],
    tiles: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    x1, y1, x2, y2 = (float(value) for value in bbox)
    area = max(1e-12, (x2 - x1) * (y2 - y1))
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    selected: list[dict[str, Any]] = []
    for tile in tiles:
        raw = tile.get("bbox_norm")
        if not isinstance(raw, list) or len(raw) != 4:
            continue
        tx1, ty1, tx2, ty2 = (float(value) for value in raw)
        overlap = max(0.0, min(x2, tx2) - max(x1, tx1)) * max(
            0.0, min(y2, ty2) - max(y1, ty1)
        )
        if overlap / area >= 0.05 or (tx1 <= cx <= tx2 and ty1 <= cy <= ty2):
            selected.append(dict(tile))
    return selected


class CalibratedCameraToolSuite(V2ToolSuite):
    """Prefer a causal calibrated contact sheet when attached to a SlotCase."""

    def _sheet(
        self, case: SlotCase
    ) -> tuple[Path | None, Mapping[str, Any]]:
        resources = _get(case, "resources", {})
        if not isinstance(resources, Mapping):
            return None, {}
        value = resources.get("calibrated_camera_sheet_path")
        return (
            Path(str(value)).resolve(strict=False) if value else None,
            resources,
        )

    def inspect_camera_context(
        self,
        scene: SceneSnapshot,
        case: SlotCase,
        output_dir: str | Path,
        round_index: int,
    ) -> EvidenceRecord:
        if not self._camera_allowed(case):
            return super().inspect_camera_context(scene, case, output_dir, round_index)
        sheet, resources = self._sheet(case)
        if sheet is None:
            return super().inspect_camera_context(scene, case, output_dir, round_index)
        if not sheet.is_file():
            return _record(
                case=case,
                tool_name="camera_context",
                round_index=round_index,
                status="failed",
                summary="The attached calibrated Camera sheet is missing.",
                metadata={
                    "reason_codes": ["calibrated_camera_sheet_missing"],
                    "case_id": _case_id(case),
                },
                modality="camera",
            )
        try:
            frames = _manifest_frames(sheet, resources, _slot_id(case))
            layout = _sheet_layout(sheet, frames)
        except (OSError, ValueError) as exc:
            return _record(
                case=case,
                tool_name="camera_context",
                round_index=round_index,
                status="failed",
                summary="The calibrated Camera sheet could not be validated.",
                metadata={
                    "reason_codes": ["calibrated_camera_sheet_invalid"],
                    "error_type": type(exc).__name__,
                },
                modality="camera",
            )

        audit_artifacts: list[Path] = []
        resource_keys: list[str] = []
        fov_map_value = resources.get("part2_fov_occupancy_map")
        if fov_map_value:
            fov_map = Path(str(fov_map_value)).resolve(strict=False)
            if fov_map.is_file():
                audit_artifacts.append(fov_map)
                resource_keys.append("part2_fov_occupancy_map")
        audit_artifacts.append(sheet)
        resource_keys.append("calibrated_camera_sheet_path")
        metadata = {
            "target_slot_id": _slot_id(case),
            "strictly_causal_to_t0": True,
            "map_to_camera_projection_used": True,
            "cyan_polygon_is_target_slot": True,
            "projection_is_occupancy_label": False,
            "camera_origin_lidar_m": [0.60, 0.36, -0.07],
            "camera_yaw_deg": -1.0,
            "bbox_norm_coordinate_space": "calibrated_contact_sheet",
            "contact_sheet_layout": layout,
            "camera_crop_guidance": (
                "camera_crop bbox_norm uses the entire contact sheet. Choose a "
                "listed tile and tighten the box around its cyan target plus the "
                "possibly occupying or occluding object."
            ),
            "model_image_paths": [str(sheet)],
            "model_image_order": ["cyan_target_camera_contact_sheet"],
            "audit_only_location_map_in_artifacts": len(audit_artifacts) == 2,
            "reason_codes": ["audited_target_projection_available"],
            "visual_task": {
                "occupied": "parked_vehicle_visibly_inside_cyan_target",
                "free": "cyan_target_visibly_clear_without_parked_vehicle",
                "unknown": "cyan_target_genuinely_occluded_clipped_or_unjudgeable",
                "metric_depth_required": False,
            },
        }
        return _record(
            case=case,
            tool_name="camera_context",
            round_index=round_index,
            status="ok",
            summary=(
                "Strictly causal multi-frame Camera sheet with the target bay "
                "projected in cyan. If a useful tile is too small, camera_crop "
                "can zoom it using the published contact-sheet coordinates."
            ),
            artifact_paths=tuple(audit_artifacts),
            metadata=metadata,
            modality="camera",
            reason_codes=["audited_target_projection_available"],
            resource_keys=resource_keys,
        )

    camera_context = inspect_camera_context

    def crop_camera(
        self,
        scene: SceneSnapshot,
        case: SlotCase,
        output_dir: str | Path,
        round_index: int,
        bbox_norm: Sequence[float],
        enhancement: str = "none",
    ) -> EvidenceRecord:
        """Crop an Agent-selected region of the calibrated contact sheet."""

        sheet, resources = self._sheet(case)
        if sheet is None:
            return super().crop_camera(
                scene, case, output_dir, round_index, bbox_norm, enhancement
            )
        mutable_resources = case.resources
        previous = mutable_resources.get("camera_image_path")
        mutable_resources["camera_image_path"] = str(sheet)
        try:
            evidence = super().crop_camera(
                scene, case, output_dir, round_index, bbox_norm, enhancement
            )
        finally:
            if previous is None:
                mutable_resources.pop("camera_image_path", None)
            else:
                mutable_resources["camera_image_path"] = previous
        if evidence.status != "ok":
            return evidence

        frames = _manifest_frames(sheet, resources, _slot_id(case))
        layout = _sheet_layout(sheet, frames)
        selected = _selected_tiles(bbox_norm, layout["tiles"])
        selected_indices = [int(item["tile_index"]) for item in selected]
        selected_frames = [int(item["lidar_frame_id"]) for item in selected]
        evidence.summary = (
            f"Agent-selected contact-sheet crop zooms tile(s) {selected_indices}, "
            f"LiDAR frame(s) {selected_frames}, with {enhancement} enhancement. "
            "The cyan polygon remains the target slot; inspect vehicle ownership, "
            "clear pavement, or genuine occlusion in the enlarged pixels."
        )
        evidence.metadata.update(
            {
                "bbox_norm_coordinate_space": "calibrated_contact_sheet",
                "selected_tile_indices": selected_indices,
                "selected_frame_ids": selected_frames,
                "cyan_polygon_is_target_slot": True,
                "agent_selected_crop": True,
                "crop_overlay_is_ground_truth": False,
                "model_image_order": [
                    "agent_crop_on_full_contact_sheet",
                    "agent_selected_zoom",
                ],
            }
        )
        evidence.metadata["reason_codes"] = ["agent_selected_contact_sheet_crop"]
        evidence.reason_codes = ["agent_selected_contact_sheet_crop"]
        return evidence

    camera_crop = crop_camera


__all__ = ["CalibratedCameraToolSuite"]
