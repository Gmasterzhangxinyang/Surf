import tempfile
import unittest
from pathlib import Path

from parking_slot_validation.models import (
    fingerprint_files,
    make_manifest_id,
    make_sample_id,
    normalize_prediction_row,
)


class PredictionModelTest(unittest.TestCase):
    def test_current_states_normalize_without_algorithm_coupling(self):
        occupied = normalize_prediction_row(
            {"slot_id": "slot_1", "state": "box_vehicle_core_supported", "score": "0.8", "anchor_frame": "10"},
            "run-a",
            "data-a",
        )
        free = normalize_prediction_row(
            {"slot_id": "slot_2", "state": "box_low_height_residual", "score": "0.2", "anchor_frame": "20"},
            "run-a",
            "data-a",
        )
        unknown = normalize_prediction_row(
            {"slot_id": "slot_3", "state": "box_boundary_conflict", "score": "0.5", "anchor_frame": "30"},
            "run-a",
            "data-a",
        )

        self.assertEqual(occupied.prediction, "occupied")
        self.assertEqual(free.prediction, "free")
        self.assertEqual(unknown.prediction, "unknown")
        self.assertEqual(occupied.audit["state"], "box_vehicle_core_supported")

    def test_normalized_prediction_column_is_accepted_for_future_algorithms(self):
        row = normalize_prediction_row(
            {"slot_id": "slot_4", "prediction": "free", "score": "0.7", "anchor_frame": "40"},
            "run-b",
            "data-a",
        )
        self.assertEqual(row.prediction, "free")

    def test_unknown_state_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unmapped prediction state"):
            normalize_prediction_row(
                {"slot_id": "slot_1", "state": "new_state", "score": "0", "anchor_frame": "1"},
                "run-a",
                "data-a",
            )

    def test_missing_identity_fields_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "slot_id"):
            normalize_prediction_row(
                {"state": "box_low_height_residual", "score": "0", "anchor_frame": "1"},
                "run-a",
                "data-a",
            )

    def test_sample_and_manifest_identities_are_canonical(self):
        self.assertEqual(
            make_sample_id("data-a", "slot_1", 12.25),
            make_sample_id("data-a", "slot_1", 12.25),
        )
        self.assertNotEqual(
            make_sample_id("data-a", "slot_1", 12.25),
            make_sample_id("data-b", "slot_1", 12.25),
        )
        self.assertEqual(
            make_manifest_id("data-a", {"max_frames": 7, "half_fov": 40}),
            make_manifest_id("data-a", {"half_fov": 40, "max_frames": 7}),
        )

    def test_file_fingerprint_changes_with_content_not_path_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "a.txt"
            second = Path(tmp) / "b.txt"
            first.write_text("alpha", encoding="utf-8")
            second.write_text("beta", encoding="utf-8")
            before = fingerprint_files([first, second])
            self.assertEqual(before, fingerprint_files([second, first]))
            second.write_text("changed", encoding="utf-8")
            self.assertNotEqual(before, fingerprint_files([first, second]))


if __name__ == "__main__":
    unittest.main()
