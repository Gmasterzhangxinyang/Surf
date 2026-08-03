from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from parking_slot_hybrid_3d.contracts import (
    DecisionState,
    FreeEvidence,
    OccupiedEvidence,
    ScopeStatus,
    SlotDecision,
)
from parking_slot_hybrid_3d.evaluation import (
    GTMetadata,
    JsonGroundTruthProvider,
    SlotGroundTruth,
    SyntheticGroundTruthProvider,
    UnavailableGroundTruthProvider,
    evaluate_against_gt,
)


def decision(slot_id: str, state: str, strength: float = 0.0) -> SlotDecision:
    occupied = OccupiedEvidence(
        strong=state == "occupied",
        weak=state == "occupied",
        strength=strength if state == "occupied" else 0.0,
    )
    free = FreeEvidence(
        strong=state == "free",
        strength=strength if state == "free" else 0.0,
    )
    return SlotDecision(
        slot_id=slot_id,
        scope_status=ScopeStatus.IN_ROUTE,
        state=DecisionState(state),
        decision_reason=f"synthetic_{state}",
        unknown_reasons=("synthetic_abstention",) if state == "unknown" else (),
        occupied_evidence=occupied,
        free_evidence=free,
    )


def label(slot_id: str, value: str) -> SlotGroundTruth:
    return SlotGroundTruth(
        dataset_id="synthetic-dataset",
        gt_version="v1",
        slot_id=slot_id,
        label=value,
        annotation_source="unit-test",
    )


class Hybrid3DEvaluationTest(unittest.TestCase):
    def test_unavailable_gt_emits_status_without_accuracy_like_numbers(self):
        result = evaluate_against_gt(
            [decision("slot-a", "unknown")],
            UnavailableGroundTruthProvider(reason="not_collected"),
            scope_slot_ids=["slot-a"],
        )

        payload = result.to_dict()
        self.assertEqual(payload, {"gt_status": "unavailable", "reason": "not_collected"})
        self.assertNotIn("metrics", payload)
        self.assertFalse(
            any(isinstance(value, (int, float)) and not isinstance(value, bool) for value in payload.values())
        )

    def test_synthetic_gt_excludes_ignore_and_reports_exact_denominators(self):
        metadata = GTMetadata(
            dataset_id="synthetic-dataset",
            gt_version="v1",
            annotation_source="unit-test",
        )
        provider = SyntheticGroundTruthProvider(
            metadata=metadata,
            labels=[
                label("a", "occupied"),
                label("b", "occupied"),
                label("c", "occupied"),
                label("d", "free"),
                label("e", "free"),
                label("f", "free"),
                label("g", "ignore"),
            ],
        )
        result = evaluate_against_gt(
            [
                decision("a", "occupied", 0.9),
                decision("b", "free", 0.8),
                decision("c", "unknown"),
                decision("d", "free", 0.7),
                decision("e", "occupied", 0.6),
                decision("f", "unknown"),
                decision("g", "occupied", 1.0),
            ],
            provider,
            scope_slot_ids=["a", "b", "c", "d", "e", "f", "g"],
        )

        payload = result.to_dict()
        self.assertEqual(payload["gt_status"], "available")
        self.assertEqual(payload["dataset_id"], "synthetic-dataset")
        self.assertEqual(payload["gt_version"], "v1")
        self.assertEqual(payload["annotation_source"], "unit-test")
        self.assertEqual(payload["scope_slot_ids"], ["a", "b", "c", "d", "e", "f", "g"])
        self.assertEqual(
            payload["counts"],
            {
                "eligible": 6,
                "ignored": 1,
                "true_occupied": 1,
                "false_occupied": 1,
                "false_free": 1,
                "true_free": 1,
                "unknown": 2,
            },
        )
        self.assertEqual(
            payload["metrics"],
            {
                "occupied_precision": {"numerator": 1, "denominator": 2, "value": 0.5},
                "occupied_recall": {"numerator": 1, "denominator": 3, "value": 1 / 3},
                "free_precision": {"numerator": 1, "denominator": 2, "value": 0.5},
                "free_recall": {"numerator": 1, "denominator": 3, "value": 1 / 3},
                "false_free_rate": {"numerator": 1, "denominator": 3, "value": 1 / 3},
                "unknown_coverage": {"numerator": 2, "denominator": 6, "value": 1 / 3},
                "terminal_coverage": {"numerator": 4, "denominator": 6, "value": 2 / 3},
            },
        )
        self.assertEqual(
            payload["risk_coverage"],
            [
                {
                    "minimum_strength": 0.9,
                    "coverage": {"numerator": 1, "denominator": 6, "value": 1 / 6},
                    "risk": {"numerator": 0, "denominator": 1, "value": 0.0},
                },
                {
                    "minimum_strength": 0.8,
                    "coverage": {"numerator": 2, "denominator": 6, "value": 1 / 3},
                    "risk": {"numerator": 1, "denominator": 2, "value": 0.5},
                },
                {
                    "minimum_strength": 0.7,
                    "coverage": {"numerator": 3, "denominator": 6, "value": 0.5},
                    "risk": {"numerator": 1, "denominator": 3, "value": 1 / 3},
                },
                {
                    "minimum_strength": 0.6,
                    "coverage": {"numerator": 4, "denominator": 6, "value": 2 / 3},
                    "risk": {"numerator": 2, "denominator": 4, "value": 0.5},
                },
            ],
        )
        self.assertEqual(payload["score_semantics"], "rule_evidence_strength_ranking_not_probability")

    def test_empty_metric_denominator_is_explicitly_null(self):
        provider = SyntheticGroundTruthProvider(
            metadata=GTMetadata("synthetic-dataset", "v1", "unit-test"),
            labels=[label("a", "free")],
        )
        payload = evaluate_against_gt(
            [decision("a", "unknown")], provider, scope_slot_ids=["a"]
        ).to_dict()

        self.assertEqual(
            payload["metrics"]["occupied_precision"],
            {"numerator": 0, "denominator": 0, "value": None},
        )
        self.assertEqual(
            payload["metrics"]["occupied_recall"],
            {"numerator": 0, "denominator": 0, "value": None},
        )

    def test_provider_boundary_rejects_duplicates_unknown_slots_and_invalid_labels(self):
        metadata = GTMetadata("synthetic-dataset", "v1", "unit-test")
        cases = (
            ([label("a", "occupied"), label("a", "free")], ["a"], "duplicate GT slot_id"),
            ([label("missing", "occupied")], ["a"], "unknown GT slot_id"),
            ([label("a", "invalid")], ["a"], "invalid GT label"),
        )
        for labels, scope, error in cases:
            with self.subTest(error=error):
                provider = SyntheticGroundTruthProvider(metadata=metadata, labels=labels)
                with self.assertRaisesRegex(ValueError, error):
                    evaluate_against_gt([], provider, scope_slot_ids=scope)

    def test_json_provider_requires_metadata_and_matching_label_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing_metadata = root / "missing_metadata.json"
            missing_metadata.write_text(json.dumps({"metadata": {}, "labels": []}), encoding="utf-8")
            provider = JsonGroundTruthProvider(missing_metadata)
            with self.assertRaisesRegex(ValueError, "metadata.dataset_id"):
                provider.load_metadata()

            mismatch = root / "mismatch.json"
            mismatch.write_text(
                json.dumps(
                    {
                        "metadata": {
                            "dataset_id": "dataset-a",
                            "gt_version": "v1",
                            "annotation_source": "human-review",
                        },
                        "labels": [
                            {
                                "dataset_id": "dataset-b",
                                "gt_version": "v1",
                                "slot_id": "a",
                                "label": "occupied",
                                "annotation_source": "human-review",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "dataset_id does not match metadata"):
                evaluate_against_gt([], JsonGroundTruthProvider(mismatch), scope_slot_ids=["a"])


if __name__ == "__main__":
    unittest.main()
