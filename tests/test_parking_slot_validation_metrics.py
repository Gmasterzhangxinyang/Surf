import unittest

from parking_slot_validation.metrics import evaluate


class MetricsTest(unittest.TestCase):
    def test_unknown_is_abstention_and_unobservable_is_excluded(self):
        result = evaluate(
            {"a": "occupied", "b": "unknown", "c": "free", "d": "occupied"},
            {
                "a": "occupied",
                "b": "occupied",
                "c": "free",
                "d": {"human_label": "unobservable", "reason": "occluded"},
            },
            4,
        )
        self.assertEqual(result["human_resolved_count"], 3)
        self.assertEqual(result["algorithm_decided_count"], 2)
        self.assertEqual(result["resolved_decided_accuracy"], 1.0)
        self.assertEqual(result["human_unobservable_count"], 1)
        self.assertEqual(result["unobservable_reason_counts"], {"occluded": 1})

    def test_confusion_counts_are_binary_only(self):
        result = evaluate(
            {"a": "occupied", "b": "occupied", "c": "free", "d": "free"},
            {"a": "occupied", "b": "free", "c": "occupied", "d": "free"},
            4,
        )
        self.assertEqual(
            [result["true_occupied"], result["false_occupied"], result["false_free"], result["true_free"]],
            [1, 1, 1, 1],
        )


if __name__ == "__main__":
    unittest.main()
