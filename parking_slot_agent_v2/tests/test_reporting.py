from __future__ import annotations

import unittest

from parking_slot_agent_v2.reporting import RunMetrics, comparison_payload


def _metrics(name: str, *, detail: float, accepted: float, errors: int) -> RunMetrics:
    return RunMetrics(
        name=name,
        processed_cases=24,
        selected_slot_id=None,
        stop_reason="candidate_queue_exhausted",
        final_state_counts={"unknown": 24},
        fov_counts={"not_visible": 24},
        case_stop_counts={"no_useful_tools_remaining": 24},
        model_turns=24,
        tool_rounds=24,
        tool_counts={"lidar_detail": 24},
        cases_with_detail=24,
        detail_coverage_pct=detail,
        cases_with_successful_detail=24,
        successful_detail_coverage_pct=detail,
        accepted_final_cases=24,
        accepted_final_coverage_pct=accepted,
        validation_errors=errors,
        validation_error_cases=int(errors > 0),
        model_outputs=24,
        model_tool_actions=0,
        model_final_actions=24,
        autonomous_tool_action_pct=0.0,
        direct_final_action_pct=100.0,
        input_tokens=None,
        output_tokens=None,
        total_tokens=None,
        total_api_seconds=None,
        median_api_seconds=None,
        max_api_seconds=None,
    )


class ReportingTest(unittest.TestCase):
    def test_workflow_claim_requires_all_three_improvements(self) -> None:
        baseline = _metrics("baseline", detail=4.0, accepted=4.0, errors=100)
        experiment = _metrics("experiment", detail=100.0, accepted=96.0, errors=1)
        payload = comparison_payload(baseline, experiment)
        self.assertTrue(payload["claims"]["workflow_reliability_improved"])
        self.assertIsNone(payload["claims"]["classification_accuracy_improved"])


if __name__ == "__main__":
    unittest.main()
