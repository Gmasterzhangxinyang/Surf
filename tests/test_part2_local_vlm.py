from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import httpx

from parking_slot_part2.local_vlm import (
    LocalImage,
    LocalOpenAICompatibleVLMAdapter,
    LocalVLMMediaError,
    LocalVLMProtocolError,
    LocalVLMRequestError,
    LocalVLMResponseTooLarge,
    LocalVLMTimeoutError,
)
from parking_slot_part2.model import (
    MODEL_TURN_SCHEMA_VERSION,
    REPLAY_SCHEMA_VERSION,
    ReplayModelAdapter,
)


EV_RGB = "ev_" + "a" * 64
EV_FAILED = "ev_" + "b" * 64
EV_LIDAR = "ev_" + "c" * 64


def _request(*, observations: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "schema_version": MODEL_TURN_SCHEMA_VERSION,
        "group_id": "group-local-a",
        "turn": 2,
        "decision_tasks": [
            {
                "task_id": "task-a",
                "slot_id": "slot_0001",
                "unknown_reasons": ["weak_vehicle_evidence"],
                "allowed_final_states": ["occupied", "free", "unknown"],
                "occupied_evidence": {"strength": 0.4},
                "free_evidence": {"strength": 0.2},
            }
        ],
        "basic_context": {
            "schema_version": "fixture/1.0",
            "evidence_catalog": [
                {
                    "evidence_id": EV_RGB,
                    "tool_name": "inspect_rgb_frame",
                    "task_id": "task-a",
                }
            ],
        },
        "observations": observations or [],
        "budget": {
            "max_model_turns": 4,
            "model_turns_used": 1,
            "model_turns_remaining_after_this": 2,
            "max_tool_attempts": 3,
            "tool_attempts_used": 1,
            "tool_attempts_remaining": 2,
        },
    }


def _final_action() -> dict[str, object]:
    return {
        "type": "final_proposal",
        "assessments": [
            {
                "task_id": "task-a",
                "slot_id": "slot_0001",
                "proposed_state": "occupied",
                "target_visibility": "clear_full",
                "target_ownership": "target",
                "semantic_finding": "vehicle_or_occupying_object",
                "resolved_unknown_reasons": ["weak_vehicle_evidence"],
                "unresolved_blockers": [],
                "evidence_refs": [EV_RGB],
                "reason_codes": ["rgb_target_vehicle"],
            }
        ],
    }


def _completion(action: object) -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": json.dumps(action)}}]},
    )


class LocalOpenAICompatibleVLMAdapterTest(unittest.TestCase):
    def test_public_adapter_rejects_remote_endpoint_without_http_call(self) -> None:
        with self.assertRaisesRegex(ValueError, "loopback"):
            LocalOpenAICompatibleVLMAdapter(
                base_url="https://example.com/v1",
                model="must-not-run",
            )

    def test_http_client_ignores_environment_proxies(self) -> None:
        with patch("parking_slot_part2.local_vlm.httpx.Client") as client_type:
            adapter = LocalOpenAICompatibleVLMAdapter(
                base_url="http://127.0.0.1:8000/v1",
                model="fixture-vlm",
            )
            adapter.close()
        self.assertIs(client_type.call_args.kwargs["trust_env"], False)

    def test_public_package_exports_stable_adapter_api(self) -> None:
        import parking_slot_part2 as part2

        self.assertEqual(
            part2.LOCAL_VLM_ADAPTER_VERSION,
            "part2-local-openai-compatible-vlm/1.0",
        )
        self.assertIs(
            part2.LocalVLMAdapter,
            part2.LocalOpenAICompatibleVLMAdapter,
        )
        for name in (
            "LocalImage",
            "LocalVLMError",
            "LocalVLMHTTPError",
            "LocalVLMMediaError",
            "LocalVLMProtocolError",
            "LocalVLMRequestError",
            "LocalVLMResponseTooLarge",
            "LocalVLMTimeoutError",
            "MediaResolver",
        ):
            with self.subTest(name=name):
                self.assertTrue(hasattr(part2, name))

    def test_posts_strict_multimodal_turn_without_credentials_and_records_replay(self) -> None:
        captured: list[httpx.Request] = []
        resolved: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return _completion(_final_action())

        def resolver(evidence_id: str) -> LocalImage | None:
            resolved.append(evidence_id)
            return LocalImage("image/png", b"private-rgb-bytes")

        observations = [
            {
                "kind": "tool_result",
                "tool_name": "inspect_rgb_frame",
                "status": "ok",
                "evidence_id": EV_RGB,
                "data": {"semantic_inference_performed": False},
            },
            {
                "kind": "tool_result",
                "tool_name": "inspect_rgb_sequence",
                "status": "failed",
                "evidence_id": EV_FAILED,
                "data": {},
            },
            {
                "kind": "tool_result",
                "tool_name": "inspect_lidar_map",
                "status": "ok",
                "evidence_id": EV_LIDAR,
                "data": {},
            },
        ]
        adapter = LocalOpenAICompatibleVLMAdapter(
            base_url="http://127.0.0.1:8000/v1",
            model="fixture-vlm",
            media_resolver=resolver,
            transport=httpx.MockTransport(handler),
        )
        self.addCleanup(adapter.close)

        action = adapter.next_action(_request(observations=observations))

        self.assertEqual(action, _final_action())
        self.assertEqual(resolved, [EV_RGB, EV_LIDAR])
        self.assertEqual(len(captured), 1)
        wire_request = captured[0]
        self.assertEqual(
            str(wire_request.url),
            "http://127.0.0.1:8000/v1/chat/completions",
        )
        self.assertNotIn("authorization", wire_request.headers)
        payload = json.loads(wire_request.content)
        self.assertEqual(payload["model"], "fixture-vlm")
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        user_parts = payload["messages"][1]["content"]
        image_parts = [part for part in user_parts if part["type"] == "image_url"]
        self.assertEqual(len(image_parts), 2)
        self.assertTrue(
            image_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")
        )
        visual_labels = [
            part["text"]
            for part in user_parts
            if part["type"] == "text"
            and part["text"].startswith("Successful visual tool evidence:")
        ]
        self.assertTrue(any(EV_RGB in label for label in visual_labels))
        self.assertTrue(any(EV_LIDAR in label for label in visual_labels))
        self.assertFalse(any(EV_FAILED in label for label in visual_labels))
        self.assertIn("group-local-a", user_parts[0]["text"])

        replay = adapter.replay_payload()
        self.assertEqual(replay["schema_version"], REPLAY_SCHEMA_VERSION)
        self.assertEqual(replay["actions"], {"group-local-a": [_final_action()]})
        ReplayModelAdapter(replay)
        serialized_replay = json.dumps(replay)
        self.assertNotIn("base_url", serialized_replay)
        self.assertNotIn("data:image", serialized_replay)
        self.assertNotIn("private-rgb-bytes", serialized_replay)

    def test_rejects_provider_leakage_without_recording(self) -> None:
        action = _final_action()
        action["provider"] = "must-not-cross-boundary"
        adapter = LocalOpenAICompatibleVLMAdapter(
            base_url="http://localhost:8000/v1",
            model="fixture-vlm",
            transport=httpx.MockTransport(lambda request: _completion(action)),
        )
        self.addCleanup(adapter.close)

        with self.assertRaisesRegex(LocalVLMProtocolError, "forbidden field"):
            adapter.next_action(_request())
        self.assertEqual(
            adapter.replay_payload(),
            {"schema_version": REPLAY_SCHEMA_VERSION, "actions": {}},
        )

    def test_rejects_nested_path_field_without_recording(self) -> None:
        action = _final_action()
        action["debug"] = {"image_path": "/secret/frame.png"}
        adapter = LocalOpenAICompatibleVLMAdapter(
            base_url="http://localhost:8000/v1",
            model="fixture-vlm",
            transport=httpx.MockTransport(lambda request: _completion(action)),
        )
        self.addCleanup(adapter.close)

        with self.assertRaisesRegex(LocalVLMProtocolError, "forbidden field") as captured:
            adapter.next_action(_request())

        self.assertNotIn("/secret", str(captured.exception))
        self.assertEqual(adapter.replay_payload()["actions"], {})

    def test_returns_and_records_repairable_invalid_enum_for_orchestrator(self) -> None:
        action = _final_action()
        action["assessments"][0]["proposed_state"] = "maybe"  # type: ignore[index]
        adapter = LocalOpenAICompatibleVLMAdapter(
            base_url="http://localhost:8000/v1",
            model="fixture-vlm",
            transport=httpx.MockTransport(lambda request: _completion(action)),
        )
        self.addCleanup(adapter.close)

        returned = adapter.next_action(_request())

        self.assertEqual(returned, action)
        self.assertEqual(
            adapter.replay_payload()["actions"],
            {"group-local-a": [action]},
        )

    def test_rejects_non_json_action_content(self) -> None:
        response = httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "```json\n{\"type\":\"tool_request\"}\n```"}}
                ]
            },
        )
        adapter = LocalOpenAICompatibleVLMAdapter(
            base_url="http://localhost:8000/v1",
            model="fixture-vlm",
            transport=httpx.MockTransport(lambda request: response),
        )
        self.addCleanup(adapter.close)
        with self.assertRaises(LocalVLMProtocolError):
            adapter.next_action(_request())

    def test_enforces_timeout_without_real_network(self) -> None:
        def timeout(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("fixture timeout", request=request)

        adapter = LocalOpenAICompatibleVLMAdapter(
            base_url="http://localhost:8000/v1",
            model="fixture-vlm",
            timeout_seconds=0.1,
            transport=httpx.MockTransport(timeout),
        )
        self.addCleanup(adapter.close)
        with self.assertRaises(LocalVLMTimeoutError):
            adapter.next_action(_request())

    def test_enforces_response_byte_limit(self) -> None:
        adapter = LocalOpenAICompatibleVLMAdapter(
            base_url="http://localhost:8000/v1",
            model="fixture-vlm",
            max_response_bytes=32,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, content=b"x" * 33)
            ),
        )
        self.addCleanup(adapter.close)
        with self.assertRaises(LocalVLMResponseTooLarge):
            adapter.next_action(_request())

    def test_enforces_total_data_url_limit_before_http_call(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return _completion(_final_action())

        adapter = LocalOpenAICompatibleVLMAdapter(
            base_url="http://localhost:8000/v1",
            model="fixture-vlm",
            media_resolver=lambda evidence_id: LocalImage("image/png", b"x" * 32),
            max_data_url_bytes=40,
            transport=httpx.MockTransport(handler),
        )
        self.addCleanup(adapter.close)
        with self.assertRaisesRegex(LocalVLMMediaError, "data URL budget"):
            adapter.next_action(
                _request(
                    observations=[
                        {
                            "kind": "tool_result",
                            "tool_name": "inspect_rgb_frame",
                            "status": "ok",
                            "evidence_id": EV_RGB,
                            "data": {},
                        }
                    ]
                )
            )
        self.assertEqual(calls, 0)

    def test_media_resolver_exception_does_not_leak_local_path(self) -> None:
        def resolver(evidence_id: str) -> LocalImage | None:
            raise FileNotFoundError("/secret/camera/frame.png")

        adapter = LocalOpenAICompatibleVLMAdapter(
            base_url="http://localhost:8000/v1",
            model="fixture-vlm",
            media_resolver=resolver,
            transport=httpx.MockTransport(lambda request: _completion(_final_action())),
        )
        self.addCleanup(adapter.close)
        with self.assertRaises(LocalVLMMediaError) as captured:
            adapter.next_action(
                _request(
                    observations=[
                        {
                            "kind": "tool_result",
                            "tool_name": "inspect_rgb_frame",
                            "status": "ok",
                            "evidence_id": EV_RGB,
                            "data": {},
                        }
                    ]
                )
            )
        self.assertNotIn("/secret", str(captured.exception))

    def test_successful_visual_tool_without_published_media_fails_closed(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return _completion(_final_action())

        adapter = LocalOpenAICompatibleVLMAdapter(
            base_url="http://localhost:8000/v1",
            model="fixture-vlm",
            media_resolver=lambda evidence_id: None,
            transport=httpx.MockTransport(handler),
        )
        self.addCleanup(adapter.close)

        with self.assertRaisesRegex(LocalVLMMediaError, "no published media"):
            adapter.next_action(
                _request(
                    observations=[
                        {
                            "kind": "tool_result",
                            "tool_name": "inspect_rgb_frame",
                            "status": "ok",
                            "evidence_id": EV_RGB,
                            "data": {},
                        }
                    ]
                )
            )
        self.assertEqual(calls, 0)

    def test_rejects_malformed_turn_before_http_call(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return _completion(_final_action())

        request = _request()
        request["unexpected"] = True
        adapter = LocalOpenAICompatibleVLMAdapter(
            base_url="http://localhost:8000/v1",
            model="fixture-vlm",
            transport=httpx.MockTransport(handler),
        )
        self.addCleanup(adapter.close)
        with self.assertRaises(LocalVLMRequestError):
            adapter.next_action(request)
        self.assertEqual(calls, 0)


if __name__ == "__main__":
    unittest.main()
