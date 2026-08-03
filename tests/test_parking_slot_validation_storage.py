import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from parking_slot_validation.storage import LabelRepository


class LabelRepositoryTest(unittest.TestCase):
    def test_label_survives_restart_and_preserves_created_time_on_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "human_labels.json"
            repo = LabelRepository(path, "data-a", "manifest-a", reviewer="tester")
            first = repo.upsert(
                {
                    "sample_id": "sample-1",
                    "slot_id": "slot_1",
                    "human_label": "occupied",
                    "reason": "",
                    "evidence_frame_refs": ["frame-a.png"],
                }
            )
            revised = repo.upsert(
                {
                    "sample_id": "sample-1",
                    "slot_id": "slot_1",
                    "human_label": "free",
                    "reason": "",
                    "evidence_frame_refs": ["frame-a.png"],
                }
            )
            reopened = LabelRepository(path, "data-a", "manifest-a")
            stored = reopened.list_labels()["sample-1"]
            self.assertEqual(stored["human_label"], "free")
            self.assertEqual(stored["created_at"], first["created_at"])
            self.assertEqual(stored["updated_at"], revised["updated_at"])
            self.assertTrue(path.with_suffix(".json.bak").exists())

    def test_unobservable_requires_allowed_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = LabelRepository(Path(tmp) / "labels.json", "data-a", "manifest-a")
            with self.assertRaisesRegex(ValueError, "reason"):
                repo.upsert(
                    {
                        "sample_id": "sample-1",
                        "slot_id": "slot_1",
                        "human_label": "unobservable",
                        "reason": "",
                    }
                )
            with self.assertRaisesRegex(ValueError, "reason"):
                repo.upsert(
                    {
                        "sample_id": "sample-1",
                        "slot_id": "slot_1",
                        "human_label": "unobservable",
                        "reason": "made_up_reason",
                    }
                )

    def test_manifest_or_dataset_mismatch_is_rejected_without_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.json"
            repo = LabelRepository(path, "data-a", "manifest-a")
            repo.upsert(
                {
                    "sample_id": "sample-1",
                    "slot_id": "slot_1",
                    "human_label": "occupied",
                    "reason": "",
                }
            )
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "dataset"):
                LabelRepository(path, "data-b", "manifest-a")
            with self.assertRaisesRegex(ValueError, "manifest"):
                LabelRepository(path, "data-a", "manifest-b")
            self.assertEqual(path.read_bytes(), before)

    def test_export_and_remove_use_persisted_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.json"
            repo = LabelRepository(path, "data-a", "manifest-a")
            repo.upsert(
                {
                    "sample_id": "sample-1",
                    "slot_id": "slot_1",
                    "human_label": "unobservable",
                    "reason": "occluded",
                }
            )
            exported = json.loads(repo.export_json())
            self.assertEqual(exported["labels"]["sample-1"]["reason"], "occluded")
            rows = list(csv.DictReader(io.StringIO(repo.export_csv())))
            self.assertEqual(rows[0]["human_label"], "unobservable")
            self.assertTrue(repo.remove("sample-1"))
            self.assertEqual(LabelRepository(path, "data-a", "manifest-a").list_labels(), {})




    def test_corrupt_primary_recovers_from_last_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.json"
            repo = LabelRepository(path, "data-a", "manifest-a")
            base = {"sample_id": "sample-1", "slot_id": "slot_1", "reason": ""}
            repo.upsert({**base, "human_label": "occupied"})
            repo.upsert({**base, "human_label": "free"})
            path.write_text("not valid json", encoding="utf-8")
            recovered = LabelRepository(path, "data-a", "manifest-a")
            self.assertEqual(recovered.list_labels()["sample-1"]["human_label"], "occupied")
