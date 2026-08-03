import tempfile
import unittest
from pathlib import Path

from scripts.build_pose_correction_review_report import embed_images


class PoseCorrectionReviewReportTest(unittest.TestCase):
    def test_embed_images_replaces_single_quoted_asset_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            report_dir = Path(temporary)
            assets = report_dir / "assets"
            assets.mkdir()
            (assets / "sample.png").write_bytes(b"png-data")

            embedded = embed_images("<img src='assets/sample.png'>", report_dir)

            self.assertIn("src='data:image/png;base64,", embedded)
            self.assertNotIn("assets/sample.png", embedded)


if __name__ == "__main__":
    unittest.main()
