import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("index", Path(__file__).parents[1] / "generate_index.py")
index = importlib.util.module_from_spec(spec)
spec.loader.exec_module(index)


class IndexTests(unittest.TestCase):
    def test_raw_folders_are_discovered_without_manual_registration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ["2026-10-19", "2026-10-26", "2026-10-26-transformer-only", "2026-02-30"]:
                folder = root / name
                folder.mkdir()
                (folder / "results.csv").write_bytes(b"group,status\n4,allowed\n")
            (root / "2026-10-26" / "games.csv").write_bytes(b"phase,round\n")
            before = {path: path.read_bytes() for path in root.rglob("*.csv")}
            manifest = index.generate(root)
            self.assertEqual(manifest["runs"], [
                {"id": "2026-10-26", "games": True},
                {"id": "2026-10-19", "games": False},
            ])
            self.assertEqual(json.loads((root / "runs.json").read_text()), manifest)
            self.assertEqual(before, {path: path.read_bytes() for path in root.rglob("*.csv")})
            folder = root / "2026-11-02"
            folder.mkdir()
            (folder / "results.csv").write_text("group,status\n2,allowed\n")
            self.assertEqual(index.generate(root)["runs"][0]["id"], "2026-11-02")

    def test_empty_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(index.generate(Path(directory)), {"runs": []})


if __name__ == "__main__":
    unittest.main()
