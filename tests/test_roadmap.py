"""The README roadmap is generated from docs/roadmap.json; marking updates every view."""

import copy
import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("roadmap", ROOT / "tools" / "roadmap.py")
roadmap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(roadmap)


class TestRoadmap(unittest.TestCase):
    def test_readme_matches_source(self):
        self.assertTrue(roadmap.sync(check=True), "README roadmap is stale: run python tools/roadmap.py")

    def test_cli_marking_updates_checkbox_diagram_and_progress(self):
        with tempfile.TemporaryDirectory() as td:
            data_path, readme = Path(td) / "roadmap.json", Path(td) / "README.md"
            shutil.copy(roadmap.DATA, data_path)
            shutil.copy(roadmap.README, readme)
            original = readme.read_bytes()
            items = roadmap.milestones(roadmap.load(data_path))
            target = next(m for m in items if m["status"] != "done")
            done = sum(m["status"] == "done" for m in items)
            mid = target["id"]
            with patch.object(roadmap, "DATA", data_path), patch.object(roadmap, "README", readme):
                self.assertEqual(roadmap.main(["done", mid.lower(), "--date", "2030-01-02"]), 0)
                text = readme.read_text(encoding="utf-8")
                self.assertIn(f"- [x] **{mid} · ", text)
                self.assertIn(f'{mid}["✅ {mid} · ', text)
                self.assertIn(f"{done + 1} of {len(items)} milestones done", text)
                self.assertIn("*(2030-01-02)*", text)
                self.assertEqual(roadmap.main(["--check"]), 0)
                self.assertEqual(roadmap.main(["set", mid, target["status"]]), 0)
                self.assertNotIn("date", roadmap.milestones(roadmap.load(data_path))[items.index(target)])
                self.assertEqual(readme.read_bytes(), original)
                self.assertEqual(roadmap.main(["done", "M999"]), 1)
                self.assertEqual(roadmap.main(["done", mid, "--evidence", "no/such/file.md"]), 1)
                self.assertEqual(readme.read_bytes(), original)

    def test_validation_rejects_inconsistent_entries(self):
        base = roadmap.load()
        first = roadmap.milestones(base)[0]
        self.assertEqual(first["status"], "done")
        mutations = {
            "duplicate id": lambda d: roadmap.milestones(d)[1].update(id=first["id"]),
            "unknown status": lambda d: roadmap.milestones(d)[0].update(status="finished"),
            "done without date": lambda d: roadmap.milestones(d)[0].pop("date"),
            "planned with date": lambda d: roadmap.milestones(d)[-1].update(date="2030-01-01"),
            "missing evidence": lambda d: roadmap.milestones(d)[0].update(evidence="no/such/file.md"),
            "unknown key": lambda d: roadmap.milestones(d)[0].update(owner="x"),
            "empty phase": lambda d: d["phases"].append({"title": "x", "milestones": []}),
            "north star without gates": lambda d: d.setdefault("north_star", {"title": "t", "statement": "s"}).update(
                gates=[]
            ),
            "gate missing today": lambda d: d.setdefault("north_star", {"title": "t", "statement": "s"}).update(
                gates=[{"gate": "g", "target": "t"}]
            ),
            "unknown top-level key": lambda d: d.update(vision="x"),
        }
        for name, mutate in mutations.items():
            data = copy.deepcopy(base)
            mutate(data)
            with self.subTest(name), self.assertRaises(ValueError):
                roadmap.validate(data, ROOT)

    def test_north_star_renders_every_gate(self):
        data = {"north_star": {"title": "T", "statement": "Do it", "gates": [
            {"gate": "Size", "target": "a | b", "today": "c"}]}, "phases": roadmap.load()["phases"]}
        roadmap.validate(data, ROOT)
        text = roadmap.render(data)
        self.assertIn("### 🎯 North Star: T", text)
        self.assertIn("| **Size** | a \\| b | c |", text)
        self.assertLess(text.index("North Star"), text.index("**Progress:**"))

    def test_readme_markers_required(self):
        with tempfile.TemporaryDirectory() as td:
            readme = Path(td) / "README.md"
            readme.write_text("# no roadmap here\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "marker"):
                roadmap.sync(roadmap.DATA, readme, root=ROOT)


if __name__ == "__main__":
    unittest.main()
