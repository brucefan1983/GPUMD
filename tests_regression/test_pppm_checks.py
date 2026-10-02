import tempfile
import unittest
from pathlib import Path

import pppm_checks


INITIAL = "PPPM mesh: 16 x 16 x 16 (target spacing 1 A; actual spacing 0.99375 0.5 0.5 A).\n"
TRACE = (
    "PPPM diagnostic: 16 16 16; thickness 15.9 8 8; rebuild 1\n"
    "PPPM diagnostic: 18 16 16; thickness 16.8 8 8; rebuild 1\n"
    "PPPM diagnostic: 20 16 16; thickness 18.6 8 8; rebuild 1\n"
    "PPPM diagnostic: 20 16 16; thickness 15.9 8 8; rebuild 0\n"
)


class PPPMCheckTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.result = {"stdout_path": str(self.root / "stdout"), "workdir": str(self.root)}
        self.spec = {"spacing": 1.0, "initial_mesh": [16, 16, 16],
                     "mesh_sequence": [[16, 16, 16], [18, 16, 16], [20, 16, 16]]}

    def check(self, text, required=True):
        Path(self.result["stdout_path"]).write_text(text)
        return pppm_checks.check(self.spec, self.result, required)

    def test_growth_then_contraction(self):
        result = self.check(INITIAL + TRACE)
        self.assertEqual(result["mesh_sequence"], self.spec["mesh_sequence"])
        self.assertTrue(result["dynamic_trace_checked"])

    def test_production_log_needs_no_trace_but_diagnostic_gate_does(self):
        result = self.check(INITIAL, required=False)
        self.assertFalse(result["dynamic_trace_checked"])
        with self.assertRaisesRegex(ValueError, "missing temporary"):
            self.check(INITIAL)

    def test_shrink_and_redundant_rebuild_fail(self):
        with self.assertRaisesRegex(ValueError, "shrank"):
            self.check(INITIAL + TRACE.replace(
                "20 16 16; thickness 15.9 8 8; rebuild 0",
                "16 16 16; thickness 15.9 8 8; rebuild 1"))
        with self.assertRaisesRegex(ValueError, "rebuilt"):
            self.check(INITIAL + TRACE.replace("rebuild 0", "rebuild 1"))

    def test_missing_growth_fails_spacing_bound(self):
        with self.assertRaisesRegex(ValueError, "exceeds"):
            self.check(INITIAL + TRACE.replace("18 16 16;", "16 16 16;"))

    def test_duplicate_initial_and_malformed_logs_fail(self):
        with self.assertRaisesRegex(ValueError, "one initial"):
            self.check(INITIAL + INITIAL + TRACE)
        with self.assertRaisesRegex(ValueError, "malformed"):
            self.check(INITIAL.replace("target spacing 1", "target spacing nan") + TRACE)

    def test_only_recognized_mesh_lines_are_removed(self):
        raw = ("other line\n" + INITIAL + TRACE + "PPPM mesh: malformed\n").encode()
        self.assertEqual(pppm_checks.strip_mesh_lines(raw), b"other line\nPPPM mesh: malformed\n")

    def test_final_cell_is_checked_independently(self):
        self.spec.update(frames=1, last_cell=[15.9, 0, 0, 0, 8, 0, 0, 0, 8])
        (self.root / "state.xyz").write_text(
            '1\nLattice="18.6 0 0 0 8 0 0 0 8" Properties=species:S:1:pos:R:3\nBa 0 0 0\n')
        with self.assertRaisesRegex(ValueError, "final cell"):
            self.check(INITIAL + TRACE)

    def test_uninitialized_lifetime(self):
        self.spec = {"spacing": 1.0, "initial_mesh": None}
        self.check("normal output\n")
        with self.assertRaisesRegex(ValueError, "must not create"):
            self.check(INITIAL)


if __name__ == "__main__":
    unittest.main()
