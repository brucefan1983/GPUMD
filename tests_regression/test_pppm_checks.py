import tempfile
import unittest
from pathlib import Path

import pppm_checks


INITIAL = "PPPM mesh: 16 x 16 x 16 (target spacing 1 A; actual spacing 0.99375 0.5 0.5 A).\n"


class PPPMCheckTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.result = {"stdout_path": str(self.root / "stdout"), "workdir": str(self.root)}
        self.spec = {"spacing": 1.0, "initial_mesh": [16, 16, 16]}

    def check(self, text):
        Path(self.result["stdout_path"]).write_text(text)
        return pppm_checks.check(self.spec, self.result)

    def test_initial_mesh(self):
        self.assertEqual(self.check(INITIAL), {"initial_mesh": [16, 16, 16]})

    def test_unexpected_intermediate_mesh_output_fails(self):
        with self.assertRaisesRegex(ValueError, "unexpected"):
            self.check(INITIAL + "PPPM mesh update: 18 x 16 x 16\n")

    def test_duplicate_initial_and_malformed_logs_fail(self):
        with self.assertRaisesRegex(ValueError, "one initial"):
            self.check(INITIAL + INITIAL)
        with self.assertRaisesRegex(ValueError, "malformed"):
            self.check(INITIAL.replace("target spacing 1", "target spacing nan"))

    def test_final_cell_is_checked_independently(self):
        self.spec.update(frames=1, last_cell=[15.9, 0, 0, 0, 8, 0, 0, 0, 8])
        (self.root / "state.xyz").write_text(
            '1\nLattice="18.6 0 0 0 8 0 0 0 8" Properties=species:S:1:pos:R:3\nBa 0 0 0\n')
        with self.assertRaisesRegex(ValueError, "final cell"):
            self.check(INITIAL)

    def test_uninitialized_lifetime(self):
        self.spec = {"spacing": 1.0, "initial_mesh": None}
        self.check("normal output\n")
        with self.assertRaisesRegex(ValueError, "must not create"):
            self.check(INITIAL)


if __name__ == "__main__":
    unittest.main()
