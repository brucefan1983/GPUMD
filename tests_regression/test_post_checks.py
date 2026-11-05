import tempfile
import unittest
from pathlib import Path

import numpy as np

import post_checks


class PostCheckTests(unittest.TestCase):
    def test_nep_energy_difference_batches(self):
        spec = {
            "name": "nep_energy_difference", "output": "loss.out",
            "generations": 10, "batches": 3, "empty_batch": 2,
        }
        post_checks.validate_spec(spec, "spec")
        data = np.column_stack([np.arange(1, 11), np.ones(10), np.full(10, 2.0)])
        data[[2, 5, 8], 1] = 0
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "loss.out"
            header = "columns generation rmse_ediff_train rmse_ediff_test"
            np.savetxt(path, data, header=header)
            result = post_checks.run([spec], root)
            self.assertEqual(result[0]["metrics"], {"rows": 10, "empty_batch_rows": 3})
            for row, column, value in ((2, 1, 1), (0, 1, 0), (0, 2, 0), (1, 0, 1), (0, 1, np.nan)):
                with self.subTest(row=row, column=column, value=value):
                    altered = data.copy()
                    altered[row, column] = value
                    np.savetxt(path, altered, header=header)
                    with self.assertRaises(post_checks.PostCheckError):
                        post_checks.run([spec], root)
            np.savetxt(path, data[:-1], header=header)
            with self.assertRaises(post_checks.PostCheckError):
                post_checks.run([spec], root)
            np.savetxt(path, data)
            with self.assertRaises(post_checks.PostCheckError):
                post_checks.run([spec], root)

    def test_nep_energy_difference_spec(self):
        spec = {
            "name": "nep_energy_difference", "output": "loss.out",
            "generations": 10, "batches": 3, "empty_batch": 2,
        }
        for key, value in (("generations", 0), ("batches", True), ("empty_batch", 3), ("empty_batch", -1)):
            with self.subTest(key=key, value=value):
                with self.assertRaises(ValueError):
                    post_checks.validate_spec(dict(spec, **{key: value}), "spec")

    def test_extxyz_parser_and_msd_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            xyz = root / "traj.xyz"
            frames = []
            for step, x in enumerate((0.0, 1.0, 2.0, 3.0, 4.0)):
                frames.append(
                    "2\n"
                    f'Time={step} Properties=species:S:1:unwrapped_position:R:3\n'
                    f"Cu {x} 0 0\n"
                    f"Cu {x} 0 0\n"
                )
            xyz.write_text("".join(frames), encoding="utf-8")
            parsed = post_checks.read_extxyz(xyz)
            self.assertEqual(len(parsed), 5)
            positions = np.asarray(
                [frame["arrays"]["unwrapped_position"] for frame in parsed]
            )
            expected = post_checks._compute_msd(positions, 3)
            out = root / "msd.out"
            np.savetxt(out, np.column_stack([np.arange(3), expected, np.zeros((3, 3))]))
            result = post_checks.run(
                [
                    {
                        "name": "msd_from_xyz",
                        "output": "msd.out",
                        "window": 3,
                        "trajectories": [{"xyz": "traj.xyz", "column_start": 1}],
                        "rtol": 1.0e-12,
                        "atol": 1.0e-12,
                    }
                ],
                root,
            )
            self.assertEqual(result[0]["name"], "msd_from_xyz")

    def test_active_uncertainty_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def observer(force):
                return (
                    "1\n"
                    "Properties=species:S:1:pos:R:3:vel:R:3:forces:R:3 energy=0\n"
                    f"C 0 0 0 0.1 0.2 0.3 {force} 0 0\n"
                )
            (root / "observer0.xyz").write_text(observer(0.0) * 2, encoding="utf-8")
            (root / "observer1.xyz").write_text(observer(2.0) * 2, encoding="utf-8")
            active = (
                "1\n"
                "Properties=species:S:1:pos:R:3:vel:R:3:forces:R:3:uncertainty:R:1 uncertainty=1\n"
                "C 0 0 0 0.1 0.2 0.3 0 0 0 1\n"
            )
            (root / "active.xyz").write_text(active * 2, encoding="utf-8")
            (root / "active.out").write_text("1 1\n2 1\n", encoding="utf-8")
            spec = {
                "name": "active_uncertainty",
                "observers": 2,
                "check_interval": 1,
                "threshold": 0.0,
                "expected_selection": "all",
                "has_velocity": True,
                "has_force": True,
                "has_atom_uncertainty": True,
                "rtol": 1.0e-12,
                "atol": 1.0e-12,
            }
            post_checks.validate_spec(spec, "spec")
            result = post_checks.run([spec], root)
            self.assertEqual(result[0]["name"], "active_uncertainty")


if __name__ == "__main__":
    unittest.main()
