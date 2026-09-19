import shutil
import tempfile
import time
import unittest
from pathlib import Path

import run_regression as runner


class RelationRunnerTests(unittest.TestCase):
    def setUp(self):
        runner.WORK_ROOT.mkdir(parents=True, exist_ok=True)
        self.invocation_root = Path(
            tempfile.mkdtemp(prefix="relation-test-", dir=runner.WORK_ROOT)
        )
        self.addCleanup(shutil.rmtree, self.invocation_root, True)
        self.recorder = runner.DifferenceRecorder(self.invocation_root)

    def make_case(self, case_id, outputs, status="PASS"):
        runs = {}
        for role in ("baseline", "candidate"):
            workdir = self.invocation_root / "cases" / case_id / role
            workdir.mkdir(parents=True)
            for output, data in outputs.items():
                output_path = workdir / output
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(data)
            runs[role] = {"workdir": str(workdir)}
        return {
            "id": case_id,
            "status": status,
            "runs": runs,
            "workdirs_retained": True,
        }

    def package_source(self, path):
        return "package:" + path.relative_to(runner.PACKAGE_ROOT).as_posix()

    def minimal_run_case(self, case_id, executable_text, mutable_inputs=None):
        source_dir = self.invocation_root / "sources" / case_id
        source_dir.mkdir(parents=True)
        model = source_dir / "model.xyz"
        model.write_text("1\nminimal\nCu 0 0 0\n", encoding="utf-8")
        potential = source_dir / "potential.txt"
        potential.write_text("unused by fake executable\n", encoding="utf-8")
        run_input = source_dir / "run.in"
        run_input.write_text("potential potential.txt\n", encoding="utf-8")
        executable = source_dir / "fake_gpumd.py"
        executable.write_text(executable_text, encoding="utf-8")
        executable.chmod(0o755)
        case = {
            "id": case_id,
            "fixture": "minimal",
            "input": self.package_source(run_input),
            "outputs": [],
            "expect": "failure",
        }
        if mutable_inputs is not None:
            case["mutable_inputs"] = mutable_inputs
        manifest = {
            "defaults": {"timeout_s": 1.0},
            "fixtures": {
                "minimal": {
                    "stage": [
                        {"source": self.package_source(model), "target": "model.xyz"},
                        {
                            "source": self.package_source(potential),
                            "target": "potential.txt",
                        },
                    ]
                }
            },
        }
        return executable, case, manifest

    def test_equal_and_concat_relations(self):
        case_results = {
            "part_a": self.make_case("part_a", {"data.out": b"alpha\n"}),
            "part_b": self.make_case("part_b", {"data.out": b"beta\n"}),
            "combined": self.make_case(
                "combined", {"data.out": b"alpha\nbeta\n", "same.out": b"state\n"}
            ),
            "control": self.make_case("control", {"same.out": b"state\n"}),
        }
        relations = [
            {
                "id": "ordered_concat",
                "description": "ordered byte concatenation",
                "roles": ["candidate"],
                "operation": "concat",
                "parts": [
                    {"case": "part_a", "output": "data.out"},
                    {"case": "part_b", "output": "data.out"},
                ],
                "result": {"case": "combined", "output": "data.out"},
            },
            {
                "id": "same_state",
                "description": "equal state",
                "roles": ["baseline", "candidate"],
                "operation": "equal",
                "members": [
                    {"case": "control", "output": "same.out"},
                    {"case": "combined", "output": "same.out"},
                ],
            },
        ]
        results = runner.evaluate_relations(
            relations, set(case_results), case_results, self.recorder
        )
        self.assertEqual([result["status"] for result in results], ["PASS", "PASS"])

    def test_relation_is_skipped_until_all_cases_are_selected(self):
        case_results = {
            "one": self.make_case("one", {"x.out": b"same\n"}),
        }
        relation = {
            "id": "needs_two",
            "description": "requires both controls",
            "roles": ["candidate"],
            "operation": "equal",
            "members": [
                {"case": "one", "output": "x.out"},
                {"case": "two", "output": "x.out"},
            ],
        }
        result = runner.evaluate_relation(
            relation, {"one"}, case_results, self.recorder
        )
        self.assertEqual(result["status"], "SKIP")
        self.assertEqual(result["missing_case_ids"], ["two"])

    def test_relation_failure_retains_only_implicated_passing_cases(self):
        case_results = {
            "left": self.make_case("left", {"x.out": b"left\n"}),
            "right": self.make_case("right", {"x.out": b"right\n"}),
            "unrelated": self.make_case("unrelated", {"x.out": b"other\n"}),
        }
        relation = {
            "id": "mismatch",
            "description": "intentional mismatch",
            "roles": ["candidate"],
            "operation": "equal",
            "members": [
                {"case": "left", "output": "x.out"},
                {"case": "right", "output": "x.out"},
            ],
        }
        relation_result = runner.evaluate_relation(
            relation, set(case_results), case_results, self.recorder
        )
        self.assertEqual(relation_result["status"], "FAIL")
        self.assertTrue((self.invocation_root / "diffs").is_dir())

        results = list(case_results.values())
        runner.cleanup_case_workdirs(
            results, [relation_result], self.invocation_root, keep_all=False
        )
        self.assertTrue((self.invocation_root / "cases" / "left").is_dir())
        self.assertTrue((self.invocation_root / "cases" / "right").is_dir())
        self.assertFalse((self.invocation_root / "cases" / "unrelated").exists())
        self.assertTrue(case_results["left"]["workdirs_retained"])
        self.assertFalse(case_results["unrelated"]["workdirs_retained"])

    def test_stream_comparison_is_lossless(self):
        self.assertEqual(runner.normalize_stdout(b"value\xff\r\n"), b"value\xff\r\n")
        self.assertNotEqual(
            runner.normalize_stdout(b"value\xff\r\n"),
            runner.normalize_stdout(b"value\xfe\n"),
        )
        with self.assertRaises(runner.ComparisonError):
            runner.compare_exact_bytes(
                b"value\xff\r\n",
                b"value\xfe\n",
                "invalid-stream-bytes",
                self.recorder,
            )

    def test_numeric_comparison_preserves_structure_and_reports_all_outputs(self):
        runner.compare_numeric_text(
            "value 1.0000000\n",
            "value 1.0000001\n",
            "within-tolerance",
            0.0,
            2.0e-7,
            self.recorder,
        )
        with self.assertRaises(runner.ComparisonError):
            runner.compare_numeric_text(
                "energy=-16011.0\n",
                "energy=-16010.999\n",
                "energy-drift",
                0.0,
                5.0e-6,
                self.recorder,
            )

        case_id = "numeric_structure"
        runs = {}
        for role in ("baseline", "candidate"):
            workdir = self.invocation_root / "cases" / case_id / role
            capture_dir = self.invocation_root / "captures" / case_id / role
            workdir.mkdir(parents=True)
            capture_dir.mkdir(parents=True)
            (capture_dir / "stdout.txt").write_bytes(b"")
            (capture_dir / "stderr.txt").write_bytes(b"")
            runs[role] = {
                "workdir": str(workdir),
                "returncode": 0,
                "generated_files": ["exact.out", "numeric.out"],
                "stdout_path": str(capture_dir / "stdout.txt"),
                "stderr_path": str(capture_dir / "stderr.txt"),
            }
        baseline_dir = Path(runs["baseline"]["workdir"])
        candidate_dir = Path(runs["candidate"]["workdir"])
        (baseline_dir / "numeric.out").write_bytes(b"1\r\n")
        (candidate_dir / "numeric.out").write_bytes(b"1\n")
        (baseline_dir / "exact.out").write_bytes(b"left\n")
        (candidate_dir / "exact.out").write_bytes(b"right\n")
        case = {
            "id": case_id,
            "comparisons": {
                "numeric.out": {
                    "mode": "numeric",
                    "rtol": 0.0,
                    "atol": 1.0,
                    "reason": "test-only tolerance",
                }
            },
        }
        with self.assertRaises(runner.ComparisonError) as context:
            runner.compare_run_pair(
                runs["baseline"], runs["candidate"], case, self.recorder
            )
        self.assertIn("exact.out", str(context.exception))
        self.assertIn("numeric.out", str(context.exception))
        self.assertEqual(len(list((self.invocation_root / "diffs").glob("*.diff"))), 2)

    def test_timeout_is_bounded_and_private_capture_exposes_name_collision(self):
        executable, case, manifest = self.minimal_run_case(
            "timeout_case",
            """#!/usr/bin/env python3
import os
import time
open("stdout.txt", "wb").write(b"executable-owned\\n")
pid = os.fork()
if pid == 0:
    os.setsid()
    open("child.pid", "w").write(str(os.getpid()))
    time.sleep(10)
    os._exit(0)
time.sleep(10)
""",
        )
        manifest["defaults"]["timeout_s"] = 0.05
        started = time.monotonic()
        result = runner.run_once(
            executable,
            case,
            manifest,
            "baseline",
            self.invocation_root,
            runner.PACKAGE_ROOT,
            {},
            1.0,
        )
        elapsed = time.monotonic() - started
        self.assertTrue(result["timed_out"])
        self.assertLess(elapsed, 0.8)
        self.assertIn("stdout.txt", result["generated_files"])
        self.assertNotEqual(
            Path(result["stdout_path"]), Path(result["workdir"]) / "stdout.txt"
        )

        child_pid_path = Path(result["workdir"]) / "child.pid"
        if child_pid_path.is_file():
            child_pid = int(child_pid_path.read_text(encoding="utf-8"))
            stat_path = Path("/proc") / str(child_pid) / "stat"
            if stat_path.is_file():
                state = stat_path.read_text(encoding="utf-8").split(")", 1)[1].split()[0]
                self.assertEqual(state, "Z")

    def test_mutable_input_may_be_modified_but_not_deleted(self):
        executable, case, manifest = self.minimal_run_case(
            "deleted_input_case",
            """#!/usr/bin/env python3
from pathlib import Path
Path("run.in").unlink()
raise SystemExit(1)
""",
            mutable_inputs=["run.in"],
        )
        result = runner.run_once(
            executable,
            case,
            manifest,
            "baseline",
            self.invocation_root,
            runner.PACKAGE_ROOT,
            {},
            1.0,
        )
        self.assertIn("missing:run.in", result["unauthorized_input_changes"])

    def test_mutable_inputs_are_compared_across_roles(self):
        case_id = "mutable_comparison"
        runs = {}
        for role, content in (("baseline", b"left\n"), ("candidate", b"right\n")):
            workdir = self.invocation_root / "cases" / case_id / role
            capture_dir = self.invocation_root / "captures" / case_id / role
            workdir.mkdir(parents=True)
            capture_dir.mkdir(parents=True)
            (workdir / "run.in").write_bytes(content)
            stdout_path = capture_dir / "stdout.txt"
            stderr_path = capture_dir / "stderr.txt"
            stdout_path.write_bytes(b"")
            stderr_path.write_bytes(b"")
            runs[role] = {
                "workdir": str(workdir),
                "returncode": 0,
                "generated_files": [],
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
            }

        case = {"id": case_id, "mutable_inputs": ["run.in"]}
        with self.assertRaises(runner.ComparisonError) as context:
            runner.compare_run_pair(
                runs["baseline"], runs["candidate"], case, self.recorder
            )
        self.assertIn("mutable:run.in", str(context.exception))

    def test_expected_failure_still_enforces_output_inventory(self):
        workdir = self.invocation_root / "cases" / "failure_inventory" / "baseline"
        capture_dir = self.invocation_root / "captures" / "failure_inventory" / "baseline"
        workdir.mkdir(parents=True)
        capture_dir.mkdir(parents=True)
        (workdir / "undeclared.out").write_text("unexpected\n", encoding="utf-8")
        stdout_path = capture_dir / "stdout.txt"
        stderr_path = capture_dir / "stderr.txt"
        stdout_path.write_bytes(b"")
        stderr_path.write_bytes(b"expected failure\n")
        case = {
            "id": "failure_inventory",
            "expect": "failure",
            "expected_returncode": 1,
            "stderr_contains": "expected failure",
            "outputs": [],
        }
        result = {
            "role": "baseline",
            "workdir": str(workdir),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "returncode": 1,
            "timed_out": False,
            "unsafe_workdir_entries": [],
            "unauthorized_input_changes": [],
            "generated_files": ["undeclared.out"],
        }
        with self.assertRaises(runner.ComparisonError):
            runner.validate_run_result(result, case)

    def test_candidate_only_contract_does_not_start_baseline(self):
        baseline, case, manifest = self.minimal_run_case(
            "candidate_only_case",
            """#!/usr/bin/env python3
from pathlib import Path
Path("baseline_was_run").write_text("unexpected\\n")
raise SystemExit(99)
""",
        )
        candidate = baseline.with_name("candidate_gpumd.py")
        candidate.write_text(
            """#!/usr/bin/env python3
import sys
sys.stderr.write("expected candidate diagnostic\\n")
raise SystemExit(1)
""",
            encoding="utf-8",
        )
        candidate.chmod(0o755)
        case.update(
            {
                "description": "candidate-only execution contract",
                "candidate_only": True,
                "expected_returncode": 1,
                "stderr_contains": "expected candidate diagnostic",
            }
        )

        result = runner.execute_case(
            case,
            manifest,
            baseline,
            candidate,
            self.invocation_root,
            runner.PACKAGE_ROOT,
            {},
            1.0,
            self.recorder,
        )

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["execution_contract"], "candidate_only")
        self.assertEqual(set(result["runs"]), {"candidate"})
        self.assertEqual(
            result["not_run"],
            {"baseline": "not run by candidate_only validation contract"},
        )
        self.assertFalse(result["cross_version_compared"])
        self.assertFalse(
            (self.invocation_root / "cases" / case["id"] / "baseline").exists()
        )

    def test_generated_symlinks_are_rejected(self):
        workdir = self.invocation_root / "unsafe"
        workdir.mkdir()
        (workdir / "outside").symlink_to(Path("/etc/passwd"))
        files, unsafe = runner.inspect_workdir(workdir)
        self.assertEqual(files, set())
        self.assertEqual(unsafe, ["symlink:outside"])

    def test_structural_parsers_reject_nonfinite_or_nonnumeric_data(self):
        with self.assertRaises(runner.ComparisonError):
            runner.validate_xyz_output("1\nframe\nCu x 0 0\n", "xyz")
        with self.assertRaises(runner.ComparisonError):
            runner.validate_csv_output("a,b\n1,nan\n", "csv")
        with self.assertRaises(runner.ComparisonError):
            runner.validate_yaml_output("value: inf\n", "yaml")
        with self.assertRaises(runner.ComparisonError):
            runner.compare_numeric_text(
                "1e999\n", "1e999\n", "numeric", 0.0, 0.0, self.recorder
            )
        text_output = self.invocation_root / "nonfinite.out"
        text_output.write_text("1.0 nan\n", encoding="utf-8")
        with self.assertRaises(runner.ComparisonError):
            runner.validate_generated_output(text_output, "nonfinite.out", "text")
        text_output.write_text("1 2\n", encoding="utf-8")
        runner.validate_generated_output(text_output, "ordinary.out", "text")
        text_output.write_text("1e999\n", encoding="utf-8")
        with self.assertRaises(runner.ComparisonError):
            runner.validate_generated_output(text_output, "overflow.out", "text")


if __name__ == "__main__":
    unittest.main()
