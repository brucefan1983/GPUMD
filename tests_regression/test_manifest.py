import hashlib
import json
import os
import re
import subprocess

import run_regression as runner
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
RUNNER = PACKAGE_ROOT / "run_regression.py"
MANIFEST = PACKAGE_ROOT / "manifest.json"


def find_repo_root() -> Path:
    """Find the GPUMD tree used to validate repo: manifest sources."""
    candidates = []
    configured = os.environ.get("GPUMD_REPO_ROOT")
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            PACKAGE_ROOT.parent,
            PACKAGE_ROOT.parent / "gpumd_repo",
            Path.cwd(),
        ]
    )
    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "src").is_dir() and (candidate / "potentials").is_dir():
            return candidate
    raise AssertionError(
        "Cannot find a GPUMD repository. Set GPUMD_REPO_ROOT to a tree "
        "containing src/ and potentials/."
    )


def test_manifest_validation_cli():
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--repo-root",
            str(find_repo_root()),
            "--check-manifest",
        ],
        cwd=PACKAGE_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert re.search(r"Manifest OK: \d+ cases, \d+ suites", result.stdout), result.stdout


def test_manifest_suite_and_input_style_contract():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cases = manifest["cases"]
    assert cases

    case_ids = [case["id"] for case in cases]
    assert len(case_ids) == len(set(case_ids))

    full_ids = {case["id"] for case in cases if "full" in case["suites"]}
    quick_ids = {case["id"] for case in cases if "quick" in case["suites"]}
    assert full_ids == set(case_ids)
    assert quick_ids
    assert quick_ids < full_ids

    focused_suites = {
        suite
        for case in cases
        for suite in case["suites"]
        if suite not in {"quick", "full"}
    }
    assert focused_suites

    gpumd_cases = [case for case in cases if case.get("program", "gpumd") == "gpumd"]
    nep_cases = [case for case in cases if case.get("program", "gpumd") == "nep"]
    assert gpumd_cases
    assert nep_cases
    assert all("input_style" not in case for case in nep_cases)

    styles = {case["input_style"] for case in gpumd_cases}
    assert {"canonical", "compatibility"} <= styles
    assert styles <= {"canonical", "compatibility", "intentional_invalid"}
    for case in gpumd_cases:
        if case["input_style"] != "intentional_invalid":
            continue
        assert case["expect"] == "failure"
        assert case["expected_returncode"] == 1
        assert "full" in case["suites"]

    for case in cases:
        assert not {"role_expectations", "compare_cross_version", "candidate_only"}.intersection(case)


def test_manifest_rejects_asymmetric_contracts():
    for key, value in (
        ("role_expectations", {"baseline": {"expect": "failure"}, "candidate": {"expect": "success"}}),
        ("compare_cross_version", False),
        ("candidate_only", True),
    ):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        manifest["cases"][0][key] = value
        try:
            runner.validate_manifest(manifest, find_repo_root())
        except runner.ConfigurationError as exc:
            assert "unknown key(s)" in str(exc) and key in str(exc), str(exc)
        else:
            raise AssertionError(f"Asymmetric contract {key} was accepted")


def test_manifest_uses_single_run_byte_exact_defaults():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert set(manifest["defaults"]) == {"timeout_s"}

    deprecated_case_keys = {
        "allow_cross_version_stdout_reordering",
        "rtol",
        "atol",
    }
    for case in manifest["cases"]:
        assert not deprecated_case_keys.intersection(case)
        comparisons = case.get("comparisons", {})
        assert isinstance(comparisons, dict)


def test_manifest_cross_case_relations_are_declarative():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cases = {case["id"]: case for case in manifest["cases"]}
    relations = manifest["relations"]
    assert relations
    assert len({relation["id"] for relation in relations}) == len(relations)
    assert {relation["operation"] for relation in relations} <= {"equal", "concat", "numeric_equal"}

    for relation in relations:
        assert relation["roles"] == ["baseline", "candidate"]
        if relation["id"].startswith("pppm_default_equivalence_"):
            assert relation["operation"] == "numeric_equal"
        if relation["operation"] in ("equal", "numeric_equal"):
            references = relation["members"]
            assert len(references) >= 2
        else:
            references = relation["parts"] + [relation["result"]]
            assert relation["parts"]
        for reference in references:
            assert reference["case"] in cases
            assert reference["output"] in cases[reference["case"]]["outputs"]
            assert "full" in cases[reference["case"]]["suites"]
            assert not cases[reference["case"]].get("candidate_only", False)

    neighbor_relation = next(
        relation
        for relation in relations
        if relation["id"] == "neighbor_alias_translation_invariance"
    )
    assert neighbor_relation["roles"] == ["baseline", "candidate"]
    assert neighbor_relation["operation"] == "equal"
    assert neighbor_relation["members"] == [
        {"case": "neighbor_alias_aliased", "output": "thermo.out"},
        {"case": "neighbor_alias_control", "output": "thermo.out"},
    ]


def test_runner_exposes_explicit_external_paths():
    result = subprocess.run(
        [sys.executable, str(RUNNER), "--help"],
        cwd=PACKAGE_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    for option in (
        "--repo-root",
        "--baseline",
        "--candidate",
        "--baseline-nep",
        "--candidate-nep",
        "--suite",
        "--case",
        "--check-manifest",
        "--keep-all",
    ):
        assert option in result.stdout
    assert "--repeats" not in result.stdout


def test_manifest_accounts_for_every_case_input():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    declared = set()
    for case in manifest["cases"]:
        source = case["input"]
        assert source.startswith("package:cases/"), source
        declared.add(source.removeprefix("package:"))
    present = {
        path.relative_to(PACKAGE_ROOT).as_posix()
        for path in (PACKAGE_ROOT / "cases").rglob("*.in")
    }
    assert declared == present


def test_package_fixture_hashes_are_pinned():
    expected = {
        "fixtures/potentials/temperature_F_nep.txt":
            "261e236885f88b785d832123ce8d7d3a8dbb220dec962ebbec26654ca481ee0b",
        "fixtures/systems/temperature_F_512.xyz":
            "44a76a943b6e1c419706e0f0b257200d38319194c80a60c589da977f7620b57c",
        "fixtures/potentials/C_2022_NEP4_MODIFIED.txt":
            "7319c9a7c21aa4cc66ba66c91cf4d4540dade2a28acab90f2529b39c26bbd459",
        "fixtures/potentials/neighbor_alias_lj.txt":
            "d058a4054474f8cc1daa260942e05e53b53654f0339bde6f72d2af331eba2a09",
        "fixtures/systems/neighbor_alias_aliased.xyz":
            "3a86e6e3b62bc15935e3e253be232dc6f2c44719d67b89823f6e019f4946e5a6",
        "fixtures/systems/neighbor_alias_control.xyz":
            "a3ab3c1eff0e29c96fdb5e706d3ae9412e9c79a7b2ac6adbf86faef27a96cfdb",
        "fixtures/systems/replicate_lj_with_velocity.xyz":
            "de3338e0caf391f62d51bb86b242f7fb7eaaec53e78e73cb2e5a764a43e3ebcd",
        "fixtures/phonon/cu_fcc_single_atom.xyz":
            "d1e93d2300c56307dee18b8a567f5f0e8b8ec38420066ad810bef5b8e3eab91d",
        "fixtures/systems/deposition_lj_no_velocity.xyz":
            "a118421c14ee9d872af2c7d7e8aea0a2855c27f491eed0147eeb4b9f06ae0f67",
        "fixtures/systems/deposition_lj_with_velocity.xyz":
            "fdb3374c3df124ffb9e14fd6b1273d96c7015b9908663298dea9d2961503fa91",
        "fixtures/deposition/add_atoms_ar.txt":
            "3f31cf5c86aa3e15f912084def616cdd67b7e7c71c347441cabb612a7bbe23cd",
        "fixtures/systems/qnep_batio3_group0.xyz":
            "f4a0e92a1b070938edd4c4b40a689aea4a8d0a81445aa4be7840e32888719c7a",
        "fixtures/systems/carbon_grouped.xyz":
            "f418c615f770a2b4f91fb08ac9771341bd7f13eac1a8b25e65e66781f104a1bf",
        "fixtures/systems/bilayer_graphene_primitive.xyz":
            "816a1a4d18aa3e15d4b758b843ffb34cd05c604674acc642f5ee39f4f61e38ea",
        "fixtures/systems/bilayer_mos2_primitive.xyz":
            "d86495b5cc3db5d98bfc678b1d97772c3ce3686d82ded53efc19e57fe23a4580",
        "fixtures/potentials/graphene_nep_ilp.txt":
            "fd19e1228ee6627802f7be6cfe4f85e523885db36ff18ffd1e383f782a0e72f0",
        "fixtures/potentials/graphene_tersoff_ilp.txt":
            "389896b88a0a901a82160f407d3b7d2146c3e0c5972c1632dcb8fd88b43d33d2",
        "fixtures/potentials/graphene_tersoff_1988_params.txt":
            "baf64d6b7f5bbfc32f7af38b64b601af9ce4b4417e8e6c2af320d00610f4f9ef",
        "fixtures/training/pbte.xyz":
            "3b9f1d55fe61def164ab87370dce66b760d6740d3c1cc35ec6bc3db057572c5c",
        "fixtures/training/pbte_water_mixed.xyz":
            "78cda019ae1243eda4764afc98b53abe287e4a84f2aeaf2c53d37a01b0c04743",
        "fixtures/training/tnep_120.xyz":
            "0d98711fcbae5cc0af3df245fb6dd3117d7b1213289790152d039c1a81ad8aee",
        "fixtures/training/water_12_fff.xyz":
            "27a92bdbb864c8c1455eb770666283c9e77c1ee663cb02f9ed1b75beb950692a",
    }
    expected["fixtures/systems/pppm_batio3_15p9.xyz"] = "6f81f06e8973b52795f8f1bc5f3e49651cc4c83fcbed0cc9b16d87f642476ed8"
    expected["fixtures/systems/pppm_batio3_8.xyz"] = "d725a5e43e6596025bd2171a0be187b6b392332a6046504759005fd97554ebb5"
    expected["fixtures/systems/pppm_batio3_above8.xyz"] = "e25849411b53c9361eb440b4d73f0bdccfbb72c70be202fc4445ae70541aa7dc"
    expected["fixtures/systems/pppm_batio3_tilt.xyz"] = "c4d06e41bccf3b33c341be6a1640326f785eefcc49ea956c8d41ffd905054375"
    expected["fixtures/systems/pppm_batio3_shear45.xyz"] = "43170c26dcfac9f4268fd1f1ed8966ab0c95a71221da2e4858a2560f6168a488"
    expected["fixtures/systems/pppm_batio3_mesh_limit.xyz"] = "b0218360043739fe7663b93954daa366f01c5eeea26aac8b241ebf11f69361b3"
    for relative_path, expected_digest in expected.items():
        digest = hashlib.sha256((PACKAGE_ROOT / relative_path).read_bytes()).hexdigest()
        assert digest == expected_digest, relative_path


def test_successful_nep_cases_declare_neighbor_output():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    nep_tags = {
        "potential:nep",
        "potential:qnep",
        "potential:multiple-nep",
        "potential:nep89",
        "potential:nep4-temperature",
        "potential:nep-ilp",
    }
    for case in manifest["cases"]:
        if case.get("program", "gpumd") != "gpumd":
            continue
        if not nep_tags.intersection(case.get("covers", [])):
            continue
        if case["id"] in {"pppm_no_force", "ewald_no_force"}:
            assert case["outputs"] == []
            assert case["stdout_contains"]
        elif case["expect"] == "success":
            assert "neighbor.out" in case["outputs"], case["id"]


def test_training_cases_use_nep_program_and_minimal_inputs():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    training_cases = [
        case for case in manifest["cases"] if "training" in case["suites"]
    ]
    assert len(training_cases) == 22
    assert all(case.get("program") == "nep" for case in training_cases)
    assert all("full" in case["suites"] for case in training_cases)
    assert all(
        case["outputs"] == ["loss.out", "nep.txt", "nep.restart"]
        for case in training_cases
    )
    assert all("stage" not in case for case in training_cases)
    assert all("mutable_inputs" not in case for case in training_cases)

    forbidden = {"population", "nep_compile"}
    for case in training_cases:
        input_path = PACKAGE_ROOT / case["input"].removeprefix("package:")
        commands = []
        for raw_line in input_path.read_text(encoding="utf-8").splitlines():
            fields = raw_line.split("#", 1)[0].split()
            if fields:
                commands.append(fields)
        keywords = [fields[0] for fields in commands]
        assert keywords.count("type") == 1, case["id"]
        assert keywords.count("batch") == 1, case["id"]
        assert keywords.count("generation") == 1, case["id"]
        assert keywords.count("output_interval") == 1, case["id"]
        assert not forbidden.intersection(keywords), case["id"]
        generation = next(fields for fields in commands if fields[0] == "generation")
        output_interval = next(
            fields for fields in commands if fields[0] == "output_interval"
        )
        assert generation == ["generation", "10"], case["id"]
        assert output_interval == ["output_interval", "10"], case["id"]

    mixed_cases = [case for case in training_cases if case["id"].endswith("_mixed")]
    assert len(mixed_cases) == 4
    for case in mixed_cases:
        input_path = PACKAGE_ROOT / case["input"].removeprefix("package:")
        text = input_path.read_text(encoding="utf-8")
        assert "type 4 Pb Te H O" in text, case["id"]

    tnep_batch_cases = {
        "train_tnep_dipole_batch",
        "train_tnep_pol_batch",
    }
    for case in training_cases:
        if case["id"] not in tnep_batch_cases:
            continue
        input_path = PACKAGE_ROOT / case["input"].removeprefix("package:")
        assert "batch 61" in input_path.read_text(encoding="utf-8"), case["id"]


def test_behavior_contract_cases_are_full_differential_cases():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cases = {case["id"]: case for case in manifest["cases"]}
    behavior_contract_ids = {
        "neighbor_alias_aliased",
        "neighbor_alias_control",
        "compute_chunk_multiple",
        "phonon_comments_before_replicate",
        "phonon_single_atom_replicate",
        "hnemdec_before_ensemble",
        "hnemdec_invalid_type",
        "observer_average_two_runs",
        "replicate_default_velocity",
        "replicate_velocity_unseeded",
        "replicate_model_velocity",
        "replicate_after_potential",
        "replicate_after_velocity",
        "replicate_after_run",
        "replicate_duplicate",
        "dftd3_duplicate",
        "kspace_duplicate",
    }
    for case_id in behavior_contract_ids:
        case = cases[case_id]
        assert "full" in case["suites"]
        assert "role_expectations" not in case
        assert "compare_cross_version" not in case

    assert cases["replicate_after_run"]["stdout_contains"] == (
        "1 steps completed."
    )
    assert cases["replicate_duplicate"]["stdout_contains"] == (
        "Replicate cell by 1 * 1 * 1."
    )


def test_default_runtime_consumers_are_full_cases():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cases = {case["id"]: case for case in manifest["cases"]}
    required = {
        "two_run_state_contract",
        "temperature_nep_two_run_ramp",
        "pimd_nvt",
        "heat_hybrid",
        "ti_liquid",
        "qnep_actions_before_potential",
        "observer_average_then_observe_control",
        "qnep_force_accessors",
        "rdf_angular_classical",
        "active_multi_nep",
        "observer_fix_move_grouping",
        "minimize_fixed_group",
        "ilp_nep_compute",
        "ilp_tersoff_compute",
        "ilp_tmd_sw_compute",
    }
    assert required <= cases.keys()
    for case_id in required:
        case = cases[case_id]
        assert "full" in case["suites"]
        assert case["expect"] == "success"
        assert "role_expectations" not in case


def test_parsing_validation_cases_match_the_accepted_baseline():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cases = {case["id"]: case for case in manifest["cases"]}
    parsing_cases = {
        case_id: case
        for case_id, case in cases.items()
        if "parsing_validation" in case["suites"]
    }
    assert len(parsing_cases) == 37

    candidate_only_ids = {
        case_id
        for case_id, case in parsing_cases.items()
        if case.get("candidate_only", False)
    }
    assert candidate_only_ids == set()

    role_case_ids = {
        case_id
        for case_id, case in parsing_cases.items()
        if "role_expectations" in case
    }
    assert role_case_ids == set()
    for case in parsing_cases.values():
        if case["input_style"] == "intentional_invalid":
            assert case["expect"] == "failure"
            assert case["expected_returncode"] == 1
            assert case.get("stdout_contains") or case.get("stderr_contains")
    for case_id in (
        "dftd3_single_late",
        "kspace_single_late",
        "dftd3_kspace_one_each",
    ):
        assert parsing_cases[case_id]["expect"] == "success"
        assert "role_expectations" not in parsing_cases[case_id]

    relation = next(
        relation
        for relation in manifest["relations"]
        if relation["id"] == "msd_option_order_equality"
    )
    assert relation["roles"] == ["baseline", "candidate"]
    assert relation["members"] == [
        {"case": "msd_options_canonical", "output": "msd.out"},
        {"case": "msd_options_reversed", "output": "msd.out"},
    ]



def test_semantic_postchecks_preserve_byte_exact_cross_version_outputs():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    post_checked = [case for case in manifest["cases"] if case.get("post_checks")]
    assert post_checked
    for case in post_checked:
        # Semantic tolerances are independent post-processing oracles only.
        # Baseline/candidate output comparison stays byte-exact.
        assert case.get("comparisons", {}) == {}
        assert "role_expectations" not in case
        assert case.get("compare_cross_version", True) is True


def test_regression_has_no_dependency_on_migrated_gpumd_test_directories():
    text = MANIFEST.read_text(encoding="utf-8")
    for directory in (
        "active",
        "dump_dipole",
        "dump_observer",
        "dump_polarizability",
        "msd",
        "ti-liquid",
    ):
        assert f"repo:tests/gpumd/{directory}/" not in text


def test_numeric_exceptions_are_explicit_per_output():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    actual = {
        case["id"]: {
            output: (comparison["rtol"], comparison["atol"])
            for output, comparison in case.get("comparisons", {}).items()
            if comparison["mode"] == "numeric"
        }
        for case in manifest["cases"]
        if any(
            comparison["mode"] == "numeric"
            for comparison in case.get("comparisons", {}).values()
        )
    }
    expected = {
        "qnep_ewald_future_bec": {
            "bec.xyz": (0.0, 3e-6),
            "dpdt.out": (0.0, 2e-7),
        },
        "qnep_actions_before_potential": {
            "bec.xyz": (0.0, 3e-6),
            "dpdt.out": (0.0, 2e-7),
        },
        "qnep_force_accessors": {
            "qnep_force.xyz": (0.0, 2e-7),
        },
        "dftd3_single_late": {
            "restart.xyz": (0.0, 1e-17),
        },
        "qnep_pppm_future_bec": {
            "bec.xyz": (0, 1e-5),
            "thermo.out": (0, 1e-7),
            "dpdt.out": (0, 2e-6),
        },
    }

    pppm_same_mesh_ids = {
        "pppm_default",
        "pppm_explicit",
        "pppm_round_power2",
        "pppm_mode2",
        "pppm_peratom",
        "pppm_mode2_peratom",
        "pppm_deform_same_mesh",
        "pppm_shear_same_mesh",
        "pppm_npt",
        "pppm_two_runs",
        "pppm_spacing_one",
        "pppm_spacing_late",
        "pppm_spacing_fine",
        "pppm_spacing_coarse",
        "pppm_round_exact",
        "pppm_round_above",
        "pppm_round_5_7",
        "pppm_grow_shrink",
        "pppm_grow_shrink_peratom",
        "pppm_triclinic_mesh",
    }
    for case_id in pppm_same_mesh_ids:
        expected[case_id] = {"thermo.out": (0, 1e-7), "state.xyz": (0, 2e-5)}
    assert actual == expected


if __name__ == "__main__":
    tests = (
        test_manifest_validation_cli,
        test_manifest_suite_and_input_style_contract,
        test_manifest_rejects_asymmetric_contracts,
        test_manifest_uses_single_run_byte_exact_defaults,
        test_manifest_cross_case_relations_are_declarative,
        test_runner_exposes_explicit_external_paths,
        test_manifest_accounts_for_every_case_input,
        test_package_fixture_hashes_are_pinned,
        test_successful_nep_cases_declare_neighbor_output,
        test_training_cases_use_nep_program_and_minimal_inputs,
        test_behavior_contract_cases_are_full_differential_cases,
        test_default_runtime_consumers_are_full_cases,
        test_parsing_validation_cases_match_the_accepted_baseline,
        test_semantic_postchecks_preserve_byte_exact_cross_version_outputs,
        test_regression_has_no_dependency_on_migrated_gpumd_test_directories,
        test_numeric_exceptions_are_explicit_per_output,
    )
    for test in tests:
        test()
    print(f"PASS: {len(tests)} manifest/runner tests")
