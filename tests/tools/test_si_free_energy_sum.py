import pathlib
import subprocess
import sys

import pytest
import yaml


REPO = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tools" / "si_free_energy_sum.py"


def write_yaml(path, data):
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def read_yaml(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def assert_summary(data):
    assert set(data) == {"stage1", "stage2", "F_ref", "F_target", "G_target", "T", "V", "P"}
    assert set(data["stage1"]) == {"W_forward", "W_backward", "delta_F"}
    assert set(data["stage2"]) == {"W_forward", "W_backward", "delta_F"}
    assert data["stage1"]["W_forward"] == pytest.approx(0.45)
    assert data["stage1"]["W_backward"] == pytest.approx(-0.05)
    assert data["stage1"]["delta_F"] == pytest.approx(0.25)
    assert data["stage2"]["W_forward"] == pytest.approx(0.95)
    assert data["stage2"]["W_backward"] == pytest.approx(-0.55)
    assert data["stage2"]["delta_F"] == pytest.approx(0.75)
    assert data["F_ref"] == pytest.approx(-1.5)
    assert data["F_target"] == pytest.approx(-0.5)
    assert data["G_target"] == pytest.approx(-0.42)
    assert data["T"] == pytest.approx(3000.0)
    assert data["V"] == pytest.approx(8.0)
    assert data["P"] == pytest.approx(0.01)


def base_stage(stage, delta_f):
    return {
        "stage": stage,
        "T": 3000.0,
        "V": 8.0,
        "P": 0.01,
        "N_total": 4,
        "spring_species": ["O"],
        "uf_self_pairs": [{"element_i": "H", "element_j": "H", "p": 25.0, "sigma": 1.0}],
        "uf_cross_pairs": [{"element_i": "O", "element_j": "H", "p": 10.0, "sigma": 1.0}],
        "W_forward": delta_f + 0.2,
        "W_backward": 0.2 - delta_f,
        "delta_F": delta_f,
        "F_Einstein": -0.5,
        "F_UF_self": -1.0,
        "F_ref": -1.5,
    }


def test_combines_two_stage_yaml_files(tmp_path):
    stage1 = tmp_path / "stage1.yaml"
    stage2 = tmp_path / "stage2.yaml"
    output = tmp_path / "summary.yaml"
    write_yaml(stage1, base_stage(1, 0.25))
    write_yaml(stage2, base_stage(2, 0.75))

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(stage1), str(stage2), "-o", str(output)],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    data = read_yaml(output)
    assert_summary(data)


def test_orders_stage_output_when_input_files_are_reversed(tmp_path):
    stage1 = tmp_path / "stage1.yaml"
    stage2 = tmp_path / "stage2.yaml"
    output = tmp_path / "summary.yaml"
    write_yaml(stage1, base_stage(1, 0.25))
    write_yaml(stage2, base_stage(2, 0.75))

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(stage2), str(stage1), "-o", str(output)],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    data = read_yaml(output)
    assert_summary(data)


def test_rejects_duplicate_or_non_1_2_stage_values(tmp_path):
    stage1 = tmp_path / "stage1.yaml"
    stage2 = tmp_path / "stage2.yaml"
    output = tmp_path / "summary.yaml"
    write_yaml(stage1, base_stage(1, 0.25))
    write_yaml(stage2, base_stage(1, 0.75))

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(stage1), str(stage2), "-o", str(output)],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "stage values must be 1 and 2" in result.stderr


@pytest.mark.parametrize("stage_a, stage_b", [(1.9, 2.1), (True, 2)])
def test_rejects_non_exact_stage_values(tmp_path, stage_a, stage_b):
    stage1 = tmp_path / "stage1.yaml"
    stage2 = tmp_path / "stage2.yaml"
    output = tmp_path / "summary.yaml"
    write_yaml(stage1, base_stage(stage_a, 0.25))
    write_yaml(stage2, base_stage(stage_b, 0.75))

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(stage1), str(stage2), "-o", str(output)],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "stage values must be 1 and 2" in result.stderr


@pytest.mark.parametrize(
    "key, value",
    [
        ("spring_species", ["Al", "O"]),
        ("uf_self_pairs", [{"element_i": "H", "element_j": "H", "p": 30.0, "sigma": 1.0}]),
        ("uf_cross_pairs", [{"element_i": "O", "element_j": "H", "p": 12.0, "sigma": 1.0}]),
    ],
)
def test_rejects_mismatched_reference_definitions(tmp_path, key, value):
    stage1_data = base_stage(1, 0.25)
    stage2_data = base_stage(2, 0.75)
    stage2_data[key] = value
    stage1 = tmp_path / "stage1.yaml"
    stage2 = tmp_path / "stage2.yaml"
    output = tmp_path / "summary.yaml"
    write_yaml(stage1, stage1_data)
    write_yaml(stage2, stage2_data)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(stage1), str(stage2), "-o", str(output)],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert f"reference definitions differ for '{key}'" in result.stderr


@pytest.mark.parametrize(
    "key, value",
    [
        ("T", 3100.0),
        ("V", 9.0),
        ("P", 0.02),
        ("N_total", 5),
        ("F_ref", -1.4),
    ],
)
def test_rejects_mismatched_numeric_metadata(tmp_path, key, value):
    stage1_data = base_stage(1, 0.25)
    stage2_data = base_stage(2, 0.75)
    stage2_data[key] = value
    stage1 = tmp_path / "stage1.yaml"
    stage2 = tmp_path / "stage2.yaml"
    output = tmp_path / "summary.yaml"
    write_yaml(stage1, stage1_data)
    write_yaml(stage2, stage2_data)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(stage1), str(stage2), "-o", str(output)],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert f"{key} differs between stage files" in result.stderr
