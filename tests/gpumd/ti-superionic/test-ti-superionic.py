import pathlib
import shutil
import subprocess

import pytest
import yaml


REPO = pathlib.Path(__file__).resolve().parents[3]
FIXTURE = REPO / "tests" / "gpumd" / "ti-superionic" / "self_consistent"
GPUMD = REPO / "src" / "gpumd"
REQUIRED_YAML_KEYS = {
    "stage",
    "T",
    "V",
    "P",
    "N_total",
    "spring_species",
    "spring_constants",
    "uf_self_pairs",
    "uf_cross_pairs",
    "W_forward",
    "W_backward",
    "delta_F",
    "F_Einstein",
    "F_UF_self",
    "F_ref",
}


def run_gpumd(tmp_path, run_in):
    for name in ("model.xyz", "nep.txt"):
        shutil.copy(FIXTURE / name, tmp_path / name)
    shutil.copy(FIXTURE / run_in, tmp_path / "run.in")
    return subprocess.run([str(GPUMD)], cwd=tmp_path, text=True, capture_output=True, check=False)


def run_gpumd_with_ensemble(tmp_path, ensemble_line):
    for name in ("model.xyz", "nep.txt"):
        shutil.copy(FIXTURE / name, tmp_path / name)
    (tmp_path / "run.in").write_text(
        "\n".join(
            [
                "potential nep.txt",
                "velocity 300",
                "time_step 1",
                ensemble_line,
                "run 12",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return subprocess.run([str(GPUMD)], cwd=tmp_path, text=True, capture_output=True, check=False)


def write_small_model(path):
    path.write_text(
        "\n".join(
            [
                "4",
                'Lattice="12 0 0 0 12 0 0 0 12" Properties=species:S:1:pos:R:3',
                "C 0.0 0.0 0.0",
                "C 3.0 0.0 0.0",
                "H 0.0 3.0 0.0",
                "H 3.0 3.0 0.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "run_in, yaml_name, stage, csv_name, csv_header",
    [
        (
            "run_stage1.in",
            "ti_superionic_stage1.yaml",
            1,
            "ti_superionic_stage1.csv",
            "lambda,dlambda,U_einstein,U_uf_self,U_uf_cross,dHdlambda",
        ),
        (
            "run_stage2.in",
            "ti_superionic_stage2.yaml",
            2,
            "ti_superionic_stage2.csv",
            "lambda,dlambda,U_target,U_einstein,U_uf_self,U_uf_cross,U_aux,dHdlambda",
        ),
    ],
)
def test_stage_command_writes_yaml(tmp_path, run_in, yaml_name, stage, csv_name, csv_header):
    result = run_gpumd(tmp_path, run_in)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / yaml_name).exists()
    yaml_text = (tmp_path / yaml_name).read_text(encoding="utf-8")
    data = yaml.safe_load(yaml_text)
    assert data["stage"] == stage
    assert REQUIRED_YAML_KEYS <= data.keys()
    assert data["spring_species"] == ["C"]
    assert data["spring_constants"] == [{"element": "C", "k": pytest.approx(1.0)}]
    assert data["uf_self_pairs"] == [
        {"element_i": "H", "element_j": "H", "p": 25.0, "sigma": 1.0}
    ]
    assert data["uf_cross_pairs"] == [
        {"element_i": "C", "element_j": "H", "p": 10.0, "sigma": 1.0}
    ]
    assert isinstance(data["spring_species"][0], str)
    assert isinstance(data["spring_constants"][0]["element"], str)
    assert isinstance(data["uf_self_pairs"][0]["element_i"], str)
    assert isinstance(data["uf_cross_pairs"][0]["element_j"], str)
    assert '  - "C"' in yaml_text
    assert 'element: "C", k: 1' in yaml_text
    assert 'element_i: "H", element_j: "H"' in yaml_text
    assert 'element_i: "C", element_j: "H"' in yaml_text
    assert data["uf_self_pairs"][0]["p"] == pytest.approx(25.0)
    assert data["uf_self_pairs"][0]["sigma"] == pytest.approx(1.0)
    assert data["uf_cross_pairs"][0]["p"] == pytest.approx(10.0)
    assert data["uf_cross_pairs"][0]["sigma"] == pytest.approx(1.0)
    assert isinstance(data["F_Einstein"], float)
    assert isinstance(data["F_UF_self"], float)
    assert isinstance(data["F_ref"], float)
    assert (tmp_path / csv_name).exists()
    assert (tmp_path / csv_name).read_text(encoding="utf-8").splitlines()[0] == csv_header


def test_stage_yaml_contains_reference_free_energy(tmp_path):
    result = run_gpumd(tmp_path, "run_stage1.in")
    assert result.returncode == 0, result.stderr
    data = yaml.safe_load((tmp_path / "ti_superionic_stage1.yaml").read_text(encoding="utf-8"))
    assert data["F_Einstein"] != 0.0
    assert data["F_UF_self"] != 0.0
    assert data["F_ref"] == pytest.approx(data["F_Einstein"] + data["F_UF_self"])
    assert data["spring_species"] == ["C"]
    assert data["spring_constants"] == [{"element": "C", "k": pytest.approx(1.0)}]
    assert data["uf_self_pairs"][0]["element_i"] == "H"
    assert data["uf_cross_pairs"][0]["element_i"] == "C"


def test_auto_spring_estimates_finite_reference(tmp_path):
    result = run_gpumd(tmp_path, "run_stage1_auto.in")
    assert result.returncode == 0, result.stderr
    data = yaml.safe_load((tmp_path / "ti_superionic_stage1.yaml").read_text(encoding="utf-8"))
    assert data["spring_species"] == ["C"]
    assert data["spring_constants"][0]["element"] == "C"
    assert data["spring_constants"][0]["k"] > 0.0
    assert data["F_Einstein"] != 0.0
    rows = read_csv_rows(tmp_path / "ti_superionic_stage1.csv")
    assert rows


@pytest.mark.parametrize(
    "ensemble_line, error_substring",
    [
        (
            "ensemble ti_superionic_stage1 tperiod 100 tequil 2 tswitch 4 press 0 "
            "spring C 1.0 uf H H 25 1.0",
            "Please specify temp.",
        ),
        (
            "ensemble ti_superionic_stage1 temp 0 tperiod 100 tequil 2 tswitch 4 press 0 "
            "spring C 1.0 uf H H 25 1.0",
            "Temperature should > 0.",
        ),
        (
            "ensemble ti_superionic_stage1 temp 300 tperiod 0 tequil 2 tswitch 4 press 0 "
            "spring C 1.0 uf H H 25 1.0",
            "Temperature coupling should >= 1.",
        ),
    ],
)
def test_stage_command_rejects_invalid_thermostat_inputs(
    tmp_path, ensemble_line, error_substring
):
    result = run_gpumd_with_ensemble(tmp_path, ensemble_line)
    assert result.returncode != 0
    assert error_substring in result.stderr


@pytest.mark.parametrize(
    "ensemble_line, message",
    [
        (
            "ensemble ti_superionic_stage1 temp 300 tperiod 100 tequil 2 tswitch 4 press 0 "
            "spring X 1.0 uf H H 25 1.0",
            "spring element does not exist",
        ),
        (
            "ensemble ti_superionic_stage1 temp 300 tperiod 100 tequil 2 tswitch 4 press 0 "
            "spring C 1.0 uf H H 10 1.0",
            "Self UF p must be 1, 25, 50, 75, or 100",
        ),
        (
            "ensemble ti_superionic_stage1 temp 300 tperiod 100 tequil 2 tswitch 4 press 0 "
            "spring C 1.0 uf C H 10 1.0",
            "Please specify at least one self uf pair",
        ),
        (
            "ensemble ti_superionic_stage1 temp 300 tperiod 100 tequil 2 tswitch 4 press 0 "
            "spring auto C spring H 1.0 uf H H 25 1.0",
            "Cannot mix auto and explicit spring inputs.",
        ),
        (
            "ensemble ti_superionic_stage1 temp 300 tperiod 100 tequil 0 tswitch 4 press 0 "
            "spring auto C uf H H 25 1.0",
            "tequil should be > 0 for auto spring constants.",
        ),
        (
            "ensemble ti_superionic_stage1 temp 300 tperiod 100 tequil 2 tswitch 4 press 0 "
            "spring auto C C uf H H 25 1.0",
            "Duplicate auto spring species.",
        ),
        (
            "ensemble ti_superionic_stage1 temp 300 tperiod 100 tequil 2 tswitch 4 press 0 "
            "spring C 1.0 uf H H 25 1.0 uf H H 50 1.0",
            "Duplicate UF pair.",
        ),
        (
            "ensemble ti_superionic_stage1 temp 300 tperiod 100 tequil 2 tswitch 4 press 0 "
            "spring C 1.0 uf H H 25 1.0 uf C H 10 1.0 uf H C 12 1.0",
            "Duplicate UF pair.",
        ),
        (
            "ensemble ti_superionic_stage1 temp 300 tperiod 100 tequil 2 tswitch 4 press 0 "
            "spring C 0 uf H H 25 1.0",
            "Spring constant must be positive.",
        ),
        (
            "ensemble ti_superionic_stage1 temp 300 tperiod 100 tequil 2 tswitch 4 press 0 "
            "spring C 1.0 uf H H -25 1.0",
            "UF p and sigma must be positive.",
        ),
        (
            "ensemble ti_superionic_stage1 temp 300 tperiod 100 tequil 2 tswitch 4 press 0 "
            "spring C 1.0 uf H H 25 0",
            "UF p and sigma must be positive.",
        ),
    ],
)
def test_rejects_invalid_reference_inputs(tmp_path, ensemble_line, message):
    result = run_gpumd_with_ensemble(tmp_path, ensemble_line)
    assert result.returncode != 0
    assert message in result.stderr


def test_rejects_nep_small_box_radial_list(tmp_path):
    for name in ("model.xyz", "nep.txt", "run_stage1.in"):
        shutil.copy(FIXTURE / name, tmp_path / name)
    write_small_model(tmp_path / "model.xyz")
    shutil.copy(tmp_path / "run_stage1.in", tmp_path / "run.in")

    result = subprocess.run(
        [str(GPUMD)], cwd=tmp_path, text=True, capture_output=True, check=False
    )

    assert result.returncode != 0
    assert (
        "ti_superionic requires the main NEP potential to expose the active radial neighbor list"
        in result.stderr
    )


def read_csv_rows(path):
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    header = lines[0].split(",")
    rows = []
    for line in lines[1:]:
        rows.append(dict(zip(header, [float(x) for x in line.split(",")])))
    return rows


def test_stage1_csv_has_cross_driving_force(tmp_path):
    result = run_gpumd(tmp_path, "run_stage1.in")
    assert result.returncode == 0, result.stderr
    rows = read_csv_rows(tmp_path / "ti_superionic_stage1.csv")
    assert rows
    assert any(abs(row["U_uf_cross"]) > 0.0 for row in rows)
    for row in rows:
        assert row["dHdlambda"] == pytest.approx(row["U_uf_cross"])


def test_stage2_csv_has_aux_and_target_terms(tmp_path):
    result = run_gpumd(tmp_path, "run_stage2.in")
    assert result.returncode == 0, result.stderr
    rows = read_csv_rows(tmp_path / "ti_superionic_stage2.csv")
    assert rows
    assert any(abs(row["U_aux"]) > 0.0 for row in rows)
    for row in rows:
        assert row["dHdlambda"] == pytest.approx(row["U_target"] - row["U_aux"])
