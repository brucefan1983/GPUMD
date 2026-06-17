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
    data = yaml.safe_load((tmp_path / yaml_name).read_text(encoding="utf-8"))
    assert data["stage"] == stage
    assert REQUIRED_YAML_KEYS <= data.keys()
    assert data["spring_species"] == ["C"]
    assert data["F_Einstein"] == 0
    assert data["F_UF_self"] == 0
    assert data["F_ref"] == 0
    assert (tmp_path / csv_name).exists()
    assert (tmp_path / csv_name).read_text(encoding="utf-8").splitlines()[0] == csv_header


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
