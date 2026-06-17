# Superionic Free Energy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the approved two-stage superionic free-energy workflow with `ti_superionic_stage1`, `ti_superionic_stage2`, and a YAML post-processing tool.

**Architecture:** Add one shared CUDA ensemble class, `Ensemble_TI_Superionic`, that is instantiated with a stage flag for both public commands. The class keeps reference-state parameters as numeric GPU buffers: per-atom spring masks/constants and per-type-pair UF parameter matrices. A separate Python tool combines the two independent stage YAML files into the final `F_target` and `G_target`.

**Tech Stack:** CUDA C++14 in GPUMD `src/integrate`, existing `Ensemble_LAN`/`Force`/`GPU_Vector` APIs, Python 3 standard library plus PyYAML for post-processing tests when available.

---

## File Map

- Create: `src/integrate/ensemble_ti_superionic.cuh`
  - Declares `Ensemble_TI_Superionic`, `SuperionicStage`, `SuperionicUFPair`, parser state, GPU buffers, work accumulators, and helper methods.
- Create: `src/integrate/ensemble_ti_superionic.cu`
  - Implements command parsing, lambda schedule, spring/UF CUDA kernels, stage force mixing, CSV/YAML output, and reference free-energy helpers.
- Create: `src/integrate/uf_reference.cuh`
  - Holds the existing UF spline tables and interpolation helper currently embedded in `ensemble_ti_liquid.cuh`, so `ti_liquid` and `ti_superionic` share one analytic UF reference implementation.
- Modify: `src/integrate/ensemble_ti_liquid.cuh`
  - Removes the embedded UF spline table members after they are moved into `uf_reference.cuh`.
- Modify: `src/integrate/ensemble_ti_liquid.cu`
  - Calls the shared `uf_reference` helper when computing `E_UFmodel`.
- Modify: `src/integrate/integrate.cu`
  - Includes the new header, parses `ti_superionic_stage1` and `ti_superionic_stage2`, assigns new negative type IDs, and routes both through `compute3`.
- Create: `tools/si_free_energy_sum.py`
  - Reads stage YAML files, validates metadata/reference state compatibility, and writes a summary YAML.
- Create: `tests/tools/test_si_free_energy_sum.py`
  - Tests post-processing formulas and compatibility checks with YAML fixtures.
- Create: `tests/gpumd/ti-superionic/test-ti-superionic.py`
  - Adds parser/smoke tests for stage commands and finite CSV/YAML output.
- Create test fixtures under `tests/gpumd/ti-superionic/self_consistent/`
  - Minimal C/H `model.xyz`, copied `nep.txt`, and `run_stage*.in` files for command-level checks.
- Reference only: `docs/superpowers/specs/2026-06-17-superionic-free-energy-design.md`
  - Approved behavior source.

The repository `src/makefile` already uses `$(wildcard integrate/*.cu)` and `$(wildcard integrate/*.cuh)`, so adding the new `.cu/.cuh` files is enough for compilation.

---

### Task 1: Post-Processing Tool

**Files:**
- Create: `GPUMD-dev-v4.9.1/tools/si_free_energy_sum.py`
- Create: `GPUMD-dev-v4.9.1/tests/tools/test_si_free_energy_sum.py`

- [ ] **Step 1: Write the failing post-processing tests**

Create `tests/tools/test_si_free_energy_sum.py` with:

```python
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
    assert data["stage1"]["delta_F"] == pytest.approx(0.25)
    assert data["stage2"]["delta_F"] == pytest.approx(0.75)
    assert data["F_ref"] == pytest.approx(-1.5)
    assert data["F_target"] == pytest.approx(-0.5)
    assert data["G_target"] == pytest.approx(-0.42)


def test_rejects_swapped_or_duplicate_stages(tmp_path):
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


def test_rejects_mismatched_reference_state(tmp_path):
    stage1_data = base_stage(1, 0.25)
    stage2_data = base_stage(2, 0.75)
    stage2_data["spring_species"] = ["Al", "O"]
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
    assert "reference definitions differ" in result.stderr
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /home/asus/projects/GPUMD-dev-v4.9.1
pytest tests/tools/test_si_free_energy_sum.py -q
```

Expected: FAIL because `tools/si_free_energy_sum.py` does not exist.

- [ ] **Step 3: Implement the post-processing tool**

Create `tools/si_free_energy_sum.py` with:

```python
#!/usr/bin/env python3
import argparse
import math
import sys

import yaml


REFERENCE_KEYS = ("spring_species", "uf_self_pairs", "uf_cross_pairs")
META_KEYS = ("T", "V", "P", "N_total", "F_ref")


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a YAML mapping")
    return data


def close(a, b, name):
    if not math.isclose(float(a), float(b), rel_tol=1.0e-8, abs_tol=1.0e-10):
        raise ValueError(f"{name} differs between stage files: {a} vs {b}")


def require_keys(data, path):
    required = (
        "stage",
        "T",
        "V",
        "P",
        "N_total",
        "W_forward",
        "W_backward",
        "delta_F",
        "F_ref",
        "spring_species",
        "uf_self_pairs",
        "uf_cross_pairs",
    )
    for key in required:
        if key not in data:
            raise ValueError(f"{path} is missing required key '{key}'")


def ordered_stages(first, second):
    stages = {int(first["stage"]), int(second["stage"])}
    if stages != {1, 2}:
        raise ValueError("stage values must be 1 and 2")
    return (first, second) if int(first["stage"]) == 1 else (second, first)


def validate_compatible(stage1, stage2):
    for key in REFERENCE_KEYS:
        if stage1[key] != stage2[key]:
            raise ValueError(f"reference definitions differ for '{key}'")
    for key in META_KEYS:
        close(stage1[key], stage2[key], key)


def summarize(stage1, stage2):
    f_ref = float(stage1["F_ref"])
    delta1 = float(stage1["delta_F"])
    delta2 = float(stage2["delta_F"])
    volume = float(stage1["V"])
    pressure = float(stage1["P"])
    f_target = f_ref + delta1 + delta2
    return {
        "stage1": {
            "W_forward": float(stage1["W_forward"]),
            "W_backward": float(stage1["W_backward"]),
            "delta_F": delta1,
        },
        "stage2": {
            "W_forward": float(stage2["W_forward"]),
            "W_backward": float(stage2["W_backward"]),
            "delta_F": delta2,
        },
        "F_ref": f_ref,
        "F_target": f_target,
        "G_target": f_target + pressure * volume,
        "T": float(stage1["T"]),
        "V": volume,
        "P": pressure,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize two-stage superionic TI YAML files.")
    parser.add_argument("stage_a")
    parser.add_argument("stage_b")
    parser.add_argument("-o", "--output", default="ti_superionic_summary.yaml")
    args = parser.parse_args(argv)

    try:
        first = load_yaml(args.stage_a)
        second = load_yaml(args.stage_b)
        require_keys(first, args.stage_a)
        require_keys(second, args.stage_b)
        stage1, stage2 = ordered_stages(first, second)
        validate_compatible(stage1, stage2)
        data = summarize(stage1, stage2)
        with open(args.output, "w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, sort_keys=False)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run post-processing tests**

Run:

```bash
cd /home/asus/projects/GPUMD-dev-v4.9.1
pytest tests/tools/test_si_free_energy_sum.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
cd /home/asus/projects/GPUMD-dev-v4.9.1
git add tools/si_free_energy_sum.py tests/tools/test_si_free_energy_sum.py
git commit -m "Add superionic free energy summary tool"
```

Expected: commit succeeds.

---

### Task 2: Ensemble Command Skeleton

**Files:**
- Create: `GPUMD-dev-v4.9.1/src/integrate/ensemble_ti_superionic.cuh`
- Create: `GPUMD-dev-v4.9.1/src/integrate/ensemble_ti_superionic.cu`
- Modify: `GPUMD-dev-v4.9.1/src/integrate/integrate.cu`

- [ ] **Step 1: Write a failing command parser smoke test**

Create `tests/gpumd/ti-superionic/test-ti-superionic.py` with:

```python
import pathlib
import shutil
import subprocess

import pytest
import yaml


REPO = pathlib.Path(__file__).resolve().parents[3]
FIXTURE = REPO / "tests" / "gpumd" / "ti-superionic" / "self_consistent"
GPUMD = REPO / "src" / "gpumd"


def run_gpumd(tmp_path, run_in):
    for name in ("model.xyz", "nep.txt"):
        shutil.copy(FIXTURE / name, tmp_path / name)
    shutil.copy(FIXTURE / run_in, tmp_path / "run.in")
    return subprocess.run([str(GPUMD)], cwd=tmp_path, text=True, capture_output=True, check=False)


@pytest.mark.parametrize(
    "run_in, yaml_name",
    [
        ("run_stage1.in", "ti_superionic_stage1.yaml"),
        ("run_stage2.in", "ti_superionic_stage2.yaml"),
    ],
)
def test_stage_command_writes_yaml(tmp_path, run_in, yaml_name):
    result = run_gpumd(tmp_path, run_in)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / yaml_name).exists()
    data = yaml.safe_load((tmp_path / yaml_name).read_text(encoding="utf-8"))
    assert data["stage"] in (1, 2)
```

Create fixtures:

`tests/gpumd/ti-superionic/self_consistent/model.xyz`

```text
4
Lattice="12 0 0 0 12 0 0 0 12" Properties=species:S:1:pos:R:3
C 0.0 0.0 0.0
C 3.0 0.0 0.0
H 0.0 3.0 0.0
H 3.0 3.0 0.0
```

Create `tests/gpumd/ti-superionic/self_consistent/nep.txt` by copying the existing C/H NEP test
potential:

```bash
cd /home/asus/projects/GPUMD-dev-v4.9.1
cp tests/gpumd/active/model/nep_full.txt tests/gpumd/ti-superionic/self_consistent/nep.txt
```

Expected: `head -1 tests/gpumd/ti-superionic/self_consistent/nep.txt` prints `nep3 2 C H`.

`tests/gpumd/ti-superionic/self_consistent/run_stage1.in`

```text
potential nep.txt
velocity 300
time_step 1
ensemble ti_superionic_stage1 temp 300 tperiod 100 tequil 2 tswitch 4 press 0 spring C 1.0 uf H H 25 1.0 uf C H 10 1.0
run 12
```

`tests/gpumd/ti-superionic/self_consistent/run_stage2.in`

```text
potential nep.txt
velocity 300
time_step 1
ensemble ti_superionic_stage2 temp 300 tperiod 100 tequil 2 tswitch 4 press 0 spring C 1.0 uf H H 25 1.0 uf C H 10 1.0
run 12
```

- [ ] **Step 2: Run smoke test to verify it fails**

Run:

```bash
cd /home/asus/projects/GPUMD-dev-v4.9.1
pytest tests/gpumd/ti-superionic/test-ti-superionic.py::test_stage_command_writes_yaml -q
```

Expected: FAIL with `Invalid ensemble type` or missing `src/gpumd`.

- [ ] **Step 3: Add the header skeleton**

Create `src/integrate/ensemble_ti_superionic.cuh` with:

```cpp
/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once
#include "ensemble_lan.cuh"
#include "force/force.cuh"
#include "langevin_utilities.cuh"
#include "model/box.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/read_file.cuh"
#include <map>
#include <string>
#include <vector>

enum class SuperionicStage { stage1 = 1, stage2 = 2 };

struct SuperionicUFPair
{
  std::string element_i;
  std::string element_j;
  double p = 0.0;
  double sigma = 0.0;
};

class Ensemble_TI_Superionic : public Ensemble_LAN
{
public:
  Ensemble_TI_Superionic(const char** params, int num_params, SuperionicStage stage);
  virtual ~Ensemble_TI_Superionic(void);

  virtual void compute1(
    const double time_step,
    const std::vector<Group>& group,
    Box& box,
    Atom& atoms,
    GPU_Vector<double>& thermo);

  virtual void compute3(
    const double time_step,
    const std::vector<Group>& group,
    Box& box,
    Atom& atoms,
    GPU_Vector<double>& thermo,
    Force& force);

  void init();
  void find_lambda();
  double switch_func(double t);
  double dswitch_func(double t);

protected:
  SuperionicStage stage;
  FILE* output_file = nullptr;
  double lambda = 0.0;
  double dlambda = 0.0;
  int t_equil = -1;
  int t_switch = -1;
  double target_pressure = 0.0;
  double V = 0.0;
  double W_forward = 0.0;
  double W_backward = 0.0;
  double delta_F = 0.0;
  bool auto_k = false;
  bool initialized = false;
  bool lambda_active = false;

  std::map<std::string, double> spring_map;
  std::vector<std::string> auto_spring_species;
  std::vector<SuperionicUFPair> uf_pairs;
  std::vector<double> thermo_cpu;
};
```

- [ ] **Step 4: Add a minimal implementation skeleton**

Create `src/integrate/ensemble_ti_superionic.cu` with:

```cpp
/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "ensemble_ti_superionic.cuh"
#include "utilities/gpu_macro.cuh"
#include <cstring>

Ensemble_TI_Superionic::Ensemble_TI_Superionic(
  const char** params, int num_params, SuperionicStage stage_input)
{
  stage = stage_input;
  temperature_coupling = 100;
  int i = 2;
  while (i < num_params) {
    if (strcmp(params[i], "tswitch") == 0) {
      if (!is_valid_int(params[i + 1], &t_switch))
        PRINT_INPUT_ERROR("Wrong inputs for tswitch keyword.");
      i += 2;
    } else if (strcmp(params[i], "tequil") == 0) {
      if (!is_valid_int(params[i + 1], &t_equil))
        PRINT_INPUT_ERROR("Wrong inputs for tequil keyword.");
      i += 2;
    } else if (strcmp(params[i], "temp") == 0) {
      if (!is_valid_real(params[i + 1], &temperature))
        PRINT_INPUT_ERROR("Wrong inputs for temp keyword.");
      i += 2;
    } else if (strcmp(params[i], "press") == 0) {
      if (!is_valid_real(params[i + 1], &target_pressure))
        PRINT_INPUT_ERROR("Wrong inputs for press keyword.");
      target_pressure /= PRESSURE_UNIT_CONVERSION;
      i += 2;
    } else if (strcmp(params[i], "tperiod") == 0) {
      if (!is_valid_real(params[i + 1], &temperature_coupling))
        PRINT_INPUT_ERROR("Wrong inputs for tperiod keyword.");
      i += 2;
    } else if (strcmp(params[i], "spring") == 0) {
      i++;
      if (i < num_params && strcmp(params[i], "auto") == 0) {
        auto_k = true;
        i++;
        while (i < num_params && strcmp(params[i], "uf") != 0) {
          auto_spring_species.push_back(params[i]);
          i++;
        }
      } else {
        double k = 0.0;
        while (i < num_params && strcmp(params[i], "uf") != 0) {
          if (i + 1 >= num_params || !is_valid_real(params[i + 1], &k))
            PRINT_INPUT_ERROR("Wrong inputs for spring keyword.");
          spring_map[params[i]] = k;
          i += 2;
        }
      }
    } else if (strcmp(params[i], "uf") == 0) {
      if (i + 4 >= num_params)
        PRINT_INPUT_ERROR("Wrong inputs for uf keyword.");
      SuperionicUFPair pair;
      pair.element_i = params[i + 1];
      pair.element_j = params[i + 2];
      if (!is_valid_real(params[i + 3], &pair.p))
        PRINT_INPUT_ERROR("Wrong inputs for uf p value.");
      if (!is_valid_real(params[i + 4], &pair.sigma))
        PRINT_INPUT_ERROR("Wrong inputs for uf sigma value.");
      uf_pairs.push_back(pair);
      i += 5;
    } else {
      PRINT_INPUT_ERROR("Unknown keyword.");
    }
  }

  if (t_switch < 0 || t_equil < 0)
    PRINT_INPUT_ERROR("Please specify both tswitch and tequil.");
  if (spring_map.empty() && auto_spring_species.empty())
    PRINT_INPUT_ERROR("Please specify spring species.");
  if (uf_pairs.empty())
    PRINT_INPUT_ERROR("Please specify at least one uf pair.");

  type = 3;
  c1 = exp(-0.5 / temperature_coupling);
  c2 = sqrt((1 - c1 * c1) * K_B * temperature);
}

void Ensemble_TI_Superionic::init()
{
  printf("The number of steps should be set to %d!\n", 2 * (t_equil + t_switch));
  printf(
    "Superionic thermodynamic integration stage %d: t_switch is %d timestep, t_equil is %d timesteps.\n",
    static_cast<int>(stage),
    t_switch,
    t_equil);
  output_file = my_fopen(
    stage == SuperionicStage::stage1 ? "ti_superionic_stage1.csv" : "ti_superionic_stage2.csv",
    "w");
  if (stage == SuperionicStage::stage1) {
    fprintf(output_file, "lambda,dlambda,U_einstein,U_uf_self,U_uf_cross,dHdlambda\n");
  } else {
    fprintf(output_file, "lambda,dlambda,U_target,U_einstein,U_uf_self,U_uf_cross,U_aux,dHdlambda\n");
  }
  int N = atom->number_of_atoms;
  curand_states.resize(N);
  initialize_curand_states<<<(N - 1) / 128 + 1, 128>>>(curand_states.data(), N, rand());
  GPU_CHECK_KERNEL
  thermo_cpu.resize(thermo->size());
  initialized = true;
}

Ensemble_TI_Superionic::~Ensemble_TI_Superionic(void)
{
  const char* yaml_name =
    stage == SuperionicStage::stage1 ? "ti_superionic_stage1.yaml" : "ti_superionic_stage2.yaml";
  FILE* yaml_file = my_fopen(yaml_name, "w");
  fprintf(yaml_file, "stage: %d\n", static_cast<int>(stage));
  fprintf(yaml_file, "T: %f\n", temperature);
  fprintf(yaml_file, "V: %f\n", V);
  fprintf(yaml_file, "P: %f\n", target_pressure);
  fprintf(yaml_file, "N_total: %d\n", atom ? atom->number_of_atoms : 0);
  fprintf(yaml_file, "spring_species: []\n");
  fprintf(yaml_file, "uf_self_pairs: []\n");
  fprintf(yaml_file, "uf_cross_pairs: []\n");
  fprintf(yaml_file, "W_forward: %f\n", W_forward);
  fprintf(yaml_file, "W_backward: %f\n", W_backward);
  fprintf(yaml_file, "delta_F: %f\n", delta_F);
  fprintf(yaml_file, "F_Einstein: 0.000000\n");
  fprintf(yaml_file, "F_UF_self: 0.000000\n");
  fprintf(yaml_file, "F_ref: 0.000000\n");
  if (output_file)
    fclose(output_file);
  fclose(yaml_file);
}

void Ensemble_TI_Superionic::compute1(
  const double time_step,
  const std::vector<Group>& group,
  Box& box,
  Atom& atoms,
  GPU_Vector<double>& thermo)
{
  if (*current_step == 0)
    init();
  Ensemble_LAN::compute1(time_step, group, box, atoms, thermo);
}

void Ensemble_TI_Superionic::compute3(
  const double time_step,
  const std::vector<Group>& group,
  Box& box,
  Atom& atoms,
  GPU_Vector<double>& thermo,
  Force& force)
{
  find_lambda();
  Ensemble_LAN::compute2(time_step, group, box, atoms, thermo);
}

void Ensemble_TI_Superionic::find_lambda()
{
  V = box->get_volume() / atom->number_of_atoms;
  const int t = *current_step - t_equil;
  const double r_switch = 1.0 / t_switch;
  bool need_output = false;
  if ((t >= 0) && (t <= t_switch)) {
    lambda = switch_func(t * r_switch);
    dlambda = dswitch_func(t * r_switch);
    need_output = true;
  } else if ((t >= t_equil + t_switch) && (t <= (t_equil + 2 * t_switch))) {
    lambda = switch_func(1.0 - (t - t_switch - t_equil) * r_switch);
    dlambda = -dswitch_func(1.0 - (t - t_switch - t_equil) * r_switch);
    need_output = true;
  }
  if (need_output) {
    if (stage == SuperionicStage::stage1) {
      fprintf(output_file, "%e,%e,0,0,0,0\n", lambda, dlambda);
    } else {
      fprintf(output_file, "%e,%e,0,0,0,0,0,0\n", lambda, dlambda);
    }
  }
}

double Ensemble_TI_Superionic::switch_func(double t)
{
  double t2 = t * t;
  double t5 = t2 * t2 * t;
  return ((70.0 * t2 * t2 - 315.0 * t2 * t + 540.0 * t2 - 420.0 * t + 126.0) * t5);
}

double Ensemble_TI_Superionic::dswitch_func(double t)
{
  double t2 = t * t;
  double t4 = t2 * t2;
  return ((630 * t2 * t2 - 2520 * t2 * t + 3780 * t2 - 2520 * t + 630) * t4) / t_switch;
}
```

- [ ] **Step 5: Wire the commands into `integrate.cu`**

Modify `src/integrate/integrate.cu`:

Add this include near the existing TI includes:

```cpp
#include "ensemble_ti_superionic.cuh"
```

In `Integrate::initialize`, add two no-op cases after `case -11`:

```cpp
    case -12: // ti_superionic_stage1
      break;
    case -13: // ti_superionic_stage2
      break;
```

In `Integrate::compute2`, change:

```cpp
  } else if (type == -11) {
    ensemble->compute3(time_step, group, box, atom, thermo, force);
    return;
  }
```

to:

```cpp
  } else if (type == -11 || type == -12 || type == -13) {
    ensemble->compute3(time_step, group, box, atom, thermo, force);
    return;
  }
```

In `Integrate::parse_ensemble`, add after the `ti_liquid` branch:

```cpp
  } else if (strcmp(param[1], "ti_superionic_stage1") == 0) {
    type = -12;
    ensemble.reset(new Ensemble_TI_Superionic(param, num_param, SuperionicStage::stage1));
  } else if (strcmp(param[1], "ti_superionic_stage2") == 0) {
    type = -13;
    ensemble.reset(new Ensemble_TI_Superionic(param, num_param, SuperionicStage::stage2));
```

- [ ] **Step 6: Build GPUMD**

Run:

```bash
cd /home/asus/projects/GPUMD-dev-v4.9.1/src
make gpumd
```

Expected: `The gpumd executable is successfully compiled!`

- [ ] **Step 7: Run the stage command smoke test**

Run:

```bash
cd /home/asus/projects/GPUMD-dev-v4.9.1
pytest tests/gpumd/ti-superionic/test-ti-superionic.py::test_stage_command_writes_yaml -q
```

Expected: PASS. The physics is still stubbed; this task only proves command routing and output creation.

- [ ] **Step 8: Commit Task 2**

Run:

```bash
cd /home/asus/projects/GPUMD-dev-v4.9.1
git add src/integrate/ensemble_ti_superionic.cuh src/integrate/ensemble_ti_superionic.cu src/integrate/integrate.cu tests/gpumd/ti-superionic
git commit -m "Add superionic TI ensemble command skeleton"
```

Expected: commit succeeds.

---

### Task 3: Parameter Validation And Reference-State Buffers

**Files:**
- Modify: `GPUMD-dev-v4.9.1/src/integrate/ensemble_ti_superionic.cuh`
- Modify: `GPUMD-dev-v4.9.1/src/integrate/ensemble_ti_superionic.cu`
- Modify: `GPUMD-dev-v4.9.1/tests/gpumd/ti-superionic/test-ti-superionic.py`

- [ ] **Step 1: Extend tests for rejected inputs**

Append to `tests/gpumd/ti-superionic/test-ti-superionic.py`:

```python
def write_run(tmp_path, ensemble_line):
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
                "",
            ]
        ),
        encoding="utf-8",
    )


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
    ],
)
def test_rejects_invalid_reference_inputs(tmp_path, ensemble_line, message):
    write_run(tmp_path, ensemble_line)
    result = subprocess.run([str(GPUMD)], cwd=tmp_path, text=True, capture_output=True, check=False)
    assert result.returncode != 0
    assert message in result.stderr
```

- [ ] **Step 2: Run validation tests to verify they fail**

Run:

```bash
cd /home/asus/projects/GPUMD-dev-v4.9.1
pytest tests/gpumd/ti-superionic/test-ti-superionic.py::test_rejects_invalid_reference_inputs -q
```

Expected: FAIL because the skeleton does not validate against atom species or self UF restrictions yet.

- [ ] **Step 3: Add CPU/GPU data members**

Update `src/integrate/ensemble_ti_superionic.cuh` by adding these protected members:

```cpp
  int num_types = 0;
  double beta = 0.0;
  double pe = 0.0;
  double U_einstein = 0.0;
  double U_uf_self = 0.0;
  double U_uf_cross = 0.0;
  double U_aux = 0.0;
  double dHdlambda = 0.0;
  double F_Einstein = 0.0;
  double F_UF_self = 0.0;
  double F_ref = 0.0;

  std::vector<double> cpu_k;
  std::vector<double> cpu_spring_mask;
  std::vector<double> cpu_uf_p;
  std::vector<double> cpu_uf_sigma_sqrd;
  std::vector<int> cpu_uf_kind;
  GPU_Vector<double> gpu_k;
  GPU_Vector<double> gpu_spring_mask;
  GPU_Vector<double> gpu_uf_p;
  GPU_Vector<double> gpu_uf_sigma_sqrd;
  GPU_Vector<int> gpu_uf_kind;
  GPU_Vector<double> gpu_einstein;
  GPU_Vector<double> gpu_uf_self;
  GPU_Vector<double> gpu_uf_cross;
  GPU_Vector<double> gpu_aux_fx;
  GPU_Vector<double> gpu_aux_fy;
  GPU_Vector<double> gpu_aux_fz;
  GPU_Vector<double> gpu_cross_fx;
  GPU_Vector<double> gpu_cross_fy;
  GPU_Vector<double> gpu_cross_fz;
  GPU_Vector<double> position_0;

  void prepare_reference_state();
  void validate_species();
  bool is_supported_self_p(double p) const;
  int find_type_for_symbol(const std::string& symbol) const;
  void write_yaml_pair_list(FILE* file, const char* key, bool self_pairs) const;
```

Use `cpu_uf_kind` values:

```text
0 = disabled pair
1 = UF self pair
2 = UF cross pair
```

- [ ] **Step 4: Implement species validation and buffer preparation**

In `ensemble_ti_superionic.cu`, add helper methods before `init()`:

```cpp
bool Ensemble_TI_Superionic::is_supported_self_p(double p) const
{
  return p == 1 || p == 25 || p == 50 || p == 75 || p == 100;
}

int Ensemble_TI_Superionic::find_type_for_symbol(const std::string& symbol) const
{
  for (int i = 0; i < atom->number_of_atoms; ++i) {
    if (atom->cpu_atom_symbol[i] == symbol)
      return atom->cpu_type[i];
  }
  return -1;
}

void Ensemble_TI_Superionic::validate_species()
{
  if (auto_k) {
    for (const auto& symbol : auto_spring_species) {
      if (find_type_for_symbol(symbol) < 0)
        PRINT_INPUT_ERROR("spring element does not exist in the structure.");
    }
  } else {
    for (const auto& entry : spring_map) {
      if (find_type_for_symbol(entry.first) < 0)
        PRINT_INPUT_ERROR("spring element does not exist in the structure.");
    }
  }

  bool has_self_pair = false;
  for (const auto& pair : uf_pairs) {
    int type_i = find_type_for_symbol(pair.element_i);
    int type_j = find_type_for_symbol(pair.element_j);
    if (type_i < 0 || type_j < 0)
      PRINT_INPUT_ERROR("uf element does not exist in the structure.");
    if (pair.p <= 0.0 || pair.sigma <= 0.0)
      PRINT_INPUT_ERROR("UF p and sigma must be positive.");
    if (pair.element_i == pair.element_j) {
      has_self_pair = true;
      if (!is_supported_self_p(pair.p))
        PRINT_INPUT_ERROR("Self UF p must be 1, 25, 50, 75, or 100.");
    }
  }
  if (!has_self_pair)
    PRINT_INPUT_ERROR("Please specify at least one self uf pair.");
}

void Ensemble_TI_Superionic::prepare_reference_state()
{
  validate_species();
  int N = atom->number_of_atoms;
  num_types = static_cast<int>(atom->cpu_type_size.size());
  cpu_k.assign(N, 0.0);
  cpu_spring_mask.assign(N, 0.0);
  cpu_uf_p.assign(num_types * num_types, 0.0);
  cpu_uf_sigma_sqrd.assign(num_types * num_types, 1.0);
  cpu_uf_kind.assign(num_types * num_types, 0);

  for (int i = 0; i < N; ++i) {
    std::string symbol = atom->cpu_atom_symbol[i];
    if (auto_k) {
      for (const auto& auto_symbol : auto_spring_species) {
        if (symbol == auto_symbol)
          cpu_spring_mask[i] = 1.0;
      }
    } else if (spring_map.find(symbol) != spring_map.end()) {
      cpu_spring_mask[i] = 1.0;
      cpu_k[i] = spring_map[symbol];
    }
  }

  for (const auto& pair : uf_pairs) {
    int type_i = find_type_for_symbol(pair.element_i);
    int type_j = find_type_for_symbol(pair.element_j);
    int kind = pair.element_i == pair.element_j ? 1 : 2;
    int ij = type_i * num_types + type_j;
    int ji = type_j * num_types + type_i;
    cpu_uf_p[ij] = pair.p;
    cpu_uf_p[ji] = pair.p;
    cpu_uf_sigma_sqrd[ij] = pair.sigma * pair.sigma;
    cpu_uf_sigma_sqrd[ji] = pair.sigma * pair.sigma;
    cpu_uf_kind[ij] = kind;
    cpu_uf_kind[ji] = kind;
  }

  gpu_k.resize(N);
  gpu_k.copy_from_host(cpu_k.data());
  gpu_spring_mask.resize(N);
  gpu_spring_mask.copy_from_host(cpu_spring_mask.data());
  gpu_uf_p.resize(num_types * num_types);
  gpu_uf_p.copy_from_host(cpu_uf_p.data());
  gpu_uf_sigma_sqrd.resize(num_types * num_types);
  gpu_uf_sigma_sqrd.copy_from_host(cpu_uf_sigma_sqrd.data());
  gpu_uf_kind.resize(num_types * num_types);
  gpu_uf_kind.copy_from_host(cpu_uf_kind.data());
  gpu_einstein.resize(N, 0.0);
  gpu_uf_self.resize(N, 0.0);
  gpu_uf_cross.resize(N, 0.0);
  gpu_aux_fx.resize(N, 0.0);
  gpu_aux_fy.resize(N, 0.0);
  gpu_aux_fz.resize(N, 0.0);
  gpu_cross_fx.resize(N, 0.0);
  gpu_cross_fy.resize(N, 0.0);
  gpu_cross_fz.resize(N, 0.0);
  position_0.resize(3 * N);
  CHECK(gpuMemcpy(
    position_0.data(),
    atom->position_per_atom.data(),
    sizeof(double) * position_0.size(),
    gpuMemcpyDeviceToDevice));
}
```

Call `prepare_reference_state();` from `init()` after `thermo_cpu.resize(thermo->size());`.

- [ ] **Step 5: Improve YAML pair/species output**

Replace skeleton `spring_species`, `uf_self_pairs`, and `uf_cross_pairs` output in the destructor with helper output. Add this method:

```cpp
void Ensemble_TI_Superionic::write_yaml_pair_list(FILE* file, const char* key, bool self_pairs) const
{
  bool has_pair = false;
  for (const auto& pair : uf_pairs) {
    bool is_self = pair.element_i == pair.element_j;
    if (is_self == self_pairs)
      has_pair = true;
  }
  if (!has_pair) {
    fprintf(file, "%s: []\n", key);
    return;
  }

  fprintf(file, "%s:\n", key);
  for (const auto& pair : uf_pairs) {
    bool is_self = pair.element_i == pair.element_j;
    if (is_self == self_pairs) {
      fprintf(
        file,
        "  - {element_i: %s, element_j: %s, p: %f, sigma: %f}\n",
        pair.element_i.c_str(),
        pair.element_j.c_str(),
        pair.p,
        pair.sigma);
    }
  }
}
```

In the destructor, write spring species with:

```cpp
  fprintf(yaml_file, "spring_species:\n");
  if (auto_k) {
    for (const auto& symbol : auto_spring_species)
      fprintf(yaml_file, "  - %s\n", symbol.c_str());
  } else {
    for (const auto& entry : spring_map)
      fprintf(yaml_file, "  - %s\n", entry.first.c_str());
  }
  write_yaml_pair_list(yaml_file, "uf_self_pairs", true);
  write_yaml_pair_list(yaml_file, "uf_cross_pairs", false);
```

- [ ] **Step 6: Build and run validation tests**

Run:

```bash
cd /home/asus/projects/GPUMD-dev-v4.9.1/src
make gpumd
cd /home/asus/projects/GPUMD-dev-v4.9.1
pytest tests/gpumd/ti-superionic/test-ti-superionic.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 3**

Run:

```bash
cd /home/asus/projects/GPUMD-dev-v4.9.1
git add src/integrate/ensemble_ti_superionic.cuh src/integrate/ensemble_ti_superionic.cu tests/gpumd/ti-superionic/test-ti-superionic.py
git commit -m "Validate superionic reference state inputs"
```

Expected: commit succeeds.

---

### Task 4: Spring/UF Kernels, Work Accumulation, And Stage Mixing

**Files:**
- Modify: `GPUMD-dev-v4.9.1/src/integrate/ensemble_ti_superionic.cuh`
- Modify: `GPUMD-dev-v4.9.1/src/integrate/ensemble_ti_superionic.cu`
- Modify: `GPUMD-dev-v4.9.1/tests/gpumd/ti-superionic/test-ti-superionic.py`

- [ ] **Step 1: Extend smoke tests to assert non-stub CSV data**

Append to `tests/gpumd/ti-superionic/test-ti-superionic.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail on stub data**

Run:

```bash
cd /home/asus/projects/GPUMD-dev-v4.9.1
pytest tests/gpumd/ti-superionic/test-ti-superionic.py::test_stage1_csv_has_cross_driving_force tests/gpumd/ti-superionic/test-ti-superionic.py::test_stage2_csv_has_aux_and_target_terms -q
```

Expected: FAIL because the CSV currently contains zeros.

- [ ] **Step 3: Add GPU kernels for spring, UF, force mixing, and reductions**

In `ensemble_ti_superionic.cu`, add an anonymous namespace before the constructor with kernels following the existing `ti_spring`/`ti_liquid` style:

```cpp
namespace
{
static __global__ void gpu_zero_superionic_arrays(
  int N,
  double* einstein,
  double* uf_self,
  double* uf_cross,
  double* aux_fx,
  double* aux_fy,
  double* aux_fz,
  double* cross_fx,
  double* cross_fy,
  double* cross_fz)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    einstein[i] = 0.0;
    uf_self[i] = 0.0;
    uf_cross[i] = 0.0;
    aux_fx[i] = 0.0;
    aux_fy[i] = 0.0;
    aux_fz[i] = 0.0;
    cross_fx[i] = 0.0;
    cross_fy[i] = 0.0;
    cross_fz[i] = 0.0;
  }
}

static __global__ void gpu_find_superionic_spring(
  int N,
  Box box,
  const double* k,
  const double* spring_mask,
  const double* x,
  const double* y,
  const double* z,
  const double* x0,
  const double* y0,
  const double* z0,
  double* einstein,
  double* aux_fx,
  double* aux_fy,
  double* aux_fz)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N && spring_mask[i] > 0.5) {
    double dx = x[i] - x0[i];
    double dy = y[i] - y0[i];
    double dz = z[i] - z0[i];
    apply_mic(box, dx, dy, dz);
    einstein[i] = 0.5 * k[i] * (dx * dx + dy * dy + dz * dz);
    aux_fx[i] += -k[i] * dx;
    aux_fy[i] += -k[i] * dy;
    aux_fz[i] += -k[i] * dz;
  }
}

static __global__ void gpu_find_superionic_uf(
  int N,
  int num_types,
  Box box,
  double beta,
  const int* g_type,
  const int* g_NN,
  const int* g_NL,
  const double* uf_p,
  const double* uf_sigma_sqrd,
  const int* uf_kind,
  const double* x,
  const double* y,
  const double* z,
  double* uf_self,
  double* uf_cross,
  double* aux_fx,
  double* aux_fy,
  double* aux_fz,
  double* cross_fx,
  double* cross_fy,
  double* cross_fz)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    int type_i = g_type[i];
    double x1 = x[i];
    double y1 = y[i];
    double z1 = z[i];
    for (int n = 0; n < g_NN[i]; ++n) {
      int j = g_NL[i + N * n];
      int type_j = g_type[j];
      int pair_index = type_i * num_types + type_j;
      int kind = uf_kind[pair_index];
      if (kind == 0)
        continue;
      double dx = x[j] - x1;
      double dy = y[j] - y1;
      double dz = z[j] - z1;
      apply_mic(box, dx, dy, dz);
      double r2 = dx * dx + dy * dy + dz * dz;
      double sigma2 = uf_sigma_sqrd[pair_index];
      double p = uf_p[pair_index];
      double exp_value = exp(r2 / sigma2);
      double factor = -2.0 * p / (beta * sigma2 * (exp_value - 1.0));
      double pair_energy = -p / beta * log(1.0 - exp(-r2 / sigma2));
      if (kind == 1) {
        aux_fx[i] += dx * factor;
        aux_fy[i] += dy * factor;
        aux_fz[i] += dz * factor;
        uf_self[i] += 0.5 * pair_energy;
      } else {
        cross_fx[i] += dx * factor;
        cross_fy[i] += dy * factor;
        cross_fz[i] += dz * factor;
        uf_cross[i] += 0.5 * pair_energy;
      }
    }
  }
}

static __global__ void gpu_apply_superionic_stage1(
  int N,
  double lambda,
  const double* aux_fx,
  const double* aux_fy,
  const double* aux_fz,
  const double* cross_fx,
  const double* cross_fy,
  const double* cross_fz,
  double* fx,
  double* fy,
  double* fz)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    fx[i] = aux_fx[i] + lambda * cross_fx[i];
    fy[i] = aux_fy[i] + lambda * cross_fy[i];
    fz[i] = aux_fz[i] + lambda * cross_fz[i];
  }
}

static __global__ void gpu_add_cross_to_aux(
  int N,
  const double* cross_fx,
  const double* cross_fy,
  const double* cross_fz,
  double* aux_fx,
  double* aux_fy,
  double* aux_fz)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    aux_fx[i] += cross_fx[i];
    aux_fy[i] += cross_fy[i];
    aux_fz[i] += cross_fz[i];
  }
}

static __global__ void gpu_apply_superionic_stage2(
  int N,
  double lambda,
  const double* aux_fx,
  const double* aux_fy,
  const double* aux_fz,
  double* fx,
  double* fy,
  double* fz)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    fx[i] = (1.0 - lambda) * aux_fx[i] + lambda * fx[i];
    fy[i] = (1.0 - lambda) * aux_fy[i] + lambda * fy[i];
    fz[i] = (1.0 - lambda) * aux_fz[i] + lambda * fz[i];
  }
}

static __global__ void gpu_sum_array(const int N, double* data)
{
  int tid = threadIdx.x;
  int number_of_patches = (N - 1) / 1024 + 1;
  __shared__ double s_data[1024];
  s_data[tid] = 0.0;
  for (int patch = 0; patch < number_of_patches; ++patch) {
    int n = tid + patch * 1024;
    if (n < N)
      s_data[tid] += data[n];
  }
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset)
      s_data[tid] += s_data[tid + offset];
    __syncthreads();
  }
  if (tid == 0)
    data[0] = s_data[0];
}
} // namespace
```

Then add class methods in the header and implementation:

```cpp
  void find_thermo();
  void find_reference_forces(Force& force);
  double get_sum(GPU_Vector<double>& data);
  void accumulate_work();
```

Implement them:

```cpp
void Ensemble_TI_Superionic::find_thermo()
{
  Ensemble::find_thermo(
    false,
    box->get_volume(),
    *group,
    atom->mass,
    atom->potential_per_atom,
    atom->velocity_per_atom,
    atom->virial_per_atom,
    *thermo);
  thermo->copy_to_host(thermo_cpu.data());
  pe = thermo_cpu[1];
}

double Ensemble_TI_Superionic::get_sum(GPU_Vector<double>& data)
{
  double value = 0.0;
  gpu_sum_array<<<1, 1024>>>(atom->number_of_atoms, data.data());
  data.copy_to_host(&value, 1);
  return value;
}

void Ensemble_TI_Superionic::find_reference_forces(Force& force)
{
  int N = atom->number_of_atoms;
  gpu_zero_superionic_arrays<<<(N - 1) / 128 + 1, 128>>>(
    N,
    gpu_einstein.data(),
    gpu_uf_self.data(),
    gpu_uf_cross.data(),
    gpu_aux_fx.data(),
    gpu_aux_fy.data(),
    gpu_aux_fz.data(),
    gpu_cross_fx.data(),
    gpu_cross_fy.data(),
    gpu_cross_fz.data());
  GPU_CHECK_KERNEL

  gpu_find_superionic_spring<<<(N - 1) / 128 + 1, 128>>>(
    N,
    *box,
    gpu_k.data(),
    gpu_spring_mask.data(),
    atom->position_per_atom.data(),
    atom->position_per_atom.data() + N,
    atom->position_per_atom.data() + 2 * N,
    position_0.data(),
    position_0.data() + N,
    position_0.data() + 2 * N,
    gpu_einstein.data(),
    gpu_aux_fx.data(),
    gpu_aux_fy.data(),
    gpu_aux_fz.data());
  GPU_CHECK_KERNEL

  const GPU_Vector<int>& NN = force.potentials[0]->get_NN_radial_ptr();
  const GPU_Vector<int>& NL = force.potentials[0]->get_NL_radial_ptr();
  if (NN.size() == 0 || NL.size() == 0)
    PRINT_INPUT_ERROR("The main potential must provide a radial neighbor list for ti_superionic.");
  gpu_find_superionic_uf<<<(N - 1) / 128 + 1, 128>>>(
    N,
    num_types,
    *box,
    beta,
    atom->type.data(),
    NN.data(),
    NL.data(),
    gpu_uf_p.data(),
    gpu_uf_sigma_sqrd.data(),
    gpu_uf_kind.data(),
    atom->position_per_atom.data(),
    atom->position_per_atom.data() + N,
    atom->position_per_atom.data() + 2 * N,
    gpu_uf_self.data(),
    gpu_uf_cross.data(),
    gpu_aux_fx.data(),
    gpu_aux_fy.data(),
    gpu_aux_fz.data(),
    gpu_cross_fx.data(),
    gpu_cross_fy.data(),
    gpu_cross_fz.data());
  GPU_CHECK_KERNEL

  U_einstein = get_sum(gpu_einstein);
  U_uf_self = get_sum(gpu_uf_self);
  U_uf_cross = get_sum(gpu_uf_cross);
  U_aux = U_einstein + U_uf_self + U_uf_cross;
}

void Ensemble_TI_Superionic::accumulate_work()
{
  double increment = dHdlambda * dlambda / atom->number_of_atoms;
  if (dlambda > 0.0) {
    W_forward += increment;
  } else if (dlambda < 0.0) {
    W_backward += increment;
  }
  delta_F = 0.5 * (W_forward - W_backward);
}
```

- [ ] **Step 4: Replace compute path with real physics**

In `init()`, set:

```cpp
  beta = 1.0 / (temperature * K_B);
```

In `compute3()`, replace the stub body with:

```cpp
  find_lambda();
  find_thermo();
  find_reference_forces(force);

  int N = atom->number_of_atoms;
  if (stage == SuperionicStage::stage1) {
    dHdlambda = U_uf_cross;
    gpu_apply_superionic_stage1<<<(N - 1) / 128 + 1, 128>>>(
      N,
      lambda,
      gpu_aux_fx.data(),
      gpu_aux_fy.data(),
      gpu_aux_fz.data(),
      gpu_cross_fx.data(),
      gpu_cross_fy.data(),
      gpu_cross_fz.data(),
      atom->force_per_atom.data(),
      atom->force_per_atom.data() + N,
      atom->force_per_atom.data() + 2 * N);
  } else {
    dHdlambda = pe - U_aux;
    gpu_add_cross_to_aux<<<(N - 1) / 128 + 1, 128>>>(
      N,
      gpu_cross_fx.data(),
      gpu_cross_fy.data(),
      gpu_cross_fz.data(),
      gpu_aux_fx.data(),
      gpu_aux_fy.data(),
      gpu_aux_fz.data());
    gpu_apply_superionic_stage2<<<(N - 1) / 128 + 1, 128>>>(
      N,
      lambda,
      gpu_aux_fx.data(),
      gpu_aux_fy.data(),
      gpu_aux_fz.data(),
      atom->force_per_atom.data(),
      atom->force_per_atom.data() + N,
      atom->force_per_atom.data() + 2 * N);
  }
  GPU_CHECK_KERNEL

  if (lambda_active) {
    accumulate_work();
    if (stage == SuperionicStage::stage1) {
      fprintf(
        output_file,
        "%e,%e,%e,%e,%e,%e\n",
        lambda,
        dlambda,
        U_einstein / N,
        U_uf_self / N,
        U_uf_cross / N,
        dHdlambda / N);
    } else {
      fprintf(
        output_file,
        "%e,%e,%e,%e,%e,%e,%e,%e\n",
        lambda,
        dlambda,
        pe / N,
        U_einstein / N,
        U_uf_self / N,
        U_uf_cross / N,
        U_aux / N,
        dHdlambda / N);
    }
  }

  Ensemble_LAN::compute2(time_step, group, box, atoms, thermo);
```

Replace the skeleton `find_lambda()` with this version. It is responsible only for setting
`lambda`, `dlambda`, and `V`; it does not write CSV rows.

```cpp
void Ensemble_TI_Superionic::find_lambda()
{
  V = box->get_volume() / atom->number_of_atoms;
  lambda = 0.0;
  dlambda = 0.0;
  lambda_active = false;

  const int t = *current_step - t_equil;
  const double r_switch = 1.0 / t_switch;

  if ((t >= 0) && (t <= t_switch)) {
    lambda = switch_func(t * r_switch);
    dlambda = dswitch_func(t * r_switch);
    lambda_active = true;
  } else if ((t >= t_equil + t_switch) && (t <= (t_equil + 2 * t_switch))) {
    lambda = switch_func(1.0 - (t - t_switch - t_equil) * r_switch);
    dlambda = -dswitch_func(1.0 - (t - t_switch - t_equil) * r_switch);
    lambda_active = true;
  }
}
```

This replacement removes the stub CSV writes from `find_lambda()` and prevents stale `dlambda`
values from being reused during the equilibration plateau between forward and backward switching.

- [ ] **Step 5: Build and run non-stub CSV tests**

Run:

```bash
cd /home/asus/projects/GPUMD-dev-v4.9.1/src
make gpumd
cd /home/asus/projects/GPUMD-dev-v4.9.1
pytest tests/gpumd/ti-superionic/test-ti-superionic.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 4**

Run:

```bash
cd /home/asus/projects/GPUMD-dev-v4.9.1
git add src/integrate/ensemble_ti_superionic.cuh src/integrate/ensemble_ti_superionic.cu tests/gpumd/ti-superionic/test-ti-superionic.py
git commit -m "Implement superionic TI stage forces"
```

Expected: commit succeeds.

---

### Task 5: Reference Free Energies And YAML Completeness

**Files:**
- Create: `GPUMD-dev-v4.9.1/src/integrate/uf_reference.cuh`
- Modify: `GPUMD-dev-v4.9.1/src/integrate/ensemble_ti_liquid.cuh`
- Modify: `GPUMD-dev-v4.9.1/src/integrate/ensemble_ti_liquid.cu`
- Modify: `GPUMD-dev-v4.9.1/src/integrate/ensemble_ti_superionic.cuh`
- Modify: `GPUMD-dev-v4.9.1/src/integrate/ensemble_ti_superionic.cu`
- Modify: `GPUMD-dev-v4.9.1/tests/gpumd/ti-superionic/test-ti-superionic.py`

- [ ] **Step 1: Extend tests for reference free-energy metadata**

Append to `tests/gpumd/ti-superionic/test-ti-superionic.py`:

```python
def test_stage_yaml_contains_reference_free_energy(tmp_path):
    result = run_gpumd(tmp_path, "run_stage1.in")
    assert result.returncode == 0, result.stderr
    data = yaml.safe_load((tmp_path / "ti_superionic_stage1.yaml").read_text(encoding="utf-8"))
    assert data["F_Einstein"] != 0.0
    assert data["F_UF_self"] != 0.0
    assert data["F_ref"] == pytest.approx(data["F_Einstein"] + data["F_UF_self"])
    assert data["spring_species"] == ["C"]
    assert data["uf_self_pairs"][0]["element_i"] == "H"
    assert data["uf_cross_pairs"][0]["element_i"] == "C"
```

- [ ] **Step 2: Run the new test to verify it fails**

Run:

```bash
cd /home/asus/projects/GPUMD-dev-v4.9.1
pytest tests/gpumd/ti-superionic/test-ti-superionic.py::test_stage_yaml_contains_reference_free_energy -q
```

Expected: FAIL because `F_Einstein` and `F_UF_self` are still zero.

- [ ] **Step 3: Extract shared UF spline/free-energy helpers**

Run this one-time mechanical migration from the repository root. It extracts the ten UF spline
tables from `ensemble_ti_liquid.cuh`, creates `uf_reference.cuh`, and removes the old table members
and `fe()` declaration from `Ensemble_TI_Liquid`.

```bash
cd /home/asus/projects/GPUMD-dev-v4.9.1
python3 - <<'PY'
from pathlib import Path
import re

root = Path.cwd()
liquid_h = root / "src" / "integrate" / "ensemble_ti_liquid.cuh"
uf_h = root / "src" / "integrate" / "uf_reference.cuh"
text = liquid_h.read_text(encoding="utf-8")

tables = [
    ("std::vector<double>", "sum_spline1"),
    ("std::vector<double>", "sum_spline25"),
    ("std::vector<double>", "sum_spline50"),
    ("std::vector<double>", "sum_spline75"),
    ("std::vector<double>", "sum_spline100"),
    (
        "std::vector<std::vector<double>>",
        "spline1",
    ),
    (
        "std::vector<std::vector<double>>",
        "spline25",
    ),
    (
        "std::vector<std::vector<double>>",
        "spline50",
    ),
    (
        "std::vector<std::vector<double>>",
        "spline75",
    ),
    (
        "std::vector<std::vector<double>>",
        "spline100",
    ),
]

blocks = {}
for decl, name in tables:
    pattern = re.compile(
        r"  " + re.escape(decl) + r" " + name + r" = \{(?P<body>.*?)\};\n",
        re.DOTALL,
    )
    match = pattern.search(text)
    if match is None:
        raise SystemExit(f"Could not find table {name}")
    blocks[name] = (decl, match.group("body"))
    text = pattern.sub("", text)

text = re.sub(
    r"\n  double fe\(double x, const double coef\[4\], const double sum_spline\[106\], int index\);",
    "",
    text,
)
liquid_h.write_text(text, encoding="utf-8")

accessors = []
for name, (decl, body) in blocks.items():
    accessors.append(
        f"inline const {decl}& {name}()\n"
        "{\n"
        f"  static const {decl} data = {{{body}}};\n"
        "  return data;\n"
        "}\n"
    )

header = """/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once
#include "utilities/error.cuh"
#include <cmath>
#include <vector>

namespace uf_reference
{
struct UFReferenceData
{
  const std::vector<double>& sum_spline;
  const std::vector<std::vector<double>>& spline;
};

""" + "\n".join(accessors) + """
inline double fe(double x, const double coef[4], const std::vector<double>& sum_spline, int index)
{
  double result;
  double x_0 = 0.0;

  if (x < 0.0025) {
    result = coef[0] * (x * x) / 2.0 + coef[1] * x;
    return result;
  } else if (x < 0.1) {
    if (static_cast<int>(x * 10000) % 25 == 0) {
      return sum_spline[index - 1];
    } else {
      x_0 = 0.0025 * static_cast<int>(x * 400);
    }
  } else if (x < 1) {
    if (static_cast<int>(x * 1000) % 25 == 0) {
      return sum_spline[index - 1];
    } else {
      x_0 = 0.025 * static_cast<int>(x * 40);
    }
  } else if (x < 4) {
    if (static_cast<int>(x * 100) % 10 == 0) {
      return sum_spline[index - 1];
    } else {
      x_0 = 0.1 * static_cast<int>(x * 10);
    }
  } else {
    return sum_spline[index];
  }

  result = sum_spline[index - 1] + coef[0] * (x * x - x_0 * x_0) / 2.0 +
           coef[1] * (x - x_0) + (coef[2] - 1.0) * std::log(x / x_0) -
           coef[3] * (1.0 / x - 1.0 / x_0);

  return result;
}

inline bool supports_p(double p) { return p == 1 || p == 25 || p == 50 || p == 75 || p == 100; }

inline UFReferenceData get_data(double p)
{
  if (p == 1)
    return {sum_spline1(), spline1()};
  if (p == 25)
    return {sum_spline25(), spline25()};
  if (p == 50)
    return {sum_spline50(), spline50()};
  if (p == 75)
    return {sum_spline75(), spline75()};
  if (p == 100)
    return {sum_spline100(), spline100()};
  PRINT_INPUT_ERROR("Self UF p must be 1, 25, 50, 75, or 100.");
  return {sum_spline1(), spline1()};
}
} // namespace uf_reference
"""
uf_h.write_text(header, encoding="utf-8")
PY
```

Expected: `src/integrate/uf_reference.cuh` exists, and `src/integrate/ensemble_ti_liquid.cuh`
no longer contains the UF spline table data.

Delete the old `Ensemble_TI_Liquid::fe(...)` method from `src/integrate/ensemble_ti_liquid.cu`.
Run this mechanical edit:

```bash
cd /home/asus/projects/GPUMD-dev-v4.9.1
python3 - <<'PY'
from pathlib import Path
import re

path = Path("src/integrate/ensemble_ti_liquid.cu")
text = path.read_text(encoding="utf-8")

text = text.replace(
    '#include "ensemble_ti_liquid.cuh"\n',
    '#include "ensemble_ti_liquid.cuh"\n#include "uf_reference.cuh"\n',
    1,
)

text = re.sub(
    r"\ndouble\nEnsemble_TI_Liquid::fe\(double x, const double coef\[4\], const double sum_spline\[106\], int index\)\n\{.*?\n\}\n\nvoid Ensemble_TI_Liquid::init\(\)",
    "\nvoid Ensemble_TI_Liquid::init()",
    text,
    count=1,
    flags=re.DOTALL,
)

old = r"""  double coef[4] = {0.0};
  for (int n = 0; n < 4; n++) {
    if (p == 1) {
      coef[n] = spline1[index][n];
    } else if (p == 25) {
      coef[n] = spline25[index][n];
    } else if (p == 50) {
      coef[n] = spline50[index][n];
    } else if (p == 75) {
      coef[n] = spline75[index][n];
    } else if (p == 100) {
      coef[n] = spline100[index][n];
    }
  }

  double sum_spline[106] = {0.0};
  for (int n = 0; n < 106; n++) {
    if (p == 1) {
      sum_spline[n] = sum_spline1[n];
    } else if (p == 25) {
      sum_spline[n] = sum_spline25[n];
    } else if (p == 50) {
      sum_spline[n] = sum_spline50[n];
    } else if (p == 75) {
      sum_spline[n] = sum_spline75[n];
    } else {
      sum_spline[n] = sum_spline100[n];
    }
  }
  double F_UF = 0;

  F_UF = fe(x_UF, coef, sum_spline, index) * kT * N;
"""
new = r"""  const auto uf_data = uf_reference::get_data(p);
  const std::vector<double>& sum_spline = uf_data.sum_spline;
  const std::vector<std::vector<double>>& spline = uf_data.spline;
  double coef[4] = {0.0};
  for (int n = 0; n < 4; n++)
    coef[n] = spline[index][n];

  double F_UF = uf_reference::fe(x_UF, coef, sum_spline, index) * kT * N;
"""
if old not in text:
    raise SystemExit("Could not find ti_liquid destructor UF free-energy block")
text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")
PY
```

Expected: `src/integrate/ensemble_ti_liquid.cu` now includes:

```cpp
#include "uf_reference.cuh"
```
and `Ensemble_TI_Liquid::~Ensemble_TI_Liquid()` contains:

```cpp
  const auto uf_data = uf_reference::get_data(p);
  const std::vector<double>& sum_spline = uf_data.sum_spline;
  const std::vector<std::vector<double>>& spline = uf_data.spline;
  double coef[4] = {0.0};
  for (int n = 0; n < 4; n++)
    coef[n] = spline[index][n];

  double F_UF = uf_reference::fe(x_UF, coef, sum_spline, index) * kT * N;
```

Run:

```bash
cd /home/asus/projects/GPUMD-dev-v4.9.1
rg -n "sum_spline1|spline100|Ensemble_TI_Liquid::fe|double sum_spline\\[106\\]" src/integrate/ensemble_ti_liquid.cuh src/integrate/ensemble_ti_liquid.cu src/integrate/uf_reference.cuh
```

Expected: matches for `sum_spline1` and `spline100` appear only in
`src/integrate/uf_reference.cuh`; there are no matches for `Ensemble_TI_Liquid::fe` or
`double sum_spline[106]`.

In `ensemble_ti_superionic.cuh`, add:

```cpp
  double get_uf_fe_for_pair(const SuperionicUFPair& pair, int count);
  void compute_reference_free_energy();
```

Implement:

```cpp
double Ensemble_TI_Superionic::get_uf_fe_for_pair(const SuperionicUFPair& pair, int count)
{
  double kT = K_B * temperature;
  double volume_per_atom = box->get_volume() / atom->number_of_atoms;
  double species_volume = box->get_volume() / count;
  double sigma_sqrd = pair.sigma * pair.sigma;
  double x_UF = pow(PI * sigma_sqrd, 1.5) / (2.0 * species_volume);
  int index = 0;
  if (x_UF < 0.1) {
    index = static_cast<int>(x_UF * 400);
  } else if (x_UF < 1) {
    index = 40 + static_cast<int>(x_UF * 40 - 4);
  } else if (x_UF < 4) {
    index = 76 + static_cast<int>(x_UF * 10 - 10);
  } else {
    index = 105;
  }

  const auto uf_data = uf_reference::get_data(pair.p);
  const std::vector<double>& sum_spline = uf_data.sum_spline;
  const std::vector<std::vector<double>>& spline = uf_data.spline;
  double coef[4] = {spline[index][0], spline[index][1], spline[index][2], spline[index][3]};

  double F_UF = uf_reference::fe(x_UF, coef, sum_spline, index) * kT * count;
  double mass = 0.0;
  int type = find_type_for_symbol(pair.element_i);
  for (int i = 0; i < atom->number_of_atoms; ++i) {
    if (atom->cpu_type[i] == type) {
      mass = atom->cpu_mass[i];
      break;
    }
  }
  double de_broglie = log(HBAR * sqrt(2 * PI / (mass * kT)));
  double F_IG = count * kT * (log(1.0 / species_volume) - 1.0) + 3.0 * kT * count * de_broglie;
  return (F_UF + F_IG) / atom->number_of_atoms;
}
```

- [ ] **Step 4: Add Einstein reference free energy**

Implement:

```cpp
void Ensemble_TI_Superionic::compute_reference_free_energy()
{
  double kT = K_B * temperature;
  F_Einstein = 0.0;
  for (int i = 0; i < atom->number_of_atoms; ++i) {
    if (cpu_spring_mask[i] > 0.5) {
      double omega = sqrt(cpu_k[i] / atom->cpu_mass[i]);
      F_Einstein += log(omega * HBAR / kT);
    }
  }
  F_Einstein = 3.0 * kT * F_Einstein / atom->number_of_atoms;

  F_UF_self = 0.0;
  for (const auto& pair : uf_pairs) {
    if (pair.element_i == pair.element_j) {
      int count = 0;
      int type = find_type_for_symbol(pair.element_i);
      for (int i = 0; i < atom->number_of_atoms; ++i) {
        if (atom->cpu_type[i] == type)
          count++;
      }
      F_UF_self += get_uf_fe_for_pair(pair, count);
    }
  }
  F_ref = F_Einstein + F_UF_self;
}
```

Call `compute_reference_free_energy()` in the destructor before writing YAML. If `auto_k` is used, make sure spring constants have already been estimated before the destructor. Task 6 finalizes auto spring.

- [ ] **Step 5: Write computed values into YAML**

Replace zero values in the destructor:

```cpp
  fprintf(yaml_file, "F_Einstein: %f\n", F_Einstein);
  fprintf(yaml_file, "F_UF_self: %f\n", F_UF_self);
  fprintf(yaml_file, "F_ref: %f\n", F_ref);
```

Ensure `W_forward`, `W_backward`, and `delta_F` are written per atom. They already are if Task 4 accumulated per atom.

- [ ] **Step 6: Build and run YAML tests**

Run:

```bash
cd /home/asus/projects/GPUMD-dev-v4.9.1/src
make gpumd
cd /home/asus/projects/GPUMD-dev-v4.9.1
pytest tests/gpumd/ti-superionic/test-ti-superionic.py -q
pytest tests/tools/test_si_free_energy_sum.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 5**

Run:

```bash
cd /home/asus/projects/GPUMD-dev-v4.9.1
git add src/integrate/uf_reference.cuh src/integrate/ensemble_ti_liquid.cuh src/integrate/ensemble_ti_liquid.cu src/integrate/ensemble_ti_superionic.cuh src/integrate/ensemble_ti_superionic.cu tests/gpumd/ti-superionic/test-ti-superionic.py
git commit -m "Report superionic reference free energies"
```

Expected: commit succeeds.

---

### Task 6: Auto Spring Constants, Docs, And Final Verification

**Files:**
- Modify: `GPUMD-dev-v4.9.1/src/integrate/ensemble_ti_superionic.cuh`
- Modify: `GPUMD-dev-v4.9.1/src/integrate/ensemble_ti_superionic.cu`
- Modify: `GPUMD-dev-v4.9.1/tests/gpumd/ti-superionic/test-ti-superionic.py`

- [ ] **Step 1: Add an auto spring fixture and test**

Create `tests/gpumd/ti-superionic/self_consistent/run_stage1_auto.in`:

```text
potential nep.txt
velocity 300
time_step 1
ensemble ti_superionic_stage1 temp 300 tperiod 100 tequil 4 tswitch 4 press 0 spring auto C uf H H 25 1.0 uf C H 10 1.0
run 16
```

Append to `tests/gpumd/ti-superionic/test-ti-superionic.py`:

```python
def test_auto_spring_estimates_finite_reference(tmp_path):
    result = run_gpumd(tmp_path, "run_stage1_auto.in")
    assert result.returncode == 0, result.stderr
    data = yaml.safe_load((tmp_path / "ti_superionic_stage1.yaml").read_text(encoding="utf-8"))
    assert data["spring_species"] == ["C"]
    assert data["F_Einstein"] != 0.0
```

- [ ] **Step 2: Run auto spring test to verify it fails**

Run:

```bash
cd /home/asus/projects/GPUMD-dev-v4.9.1
pytest tests/gpumd/ti-superionic/test-ti-superionic.py::test_auto_spring_estimates_finite_reference -q
```

Expected: FAIL because auto spring constants are not computed yet.

- [ ] **Step 3: Add MSD accumulation for auto spring**

In the header, add:

```cpp
  GPU_Vector<double> gpu_msd;
  void accumulate_msd_for_auto_k();
  void finalize_auto_k();
```

Add kernels:

```cpp
static __global__ void gpu_add_superionic_msd(
  int N,
  Box box,
  const double* spring_mask,
  double* msd,
  const double* x,
  const double* y,
  const double* z,
  const double* x0,
  const double* y0,
  const double* z0)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N && spring_mask[i] > 0.5) {
    double dx = x[i] - x0[i];
    double dy = y[i] - y0[i];
    double dz = z[i] - z0[i];
    apply_mic(box, dx, dy, dz);
    msd[i] += dx * dx + dy * dy + dz * dz;
  }
}
```

Resize `gpu_msd` in `prepare_reference_state()`:

```cpp
  gpu_msd.resize(N, 0.0);
```

Implement:

```cpp
void Ensemble_TI_Superionic::accumulate_msd_for_auto_k()
{
  if (!auto_k || *current_step >= t_equil)
    return;
  int N = atom->number_of_atoms;
  gpu_add_superionic_msd<<<(N - 1) / 128 + 1, 128>>>(
    N,
    *box,
    gpu_spring_mask.data(),
    gpu_msd.data(),
    atom->position_per_atom.data(),
    atom->position_per_atom.data() + N,
    atom->position_per_atom.data() + 2 * N,
    position_0.data(),
    position_0.data() + N,
    position_0.data() + 2 * N);
  GPU_CHECK_KERNEL
}

void Ensemble_TI_Superionic::finalize_auto_k()
{
  if (!auto_k || *current_step != t_equil - 1)
    return;
  std::vector<double> msd(atom->number_of_atoms);
  gpu_msd.copy_to_host(msd.data());
  std::map<std::string, double> msd_by_symbol;
  std::map<std::string, int> count_by_symbol;
  for (int i = 0; i < atom->number_of_atoms; ++i) {
    if (cpu_spring_mask[i] > 0.5) {
      const std::string& symbol = atom->cpu_atom_symbol[i];
      msd_by_symbol[symbol] += msd[i];
      count_by_symbol[symbol] += 1;
    }
  }
  for (const auto& entry : count_by_symbol) {
    const std::string& symbol = entry.first;
    double mean_msd = msd_by_symbol[symbol] / (entry.second * t_equil);
    if (mean_msd <= 0.0)
      PRINT_INPUT_ERROR("Cannot estimate spring constant from zero MSD.");
    spring_map[symbol] = 3.0 * K_B * temperature / mean_msd;
    printf("  %s --- %f eV/A^2\n", symbol.c_str(), spring_map[symbol]);
  }
  for (int i = 0; i < atom->number_of_atoms; ++i) {
    if (cpu_spring_mask[i] > 0.5) {
      const std::string& symbol = atom->cpu_atom_symbol[i];
      cpu_k[i] = spring_map[symbol];
    }
  }
  gpu_k.copy_from_host(cpu_k.data());
}
```

Call both from `find_lambda()` before the switching interval:

```cpp
  accumulate_msd_for_auto_k();
  finalize_auto_k();
```

- [ ] **Step 4: Ensure reference free energy waits for auto spring**

In `compute_reference_free_energy()`, before using `cpu_k`, reject unset spring constants:

```cpp
      if (cpu_k[i] <= 0.0)
        PRINT_INPUT_ERROR("Spring constant is not available for a spring atom.");
```

This catches too-short runs where `t_equil` was never completed.

- [ ] **Step 5: Run all local tests**

Run:

```bash
cd /home/asus/projects/GPUMD-dev-v4.9.1/src
make gpumd
cd /home/asus/projects/GPUMD-dev-v4.9.1
pytest tests/gpumd/ti-superionic/test-ti-superionic.py -q
pytest tests/tools/test_si_free_energy_sum.py -q
pytest tests/gpumd/ti-liquid/test-ti-liquid.py -q
```

Expected: PASS. If `tests/gpumd/ti-liquid/test-ti-liquid.py` cannot run because local Python packages such as `ase` or `calorine` are missing, record the missing dependency and run at least the superionic and tool tests.

- [ ] **Step 6: Run a source-level sanity scan**

Run:

```bash
cd /home/asus/projects/GPUMD-dev-v4.9.1
pattern='TO''DO|TB''D|FIX''ME|printf\(\"debug|cout'
rg -n "$pattern" src/integrate/ensemble_ti_superionic.cu src/integrate/ensemble_ti_superionic.cuh tools/si_free_energy_sum.py tests/gpumd/ti-superionic tests/tools
```

Expected: no unwanted deferred-work markers or debug output. Existing words in comments are
acceptable only if they describe a real documented behavior, not unfinished implementation code.

- [ ] **Step 7: Commit Task 6**

Run:

```bash
cd /home/asus/projects/GPUMD-dev-v4.9.1
git add src/integrate/ensemble_ti_superionic.cuh src/integrate/ensemble_ti_superionic.cu tests/gpumd/ti-superionic/test-ti-superionic.py tests/gpumd/ti-superionic/self_consistent/run_stage1_auto.in
git commit -m "Finalize superionic TI validation"
```

Expected: commit succeeds.

---

## Final Verification

Run:

```bash
cd /home/asus/projects/GPUMD-dev-v4.9.1/src
make clean
make gpumd
cd /home/asus/projects/GPUMD-dev-v4.9.1
pytest tests/tools/test_si_free_energy_sum.py -q
pytest tests/gpumd/ti-superionic/test-ti-superionic.py -q
pytest tests/gpumd/ti-liquid/test-ti-liquid.py -q
git status --short
```

Expected:

- `make gpumd` succeeds.
- New post-processing tests pass.
- New superionic stage tests pass.
- Existing `ti_liquid` test passes, or a missing local test dependency is explicitly reported.
- `git status --short` shows no uncommitted source/test/doc changes after the final commit.

## Plan Self-Review Checklist

- Spec coverage: The tasks cover both stage commands, shared `.cu/.cuh`, direct ensemble syntax, no UF cutoff argument, explicit and auto spring constants, Stage 1 `ref -> aux`, Stage 2 `aux -> target`, CSV/YAML outputs, no error estimates, independent stage YAML, and summary-tool aggregation of `F_target`/`G_target`.
- Scope check: The work is one coherent GPUMD feature plus a small post-processing tool. It does not mix in RS/NPT, replica statistics, or automatic cutoff validation.
- Type consistency: `SuperionicStage`, `SuperionicUFPair`, and `Ensemble_TI_Superionic` are introduced before later tasks reference them. GPU buffers use existing `GPU_Vector<T>` APIs.
