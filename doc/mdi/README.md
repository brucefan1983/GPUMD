## MDI interface to external codes

`GPUMD` can act as an MDI ENGINE and be driven by external DRIVER codes via the [MolSSI Driver Interface (MDI)](https://molssi-mdi.github.io/MDI_Library/user_guide/installation.html).
This repository includes a minimal example that couples `GPUMD` to `VASP` for QM/MM-style simulations.

This document collects all information specific to the MDI integration so that the top-level `README` can remain focused on the core `GPUMD` package.

---

## 1. Prerequisites

- **MDI library**
  - Install the MDI library following the official instructions:  
    [MDI installation guide](https://molssi-mdi.github.io/MDI_Library/user_guide/installation.html).
  - Note the installation prefix, in particular:
    - `MDI_INC_PATH`: directory that contains `mdi.h`  
      (for example, `/path/to/MDI_Library/install/include`)
    - `MDI_LIB_PATH`: directory that contains the MDI library  
      (for example, `/path/to/MDI_Library/install/lib64`)

- **VASP**
  - A working VASP executable (for example, `vasp_std` or `mpirun -n N vasp_std`) available in your `PATH` or referenced via a full path.

- **Python**
  - Python 3
  - Python MDI interface:

    ```bash
    pip install mdi
    ```

  - `numpy`

---

## 2. Compiling `GPUMD` with MDI support

From the `src` directory:

```bash
cd src
make USE_MDI=1 MDI_LIB=1 \
  MDI_LIB_PATH=/path/to/MDI_Library/install/lib64 \
  MDI_INC_PATH=/path/to/MDI_Library/install/include
```

Adjust `MDI_LIB_PATH` and `MDI_INC_PATH` to match your local MDI installation.  
This will build a `gpumd` executable that is linked against MDI and can run in ENGINE mode.

---

## 3. Files and layout for the VASP–GPUMD example

MDI-related helper scripts are collected under:

- `tools/mdi/run_mdi_vasp_gpumd.sh`  
  Orchestrates a QM/MM MD simulation where:
  - `GPUMD` runs as MDI ENGINE (MD integrator)
  - `vasp_mdi_driver.py` runs as DRIVER (QM calculator)

- `tools/mdi/vasp_mdi_driver.py`  
  Python MDI DRIVER that:
  - connects to the `GPUMD` ENGINE over MDI,
  - launches VASP for QM calculations,
  - reads forces (and energy) from `vasprun.xml`,
  - sends QM forces and energy back to `GPUMD`.

An example input system is provided in:

- `examples/mdi-interface/` (Cu dimer), containing
  - `INCAR`, `INCAR_MDI_TEMPLATE`, `POSCAR`, `POTCAR`, `run.in`, `model.xyz`

---

## 4. Running the minimal VASP–GPUMD MDI example

Assuming:
- `GPUMD` has been compiled with MDI support as described above, and
- you have a working VASP executable (e.g. `vasp_std`),

you can run the minimal example as follows (from the repository root):

```bash
cd examples/mdi-interface

bash ../../tools/mdi/run_mdi_vasp_gpumd.sh \
  --gpumd-bin ../../src/gpumd \
  --run-in run.in \
  --vasp-cmd "vasp_std" \
  --poscar POSCAR \
  --steps 3
  --no-cleanup
```

Key options:

- `--gpumd-bin`  
  Path to the `gpumd` binary compiled with MDI support.

- `--run-in`  
  GPUMD input file (default: `run.in`).

- `--vasp-cmd`  
  Command used to run VASP (`vasp_std`, `mpirun -n 8 vasp_std`, etc.).

- `--poscar`  
  POSCAR template file (default: `POSCAR_template`).  
  In the minimal test, you can simply use `POSCAR`.

- `--steps`  
  Number of MD steps controlled by the DRIVER.

- `--no-cleanup`
  Keep all the log directories and files

The script:

1. Starts `GPUMD` as MDI ENGINE in the background.
2. Starts the Python VASP DRIVER (`vasp_mdi_driver.py`) as MDI DRIVER.
3. Runs VASP at each MD step to compute QM energies and forces.
4. Sends QM forces and energy back to `GPUMD` via MDI.
5. Lets `GPUMD` perform the MD time integration.

Standard `GPUMD` dump and compute keywords in `run.in` (for example, `dump_force`, `dump_thermo`, etc.) work as usual, so forces and thermodynamic quantities can be written to the standard output files while using QM forces from VASP.

---

## 5. Logs, troubleshooting, and tips

- **Logs**
  - Temporary files are stored under `.gpumd_vasp_tmp/` in the run directory.
  - Persistent logs are saved under `.gpumd_logs/`:
    - `gpumd_*.log` for the ENGINE
    - `vasp_driver_*.log` for the DRIVER

- **MDI / VASP issues**
  - If `vasp_mdi_driver.py` reports missing `mdi`:

    ```bash
    pip install mdi
    ```

  - If VASP fails to converge in one step:
    - Try increasing `NELM` in your INCAR.
    - Use more conservative smearing (`SIGMA`) and a higher `ENCUT` for production runs.

- **INCAR template**
  - The `INCAR_MDI_TEMPLATE` file in `examples/mdi-interface/` is tuned for fast, reliable integration testing.
  - For real systems, start from your production INCAR and make sure:
    - `IBRION = -1` (external MD driver)
    - `NSW = 1` (one ionic step per driver call)

---

## 6. Extending to other DRIVER codes

The current example focuses on `VASP` as the QM DRIVER.  
In principle, any external code that implements the MDI protocol can be used as a DRIVER:

- Replace `vasp_mdi_driver.py` with a DRIVER that:
  - connects to `GPUMD` over MDI,
  - requests coordinates from `GPUMD`,
  - computes energies and forces,
  - sends them back via the appropriate MDI commands.

The MDI-specific logic inside `GPUMD` is generic and not tied to VASP.

