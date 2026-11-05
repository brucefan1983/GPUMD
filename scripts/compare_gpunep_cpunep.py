"""Standalone GPUNEP vs. CPUNEP comparison utility.

Not part of the pytest suite (tests_pytest/) -- CPUNEP has no role there (see
tests_pytest/conftest.py's module docstring): it's a separately maintained, independently
implemented CPU evaluator not guaranteed to support the same features as GPUMD. This script is
preserved as an opt-in developer utility because it once caught a real bug in calorine's nep_cpu
charge-model force formula (charge neutrality shifts predicted charges by -mean(charge), but the
corresponding dE/dQ term wasn't shifted to match, breaking the force chain rule for qNEP
charge_mode 1/2) -- fixed in calorine commit bf5cac1 by mirroring GPUMD's own zero_mean_D_real
(src/force/nep_charge.cu:608-635).

Run directly: python3 scripts/compare_gpunep_cpunep.py
"""
import sys
from pathlib import Path

import numpy as np
from calorine.calculators import CPUNEP

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'tests_pytest'))

import conftest  # noqa: E402

ENERGY_TOLERANCE = dict(rtol=1e-3, atol=2e-4)
FORCE_TOLERANCE = dict(rtol=1e-3, atol=5e-5)
BEC_TOLERANCE = dict(rtol=1e-3, atol=1e-5)

GPUMD_COMMAND = str(conftest.GPUMD_EXECUTABLE)


def _combinations():
    for structure_name, builder in conftest._STRUCTURE_BUILDERS.items():
        for model_type in ('nep', 'qnep_mode1', 'qnep_mode2'):
            filename = conftest._MODEL_FILES.get((structure_name, model_type))
            if filename is not None:
                yield structure_name, model_type, builder, conftest.MODELS_DIR / filename


def _report(label, ok):
    print(f'  [{"OK" if ok else "FAIL"}] {label}')
    return ok


def main():
    all_ok = True
    for structure_name, model_type, builder, model_path in _combinations():
        print(f'{structure_name} / {model_type}:')

        gpu_atoms = builder()
        gpu_atoms.calc = conftest.make_gpunep(model_path, GPUMD_COMMAND, model_type)
        gpu_energy = gpu_atoms.get_potential_energy()
        gpu_forces = gpu_atoms.get_forces()

        cpu_atoms = builder()
        cpu_atoms.calc = CPUNEP(str(model_path))
        cpu_energy = cpu_atoms.get_potential_energy()
        cpu_forces = cpu_atoms.get_forces()

        all_ok &= _report('energy', bool(np.isclose(gpu_energy, cpu_energy, **ENERGY_TOLERANCE)))
        all_ok &= _report('forces', np.allclose(gpu_forces, cpu_forces, **FORCE_TOLERANCE))

        if model_type != 'nep':
            gpu_bec_atoms = builder()
            gpu_calc = conftest.make_gpunep(model_path, GPUMD_COMMAND, model_type)
            gpu_bec_atoms.calc = gpu_calc
            gpu_bec = gpu_calc.get_born_effective_charges(gpu_bec_atoms)

            cpu_bec_atoms = builder()
            cpu_calc = CPUNEP(str(model_path))
            cpu_bec_atoms.calc = cpu_calc
            cpu_bec = cpu_calc.get_born_effective_charges(cpu_bec_atoms)

            all_ok &= _report(
                'born effective charges', np.allclose(gpu_bec, cpu_bec, **BEC_TOLERANCE))

    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
