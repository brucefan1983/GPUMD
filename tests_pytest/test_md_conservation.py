"""Short NVE run_custom_md: energy and total-momentum drift bounds.

GPUMD's NVE integrator is velocity-Verlet: local per-step error is O(time_step**3), and over a
FIXED NUMBER OF STEPS the accumulated (global) energy drift scales as O(time_step**2). Energy is
also extensive, so the same per-step numerical error accumulates across more degrees of freedom
in a larger system -- the drift bound below is expressed as
ENERGY_DRIFT_COEFFICIENT * TIME_STEP**2 * n_atoms (eV/fs**2 * fs**2 = eV, scaled per atom) rather
than a bare constant, so it's explicit that both halving TIME_STEP and using a smaller structure
should relax this bound. `velocity <T>` zeros total linear/angular momentum at initialization
(see doc/gpumd/input_parameters/velocity.rst), and NVE has no external forces/torques, so
momentum should stay at ~0 throughout; MOMENTUM_DRIFT_BOUND is an absolute bound on the largest
total-momentum norm observed over the trajectory, not scaled at all, since momentum conservation
is exact at the equations-of-motion level regardless of step size or system size -- any drift
here is purely numerical.
"""
import numpy as np
import pytest
from calorine.calculators import GPUNEP

pytestmark = pytest.mark.slow

TIME_STEP = 1.0  # fs
N_STEPS = 200
DUMP_INTERVAL = 1

ENERGY_DRIFT_COEFFICIENT = 2e-3  # eV / (fs**2 * atom); qNEP charge models carry an extra
# electrostatic energy term evaluated every step on top of the short-range descriptor energy
# plain NEP models already have, so some extra drift relative to plain NEP is expected here.
# Lighter/faster-vibrating structures (e.g. water's H atoms) also drift more per atom than
# heavier, slower-vibrating ones at the same time_step -- this coefficient is calibrated to
# cover the fastest-drifting structure/model combination in this suite with margin.
MOMENTUM_DRIFT_BOUND = 1e-3  # Dalton * Angstrom / fs


def _run_nve(tmp_path, structure, model_path, gpumd_command):
    atoms = structure.copy()
    calc = GPUNEP(str(model_path), command=gpumd_command, directory=str(tmp_path))
    atoms.calc = calc
    params = [
        ('velocity', [300, 'seed', 42]),  # fixed seed: drift is sensitive to the random initial
        # velocity draw, especially for the smallest structure fixtures, so an unseeded run
        # would make this test flaky.
        ('ensemble', 'nve'),
        ('time_step', TIME_STEP),
        ('dump_thermo', DUMP_INTERVAL),
        ('dump_velocity', DUMP_INTERVAL),
        ('run', N_STEPS),
    ]
    calc.run_custom_md(params)
    thermo = np.loadtxt(tmp_path / 'thermo.out')
    velocity = np.loadtxt(tmp_path / 'velocity.out')
    return thermo, velocity


def test_nve_conservation(tmp_path, structure, model_path, gpumd_command):
    thermo, velocity = _run_nve(tmp_path, structure, model_path, gpumd_command)

    n_atoms = len(structure)
    total_energy = thermo[:, 1] + thermo[:, 2]  # kinetic (K) + potential (U)
    energy_drift = np.max(np.abs(total_energy - total_energy[0]))
    energy_bound = ENERGY_DRIFT_COEFFICIENT * TIME_STEP**2 * n_atoms
    assert energy_drift < energy_bound

    n_frames = velocity.shape[0] // n_atoms
    masses = structure.get_masses()
    momentum_per_frame = np.array([
        (masses[:, None] * velocity[frame * n_atoms:(frame + 1) * n_atoms]).sum(axis=0)
        for frame in range(n_frames)
    ])
    momentum_drift = np.max(np.linalg.norm(momentum_per_frame, axis=1))
    assert momentum_drift < MOMENTUM_DRIFT_BOUND
