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

/*----------------------------------------------------------------------------80
Interface to the PLUMED plugin: https://www.plumed.org
------------------------------------------------------------------------------*/

#ifdef USE_PLUMED

#include "plumed.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/read_file.cuh"
#include <cstring>

#define E_C 1.602176634E-19 // Elementary charge
#define N_A 6.0221367E23    // Avogadro constant

const double ENERGY_UNIT_CONVERSION = N_A * E_C / 1000; // from eV to kJ/mol

static __global__ void gpu_add_plumed_virial(
  const int N,
  const double vxx,
  const double vyy,
  const double vzz,
  const double vxy,
  const double vxz,
  const double vyz,
  const double vyx,
  const double vzx,
  const double vzy,
  double* g_sxx,
  double* g_syy,
  double* g_szz,
  double* g_sxy,
  double* g_sxz,
  double* g_syz,
  double* g_syx,
  double* g_szx,
  double* g_szy)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    const double inv_N = 1.0 / N;
    g_sxx[i] += vxx * inv_N;
    g_syy[i] += vyy * inv_N;
    g_szz[i] += vzz * inv_N;
    g_sxy[i] += vxy * inv_N;
    g_sxz[i] += vxz * inv_N;
    g_syz[i] += vyz * inv_N;
    g_syx[i] += vyx * inv_N;
    g_szx[i] += vzx * inv_N;
    g_szy[i] += vzy * inv_N;
  }
}

void PLUMED::preprocess(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  n_atom = atom.number_of_atoms;
  cpu_b_vector = std::vector<double>(9);
  cpu_v_vector = std::vector<double>(9);
  cpu_m_vector = std::vector<double>(3 * n_atom);
  cpu_f_vector = std::vector<double>(3 * n_atom);
  cpu_q_vector = std::vector<double>(3 * n_atom);
  memcpy(cpu_m_vector.data(), atom.cpu_mass.data(), n_atom * sizeof(double));

  init(time_step, integrate.temperature);
}

PLUMED::PLUMED(const char** param, int num_param)
{
  parse(param, num_param);
  action_name = "plumed";
}

void PLUMED::parse(const char** param, int num_param)
{
  use_plumed = 1;
  memset(input_file, 0, 80);
  if (!plumed_installed()) {
    PRINT_INPUT_ERROR("PLUMED not installed!\n");
  }
  if (num_param != 4) {
    PRINT_INPUT_ERROR("plumed should have 3 parameters.");
  }
  sprintf(input_file, "%s", param[1]);
  sprintf(output_file, "%s.out", param[1]);
  if (!is_valid_int(param[2], &interval)) {
    PRINT_INPUT_ERROR("plumed invoke interval should be an integer.");
  }
  if (interval <= 0) {
    PRINT_INPUT_ERROR("plumed invoke interval should > 0.");
  }
  if (!is_valid_int(param[3], &restart)) {
    PRINT_INPUT_ERROR("plumed restart parameter should be 0 or 1.");
  }
  printf("Use PLUMED for this run.\n");
  printf("    input  file: '%s'.\n", input_file);
  printf("    output file: '%s'.\n", output_file);
  printf("    invoke freq: every %d steps.\n", interval);
  if (restart) {
    printf("    will restart calculations from old files.\n");
  }
}

void PLUMED::init(const double ts, const double T)
{
  step = 0;
  time_step = ts;

  const char engine_name[7] = "GPUMD\0";                // my name
  const double KbT = K_B * T;                           // eV
  const double time_unit = TIME_UNIT_CONVERSION / 1000; // natural -> ps
  const double mass_unit = 1.0;                         // amu. -> amu.
  const double energy_unit = ENERGY_UNIT_CONVERSION;    // ev -> kJ/mol
  const double length_unit = 0.1;                       // Ang -> nm
  const double charge_unit = 1.0;                       // e -> e

  plumed_main = plumed_create();
  plumed_cmd(plumed_main, "setKbT", &KbT);
  plumed_cmd(plumed_main, "setMDEngine", engine_name);
  plumed_cmd(plumed_main, "setMDTimeUnits", &time_unit);
  plumed_cmd(plumed_main, "setMDMassUnits", &mass_unit);
  plumed_cmd(plumed_main, "setMDEnergyUnits", &energy_unit);
  plumed_cmd(plumed_main, "setMDLengthUnits", &length_unit);
  plumed_cmd(plumed_main, "setMDChargeUnits", &charge_unit);
  plumed_cmd(plumed_main, "setPlumedDat", input_file);
  plumed_cmd(plumed_main, "setLogFile", output_file);
  plumed_cmd(plumed_main, "setTimestep", &time_step);
  plumed_cmd(plumed_main, "setRestart", &restart);
  plumed_cmd(plumed_main, "setNatoms", &n_atom);
  plumed_cmd(plumed_main, "init", NULL);
}

void PLUMED::calculate(int plumed_step, Box& box, Atom& atom)
{
  atom.force_per_atom.copy_to_host(cpu_f_vector.data());
  atom.position_per_atom.copy_to_host(cpu_q_vector.data());

  cpu_b_vector[0] = box.cpu_h[0];
  cpu_b_vector[1] = box.cpu_h[3];
  cpu_b_vector[2] = box.cpu_h[6];
  cpu_b_vector[3] = box.cpu_h[1];
  cpu_b_vector[4] = box.cpu_h[4];
  cpu_b_vector[5] = box.cpu_h[7];
  cpu_b_vector[6] = box.cpu_h[2];
  cpu_b_vector[7] = box.cpu_h[5];
  cpu_b_vector[8] = box.cpu_h[8];

  fill(cpu_v_vector.begin(), cpu_v_vector.end(), 0.0);
  stop_flag = 0;

  plumed_cmd(plumed_main, "setStep", &plumed_step);
  plumed_cmd(plumed_main, "setMasses", cpu_m_vector.data());
  plumed_cmd(plumed_main, "setBox", cpu_b_vector.data());
  plumed_cmd(plumed_main, "setVirial", cpu_v_vector.data());
  plumed_cmd(plumed_main, "setForcesX", &(cpu_f_vector.data()[0 * n_atom]));
  plumed_cmd(plumed_main, "setForcesY", &(cpu_f_vector.data()[1 * n_atom]));
  plumed_cmd(plumed_main, "setForcesZ", &(cpu_f_vector.data()[2 * n_atom]));
  plumed_cmd(plumed_main, "setPositionsX", &(cpu_q_vector.data()[0 * n_atom]));
  plumed_cmd(plumed_main, "setPositionsY", &(cpu_q_vector.data()[1 * n_atom]));
  plumed_cmd(plumed_main, "setPositionsZ", &(cpu_q_vector.data()[2 * n_atom]));
  plumed_cmd(plumed_main, "setStopFlag", &stop_flag);
  plumed_cmd(plumed_main, "prepareCalc", NULL);
  plumed_cmd(plumed_main, "performCalc", NULL);
  plumed_cmd(plumed_main, "getBias", &bias_energy);

  atom.force_per_atom.copy_from_host(cpu_f_vector.data());

  // PLUMED and GPUMD use opposite virial sign conventions. Distribute the
  // global PLUMED bias virial uniformly over atoms so that the total virial
  // seen by the GPUMD pressure/barostat is correct without dividing by the
  // pre-existing physical virial.
  gpu_add_plumed_virial<<<(n_atom - 1) / 128 + 1, 128>>>(
    n_atom,
    -cpu_v_vector[0],
    -cpu_v_vector[4],
    -cpu_v_vector[8],
    -cpu_v_vector[1],
    -cpu_v_vector[2],
    -cpu_v_vector[5],
    -cpu_v_vector[3],
    -cpu_v_vector[6],
    -cpu_v_vector[7],
    atom.virial_per_atom.data() + n_atom * 0,
    atom.virial_per_atom.data() + n_atom * 1,
    atom.virial_per_atom.data() + n_atom * 2,
    atom.virial_per_atom.data() + n_atom * 3,
    atom.virial_per_atom.data() + n_atom * 4,
    atom.virial_per_atom.data() + n_atom * 5,
    atom.virial_per_atom.data() + n_atom * 6,
    atom.virial_per_atom.data() + n_atom * 7,
    atom.virial_per_atom.data() + n_atom * 8);
  GPU_CHECK_KERNEL
}

void PLUMED::process_dynamics(
  const int md_step,
  Box& box,
  Atom& atom)
{
  // Biased PLUMED simulations require interval = 1. In this case PLUMED must
  // modify force/virial before the second integration half-step. The initial
  // md_step = 0 call supplies the bias force for the first half-step.
  if (interval != 1) {
    return;
  }
  dynamics_mode = true;
  step = md_step;
  calculate(step, box, atom);
}

void PLUMED::process(
  const int number_of_steps,
  int step_input,
  const int fixed_group,
  const int move_group,
  const double global_time,
  const double temperature,
  Integrate& integrate,
  Box& box,
  std::vector<Group>& group,
  GPU_Vector<double>& thermo,
  Atom& atom,
  Force& force)
{
  // interval = 1 is handled in process_dynamics() so the PLUMED bias enters
  // the velocity-Verlet integration at the correct point.
  if (dynamics_mode || step_input % interval != 0) {
    return;
  }

  // Keep the historical invocation schedule when the dynamics hook is not used
  // (for example in other executables) and for analysis-only interval > 1.
  step += interval;
  calculate(step, box, atom);
}

void PLUMED::postprocess(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature)
{
  if (use_plumed == 1) {
    use_plumed = 0;
    plumed_finalize(plumed_main);
  }
}

#endif
