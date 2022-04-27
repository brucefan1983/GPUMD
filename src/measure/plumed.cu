/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, Mikko Ervasti, and Ari Harju
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

#include "plumed.cuh"
#include "utilities/error.cuh"
#include "utilities/common.cuh"
#include "utilities/read_file.cuh"
#include "utilities/gpu_vector.cuh"

static __global__ void gpu_sum(
  const int N,
  const double* g_data,
  double* g_data_sum)
{
  int number_of_rounds = (N - 1) / 1024 + 1;
  __shared__ double s_data[1024];
  s_data[threadIdx.x] = 0.0;
  for (int round = 0; round < number_of_rounds; ++round) {
    int n = threadIdx.x + round * 1024;
    if (n < N) {
      s_data[threadIdx.x] += g_data[n + blockIdx.x * N];
    }
  }
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      s_data[threadIdx.x] += s_data[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    g_data_sum[blockIdx.x] = s_data[0];
  }
}

static void __global__ gpu_scale_virial(
  const int N,
  const double *factors,
  double* g_sxx,
  double* g_syy,
  double* g_szz,
  double* g_sxy,
  double* g_sxz,
  double* g_syz)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    g_sxx[i] *= factors[0];
    g_syy[i] *= factors[4];
    g_szz[i] *= factors[8];
    g_sxy[i] *= factors[1];
    g_sxz[i] *= factors[2];
    g_syz[i] *= factors[5];
  }
}

void PLUMED::preprocess(const std::vector<double>& cpu_mass)
{
  n_atom = cpu_mass.size();
  gpu_v_vector.resize(6);
  gpu_v_factor.resize(9);
  cpu_b_vector = std::vector<double>(9);
  cpu_v_vector = std::vector<double>(9);
  cpu_v_factor = std::vector<double>(9);
  cpu_m_vector = std::vector<double>(3 * n_atom);
  cpu_f_vector = std::vector<double>(3 * n_atom);
  cpu_q_vector = std::vector<double>(3 * n_atom);
  memcpy(cpu_m_vector.data(), cpu_mass.data(), n_atom * sizeof(double));
}

void PLUMED::parse(char **param, int num_param)
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
  plumed_cmd(plumed_main, "setKbT",           &KbT);
  plumed_cmd(plumed_main, "setMDEngine",      engine_name);
  plumed_cmd(plumed_main, "setMDTimeUnits",   &time_unit);
  plumed_cmd(plumed_main, "setMDMassUnits",   &mass_unit);
  plumed_cmd(plumed_main, "setMDEnergyUnits", &energy_unit);
  plumed_cmd(plumed_main, "setMDLengthUnits", &length_unit);
  plumed_cmd(plumed_main, "setMDChargeUnits", &charge_unit);
  plumed_cmd(plumed_main, "setPlumedDat",     input_file);
  plumed_cmd(plumed_main, "setLogFile",       output_file);
  plumed_cmd(plumed_main, "setTimestep",      &time_step);
  plumed_cmd(plumed_main, "setRestart",       &restart);
  plumed_cmd(plumed_main, "setNatoms",        &n_atom);
  plumed_cmd(plumed_main, "init",             NULL);
}

void PLUMED::process(
  Box& box,
  GPU_Vector<double>& thermo,
  GPU_Vector<double>& position,
  GPU_Vector<double>& force,
  GPU_Vector<double>& virial)
{
  std::vector<double> tmp(6);
  step += interval;

  force.copy_to_host(cpu_f_vector.data());
  position.copy_to_host(cpu_q_vector.data());

  if (box.triclinic == 0) {
    cpu_b_vector.data()[0] = box.cpu_h[0];
    cpu_b_vector.data()[1] = 0.0;
    cpu_b_vector.data()[2] = 0.0;
    cpu_b_vector.data()[3] = 0.0;
    cpu_b_vector.data()[4] = box.cpu_h[1];
    cpu_b_vector.data()[5] = 0.0;
    cpu_b_vector.data()[6] = 0.0;
    cpu_b_vector.data()[7] = 0.0;
    cpu_b_vector.data()[8] = box.cpu_h[2];
  } else {
    cpu_b_vector.data()[0] = box.cpu_h[0];
    cpu_b_vector.data()[1] = box.cpu_h[3];
    cpu_b_vector.data()[2] = box.cpu_h[6];
    cpu_b_vector.data()[3] = box.cpu_h[1];
    cpu_b_vector.data()[4] = box.cpu_h[4];
    cpu_b_vector.data()[5] = box.cpu_h[7];
    cpu_b_vector.data()[6] = box.cpu_h[2];
    cpu_b_vector.data()[7] = box.cpu_h[5];
    cpu_b_vector.data()[8] = box.cpu_h[8];
  }

  gpu_sum<<<6, 1024>>>(n_atom, virial.data(), gpu_v_vector.data());
  CUDA_CHECK_KERNEL
  gpu_v_vector.copy_to_host(tmp.data());
  cpu_v_vector.data()[0] = tmp.data()[0];
  cpu_v_vector.data()[1] = tmp.data()[3];
  cpu_v_vector.data()[2] = tmp.data()[4];
  cpu_v_vector.data()[3] = tmp.data()[3];
  cpu_v_vector.data()[4] = tmp.data()[1];
  cpu_v_vector.data()[5] = tmp.data()[5];
  cpu_v_vector.data()[6] = tmp.data()[4];
  cpu_v_vector.data()[7] = tmp.data()[5];
  cpu_v_vector.data()[8] = tmp.data()[2];

  plumed_cmd(plumed_main, "setStep",       &step);
  plumed_cmd(plumed_main, "setMasses",     cpu_m_vector.data());
  plumed_cmd(plumed_main, "setBox",        cpu_b_vector.data());
  plumed_cmd(plumed_main, "setVirial",     cpu_v_vector.data());
  plumed_cmd(plumed_main, "setForcesX",    &(cpu_f_vector.data()[0 * n_atom]));
  plumed_cmd(plumed_main, "setForcesY",    &(cpu_f_vector.data()[1 * n_atom]));
  plumed_cmd(plumed_main, "setForcesZ",    &(cpu_f_vector.data()[2 * n_atom]));
  plumed_cmd(plumed_main, "setPositionsX", &(cpu_q_vector.data()[0 * n_atom]));
  plumed_cmd(plumed_main, "setPositionsY", &(cpu_q_vector.data()[1 * n_atom]));
  plumed_cmd(plumed_main, "setPositionsZ", &(cpu_q_vector.data()[2 * n_atom]));
  plumed_cmd(plumed_main, "prepareCalc",   NULL);
  plumed_cmd(plumed_main, "performCalc",   NULL);
  plumed_cmd(plumed_main, "getBias",       &bias_energy);
  plumed_cmd(plumed_main, "setStopFlag",   &stop_flag);

  force.copy_from_host(cpu_f_vector.data());

  cpu_v_factor.data()[0] = cpu_v_vector.data()[0] / tmp.data()[0];
  cpu_v_factor.data()[1] = cpu_v_vector.data()[1] / tmp.data()[3];
  cpu_v_factor.data()[2] = cpu_v_vector.data()[2] / tmp.data()[4];
  cpu_v_factor.data()[3] = cpu_v_vector.data()[3] / tmp.data()[3];
  cpu_v_factor.data()[4] = cpu_v_vector.data()[4] / tmp.data()[1];
  cpu_v_factor.data()[5] = cpu_v_vector.data()[5] / tmp.data()[5];
  cpu_v_factor.data()[6] = cpu_v_vector.data()[6] / tmp.data()[4];
  cpu_v_factor.data()[7] = cpu_v_vector.data()[7] / tmp.data()[5];
  cpu_v_factor.data()[8] = cpu_v_vector.data()[8] / tmp.data()[2];
  gpu_v_factor.copy_from_host(cpu_v_factor.data());
  gpu_scale_virial<<<(n_atom - 1) / 128 + 1, n_atom>>>(
    n_atom, gpu_v_factor.data(),
    virial.data() + n_atom * 0, virial.data() + n_atom * 1,
    virial.data() + n_atom * 2, virial.data() + n_atom * 3,
    virial.data() + n_atom * 4, virial.data() + n_atom * 5);
  CUDA_CHECK_KERNEL
}

PLUMED::~PLUMED(void)
{
  plumed_finalize(plumed_main);
}