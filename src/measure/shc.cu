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
Spectral heat current (SHC) calculations. Referene:
[1] Z. Fan, H. Dong, A. Harju, T. Ala-Nissila, Homogeneous nonequilibrium
molecular dynamics method for heat transport and spectral decomposition
with many-body potentials, Phys. Rev. B 99, 064308 (2019).
------------------------------------------------------------------------------*/

#include "model/group.cuh"
#include "parse_utilities.cuh"
#include "shc.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <cstring>

const int BLOCK_SIZE_SHC = 128;

void SHC::preprocess(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  if (!compute) {
    return;
  }

  num_time_origins = 0;
  if (-1 == group_method) {
    group_size = atom.number_of_atoms;
    group_num = 1;
  } else {
    if (group_id == -1) {
      group_size = atom.number_of_atoms;
      group_num = group[group_method].number;
    } else {
      group_size = group[group_method].cpu_size[group_id];
      group_num = 1;
    }
  }

  vx.resize(group_size * Nc);
  vy.resize(group_size * Nc);
  vz.resize(group_size * Nc);
  sx.resize(group_size * Nc);
  sy.resize(group_size * Nc);
  sz.resize(group_size * Nc);
  ki_negative.resize(Nc * group_num, 0.0);
  ko_negative.resize(Nc * group_num, 0.0);
  ki_positive.resize(Nc * group_num, 0.0);
  ko_positive.resize(Nc * group_num, 0.0);
  ki.resize((Nc * 2 - 1) * group_num);
  ko.resize((Nc * 2 - 1) * group_num);
  ki.assign((Nc * 2 - 1) * group_num, 0.0);
  ko.assign((Nc * 2 - 1) * group_num, 0.0);
  shc_i.resize(num_omega * group_num);
  shc_o.resize(num_omega * group_num);
  shc_i.assign(num_omega * group_num, 0.0);
  shc_o.assign(num_omega * group_num, 0.0);
}

static __global__ void gpu_find_k(
  const int group_size,
  const int correlation_step,
  const double* g_sx,
  const double* g_sy,
  const double* g_sz,
  const double* g_vx,
  const double* g_vy,
  const double* g_vz,
  double* g_ki,
  double* g_ko)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int size_sum = bid * group_size;
  int number_of_rounds = (group_size - 1) / BLOCK_SIZE_SHC + 1;
  __shared__ double s_ki[BLOCK_SIZE_SHC];
  __shared__ double s_ko[BLOCK_SIZE_SHC];
  double ki = 0.0;
  double ko = 0.0;

  for (int round = 0; round < number_of_rounds; ++round) {
    int n = tid + round * BLOCK_SIZE_SHC;
    if (n < group_size) {
      ki += g_sx[n] * g_vx[size_sum + n] + g_sy[n] * g_vy[size_sum + n];
      ko += g_sz[n] * g_vz[size_sum + n];
    }
  }
  s_ki[tid] = ki;
  s_ko[tid] = ko;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_ki[tid] += s_ki[tid + offset];
      s_ko[tid] += s_ko[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    if (bid <= correlation_step) {
      g_ki[correlation_step - bid] += s_ki[0];
      g_ko[correlation_step - bid] += s_ko[0];
    } else {
      g_ki[correlation_step + gridDim.x - bid] += s_ki[0];
      g_ko[correlation_step + gridDim.x - bid] += s_ko[0];
    }
  }
}

static __global__ void gpu_copy_data(
  const int group_size,
  const int offset,
  const int* g_group_contents,
  double* g_sx_o,
  double* g_sy_o,
  double* g_sz_o,
  double* g_vx_o,
  double* g_vy_o,
  double* g_vz_o,
  const double* g_sx_i,
  const double* g_sy_i,
  const double* g_sz_i,
  const double* g_vx_i,
  const double* g_vy_i,
  const double* g_vz_i)
{
  int n = threadIdx.x + blockIdx.x * blockDim.x;

  if (n < group_size) {
    int m = g_group_contents[offset + n];
    g_sx_o[n] = g_sx_i[m];
    g_sy_o[n] = g_sy_i[m];
    g_sz_o[n] = g_sz_i[m];
    g_vx_o[n] = g_vx_i[m];
    g_vy_o[n] = g_vy_i[m];
    g_vz_o[n] = g_vz_i[m];
  }
}

void SHC::process(
  const int number_of_steps,
  int step,
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
  if (!compute) {
    return;
  }
  if ((step + 1) % sample_interval != 0) {
    return;
  }
  int sample_step = step / sample_interval; // 0, 1, ..., Nc-1, Nc, Nc+1, ...
  int correlation_step = sample_step % Nc;  // 0, 1, ..., Nc-1, 0, 1, ...
  int offset = correlation_step * group_size;

  const int N = atom.number_of_atoms;

  const int tensor[3][3] = {0, 3, 4, 6, 1, 5, 7, 8, 2};
  const double* sx_tmp = atom.virial_per_atom.data() + N * tensor[direction][0];
  const double* sy_tmp = atom.virial_per_atom.data() + N * tensor[direction][1];
  const double* sz_tmp = atom.virial_per_atom.data() + N * tensor[direction][2];
  const double* vx_tmp = atom.velocity_per_atom.data();
  const double* vy_tmp = atom.velocity_per_atom.data() + N;
  const double* vz_tmp = atom.velocity_per_atom.data() + N * 2;

  if (-1 == group_method) {
    CHECK(gpuMemcpy(sx.data() + offset, sx_tmp, sizeof(double) * N, gpuMemcpyDeviceToDevice));
    CHECK(gpuMemcpy(sy.data() + offset, sy_tmp, sizeof(double) * N, gpuMemcpyDeviceToDevice));
    CHECK(gpuMemcpy(sz.data() + offset, sz_tmp, sizeof(double) * N, gpuMemcpyDeviceToDevice));
    CHECK(gpuMemcpy(vx.data() + offset, vx_tmp, sizeof(double) * N, gpuMemcpyDeviceToDevice));
    CHECK(gpuMemcpy(vy.data() + offset, vy_tmp, sizeof(double) * N, gpuMemcpyDeviceToDevice));
    CHECK(gpuMemcpy(vz.data() + offset, vz_tmp, sizeof(double) * N, gpuMemcpyDeviceToDevice));
  } else {
    if (group_id == -1) {
      for (int n = 1; n < group_num; ++n) {
        int offset_s = Nc * group[group_method].cpu_size_sum[n] +
                       correlation_step * group[group_method].cpu_size[n];
        gpu_copy_data<<<
          (group[group_method].cpu_size[n] - 1) / BLOCK_SIZE_SHC + 1,
          BLOCK_SIZE_SHC>>>(
          group[group_method].cpu_size[n],
          group[group_method].cpu_size_sum[n],
          group[group_method].contents.data(),
          sx.data() + offset_s,
          sy.data() + offset_s,
          sz.data() + offset_s,
          vx.data() + offset_s,
          vy.data() + offset_s,
          vz.data() + offset_s,
          sx_tmp,
          sy_tmp,
          sz_tmp,
          vx_tmp,
          vy_tmp,
          vz_tmp);
        GPU_CHECK_KERNEL
      }
    } else {
      gpu_copy_data<<<(group_size - 1) / BLOCK_SIZE_SHC + 1, BLOCK_SIZE_SHC>>>(
        group_size,
        group[group_method].cpu_size_sum[group_id],
        group[group_method].contents.data(),
        sx.data() + offset,
        sy.data() + offset,
        sz.data() + offset,
        vx.data() + offset,
        vy.data() + offset,
        vz.data() + offset,
        sx_tmp,
        sy_tmp,
        sz_tmp,
        vx_tmp,
        vy_tmp,
        vz_tmp);
    }
    GPU_CHECK_KERNEL
  }

  if (sample_step >= Nc - 1) {
    ++num_time_origins;

    if (group_id == -1) {
      for (int n = 1; n < group_num; ++n) {
        int offset_s = Nc * group[group_method].cpu_size_sum[n];
        int offset_e = offset_s + correlation_step * group[group_method].cpu_size[n];
        gpu_find_k<<<Nc, BLOCK_SIZE_SHC>>>(
          group[group_method].cpu_size[n],
          correlation_step,
          sx.data() + offset_e,
          sy.data() + offset_e,
          sz.data() + offset_e,
          vx.data() + offset_s,
          vy.data() + offset_s,
          vz.data() + offset_s,
          ki_negative.data() + Nc * n,
          ko_negative.data() + Nc * n);
        GPU_CHECK_KERNEL

        gpu_find_k<<<Nc, BLOCK_SIZE_SHC>>>(
          group[group_method].cpu_size[n],
          correlation_step,
          vx.data() + offset_e,
          vy.data() + offset_e,
          vz.data() + offset_e,
          sx.data() + offset_s,
          sy.data() + offset_s,
          sz.data() + offset_s,
          ki_positive.data() + Nc * n,
          ko_positive.data() + Nc * n);
        GPU_CHECK_KERNEL
      }
    } else {
      gpu_find_k<<<Nc, BLOCK_SIZE_SHC>>>(
        group_size,
        correlation_step,
        sx.data() + offset,
        sy.data() + offset,
        sz.data() + offset,
        vx.data(),
        vy.data(),
        vz.data(),
        ki_negative.data(),
        ko_negative.data());
      GPU_CHECK_KERNEL

      gpu_find_k<<<Nc, BLOCK_SIZE_SHC>>>(
        group_size,
        correlation_step,
        vx.data() + offset,
        vy.data() + offset,
        vz.data() + offset,
        sx.data(),
        sy.data(),
        sz.data(),
        ki_positive.data(),
        ko_positive.data());
      GPU_CHECK_KERNEL
    }
  }
}

void SHC::average_k()
{
  std::vector<double> ki_negative_cpu(Nc * group_num);
  std::vector<double> ko_negative_cpu(Nc * group_num);
  std::vector<double> ki_positive_cpu(Nc * group_num);
  std::vector<double> ko_positive_cpu(Nc * group_num);
  ki_negative.copy_to_host(ki_negative_cpu.data());
  ko_negative.copy_to_host(ko_negative_cpu.data());
  ki_positive.copy_to_host(ki_positive_cpu.data());
  ko_positive.copy_to_host(ko_positive_cpu.data());

  const double scalar = 1000.0 / TIME_UNIT_CONVERSION / num_time_origins;
  if (group_id == -1) {
    for (int n = 1; n < group_num; ++n) {
      const int offset_k = (Nc * 2 - 1) * n;
      const int offset_n = Nc * n;
      for (int nc = 0; nc < Nc - 1; ++nc) {
        ki[nc + offset_k] = ki_negative_cpu[Nc - nc - 1 + offset_n] * scalar;
        ko[nc + offset_k] = ko_negative_cpu[Nc - nc - 1 + offset_n] * scalar;
      }
      for (int nc = 0; nc < Nc; ++nc) {
        ki[nc + Nc - 1 + offset_k] = ki_positive_cpu[nc + offset_n] * scalar;
        ko[nc + Nc - 1 + offset_k] = ko_positive_cpu[nc + offset_n] * scalar;
      }
    }
  } else {
    for (int nc = 0; nc < Nc - 1; ++nc) {
      ki[nc] = ki_negative_cpu[Nc - nc - 1] * scalar;
      ko[nc] = ko_negative_cpu[Nc - nc - 1] * scalar;
    }

    for (int nc = 0; nc < Nc; ++nc) {
      ki[nc + Nc - 1] = ki_positive_cpu[nc] * scalar;
      ko[nc + Nc - 1] = ko_positive_cpu[nc] * scalar;
    }
  }
}

void SHC::find_shc(const double dt_in_ps, const double d_omega)
{
  if (group_id == -1) {
    for (int n = 1; n < group_num; ++n) {
      const int offset_k = (Nc * 2 - 1) * n;
      for (int nc = 0; nc < Nc * 2 - 1; ++nc) {
        const double hann_window = (cos(PI * (nc + 1 - Nc) / Nc) + 1.0) * 0.5;
        ki[nc + offset_k] *= hann_window;
        ko[nc + offset_k] *= hann_window;
      }
    }

    for (int n = 1; n < group_num; ++n) {
      const int offset_s = num_omega * n;
      const int offset_k = (Nc * 2 - 1) * n;
      for (int nw = 0; nw < num_omega; ++nw) {
        const double omega = (nw + 1) * d_omega;
        for (int nc = 0; nc < Nc * 2 - 1; nc++) {
          const double t_in_ps = (nc + 1 - Nc) * dt_in_ps;
          const double cos_factor = cos(omega * t_in_ps);
          shc_i[nw + offset_s] += ki[nc + offset_k] * cos_factor;
          shc_o[nw + offset_s] += ko[nc + offset_k] * cos_factor;
        }
        shc_i[nw + offset_s] *= 2.0 * dt_in_ps;
        shc_o[nw + offset_s] *= 2.0 * dt_in_ps;
      }
    }
  } else {
    for (int nc = 0; nc < Nc * 2 - 1; ++nc) {
      const double hann_window = (cos(PI * (nc + 1 - Nc) / Nc) + 1.0) * 0.5;
      ki[nc] *= hann_window;
      ko[nc] *= hann_window;
    }

    for (int nw = 0; nw < num_omega; ++nw) {
      const double omega = (nw + 1) * d_omega;
      for (int nc = 0; nc < Nc * 2 - 1; nc++) {
        const double t_in_ps = (nc + 1 - Nc) * dt_in_ps;
        const double cos_factor = cos(omega * t_in_ps);
        shc_i[nw] += ki[nc] * cos_factor;
        shc_o[nw] += ko[nc] * cos_factor;
      }
      shc_i[nw] *= 2.0 * dt_in_ps;
      shc_o[nw] *= 2.0 * dt_in_ps;
    }
  }
}

void SHC::postprocess(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature)
{
  if (!compute) {
    return;
  }

  const double dt_in_ps = time_step * sample_interval * TIME_UNIT_CONVERSION / 1000.0;
  const double d_omega = max_omega / num_omega;

  FILE* fid = my_fopen("shc.out", "a");

  average_k();
  find_shc(dt_in_ps, d_omega);
  if (group_id == -1) {
    for (int n = 1; n < group_num; ++n) {
      const int offset_k = (Nc * 2 - 1) * n;
      // ki and ko are in units of A*eV/ps
      for (int nc = 0; nc < Nc * 2 - 1; ++nc) {
        fprintf(fid, "%g %g %g\n", (nc + 1 - Nc) * dt_in_ps, ki[nc + offset_k], ko[nc + offset_k]);
      }
      const int offset_s = num_omega * n;
      // shc_i and shc_o are in units of A*eV = A*eV/ps/THz
      for (int nc = 0; nc < num_omega; ++nc) {
        fprintf(fid, "%g %g %g\n", (nc + 1) * d_omega, shc_i[nc + offset_s], shc_o[nc + offset_s]);
      }
    }
  } else {
    for (int nc = 0; nc < Nc * 2 - 1; ++nc) {
      fprintf(fid, "%g %g %g\n", (nc + 1 - Nc) * dt_in_ps, ki[nc], ko[nc]);
    }

    for (int nc = 0; nc < num_omega; ++nc) {
      fprintf(fid, "%g %g %g\n", (nc + 1) * d_omega, shc_i[nc], shc_o[nc]);
    }
  }

  fflush(fid);
  fclose(fid);

  compute = 0;
  group_method = -1;
}

void SHC::parse(const char** param, int num_param, const std::vector<Group>& groups)
{
  printf("Compute SHC.\n");
  compute = 1;

  if ((num_param != 6) && (num_param != 9)) {
    PRINT_INPUT_ERROR("compute_shc should have 5 or 8 parameters.");
  }

  if (!is_valid_int(param[1], &sample_interval)) {
    PRINT_INPUT_ERROR("Sampling interval for SHC should be an integer.");
  }
  if (sample_interval < 1) {
    PRINT_INPUT_ERROR("Sampling interval for SHC should >= 1.");
  }
  if (sample_interval > 50) {
    PRINT_INPUT_ERROR("Sampling interval for SHC should <= 50 (trust me).");
  }
  printf("    sampling interval for SHC is %d.\n", sample_interval);

  if (!is_valid_int(param[2], &Nc)) {
    PRINT_INPUT_ERROR("Nc for SHC should be an integer.");
  }
  if (Nc < 100) {
    PRINT_INPUT_ERROR("Nc for SHC should >= 100 (trust me).");
  }
  if (Nc > 1000) {
    PRINT_INPUT_ERROR("Nc for SHC should <= 1000 (trust me).");
  }
  printf("    number of correlation data is %d.\n", Nc);

  if (!is_valid_int(param[3], &direction)) {
    PRINT_INPUT_ERROR("direction for SHC should be an integer.");
  }
  if (direction == 0) {
    printf("    transport direction is x.\n");
  } else if (direction == 1) {
    printf("    transport direction is y.\n");
  } else if (direction == 2) {
    printf("    transport direction is z.\n");
  } else {
    PRINT_INPUT_ERROR("Transport direction should be x or y or z.");
  }

  if (!is_valid_int(param[4], &num_omega)) {
    PRINT_INPUT_ERROR("num_omega for SHC should be an integer.");
  }
  if (num_omega < 0) {
    PRINT_INPUT_ERROR("num_omega for SHC should >= 0.");
  }
  printf("    num_omega for SHC is %d.\n", num_omega);

  if (!is_valid_real(param[5], &max_omega)) {
    PRINT_INPUT_ERROR("max_omega for SHC should be a number.");
  }
  if (max_omega <= 0) {
    PRINT_INPUT_ERROR("max_omega for SHC should > 0.");
  }
  printf("    max_omega for SHC is %g.\n", max_omega);

  for (int k = 6; k < num_param; k++) {
    if (strcmp(param[k], "group") == 0) {
      parse_group(param, num_param, true, groups, k, group_method, group_id);
    } else {
      PRINT_INPUT_ERROR("Unrecognized argument in compute_shc.\n");
    }
  }

  if (group_id == -1) {
    printf(
      "    compute SHC for all group IDs except for group ID 0 in grouping method %d.\n",
      group_method);
  }
  if (group_id < -1) {
    PRINT_INPUT_ERROR("group ID should >= -1 for computing SHC.");
  }
}

SHC::SHC(const char** param, int num_param, const std::vector<Group>& groups)
{
  parse(param, num_param, groups);
  property_name = "compute_shc";
}