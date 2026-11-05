/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD and is distributed under LGPLv3 or later.
*/

#include "main_nep/mic.cuh"
#include "spin_data.cuh"
#include "utilities/error.cuh"
#include <algorithm>
#include <cmath>
#include <limits>
// Both passes traverse the same periodic geometry; structural neighbors belong to Dataset.
template <bool Store>
static __global__ void gpu_find_spin_neighbors(
  const int N,
  const int* Na,
  const int* Na_sum,
  const float rc_spin,
  const float* __restrict__ g_box,
  const float* __restrict__ g_box_original,
  const int* __restrict__ g_num_cell,
  const float* x,
  const float* y,
  const float* z,
  const int plane_size,
  int* NN_spin,
  int* NL_spin,
  double* r12_spin)
{
  const int N1 = Na_sum[blockIdx.x];
  const int N2 = N1 + Na[blockIdx.x];
  const float* box = g_box + 18 * blockIdx.x;
  const float* box_original = g_box_original + 9 * blockIdx.x;
  const int* num_cell = g_num_cell + 3 * blockIdx.x;
  for (int n1 = N1 + threadIdx.x; n1 < N2; n1 += blockDim.x) {
    const float x1 = x[n1], y1 = y[n1], z1 = z[n1];
    int count = 0;
    for (int n2 = N1; n2 < N2; ++n2) {
      for (int ia = 0; ia < num_cell[0]; ++ia) {
        for (int ib = 0; ib < num_cell[1]; ++ib) {
          for (int ic = 0; ic < num_cell[2]; ++ic) {
            if (ia == 0 && ib == 0 && ic == 0 && n1 == n2)
              continue;
            float dx = box_original[0] * ia + box_original[1] * ib + box_original[2] * ic;
            float dy = box_original[3] * ia + box_original[4] * ib + box_original[5] * ic;
            float dz = box_original[6] * ia + box_original[7] * ib + box_original[8] * ic;
            float x12 = x[n2] + dx - x1;
            float y12 = y[n2] + dy - y1;
            float z12 = z[n2] + dz - z1;
            dev_apply_mic(box, x12, y12, z12);
            if (x12 * x12 + y12 * y12 + z12 * z12 < rc_spin * rc_spin) {
              if (Store) {
                const int index = n1 + N * count;
                NL_spin[index] = n2;
                r12_spin[index] = x12;
                r12_spin[plane_size + index] = y12;
                r12_spin[2 * plane_size + index] = z12;
              }
              ++count;
            }
          }
        }
      }
    }
    NN_spin[n1] = count;
  }
}

void Spin_Data::construct(Spin_Parameters& para, Dataset& dataset, const std::vector<Labels>& input)
{
  const int N = dataset.N, Nc = dataset.Nc;
  labels.reserve(Nc);
  std::vector<double> position(3 * N), moments(3 * N);
  mforce_ref_cpu.resize(3 * N);
  mforce_cpu.resize(3 * N);
  has_mforce_cpu.resize(Nc);
  active_count_cpu.assign(Nc, 0);
  torque_active_count_cpu.assign(Nc, 0);
  for (int n = 0; n < Nc; ++n) {
    const auto& structure = dataset.structures[n];
    labels.push_back(input.at(structure.index_in_file));
    const auto& label = labels.back();
    has_mforce_cpu[n] = label.has_mforce;
    for (int a = 0; a < structure.num_atom; ++a) {
      if (para.spin_type_active[structure.type[a]]) {
        ++active_count_cpu[n];
        const double sx = label.sx[a];
        const double sy = label.sy[a];
        const double sz = label.sz[a];
        const double spin2 = sx * sx + sy * sy + sz * sz;
        if (spin2 > 1.0e-20)
          ++torque_active_count_cpu[n];
      }
      const int i = dataset.Na_sum_cpu[n] + a;
      position[i] = structure.x[a];
      position[N + i] = structure.y[a];
      position[2 * N + i] = structure.z[a];
      moments[i] = label.sx[a];
      moments[N + i] = label.sy[a];
      moments[2 * N + i] = label.sz[a];
      mforce_ref_cpu[i] = label.mfx[a];
      mforce_ref_cpu[N + i] = label.mfy[a];
      mforce_ref_cpu[2 * N + i] = label.mfz[a];
    }
  }
  r_spin.resize(3 * N);
  r_spin.copy_from_host(position.data());
  spin.resize(3 * N);
  spin.copy_from_host(moments.data());
  mforce.resize(3 * N);
  mforce.fill(0.0f);
  mforce_ref_gpu.resize(3 * N);
  mforce_ref_gpu.copy_from_host(mforce_ref_cpu.data());
  has_mforce_gpu.resize(Nc);
  has_mforce_gpu.copy_from_host(has_mforce_cpu.data());
  spin_type_active_gpu.resize(para.num_types);
  spin_type_active_gpu.copy_from_host(para.spin_type_active.data());
  NN_spin.resize(N);
  gpu_find_spin_neighbors<false><<<Nc, 256>>>(
    N,
    dataset.Na.data(),
    dataset.Na_sum.data(),
    para.spin_cutoff,
    dataset.box.data(),
    dataset.box_original.data(),
    dataset.num_cell.data(),
    dataset.r.data(),
    dataset.r.data() + N,
    dataset.r.data() + 2 * N,
    0,
    NN_spin.data(),
    nullptr,
    nullptr);
  GPU_CHECK_KERNEL
  std::vector<int> counts(N);
  NN_spin.copy_to_host(counts.data());
  for (int count : counts)
    max_NN_spin = std::max(max_NN_spin, count);
  const long long slots = static_cast<long long>(N) * max_NN_spin;
  if (slots > std::numeric_limits<int>::max())
    PRINT_INPUT_ERROR("Spin neighbor slot count exceeds the supported int range.");
  NL_spin.resize(slots);
  r12_spin.resize(3 * slots);
  gpu_find_spin_neighbors<true><<<Nc, 256>>>(
    N,
    dataset.Na.data(),
    dataset.Na_sum.data(),
    para.spin_cutoff,
    dataset.box.data(),
    dataset.box_original.data(),
    dataset.num_cell.data(),
    dataset.r.data(),
    dataset.r.data() + N,
    dataset.r.data() + 2 * N,
    static_cast<int>(slots),
    NN_spin.data(),
    NL_spin.data(),
    r12_spin.data());
  GPU_CHECK_KERNEL
}
namespace
{
enum class MagneticError { Cartesian, Torque, Parallel, Perpendicular };

template <MagneticError Channel>
static __global__ void gpu_sum_mforce_error(
  const int N,
  const int* Na,
  const int* Na_sum,
  const int* type,
  const int* active_type,
  const int* has_mforce,
  const double* spin,
  const float* mforce,
  const float* mforce_ref,
  float* error)
{
  const int configuration = blockIdx.x;
  const int begin = Na_sum[configuration];
  const int end = begin + Na[configuration];
  extern __shared__ float partial[];
  float sum = 0.0f;
  if (has_mforce[configuration]) {
    for (int atom = begin + threadIdx.x; atom < end; atom += blockDim.x) {
      if (!active_type[type[atom]]) {
        continue;
      }
      float predicted[3] = {mforce[atom], mforce[N + atom], mforce[2 * N + atom]};
      float reference[3] = {mforce_ref[atom], mforce_ref[N + atom], mforce_ref[2 * N + atom]};
      const double sx = spin[atom];
      const double sy = spin[N + atom];
      const double sz = spin[2 * N + atom];
      const double spin2 = sx * sx + sy * sy + sz * sz;
      if (Channel == MagneticError::Torque && spin2 <= 1.0e-20) {
        continue;
      }
      if (Channel == MagneticError::Torque) {
        const float predicted_tau[3] = {
          static_cast<float>(sy * predicted[2] - sz * predicted[1]),
          static_cast<float>(sz * predicted[0] - sx * predicted[2]),
          static_cast<float>(sx * predicted[1] - sy * predicted[0])};
        const float reference_tau[3] = {
          static_cast<float>(sy * reference[2] - sz * reference[1]),
          static_cast<float>(sz * reference[0] - sx * reference[2]),
          static_cast<float>(sx * reference[1] - sy * reference[0])};
        for (int component = 0; component < 3; ++component) {
          const float difference = predicted_tau[component] - reference_tau[component];
          sum += difference * difference;
        }
      } else if (Channel == MagneticError::Cartesian) {
        for (int component = 0; component < 3; ++component) {
          const float difference = predicted[component] - reference[component];
          sum += difference * difference;
        }
      } else {
        const double dx = static_cast<double>(predicted[0]) - reference[0];
        const double dy = static_cast<double>(predicted[1]) - reference[1];
        const double dz = static_cast<double>(predicted[2]) - reference[2];
        // Normalize after scaling: tiny nonzero spins still define an axis.
        const double scale = fmax(fabs(sx), fmax(fabs(sy), fabs(sz)));
        double parallel, perpendicular;
        if (scale == 0.0) {
          // No preferred axis at zero spin: isotropic directional average.
          const double norm2 = dx * dx + dy * dy + dz * dz;
          parallel = norm2 / 3.0;
          perpendicular = 2.0 * norm2 / 3.0;
        } else {
          double ux = sx / scale, uy = sy / scale, uz = sz / scale;
          const double norm = sqrt(ux * ux + uy * uy + uz * uz);
          ux /= norm;
          uy /= norm;
          uz /= norm;
          const double projection = dx * ux + dy * uy + dz * uz;
          parallel = projection * projection;
          const double tx = dx - projection * ux;
          const double ty = dy - projection * uy;
          const double tz = dz - projection * uz;
          perpendicular = tx * tx + ty * ty + tz * tz;
        }
        sum += static_cast<float>(Channel == MagneticError::Parallel ? parallel : perpendicular);
      }
    }
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      partial[threadIdx.x] += partial[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    error[configuration] = partial[0];
  }
}

template <MagneticError Channel>
static std::vector<float> get_rmse_spin_impl(
  Spin_Data& spin_data,
  Dataset& dataset,
  Spin_Parameters& para,
  const bool use_weight,
  int device_id)
{
  CHECK(gpuSetDevice(device_id));
  const int block_size = 256;
  gpu_sum_mforce_error<Channel><<<dataset.Nc, block_size, sizeof(float) * block_size>>>(
    dataset.N,
    dataset.Na.data(),
    dataset.Na_sum.data(),
    dataset.type.data(),
    spin_data.spin_type_active_gpu.data(),
    spin_data.has_mforce_gpu.data(),
    spin_data.spin.data(),
    spin_data.mforce.data(),
    spin_data.mforce_ref_gpu.data(),
    dataset.error_gpu.data());
  GPU_CHECK_KERNEL
  dataset.error_gpu.copy_to_host(dataset.error_cpu.data());

  std::vector<float> rmse_array(para.num_types + 1, 0.0f);
  std::vector<int> count_array(para.num_types + 1, 0);
  for (int configuration = 0; configuration < dataset.Nc; ++configuration) {
    if (!spin_data.has_mforce_cpu[configuration]) {
      continue;
    }
    const int active_count = Channel == MagneticError::Torque
                               ? spin_data.torque_active_count_cpu[configuration]
                               : spin_data.active_count_cpu[configuration];
    const float weighted_error = use_weight ? dataset.weight_cpu[configuration] *
                                                dataset.weight_cpu[configuration] *
                                                dataset.error_cpu[configuration]
                                            : dataset.error_cpu[configuration];
    for (int type = 0; type <= para.num_types; ++type) {
      if (dataset.has_type[type * dataset.Nc + configuration]) {
        rmse_array[type] += weighted_error;
        count_array[type] += active_count;
      }
    }
  }
  for (int type = 0; type <= para.num_types; ++type) {
    if (count_array[type] > 0) {
      const int degrees_of_freedom =
        Channel == MagneticError::Cartesian ? 3 : (Channel == MagneticError::Parallel ? 1 : 2);
      rmse_array[type] = sqrt(rmse_array[type] / (degrees_of_freedom * count_array[type]));
    }
  }
  return rmse_array;
}

} // namespace

std::vector<float> Spin_Data::get_rmse_mforce(
  Spin_Parameters& para, Dataset& dataset, const bool use_weight, int device_id)
{
  return get_rmse_spin_impl<MagneticError::Cartesian>(*this, dataset, para, use_weight, device_id);
}

std::vector<float> Spin_Data::get_rmse_tau(
  Spin_Parameters& para, Dataset& dataset, const bool use_weight, int device_id)
{
  return get_rmse_spin_impl<MagneticError::Torque>(*this, dataset, para, use_weight, device_id);
}

std::vector<float> Spin_Data::get_rmse_mforce_parallel(
  Spin_Parameters& para, Dataset& dataset, const bool use_weight, int device_id)
{
  return get_rmse_spin_impl<MagneticError::Parallel>(*this, dataset, para, use_weight, device_id);
}

std::vector<float> Spin_Data::get_rmse_mforce_perp(
  Spin_Parameters& para, Dataset& dataset, const bool use_weight, int device_id)
{
  return get_rmse_spin_impl<MagneticError::Perpendicular>(
    *this, dataset, para, use_weight, device_id);
}
