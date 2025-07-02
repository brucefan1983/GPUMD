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

/*-----------------------------------------------------------------------------------------------100
Calculate:
    Angular distribution function (ADF)
Author:
    Yongchao Wu 2025-01-18
Email:
    yongchao_wu@bit.edu.cn
--------------------------------------------------------------------------------------------------*/

#include "adf.cuh"
#include "force/neighbor.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include "parse_utilities.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <cstring>
#include <numeric>

namespace
{

static __global__ void gpu_find_adf_global(
  const int N,
  const Box box,
  const int* __restrict__ NN,
  const int* __restrict__ NL,
  const double* __restrict__ x,
  const double* __restrict__ y,
  const double* __restrict__ z,
  int* adf,
  const int adf_bins,
  double rc_min,
  double rc_max)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const double My_PI = 3.14159265358979323846;
  const double rc_min_sq = rc_min * rc_min;
  const double rc_max_sq = rc_max * rc_max;
  const double delta_theta = 180.0 / adf_bins;
  const double delta_theta_inv = 1.0 / delta_theta;

  if (i < N) {
    const double x1 = x[i];
    const double y1 = y[i];
    const double z1 = z[i];
    const int i_neigh = NN[i];

    for (int jj = 0; jj < i_neigh; jj++) {
      const int j = NL[i + jj * N];
      double xij = x[j] - x1;
      double yij = y[j] - y1;
      double zij = z[j] - z1;
      apply_mic(box, xij, yij, zij);
      const double dij_sq = xij * xij + yij * yij + zij * zij;

      if (dij_sq >= rc_min_sq && dij_sq < rc_max_sq) {
        for (int kk = jj + 1; kk < i_neigh; kk++) {
          const int k = NL[i + kk * N];
          double xik = x[k] - x1;
          double yik = y[k] - y1;
          double zik = z[k] - z1;
          apply_mic(box, xik, yik, zik);
          const double dik_sq = xik * xik + yik * yik + zik * zik;
          if (dik_sq >= rc_min_sq && dik_sq < rc_max_sq) {
            const double cos_theta = (xij * xik + yij * yik + zij * zik) / sqrt(dij_sq * dik_sq);
            const double theta = acos(cos_theta) * 180.0 / My_PI;
            const int bin = static_cast<int>(floor(theta * delta_theta_inv));
            if (bin > adf_bins - 1) {
              atomicAdd(&adf[adf_bins - 1], 1);
            } else {
              atomicAdd(&adf[bin], 1);
            }
          }
        }
      }
    }
  }
}

static __global__ void gpu_find_adf_local(
  const int N,
  const Box box,
  const int* __restrict__ NN,
  const int* __restrict__ NL,
  const double* __restrict__ x,
  const double* __restrict__ y,
  const double* __restrict__ z,
  int* adf,
  const int Ntriples,
  const int adf_bins,
  const int* type,
  const int* itype,
  const int* jtype,
  const int* ktype,
  const double* rc_min_j,
  const double* rc_max_j,
  const double* rc_min_k,
  const double* rc_max_k)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const double My_PI = 3.14159265358979323846;
  const double delta_theta = 180.0 / adf_bins;
  const double delta_theta_inv = 1.0 / delta_theta;

  if (i < N) {
    const double x1 = x[i];
    const double y1 = y[i];
    const double z1 = z[i];
    const int i_neigh = NN[i];
    const int i_type = type[i];
    for (int m = 0; m < Ntriples; m++) {
      if (i_type == itype[m]) {
        for (int jj = 0; jj < i_neigh; jj++) {
          const int j = NL[i + jj * N];
          const int j_type = type[j];
          if (j_type == jtype[m]) {
            double xij = x[j] - x1;
            double yij = y[j] - y1;
            double zij = z[j] - z1;
            apply_mic(box, xij, yij, zij);
            const double dij_sq = xij * xij + yij * yij + zij * zij;

            if (dij_sq >= rc_min_j[m] * rc_min_j[m] && dij_sq < rc_max_j[m] * rc_max_j[m]) {
              for (int kk = jj + 1; kk < i_neigh; kk++) {
                const int k = NL[i + kk * N];
                const int k_type = type[k];
                if (k_type == ktype[m]) {
                  double xik = x[k] - x1;
                  double yik = y[k] - y1;
                  double zik = z[k] - z1;
                  apply_mic(box, xik, yik, zik);
                  const double dik_sq = xik * xik + yik * yik + zik * zik;
                  if (dik_sq >= rc_min_k[m] * rc_min_k[m] && dik_sq < rc_max_k[m] * rc_max_k[m]) {
                    const double cos_theta =
                      (xij * xik + yij * yik + zij * zik) / sqrt(dij_sq * dik_sq);
                    const double theta = acos(cos_theta) * 180.0 / My_PI;
                    const int bin = static_cast<int>(floor(theta * delta_theta_inv));
                    if (bin > adf_bins - 1) {
                      atomicAdd(&adf[m * Ntriples + adf_bins - 1], 1);
                    } else {
                      atomicAdd(&adf[m * Ntriples + bin], 1);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

} // namespace

ADF::ADF(const char** param, const int num_param, Box& box, const int number_of_types)
{
  parse(param, num_param, box, number_of_types);
  property_name = "compute_adf";
}

void ADF::preprocess(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  if (!compute_) {
    return;
  }
  num_atoms_ = atom.number_of_atoms;
  if (global_) {
    adf.resize(adf_bins_, 0);
    adf_gpu.resize(adf_bins_, 0);
  } else {
    adf.resize(adf_bins_ * num_triples_, 0);
    adf_gpu.resize(adf_bins_ * num_triples_, 0);
    itype_gpu.resize(num_triples_);
    itype_gpu.copy_from_host(itype_cpu.data());
    jtype_gpu.resize(num_triples_);
    jtype_gpu.copy_from_host(jtype_cpu.data());
    ktype_gpu.resize(num_triples_);
    ktype_gpu.copy_from_host(ktype_cpu.data());
    rc_min_j_gpu.resize(num_triples_);
    rc_min_j_gpu.copy_from_host(rc_min_j_cpu.data());
    rc_max_j_gpu.resize(num_triples_);
    rc_max_j_gpu.copy_from_host(rc_max_j_cpu.data());
    rc_min_k_gpu.resize(num_triples_);
    rc_min_k_gpu.copy_from_host(rc_min_k_cpu.data());
    rc_max_k_gpu.resize(num_triples_);
    rc_max_k_gpu.copy_from_host(rc_max_k_cpu.data());
  }

  angle.resize(adf_bins_);
  double delta_theta = 180.0 / adf_bins_;
  std::vector<double> r(adf_bins_ + 1);
  for (int i = 0; i < adf_bins_ + 1; i++) {
    r[i] = i * delta_theta;
  }
  for (int i = 1; i < adf_bins_ + 1; i++) {
    angle[i - 1] = (r[i] + r[i - 1]) / 2;
  }

  cell_count.resize(num_atoms_);
  cell_count_sum.resize(num_atoms_);
  cell_contents.resize(num_atoms_);
  NN.resize(num_atoms_);
  NL.resize(num_atoms_ * 200); // 200 is the maximum number of neighbors with bond

  fid = fopen("adf.out", "a");
};

void ADF::process(
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

  if (!compute_) {
    return;
  }

  if ((step + 1) % num_interval_ != 0) {
    return;
  }

  const int BLOCK_SIZE = 64;
  int grid_size = (num_atoms_ - 1) / BLOCK_SIZE + 1;
  find_neighbor(
    0,
    num_atoms_,
    rc_max_,
    box,
    atom.type,
    atom.position_per_atom,
    cell_count,
    cell_count_sum,
    cell_contents,
    NN,
    NL);

  if (global_) {
    gpu_find_adf_global<<<grid_size, BLOCK_SIZE>>>(
      num_atoms_,
      box,
      NN.data(),
      NL.data(),
      atom.position_per_atom.data(),
      atom.position_per_atom.data() + num_atoms_,
      atom.position_per_atom.data() + 2 * num_atoms_,
      adf_gpu.data(),
      adf_bins_,
      rc_min_,
      rc_max_);
    GPU_CHECK_KERNEL
  } else {
    gpu_find_adf_local<<<grid_size, BLOCK_SIZE>>>(
      num_atoms_,
      box,
      NN.data(),
      NL.data(),
      atom.position_per_atom.data(),
      atom.position_per_atom.data() + num_atoms_,
      atom.position_per_atom.data() + 2 * num_atoms_,
      adf_gpu.data(),
      num_triples_,
      adf_bins_,
      atom.type.data(),
      itype_gpu.data(),
      jtype_gpu.data(),
      ktype_gpu.data(),
      rc_min_j_gpu.data(),
      rc_max_j_gpu.data(),
      rc_min_k_gpu.data(),
      rc_max_k_gpu.data());
    GPU_CHECK_KERNEL
  }

  CHECK(gpuMemcpy(adf.data(), adf_gpu.data(), sizeof(int) * adf.size(), gpuMemcpyDeviceToHost));
  CHECK(gpuDeviceSynchronize()); // needed for pre-Pascal GPU

  double delta = angle[1] - angle[0];
  fprintf(fid, "#angles ");
  if (global_) {
    fprintf(fid, "total step = %d\n", step);
  } else {
    for (int m = 0; m < num_triples_; m++) {
      fprintf(fid, "triples_%d-%d-%d ", itype_cpu[m], jtype_cpu[m], ktype_cpu[m]);
    }
    fprintf(fid, "step = %d\n", step);
  }

  for (int i = 0; i < adf_bins_; i++) {
    fprintf(fid, "%g ", angle[i]);
    if (global_) {
      int total = std::accumulate(adf.begin(), adf.end(), 0);
      if (total > 0) {
        fprintf(fid, "%g\n", adf[i] / (total * delta * 1.0));
      } else {
        fprintf(fid, "%g\n", adf[i] / (delta * 1.0));
      }
    } else {
      auto start = adf.begin();
      for (int m = 0; m < num_triples_; m++) {
        int total = std::accumulate(start, start + adf_bins_, 0);
        if (total > 0) {
          fprintf(fid, "%g ", adf[adf_bins_ * m + i] / (total * delta * 1.0));
        } else {
          fprintf(fid, "%g ", adf[adf_bins_ * m + i] / (delta * 1.0));
        }
        start += adf_bins_;
      }
      fprintf(fid, "\n");
    }
  }

  fflush(fid);
};

void ADF::postprocess(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature)
{
  if (!compute_)
    return;

  fclose(fid);
  compute_ = false;
};

// compute_adf <interval> <num_bins> <rc_min> <rc_max>
// compute_adf <interval> <num_bins> <itype1> <jtype1> <ktype1> <rc_min_j1> <rc_max_j1> <rc_min_k1>
// <rc_max_k1> ...
void ADF::parse(const char** param, const int num_param, Box& box, const int number_of_types)
{
  printf("Compute angular distribution functions (ADF).\n");
  compute_ = true;

  if (num_param < 5) {
    PRINT_INPUT_ERROR("compute_rdf should have at least 4 parameters.\n");
  }

  if (num_param == 5) {

    if (!is_valid_int(param[1], &num_interval_)) {
      PRINT_INPUT_ERROR("interval step per sample should be an integer.\n");
    }
    if (num_interval_ <= 0) {
      PRINT_INPUT_ERROR("interval step per sample should be positive.\n");
    }
    if (!is_valid_int(param[2], &adf_bins_)) {
      PRINT_INPUT_ERROR("number of bins should be an integer.\n");
    }
    if (!is_valid_real(param[3], &rc_min_)) {
      PRINT_INPUT_ERROR("minimum radial cutoff should be a number.\n");
    }
    if (!is_valid_real(param[4], &rc_max_)) {
      PRINT_INPUT_ERROR("maximum radial cutoff should be a number.\n");
    }
    if (rc_min_ >= rc_max_) {
      PRINT_INPUT_ERROR("minimum radial cutoff should be less than maximum radial cutoff.\n");
    }
    if (rc_min_ < 0) {
      PRINT_INPUT_ERROR("minimum radial cutoff should be positive.\n");
    }
    global_ = true;
    double thickness_half[3] = {
      box.get_volume() / box.get_area(0) / 2.5,
      box.get_volume() / box.get_area(1) / 2.5,
      box.get_volume() / box.get_area(2) / 2.5};
    if (rc_max_ > thickness_half[0] || rc_max_ > thickness_half[1] || rc_max_ > thickness_half[2]) {
      std::string message =
        "The box has a thickness < 2.5 RDF radial cutoffs in a periodic direction.\n"
        "                Please increase the periodic direction(s).\n";
      PRINT_INPUT_ERROR(message.c_str());
    }
    printf("    Global ADF will be computed.\n");
    printf("    ADF sample interval is %d step.\n", num_interval_);
    printf("    radial cutoff will be divided into %d bins.\n", adf_bins_);
    printf("    minimal radial cutoff %g.\n", rc_min_);
    printf("    maximum radial cutoff %g.\n", rc_max_);
  } else {
    if ((num_param - 3) % 7 != 0) {
      PRINT_INPUT_ERROR("compute_adf should have 4 parameters or 2 + 7 * Ntriples parameters.\n");
    }
    if (!is_valid_int(param[1], &num_interval_)) {
      PRINT_INPUT_ERROR("interval step per sample should be an integer.\n");
    }
    if (num_interval_ <= 0) {
      PRINT_INPUT_ERROR("interval step per sample should be positive.\n");
    }
    if (!is_valid_int(param[2], &adf_bins_)) {
      PRINT_INPUT_ERROR("number of bins should be an integer.\n");
    }
    num_triples_ = (num_param - 3) / 7;
    itype_cpu.resize(num_triples_);
    jtype_cpu.resize(num_triples_);
    ktype_cpu.resize(num_triples_);
    rc_min_j_cpu.resize(num_triples_);
    rc_max_j_cpu.resize(num_triples_);
    rc_min_k_cpu.resize(num_triples_);
    rc_max_k_cpu.resize(num_triples_);
    for (int i = 0; i < num_triples_; i++) {
      if (!is_valid_int(param[3 + i * 7], &itype_cpu[i])) {
        std::string message = "itype in triples " + std::to_string(i) + " should be an integer.\n";
        PRINT_INPUT_ERROR(message.c_str());
      }
      if (itype_cpu[i] < 0) {
        std::string message =
          "itype in triples " + std::to_string(i) + " should be non-negative.\n";
        PRINT_INPUT_ERROR(message.c_str());
      }
      if (itype_cpu[i] > number_of_types - 1) {
        std::string message = "itype in triples " + std::to_string(i) +
                              " should be less than number of atomic types " +
                              std::to_string(number_of_types) + ".\n";
        PRINT_INPUT_ERROR(message.c_str());
      }
      if (!is_valid_int(param[4 + i * 7], &jtype_cpu[i])) {
        std::string message = "jtype in triples " + std::to_string(i) + " should be an integer.\n";
        PRINT_INPUT_ERROR(message.c_str());
      }
      if (jtype_cpu[i] < 0) {
        std::string message =
          "jtype in triples " + std::to_string(i) + " should be non-negative.\n";
        PRINT_INPUT_ERROR(message.c_str());
      }
      if (jtype_cpu[i] > number_of_types - 1) {
        std::string message = "jtype in triples " + std::to_string(i) +
                              " should be less than number of atomic types " +
                              std::to_string(number_of_types) + ".\n";
        PRINT_INPUT_ERROR(message.c_str());
      }
      if (!is_valid_int(param[5 + i * 7], &ktype_cpu[i])) {
        std::string message = "ktype in triples " + std::to_string(i) + " should be an integer.\n";
        PRINT_INPUT_ERROR(message.c_str());
      }
      if (ktype_cpu[i] < 0) {
        std::string message =
          "ktype in triples " + std::to_string(i) + " should be non-negative.\n";
        PRINT_INPUT_ERROR(message.c_str());
      }
      if (ktype_cpu[i] > number_of_types - 1) {
        std::string message = "ktype in triples " + std::to_string(i) +
                              " should be less than number of atomic types " +
                              std::to_string(number_of_types) + ".\n";
        PRINT_INPUT_ERROR(message.c_str());
      }
      if (!is_valid_real(param[6 + i * 7], &rc_min_j_cpu[i])) {
        std::string message =
          "minimum radial cutoff in triples " + std::to_string(i) + " should be a number.\n";
        PRINT_INPUT_ERROR(message.c_str());
      }
      if (!is_valid_real(param[7 + i * 7], &rc_max_j_cpu[i])) {
        std::string message =
          "maximum radial cutoff in triples " + std::to_string(i) + " should be a number.\n";
        PRINT_INPUT_ERROR(message.c_str());
      }
      if (rc_min_j_cpu[i] >= rc_max_j_cpu[i]) {
        std::string message =
          "minimum radial cutoff should be less than maximum radial cutoff for triples " +
          std::to_string(i) + ".\n";
        PRINT_INPUT_ERROR(message.c_str());
      }
      if (rc_min_j_cpu[i] < 0) {
        std::string message =
          "minimum radial cutoff in triples " + std::to_string(i) + " should be positive.\n";
        PRINT_INPUT_ERROR(message.c_str());
      }
      if (!is_valid_real(param[8 + i * 7], &rc_min_k_cpu[i])) {
        std::string message =
          "minimum radial cutoff in triples " + std::to_string(i) + " should be a number.\n";
        PRINT_INPUT_ERROR(message.c_str());
      }
      if (!is_valid_real(param[9 + i * 7], &rc_max_k_cpu[i])) {
        std::string message =
          "maximum radial cutoff in triples " + std::to_string(i) + " should be a number.\n";
        PRINT_INPUT_ERROR(message.c_str());
      }
      if (rc_min_k_cpu[i] >= rc_max_k_cpu[i]) {
        std::string message =
          "minimum radial cutoff should be less than maximum radial cutoff for triples " +
          std::to_string(i) + ".\n";
        PRINT_INPUT_ERROR(message.c_str());
      }
      if (rc_min_k_cpu[i] < 0) {
        std::string message =
          "minimum radial cutoff in triples " + std::to_string(i) + " should be positive.\n";
        PRINT_INPUT_ERROR(message.c_str());
      }
    }

    global_ = false;
    rc_max_ = 0.0;
    for (int i = 0; i < num_triples_; i++) {
      if (rc_max_ < rc_max_j_cpu[i]) {
        rc_max_ = rc_max_j_cpu[i];
      }
      if (rc_max_ < rc_max_k_cpu[i]) {
        rc_max_ = rc_max_k_cpu[i];
      }
    }
    double thickness_half[3] = {
      box.get_volume() / box.get_area(0) / 2.5,
      box.get_volume() / box.get_area(1) / 2.5,
      box.get_volume() / box.get_area(2) / 2.5};
    if (rc_max_ > thickness_half[0] || rc_max_ > thickness_half[1] || rc_max_ > thickness_half[2]) {
      std::string message =
        "The box has a thickness < 2.5 RDF radial cutoffs in a periodic direction.\n"
        "                Please increase the periodic direction(s).\n";
      PRINT_INPUT_ERROR(message.c_str());
    }
    printf("    Local triple ADF will be computed.\n");
    printf("    ADF sample interval is %d step.\n", num_interval_);
    printf("    radial cutoff will be divided into %d bins.\n", adf_bins_);
    for (int i = 0; i < num_triples_; i++) {
      printf(
        "    Triple %d-%d-%d: %g %g %g %g.\n",
        itype_cpu[i],
        jtype_cpu[i],
        ktype_cpu[i],
        rc_min_j_cpu[i],
        rc_max_j_cpu[i],
        rc_min_k_cpu[i],
        rc_max_k_cpu[i]);
    }
  }
}
