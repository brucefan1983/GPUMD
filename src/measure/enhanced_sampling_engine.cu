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
Coordinate collective variables and biases for enhanced sampling.
------------------------------------------------------------------------------*/

#include "enhanced_sampling_engine.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include <utility>

static __global__ void gpu_enhanced_sampling_sum_potential(
  const int number_of_atoms,
  const double* g_potential,
  double* g_sum)
{
  const int number_of_rounds = (number_of_atoms - 1) / 1024 + 1;
  __shared__ double s_sum[1024];
  s_sum[threadIdx.x] = 0.0;

  for (int round = 0; round < number_of_rounds; ++round) {
    const int atom_index = threadIdx.x + round * 1024;
    if (atom_index < number_of_atoms) {
      s_sum[threadIdx.x] += g_potential[atom_index];
    }
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      s_sum[threadIdx.x] += s_sum[threadIdx.x + offset];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    g_sum[0] = s_sum[0];
  }
}

static __global__ void gpu_enhanced_sampling_sum_bias_energy(
  const int number_of_biases,
  const double* g_bias_energies,
  double* g_total_bias_energy)
{
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    double sum = 0.0;
    for (int bias_index = 0; bias_index < number_of_biases; ++bias_index) {
      sum += g_bias_energies[bias_index];
    }
    g_total_bias_energy[0] = sum;
  }
}

static __global__ void gpu_enhanced_sampling_add_bias_energy(
  const int number_of_atoms,
  const double* g_total_bias_energy,
  double* g_potential)
{
  const int atom_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom_index < number_of_atoms) {
    g_potential[atom_index] += g_total_bias_energy[0] / number_of_atoms;
  }
}

EnhancedSamplingEngine::EnhancedSamplingEngine()
  : number_of_atoms_(0), cpu_total_bias_energy_(0.0)
{
  for (int component = 0; component < 9; ++component) {
    cpu_total_bias_virial_[component] = 0.0;
  }
  cpu_potential_before_after_[0] = 0.0;
  cpu_potential_before_after_[1] = 0.0;
}

void EnhancedSamplingEngine::add_cv(std::unique_ptr<EnhancedSamplingCV> cv)
{
  if (!cv) {
    PRINT_INPUT_ERROR("Cannot add an empty collective variable.\n");
  }
  const std::string name = cv->get_name();
  if (has_cv(name)) {
    PRINT_INPUT_ERROR("Collective variable names must be unique.\n");
  }
  const int index = static_cast<int>(cvs_.size());
  cv_indices_[name] = index;
  cvs_.emplace_back(std::move(cv));
}

void EnhancedSamplingEngine::add_bias(std::unique_ptr<EnhancedSamplingBias> bias)
{
  if (!bias) {
    PRINT_INPUT_ERROR("Cannot add an empty bias.\n");
  }
  const std::string name = bias->get_name();
  if (has_bias(name)) {
    PRINT_INPUT_ERROR("Bias names must be unique.\n");
  }
  if (
    bias->get_argument_index() < 0 ||
    bias->get_argument_index() >= static_cast<int>(cvs_.size())) {
    PRINT_INPUT_ERROR("A bias refers to an undefined collective variable.\n");
  }
  const int index = static_cast<int>(biases_.size());
  bias_indices_[name] = index;
  biases_.emplace_back(std::move(bias));
}

bool EnhancedSamplingEngine::has_cv(const std::string& name) const
{
  return cv_indices_.find(name) != cv_indices_.end();
}

bool EnhancedSamplingEngine::has_bias(const std::string& name) const
{
  return bias_indices_.find(name) != bias_indices_.end();
}

int EnhancedSamplingEngine::get_cv_index(const std::string& name) const
{
  std::map<std::string, int>::const_iterator iterator = cv_indices_.find(name);
  if (iterator == cv_indices_.end()) {
    PRINT_INPUT_ERROR("A bias refers to an undefined collective variable.\n");
  }
  return iterator->second;
}

int EnhancedSamplingEngine::get_number_of_cvs() const
{
  return static_cast<int>(cvs_.size());
}

int EnhancedSamplingEngine::get_number_of_biases() const
{
  return static_cast<int>(biases_.size());
}

void EnhancedSamplingEngine::prepare(const int number_of_atoms)
{
  if (number_of_atoms <= 0) {
    PRINT_INPUT_ERROR("Enhanced sampling requires at least one atom.\n");
  }
  if (cvs_.empty()) {
    PRINT_INPUT_ERROR("Enhanced sampling requires at least one collective variable.\n");
  }
  if (biases_.empty()) {
    PRINT_INPUT_ERROR("Enhanced sampling requires at least one bias.\n");
  }

  number_of_atoms_ = number_of_atoms;
  for (int cv_index = 0; cv_index < cvs_.size(); ++cv_index) {
    cvs_[cv_index]->prepare(number_of_atoms_);
  }

  cv_values_.resize(cvs_.size());
  cv_derivatives_.resize(cvs_.size());
  bias_energies_.resize(biases_.size());
  bias_derivatives_.resize(biases_.size());
  total_bias_energy_.resize(1);
  total_bias_virial_.resize(9);
  potential_before_after_.resize(2);

  cpu_cv_values_.resize(cvs_.size());
  cpu_cv_derivatives_.resize(cvs_.size());
  cpu_bias_energies_.resize(biases_.size());
  cpu_bias_derivatives_.resize(biases_.size());
}

void EnhancedSamplingEngine::calculate(
  const Box& box,
  Atom& atom,
  const bool collect_output_energy)
{
  if (collect_output_energy) {
    gpu_enhanced_sampling_sum_potential<<<1, 1024>>>(
      number_of_atoms_,
      atom.potential_per_atom.data(),
      potential_before_after_.data());
    GPU_CHECK_KERNEL
  }

  cv_derivatives_.fill(0.0);
  total_bias_virial_.fill(0.0);

  for (int cv_index = 0; cv_index < cvs_.size(); ++cv_index) {
    cvs_[cv_index]->compute(
      box,
      atom.position_per_atom,
      cv_values_.data() + cv_index);
  }

  // These kernels are launched in the default stream. Their order is what
  // makes plain += safe when biases share a CV or CVs share atoms.
  for (int bias_index = 0; bias_index < biases_.size(); ++bias_index) {
    biases_[bias_index]->evaluate(
      cv_values_.data(),
      cv_derivatives_.data(),
      bias_energies_.data() + bias_index,
      bias_derivatives_.data() + bias_index);
  }

  gpu_enhanced_sampling_sum_bias_energy<<<1, 1>>>(
    static_cast<int>(biases_.size()),
    bias_energies_.data(),
    total_bias_energy_.data());
  GPU_CHECK_KERNEL

  for (int cv_index = 0; cv_index < cvs_.size(); ++cv_index) {
    cvs_[cv_index]->add_force_virial(
      cv_derivatives_.data() + cv_index,
      atom.force_per_atom,
      atom.virial_per_atom,
      total_bias_virial_.data());
  }

  const int block_size = 128;
  const int grid_size = (number_of_atoms_ - 1) / block_size + 1;
  gpu_enhanced_sampling_add_bias_energy<<<grid_size, block_size>>>(
    number_of_atoms_,
    total_bias_energy_.data(),
    atom.potential_per_atom.data());
  GPU_CHECK_KERNEL

  if (collect_output_energy) {
    gpu_enhanced_sampling_sum_potential<<<1, 1024>>>(
      number_of_atoms_,
      atom.potential_per_atom.data(),
      potential_before_after_.data() + 1);
    GPU_CHECK_KERNEL
  }
}

void EnhancedSamplingEngine::check()
{
  for (int cv_index = 0; cv_index < cvs_.size(); ++cv_index) {
    cvs_[cv_index]->check();
  }
}

void EnhancedSamplingEngine::write_header(FILE* fid) const
{
  fprintf(fid, "# enhanced_sampling\n");
  fprintf(fid, "# format_version 1\n");
  fprintf(
    fid,
    "# units distance=A energy=eV derivative=eV/A virial=eV\n");
  fprintf(fid, "# columns step");
  for (int cv_index = 0; cv_index < cvs_.size(); ++cv_index) {
    fprintf(fid, " cv_%s", cvs_[cv_index]->get_name().c_str());
  }
  for (int bias_index = 0; bias_index < biases_.size(); ++bias_index) {
    fprintf(fid, " bias_%s", biases_[bias_index]->get_name().c_str());
  }
  for (int bias_index = 0; bias_index < biases_.size(); ++bias_index) {
    fprintf(
      fid,
      " dbias_%s_d_%s",
      biases_[bias_index]->get_name().c_str(),
      biases_[bias_index]->get_argument_name().c_str());
  }
  for (int cv_index = 0; cv_index < cvs_.size(); ++cv_index) {
    fprintf(fid, " derivative_%s", cvs_[cv_index]->get_name().c_str());
  }
  fprintf(fid, " bias_total potential_before potential_after");
  fprintf(fid, " virial_xx virial_xy virial_xz");
  fprintf(fid, " virial_yx virial_yy virial_yz");
  fprintf(fid, " virial_zx virial_zy virial_zz\n");
  fflush(fid);
}

void EnhancedSamplingEngine::write_output(FILE* fid, const int step)
{
  check();
  cv_values_.copy_to_host(cpu_cv_values_.data());
  cv_derivatives_.copy_to_host(cpu_cv_derivatives_.data());
  bias_energies_.copy_to_host(cpu_bias_energies_.data());
  bias_derivatives_.copy_to_host(cpu_bias_derivatives_.data());
  total_bias_energy_.copy_to_host(&cpu_total_bias_energy_);
  total_bias_virial_.copy_to_host(cpu_total_bias_virial_);
  potential_before_after_.copy_to_host(cpu_potential_before_after_);

  fprintf(fid, "%d", step);
  for (int cv_index = 0; cv_index < cpu_cv_values_.size(); ++cv_index) {
    fprintf(fid, " %.17g", cpu_cv_values_[cv_index]);
  }
  for (int bias_index = 0; bias_index < cpu_bias_energies_.size(); ++bias_index) {
    fprintf(fid, " %.17g", cpu_bias_energies_[bias_index]);
  }
  for (int bias_index = 0; bias_index < cpu_bias_derivatives_.size(); ++bias_index) {
    fprintf(fid, " %.17g", cpu_bias_derivatives_[bias_index]);
  }
  for (int cv_index = 0; cv_index < cpu_cv_derivatives_.size(); ++cv_index) {
    fprintf(fid, " %.17g", cpu_cv_derivatives_[cv_index]);
  }
  fprintf(fid, " %.17g", cpu_total_bias_energy_);
  fprintf(fid, " %.17g %.17g", cpu_potential_before_after_[0], cpu_potential_before_after_[1]);

  const int row_major_order[9] = {0, 3, 4, 6, 1, 5, 7, 8, 2};
  for (int component = 0; component < 9; ++component) {
    fprintf(fid, " %.17g", cpu_total_bias_virial_[row_major_order[component]]);
  }
  fprintf(fid, "\n");
  fflush(fid);
}
