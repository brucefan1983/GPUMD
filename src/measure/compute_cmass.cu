/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/

/*----------------------------------------------------------------------------80
GPUMD Contributing author: Zihan Yan (Westlake University)
------------------------------------------------------------------------------*/

#include "compute_cmass.cuh"
#include "integrate/integrate.cuh"
#include "parse_utilities.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"

static constexpr int block_size = 256;

static __global__ void sum_mass_and_moments(
  int atom_count,
  int selected_count,
  const int* selected_atoms,
  const double* mass,
  const double* position,
  double* sums)
{
  // One block for each Cartesian mass moment and one for the total mass.
  const int component = blockIdx.x;
  const int thread = threadIdx.x;
  double sum = 0.0;
  for (int i = thread; i < selected_count; i += block_size) {
    const int atom = selected_atoms ? selected_atoms[i] : i;
    sum += component < 3 ? mass[atom] * position[component * atom_count + atom] : mass[atom];
  }

  __shared__ double partial[block_size];
  partial[thread] = sum;
  __syncthreads();
  for (int stride = block_size / 2; stride > 0; stride /= 2) {
    if (thread < stride) {
      partial[thread] += partial[thread + stride];
    }
    __syncthreads();
  }

  if (thread == 0) {
    sums[component] = partial[0];
  }
}

ComputeCmass::ComputeCmass(
  const std::vector<std::string>& tokens, const std::vector<Group>& groups, Atom& atom)
{
  action_name = "compute_cmass";
  if (tokens.size() != 2 && (tokens.size() != 5 || tokens[2] != "group")) {
    PRINT_INPUT_ERROR("Expected compute_cmass <sample_interval> [group <group_method> <group>].\n");
  }
  if (!is_valid_int(tokens[1], &sample_interval_) || sample_interval_ <= 0) {
    PRINT_INPUT_ERROR("compute_cmass sampling interval must be a positive integer.\n");
  }

  printf("Compute center of mass every %d steps.\n", sample_interval_);
  if (tokens.size() == 5) {
    int option = 2;
    parse_group(tokens, false, groups, option, grouping_method_, group_id_);
    if (groups[grouping_method_].cpu_size[group_id_] == 0) {
      PRINT_INPUT_ERROR("compute_cmass group must contain at least one atom.\n");
    }
  }
  atom.enable_unwrapped_position();
}

void ComputeCmass::pre_run(
  int number_of_steps,
  double time_step,
  Integrate& integrate,
  std::vector<Group>& groups,
  Atom& atom,
  Box& box,
  Force& force)
{
  if (is_pimd(integrate.get_type())) {
    PRINT_INPUT_ERROR(
      "compute_cmass supports classical MD only. PIMD, RPMD, and TRPMD are unsupported.\n");
  }
  sums_.resize(4);
  file_ = my_fopen("cmass.out", "a");
  fprintf(file_, "# compute_cmass %d", sample_interval_);
  if (grouping_method_ >= 0) {
    fprintf(file_, " group %d %d", grouping_method_, group_id_);
  }
  fprintf(file_, "\n# columns time(ps) x(A) y(A) z(A)\n");
}

void ComputeCmass::end_of_step(
  int number_of_steps,
  int step,
  int fixed_group,
  int move_group,
  double global_time,
  double temperature,
  Integrate& integrate,
  Box& box,
  std::vector<Group>& groups,
  GPU_Vector<double>& thermo,
  Atom& atom,
  Force& force)
{
  if ((step + 1) % sample_interval_ != 0) {
    return;
  }

  int selected_count = atom.number_of_atoms;
  const int* selected_atoms = nullptr;
  if (grouping_method_ >= 0) {
    const Group& group = groups[grouping_method_];
    selected_count = group.cpu_size[group_id_];
    selected_atoms = group.contents.data() + group.cpu_size_sum[group_id_];
  }
  sum_mass_and_moments<<<4, block_size>>>(
    atom.number_of_atoms,
    selected_count,
    selected_atoms,
    atom.mass.data(),
    atom.unwrapped_position.data(),
    sums_.data());
  GPU_CHECK_KERNEL

  double sums[4];
  sums_.copy_to_host(sums);
  fprintf(
    file_,
    "%20.10e%20.10e%20.10e%20.10e\n",
    global_time * TIME_UNIT_CONVERSION / 1000.0,
    sums[0] / sums[3],
    sums[1] / sums[3],
    sums[2] / sums[3]);
  fflush(file_);
}

void ComputeCmass::post_run(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  int number_of_steps,
  double time_step,
  double temperature)
{
  fclose(file_);
}
