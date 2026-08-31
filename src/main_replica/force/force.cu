/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
    for more details. You should have received a copy of the GNU General
    Public License along with GPUMD. If not, see <http://www.gnu.org/licenses/>.
*/

#include "force.cuh"
#include "nep.cuh"
#include "nep_charge.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace
{

bool is_energy_nep(const char* name)
{
  return
    strcmp(name, "nep4") == 0 || strcmp(name, "nep4_zbl") == 0 ||
    strcmp(name, "nep5") == 0 || strcmp(name, "nep5_zbl") == 0;
}

bool is_charge_nep(const char* name)
{
  return
    strcmp(name, "nep4_charge1") == 0 || strcmp(name, "nep4_zbl_charge1") == 0 ||
    strcmp(name, "nep4_charge2") == 0 || strcmp(name, "nep4_zbl_charge2") == 0;
}

std::string read_potential_name(const char* file_potential)
{
  std::ifstream input(file_potential);
  if (!input.is_open())
    PRINT_INPUT_ERROR("Cannot open the NEP potential file.\n");

  std::string line;
  while (std::getline(input, line)) {
    const std::vector<std::string> tokens = get_tokens(line);
    if (tokens.empty())
      continue;
    return tokens[0];
  }
  PRINT_INPUT_ERROR("The NEP potential file is empty.\n");
  return std::string();
}

__global__ void apply_pbc(int number_of_atoms, Box box, double* x, double* y, double* z)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= number_of_atoms)
    return;

  double sx = box.cpu_h[9] * x[n] + box.cpu_h[10] * y[n] + box.cpu_h[11] * z[n];
  double sy = box.cpu_h[12] * x[n] + box.cpu_h[13] * y[n] + box.cpu_h[14] * z[n];
  double sz = box.cpu_h[15] * x[n] + box.cpu_h[16] * y[n] + box.cpu_h[17] * z[n];
  if (box.pbc_x == 1) {
    if (sx < 0.0)
      sx += 1.0;
    else if (sx > 1.0)
      sx -= 1.0;
  }
  if (box.pbc_y == 1) {
    if (sy < 0.0)
      sy += 1.0;
    else if (sy > 1.0)
      sy -= 1.0;
  }
  if (box.pbc_z == 1) {
    if (sz < 0.0)
      sz += 1.0;
    else if (sz > 1.0)
      sz -= 1.0;
  }
  x[n] = box.cpu_h[0] * sx + box.cpu_h[1] * sy + box.cpu_h[2] * sz;
  y[n] = box.cpu_h[3] * sx + box.cpu_h[4] * sy + box.cpu_h[5] * sz;
  z[n] = box.cpu_h[6] * sx + box.cpu_h[7] * sy + box.cpu_h[8] * sz;
}

__global__ void initialize_properties(
  int number_of_atoms,
  double* force_x,
  double* force_y,
  double* force_z,
  double* potential,
  double* virial)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= number_of_atoms)
    return;

  force_x[n] = 0.0;
  force_y[n] = 0.0;
  force_z[n] = 0.0;
  potential[n] = 0.0;
  for (int component = 0; component < 9; ++component)
    virial[n + component * number_of_atoms] = 0.0;
}

} // namespace

Force::Force() = default;
Force::~Force() = default;

void Force::parse_potential(
  const char** param, const int num_param, const Box& box, const int number_of_atoms)
{
  (void)box;
  if (num_param != 2)
    PRINT_INPUT_ERROR("potential should have one parameter.\n");
  if (model_type_ != Model_Type::none)
    PRINT_INPUT_ERROR("Replica dynamics supports exactly one potential.\n");

  const std::string potential_name = read_potential_name(param[1]);
  potential_file_ = param[1];
  number_of_atoms_ = number_of_atoms;
  if (is_energy_nep(potential_name.c_str())) {
    model_type_ = Model_Type::energy_nep;
    potential_.reset(new NEP(param[1], number_of_atoms, false));
    if (potential_->nep_model_type != 0)
      PRINT_INPUT_ERROR("Replica dynamics only supports energy NEP4/NEP5 models.\n");
    potential_->N1 = 0;
    potential_->N2 = number_of_atoms;
  } else if (is_charge_nep(potential_name.c_str())) {
    if (!box.pbc_x || !box.pbc_y || !box.pbc_z)
      PRINT_INPUT_ERROR("qNEP replica dynamics requires periodic boundaries.\n");
    model_type_ = Model_Type::charge_nep;
  } else {
    PRINT_INPUT_ERROR("Replica dynamics only supports energy NEP4/NEP5 and qNEP4 models.\n");
  }
}

void Force::clear()
{
  charge_potentials_.clear();
  potential_.reset();
  model_type_ = Model_Type::none;
}

bool Force::has_potential() const { return model_type_ != Model_Type::none; }

void Force::prepare_stream(const gpuStream_t stream)
{
  if (model_type_ == Model_Type::none || stream == nullptr)
    PRINT_INPUT_ERROR("Replica force evaluation requires an explicit GPU stream.\n");
  if (model_type_ == Model_Type::energy_nep) {
    potential_->prepare_stream(stream);
    return;
  }

  if (charge_potentials_.find(stream) == charge_potentials_.end()) {
    std::unique_ptr<NEP_Charge> potential(new NEP_Charge(
      potential_file_.c_str(),
      number_of_atoms_,
      stream,
      charge_potentials_.empty()));
    potential->N1 = 0;
    potential->N2 = number_of_atoms_;
    charge_potentials_.emplace(stream, std::move(potential));
  }
}

void Force::compute(
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& type,
  std::vector<Group>& group,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  (void)group;
  if (model_type_ != Model_Type::energy_nep || !potential_)
    PRINT_INPUT_ERROR("No NEP potential has been initialized.\n");

  box.set_is_orthogonal();
  const int number_of_atoms = type.size();
  apply_pbc<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms,
    box,
    position_per_atom.data(),
    position_per_atom.data() + number_of_atoms,
    position_per_atom.data() + number_of_atoms * 2);
  GPU_CHECK_KERNEL
  initialize_properties<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms,
    force_per_atom.data(),
    force_per_atom.data() + number_of_atoms,
    force_per_atom.data() + number_of_atoms * 2,
    potential_per_atom.data(),
    virial_per_atom.data());
  GPU_CHECK_KERNEL
  potential_->compute(
    box,
    type,
    position_per_atom,
    potential_per_atom,
    force_per_atom,
    virial_per_atom);
}

void Force::compute(
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& type,
  std::vector<Group>& group,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom,
  const gpuStream_t stream)
{
  (void)group;
  if (model_type_ == Model_Type::none || stream == nullptr)
    PRINT_INPUT_ERROR("Replica force evaluation requires an explicit GPU stream.\n");

  Box stream_box = box;
  stream_box.set_is_orthogonal();
  const int number_of_atoms = type.size();
  apply_pbc<<<(number_of_atoms - 1) / 128 + 1, 128, 0, stream>>>(
    number_of_atoms,
    stream_box,
    position_per_atom.data(),
    position_per_atom.data() + number_of_atoms,
    position_per_atom.data() + number_of_atoms * 2);
  GPU_CHECK_KERNEL
  initialize_properties<<<(number_of_atoms - 1) / 128 + 1, 128, 0, stream>>>(
    number_of_atoms,
    force_per_atom.data(),
    force_per_atom.data() + number_of_atoms,
    force_per_atom.data() + number_of_atoms * 2,
    potential_per_atom.data(),
    virial_per_atom.data());
  GPU_CHECK_KERNEL
  if (model_type_ == Model_Type::energy_nep) {
    potential_->compute(
      stream_box,
      type,
      position_per_atom,
      potential_per_atom,
      force_per_atom,
      virial_per_atom,
      stream);
  } else {
    const auto iterator = charge_potentials_.find(stream);
    if (iterator == charge_potentials_.end())
      PRINT_INPUT_ERROR("The qNEP stream has not been prepared.\n");
    iterator->second->compute(
      stream_box,
      type,
      position_per_atom,
      potential_per_atom,
      force_per_atom,
      virial_per_atom);
  }
}
