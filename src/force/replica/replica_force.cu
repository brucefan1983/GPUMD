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
Force evaluation of the replicas on one GPU, see replica_force.cuh.
------------------------------------------------------------------------------*/

#include "replica_force.cuh"
#include "replica_nep.cuh"
#include "replica_nep_charge.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"

namespace
{

// Wraps atoms into the box. Coordinates are rewritten only when an atom is wrapped:
// the x -> s -> x round trip is not exact, so rewriting every atom would perturb
// positions in the last bit at each extra evaluation (e.g. after a resume).
__global__ void apply_pbc(const int number_of_atoms, const Box box, double* x, double* y, double* z)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= number_of_atoms)
    return;
  double sx = box.cpu_h[9] * x[n] + box.cpu_h[10] * y[n] + box.cpu_h[11] * z[n];
  double sy = box.cpu_h[12] * x[n] + box.cpu_h[13] * y[n] + box.cpu_h[14] * z[n];
  double sz = box.cpu_h[15] * x[n] + box.cpu_h[16] * y[n] + box.cpu_h[17] * z[n];
  bool wrapped = false;
  if (box.pbc_x == 1) {
    if (sx < 0.0) {
      sx += 1.0;
      wrapped = true;
    } else if (sx > 1.0) {
      sx -= 1.0;
      wrapped = true;
    }
  }
  if (box.pbc_y == 1) {
    if (sy < 0.0) {
      sy += 1.0;
      wrapped = true;
    } else if (sy > 1.0) {
      sy -= 1.0;
      wrapped = true;
    }
  }
  if (box.pbc_z == 1) {
    if (sz < 0.0) {
      sz += 1.0;
      wrapped = true;
    } else if (sz > 1.0) {
      sz -= 1.0;
      wrapped = true;
    }
  }
  if (wrapped) {
    x[n] = box.cpu_h[0] * sx + box.cpu_h[1] * sy + box.cpu_h[2] * sz;
    y[n] = box.cpu_h[3] * sx + box.cpu_h[4] * sy + box.cpu_h[5] * sz;
    z[n] = box.cpu_h[6] * sx + box.cpu_h[7] * sy + box.cpu_h[8] * sz;
  }
}

__global__ void initialize_properties(
  const int number_of_atoms,
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

Replica_Force::Replica_Force() = default;

Replica_Force::~Replica_Force() = default;

void Replica_Force::initialize(
  const std::string& potential_file,
  const Box& box,
  const int number_of_atoms,
  const RunInput& run_input,
  const bool verbose)
{
  if (model_type_ != Model_Type::none)
    PRINT_INPUT_ERROR("Replica dynamics supports exactly one potential.\n");
  potential_file_ = potential_file;
  number_of_atoms_ = number_of_atoms;
  run_input_ = &run_input;
  verbose_ = verbose;
  if (is_nep_charge_potential(potential_file)) {
    if (!box.pbc_x || !box.pbc_y || !box.pbc_z)
      PRINT_INPUT_ERROR("qNEP replica dynamics requires periodic boundaries.\n");
    model_type_ = Model_Type::charge_nep;
    return;
  }
  energy_nep_.reset(
    new NEP_Replica(potential_file.c_str(), number_of_atoms, verbose));
  model_type_ = Model_Type::energy_nep;
}

void Replica_Force::clear()
{
  charge_neps_.clear();
  energy_nep_.reset();
  model_type_ = Model_Type::none;
}

void Replica_Force::prepare_stream(const gpuStream_t stream)
{
  if (model_type_ == Model_Type::none || stream == nullptr)
    PRINT_INPUT_ERROR("Replica force evaluation requires a potential and a GPU stream.\n");
  if (model_type_ == Model_Type::energy_nep) {
    energy_nep_->prepare_stream(stream);
    return;
  }
  if (charge_neps_.find(stream) == charge_neps_.end()) {
    const bool verbose = verbose_ && charge_neps_.empty();
    charge_neps_.emplace(
      stream,
      std::unique_ptr<NEP_Charge_Replica>(new NEP_Charge_Replica(
        potential_file_.c_str(), number_of_atoms_, *run_input_, stream, verbose)));
  }
}

void Replica_Force::compute(
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& type,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom,
  const gpuStream_t stream)
{
  if (model_type_ == Model_Type::none || stream == nullptr)
    PRINT_INPUT_ERROR("Replica force evaluation requires a potential and a GPU stream.\n");
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
    energy_nep_->compute(
      stream_box, type, position_per_atom, potential_per_atom, force_per_atom, virial_per_atom,
      stream);
    return;
  }
  const auto iterator = charge_neps_.find(stream);
  if (iterator == charge_neps_.end())
    PRINT_INPUT_ERROR("The qNEP stream has not been prepared.\n");
  iterator->second->compute(
    stream_box, type, position_per_atom, potential_per_atom, force_per_atom, virial_per_atom);
}
