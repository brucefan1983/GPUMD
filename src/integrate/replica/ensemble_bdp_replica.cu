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
The nvt_bdp thermostat of one replica, see ensemble_bdp_replica.cuh.
[1] G. Bussi et al. J. Chem. Phys. 126, 014101 (2007).
------------------------------------------------------------------------------*/

#include "ensemble_bdp_replica.cuh"
#include "replica_thermo_kernels.cuh"
#include "integrate/svr_utilities.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include <iomanip>
#include <sstream>

namespace
{

// The rescaling of resamplekin() with the kinetic energy read on the device.
__global__ void gpu_resamplekin_and_scale(
  const int number_of_atoms,
  const int ndeg,
  const double temperature,
  const double temperature_coupling,
  const double gaussian,
  const double sum_of_squares,
  const double* thermo,
  double* velocity_x,
  double* velocity_y,
  double* velocity_z)
{
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= number_of_atoms)
    return;
  const double factor = svr_velocity_factor(
    thermo[0], ndeg, temperature, temperature_coupling, gaussian, sum_of_squares);
  velocity_x[atom] *= factor;
  velocity_y[atom] *= factor;
  velocity_z[atom] *= factor;
}

} // namespace

Ensemble_BDP_Replica::Ensemble_BDP_Replica(
  const double temperature_input, const double temperature_coupling_input)
{
  type = EnsembleType::NVT_BDP;
  temperature = temperature_input;
  temperature_coupling = temperature_coupling_input;
  fixed_group = -1;
  move_group = -1;
  initialize_rng();
}

int Ensemble_BDP_Replica::get_degrees_of_freedom(const int number_of_atoms) const
{
  const int degrees_of_freedom = 3 * number_of_atoms - removed_degrees_of_freedom;
  if (degrees_of_freedom <= 0)
    PRINT_INPUT_ERROR("Temperature calculation requires positive degrees of freedom.");
  return degrees_of_freedom;
}

void Ensemble_BDP_Replica::compute1(
  const double time_step, Atom& atom, const gpuStream_t stream)
{
  const std::vector<Group> no_group;
  velocity_verlet(
    true,
    time_step,
    no_group,
    atom.mass,
    atom.force_per_atom,
    atom.position_per_atom,
    atom.velocity_per_atom,
    stream);
}

void Ensemble_BDP_Replica::compute2(
  const double time_step,
  const double volume,
  Atom& atom,
  GPU_Vector<double>& thermo,
  const gpuStream_t stream)
{
  const std::vector<Group> no_group;
  const int number_of_atoms = atom.number_of_atoms;
  velocity_verlet(
    false,
    time_step,
    no_group,
    atom.mass,
    atom.force_per_atom,
    atom.position_per_atom,
    atom.velocity_per_atom,
    stream);
  find_thermo(volume, atom, thermo, stream);
  const int ndeg = get_degrees_of_freedom(number_of_atoms);
  double gaussian = 0.0;
  double sum_of_squares = 0.0;
  draw_svr_noise(ndeg, gaussian, sum_of_squares);
  gpu_resamplekin_and_scale<<<(number_of_atoms - 1) / 128 + 1, 128, 0, stream>>>(
    number_of_atoms,
    ndeg,
    temperature,
    temperature_coupling,
    gaussian,
    sum_of_squares,
    thermo.data(),
    atom.velocity_per_atom.data(),
    atom.velocity_per_atom.data() + number_of_atoms,
    atom.velocity_per_atom.data() + 2 * number_of_atoms);
  GPU_CHECK_KERNEL
}

void Ensemble_BDP_Replica::find_thermo(
  const double volume, const Atom& atom, GPU_Vector<double>& thermo, const gpuStream_t stream)
{
  const int N = atom.number_of_atoms;
  gpu_find_thermo_replicas<<<8, 1024, 0, stream>>>(
    N,
    N,
    get_degrees_of_freedom(N),
    volume,
    atom.mass.data(),
    atom.potential_per_atom.data(),
    atom.velocity_per_atom.data(),
    atom.virial_per_atom.data(),
    thermo.data());
  GPU_CHECK_KERNEL
}

void Ensemble_BDP_Replica::draw_svr_noise(
  const int ndeg, double& gaussian, double& sum_of_squares)
{
  const SVR_Noise noise = draw_resamplekin_noise(ndeg, rng, gaussian_cache_);
  gaussian = noise.gaussian;
  sum_of_squares = noise.sum_of_squares;
}

void Ensemble_BDP_Replica::seed_rng(const unsigned int seed)
{
  rng.seed(seed);
  gaussian_cache_ = SVR_Gaussian_Cache();
}

std::string Ensemble_BDP_Replica::export_rng_state() const
{
  std::ostringstream output;
  output << 1 << ' ' << gaussian_cache_.iset << ' ' << std::setprecision(17)
         << gaussian_cache_.gset << ' ' << rng;
  return output.str();
}

bool Ensemble_BDP_Replica::import_rng_state(const std::string& state)
{
  std::istringstream input(state);
  int version = 0;
  SVR_Gaussian_Cache cache;
  std::mt19937 restored_rng;
  if (
    !(input >> version >> cache.iset >> cache.gset >> restored_rng) || version != 1 ||
    (cache.iset != 0 && cache.iset != 1)) {
    return false;
  }
  input >> std::ws;
  if (!input.eof()) {
    return false;
  }
  gaussian_cache_ = cache;
  rng = restored_rng;
  return true;
}
