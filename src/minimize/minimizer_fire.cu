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
The FIRE (fast inertial relaxation engine) minimizer
Reference: PhysRevLett 97, 170201 (2006)
           Computational Materials Science 175 (2020) 109584
------------------------------------------------------------------------------*/

#include "minimizer_fire.cuh"

void Minimizer_FIRE::compute(
  Force& force,
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& type,
  std::vector<Group>& group,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  double next_dt;
  const int size = number_of_atoms_ * 3;
  int base = (number_of_steps_ >= 10) ? (number_of_steps_ / 10) : 1;
  // create a velocity vector in GPU
  GPU_Vector<double> v(size, 0);
  GPU_Vector<double> temp1(size);
  GPU_Vector<double> temp2(size);

  printf("\nEnergy minimization started.\n");

  for (int step = 0; step < number_of_steps_; ++step) {
    force.compute(
      box, position_per_atom, type, group, potential_per_atom, force_per_atom, virial_per_atom);
    calculate_force_square_max(force_per_atom);
    const double force_max = sqrt(cpu_force_square_max_[0]);
    calculate_total_potential(potential_per_atom);

    if (step % base == 0 || force_max < force_tolerance_) {
      printf(
        "    step %d: total_potential = %.10f eV, f_max = %.10f eV/A.\n",
        step,
        cpu_total_potential_[0],
        force_max);
      if (force_max < force_tolerance_)
        break;
    }

    P = dot(v, force_per_atom);

    if (P > 0) {
      if (N_neg > N_min) {
        next_dt = dt * f_inc;
        if (next_dt < dt_max)
          dt = next_dt;
        alpha *= f_alpha;
      }
      N_neg++;
    } else {
      next_dt = dt * f_dec;
      if (next_dt > dt_min)
        dt = next_dt;
      alpha = alpha_start;
      // move position back
      scalar_multiply(-0.5 * dt, v, temp1);
      CHECK(cudaDeviceSynchronize());
      vector_sum(position_per_atom, temp1, position_per_atom);
      CHECK(cudaDeviceSynchronize());
      v.fill(0);
      N_neg = 0;
    }

    // md step
    // implicit Euler integration
    double F_modulus = sqrt(dot(force_per_atom, force_per_atom));
    double v_modulus = sqrt(dot(v, v));
    // dv = F/m*dt
    scalar_multiply(dt / m, force_per_atom, temp2);
    CHECK(cudaDeviceSynchronize());
    vector_sum(v, temp2, v);
    CHECK(cudaDeviceSynchronize());
    scalar_multiply(1 - alpha, v, temp1);
    scalar_multiply(alpha * v_modulus / F_modulus, force_per_atom, temp2);
    CHECK(cudaDeviceSynchronize());
    vector_sum(temp1, temp2, v);
    CHECK(cudaDeviceSynchronize());
    // dx = v*dt
    scalar_multiply(dt, v, temp1);
    CHECK(cudaDeviceSynchronize());
    vector_sum(position_per_atom, temp1, position_per_atom);
    CHECK(cudaDeviceSynchronize());
  }

  printf("Energy minimization finished.\n");
}