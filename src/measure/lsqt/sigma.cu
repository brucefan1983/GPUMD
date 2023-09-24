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

#include "hamiltonian.cuh"
#include "model.cuh"
#include "sigma.cuh"
#include "vector.cuh"
#include <fstream>
#include <iostream>
#define BLOCK_SIZE 512 // optimized
#define PI 3.141592653589793
//#define LORENTZ // Lorentz damping is not as good as Jackson damping

// Find the Chebyshev moments defined in Eqs. (32-34)
// in [Comput. Phys. Commun.185, 28 (2014)].
// See Algorithm 5 in [Comput. Phys. Commun.185, 28 (2014)].
void find_moments_chebyshev(
  Model& model, Hamiltonian& H, Vector& state_left, Vector& state_right, Vector& output)
{
  int n = model.number_of_atoms;
  int grid_size = (n - 1) / BLOCK_SIZE + 1;

  Vector state_0(state_right), state_1(n), state_2(n);
  Vector inner_product_1(grid_size * model.number_of_moments);

  // <left|right>
  int offset = 0 * grid_size;
  state_0.inner_product_1(n, state_left, inner_product_1, offset);

  // <left|H|right>
  H.apply(state_0, state_1);
  offset = 1 * grid_size;
  state_1.inner_product_1(n, state_left, inner_product_1, offset);

  // <left|T_m(H)|right> (m >= 2)
  for (int m = 2; m < model.number_of_moments; ++m) {
    H.kernel_polynomial(state_0, state_1, state_2);
    offset = m * grid_size;
    state_2.inner_product_1(n, state_left, inner_product_1, offset);
    // permute the pointers; do not need to copy the data
    state_0.swap(state_1);
    state_1.swap(state_2);
  }
  inner_product_1.inner_product_2(n, model.number_of_moments, output);
}

// Jackson damping in Eq. (35) of [Comput. Phys. Commun.185, 28 (2014)].
#ifdef LORENTZ
void apply_damping(Model& model, real* inner_product_real, real* inner_product_imag)
{
  real lambda = 4.0;
  real f1 = sinh(lambda);
  for (int k = 0; k < model.number_of_moments; ++k) {
    real f2 = sinh(lambda * (1.0 - k / model.number_of_moments)) / f1;
    inner_product_real[k] *= f2;
    inner_product_imag[k] *= f2;
  }
}
#else
void apply_damping(Model& model, real* inner_product_real, real* inner_product_imag)
{
  for (int k = 0; k < model.number_of_moments; ++k) {
    real factor = 1.0 / (model.number_of_moments + 1);
    real damping =
      (1 - k * factor) * cos(k * PI * factor) + sin(k * PI * factor) * factor / tan(PI * factor);
    inner_product_real[k] *= damping;
    inner_product_imag[k] *= damping;
  }
}
#endif

// Do the summation in Eqs. (29-31) in [Comput. Phys. Commun.185, 28 (2014)]
void perform_chebyshev_summation(
  Model& model, real* inner_product_real, real* inner_product_imag, real* correlation_function)
{
  for (int step1 = 0; step1 < model.number_of_energy_points; ++step1) {
    real energy_scaled = model.energy[step1] / model.energy_max;
    real chebyshev_0 = 1.0;
    real chebyshev_1 = energy_scaled;
    real chebyshev_2;
    real temp = inner_product_real[1] * chebyshev_1;
    for (int step2 = 2; step2 < model.number_of_moments; ++step2) {
      chebyshev_2 = 2.0 * energy_scaled * chebyshev_1 - chebyshev_0;
      chebyshev_0 = chebyshev_1;
      chebyshev_1 = chebyshev_2;
      temp += inner_product_real[step2] * chebyshev_2;
    }
    temp *= 2.0;
    temp += inner_product_real[0];
    temp *= 2.0 / (PI * model.volume);
    temp /= sqrt(1.0 - energy_scaled * energy_scaled);
    correlation_function[step1] = temp / model.energy_max;
  }
}

// Calculate:
// U(+t) |state> when direction = +1;
// U(-t) |state> when direction = -1.
// See Eq. (36) and Algorithm 6 in [Comput. Phys. Commun.185, 28 (2014)].
void evolve(Model& model, int direction, real time_step_scaled, Hamiltonian& H, Vector& state_in)
{
  int n = model.number_of_atoms;
  Vector state_0(state_in), state_1(n), state_2(n);

  // T_1(H) |psi> = H |psi>
  H.apply(state_in, state_1);

  // |final_state> = c_0 * T_0(H) |psi> + c_1 * T_1(H) |psi>
  real bessel_0 = j0(static_cast<double>(time_step_scaled));
  real bessel_1 = 2.0 * j1(static_cast<double>(time_step_scaled));
  H.chebyshev_01(state_0, state_1, state_in, bessel_0, bessel_1, direction);

  for (int m = 2; m < 1000000; ++m) {
    real bessel_m = jn(m, static_cast<double>(time_step_scaled));
    if (bessel_m < 1.0e-15 && bessel_m > -1.0e-15) {
      break;
    }
    bessel_m *= 2.0;
    int label;
    int m_mod_4 = m % 4;
    if (m_mod_4 == 0) {
      label = 1;
    } else if (m_mod_4 == 2) {
      label = 2;
    } else if ((m_mod_4 == 1 && direction == 1) || (m_mod_4 == 3 && direction == -1)) {
      label = 3;
    } else {
      label = 4;
    }
    H.chebyshev_2(state_0, state_1, state_2, state_in, bessel_m, label);
    // permute the pointers; do not need to copy the data
    state_0.swap(state_1);
    state_1.swap(state_2);
  }
}

// Calculate:
// [X, U(+t)] |state> when direction = +1;
// [U(-t), X] |state> when direction = -1.
// See Eq. (37) and Algorithm 7 in [Comput. Phys. Commun.185, 28 (2014)].
void evolvex(Model& model, int direction, real time_step_scaled, Hamiltonian& H, Vector& state_in)
{
  int n = model.number_of_atoms;
  Vector state_0(state_in), state_0x(n);
  Vector state_1(n), state_1x(n);
  Vector state_2(n), state_2x(n);

  // T_1(H) |psi> = H |psi>
  H.apply(state_in, state_1);

  // [X, T_1(H)] |psi> = J |psi>
  H.apply_commutator(state_in, state_1x);

  // |final_state> = c_1 * [X, T_1(H)] |psi>
  real bessel_1 = 2.0 * j1(static_cast<double>(time_step_scaled));
  H.chebyshev_1x(state_1x, state_in, bessel_1);

  for (int m = 2; m <= 1000000; ++m) {
    real bessel_m = jn(m, static_cast<double>(time_step_scaled));
    if (bessel_m < 1.0e-15 && bessel_m > -1.0e-15) {
      break;
    }
    bessel_m *= 2.0;
    int label;
    int m_mod_4 = m % 4;
    if (m_mod_4 == 1) {
      label = 3;
    } else if (m_mod_4 == 3) {
      label = 4;
    } else if ((m_mod_4 == 0 && direction == 1) || (m_mod_4 == 2 && direction == -1)) {
      label = 1;
    } else {
      label = 2;
    }
    H.chebyshev_2x(
      state_0, state_0x, state_1, state_1x, state_2, state_2x, state_in, bessel_m, label);

    // Permute the pointers; do not need to copy the data
    state_0.swap(state_1);
    state_1.swap(state_2);
    state_0x.swap(state_1x);
    state_1x.swap(state_2x);
  }
}

// calculate the DOS as a function of Fermi energy
// See Algorithm 1 in [Comput. Phys. Commun.185, 28 (2014)].
void find_dos(Model& model, Hamiltonian& H, Vector& random_state, int flag)
{
  Vector inner_product_2(model.number_of_moments);

  real* dos;
  real* inner_product_real;
  real* inner_product_imag;

  dos = new real[model.number_of_energy_points];
  inner_product_real = new real[model.number_of_moments];
  inner_product_imag = new real[model.number_of_moments];

  find_moments_chebyshev(model, H, random_state, random_state, inner_product_2);
  inner_product_2.copy_to_host(inner_product_real, inner_product_imag);

  apply_damping(model, inner_product_real, inner_product_imag);
  perform_chebyshev_summation(model, inner_product_real, inner_product_imag, dos);

  std::string filename = (flag == 0) ? "/dos.out" : "/ldos.out";
  std::ofstream output(model.input_dir + filename, std::ios::app);

  if (!output.is_open()) {
    std::cout << "Error: cannot open " + model.input_dir + filename << std::endl;
    exit(1);
  }

  for (int n = 0; n < model.number_of_energy_points; ++n) {
    output << dos[n] << " ";
  }
  output << std::endl;
  output.close();

  delete[] inner_product_real;
  delete[] inner_product_imag;
  delete[] dos;
}

// calculate the group velocity, which is sqrt{VAC(t=0)}
// as a function of Fermi energy
void find_vac0(Model& model, Hamiltonian& H, Vector& random_state)
{
  Vector inner_product_2(model.number_of_moments);
  real* inner_product_real;
  real* inner_product_imag;
  real* vac0;
  inner_product_real = new real[model.number_of_moments];
  inner_product_imag = new real[model.number_of_moments];
  vac0 = new real[model.number_of_energy_points];

  Vector state(model.number_of_atoms);
  H.apply_current(random_state, state);
  find_moments_chebyshev(model, H, state, state, inner_product_2);
  inner_product_2.copy_to_host(inner_product_real, inner_product_imag);
  apply_damping(model, inner_product_real, inner_product_imag);
  perform_chebyshev_summation(model, inner_product_real, inner_product_imag, vac0);

  std::ofstream output(model.input_dir + "/vac0.out", std::ios::app);
  if (!output.is_open()) {
    std::cout << "Error: cannot open " + model.input_dir + "/vac0.out" << std::endl;
    exit(1);
  }
  for (int n = 0; n < model.number_of_energy_points; ++n) {
    output << vac0[n] << " ";
  }
  output << std::endl;
  output.close();

  delete[] inner_product_real;
  delete[] inner_product_imag;
  delete[] vac0;
}

// calculate the VAC as a function of correlation time and Fermi energy
// See Algorithm 2 in [Comput. Phys. Commun.185, 28 (2014)].
void find_vac(Model& model, Hamiltonian& H, Vector& random_state)
{
  Vector state_left(random_state);
  Vector state_left_copy(model.number_of_atoms);
  Vector state_right(random_state);
  Vector inner_product_2(model.number_of_moments);

  real* inner_product_real;
  real* inner_product_imag;
  real* vac;

  vac = new real[model.number_of_energy_points];
  inner_product_real = new real[model.number_of_moments];
  inner_product_imag = new real[model.number_of_moments];

  H.apply_current(state_left, state_right);

  std::ofstream output(model.input_dir + "/vac.out", std::ios::app);
  if (!output.is_open()) {
    std::cout << "Error: cannot open " + model.input_dir + "/vac.out" << std::endl;
    exit(1);
  }

  for (int m = 0; m < model.number_of_steps_correlation; ++m) {
    std::cout << "- calculating VAC step " << m << std::endl;
    H.apply_current(state_left, state_left_copy);
    find_moments_chebyshev(model, H, state_right, state_left_copy, inner_product_2);
    inner_product_2.copy_to_host(inner_product_real, inner_product_imag);

    apply_damping(model, inner_product_real, inner_product_imag);
    perform_chebyshev_summation(model, inner_product_real, inner_product_imag, vac);

    for (int n = 0; n < model.number_of_energy_points; ++n) {
      output << vac[n] << " ";
    }
    output << std::endl;

    if (m < model.number_of_steps_correlation - 1) {
      real time_step_scaled = model.time_step[m] * model.energy_max;
      evolve(model, -1, time_step_scaled, H, state_left);
      evolve(model, -1, time_step_scaled, H, state_right);
    }
  }

  output.close();

  delete[] inner_product_real;
  delete[] inner_product_imag;
  delete[] vac;
}

// calculate the MSD as a function of correlation time and Fermi energy
// See Algorithm 3 in [Comput. Phys. Commun.185, 28 (2014)].
void find_msd(Model& model, Hamiltonian& H, Vector& random_state)
{
  Vector state(random_state);
  Vector state_x(random_state);
  Vector state_copy(model.number_of_atoms);
  Vector inner_product_2(model.number_of_moments);

  real* inner_product_real;
  real* inner_product_imag;
  real* msd;

  msd = new real[model.number_of_energy_points];
  inner_product_real = new real[model.number_of_moments];
  inner_product_imag = new real[model.number_of_moments];

  real time_step_scaled = model.time_step[0] * model.energy_max;
  evolve(model, 1, time_step_scaled, H, state);
  evolvex(model, 1, time_step_scaled, H, state_x);

  std::ofstream output(model.input_dir + "/msd.out", std::ios::app);
  if (!output.is_open()) {
    std::cout << "Error: cannot open " + model.input_dir + "/msd.out" << std::endl;
    exit(1);
  }

  for (int m = 0; m < model.number_of_steps_correlation; ++m) {
    std::cout << "- calculating MSD step " << m << std::endl;

    find_moments_chebyshev(model, H, state_x, state_x, inner_product_2);
    inner_product_2.copy_to_host(inner_product_real, inner_product_imag);

    apply_damping(model, inner_product_real, inner_product_imag);
    perform_chebyshev_summation(model, inner_product_real, inner_product_imag, msd);

    for (int n = 0; n < model.number_of_energy_points; ++n) {
      output << msd[n] << " ";
    }
    output << std::endl;

    if (m < model.number_of_steps_correlation - 1) {
      time_step_scaled = model.time_step[m + 1] * model.energy_max;

      // update [X, U^m] |phi> to [X, U^(m+1)] |phi>
      state_copy.copy(state);

      evolvex(model, 1, time_step_scaled, H, state_copy);
      evolve(model, 1, time_step_scaled, H, state_x);

      state_x.add(state_copy);

      // update U^m |phi> to U^(m+1) |phi>
      evolve(model, 1, time_step_scaled, H, state);
    }
  }

  output.close();

  delete[] inner_product_real;
  delete[] inner_product_imag;
  delete[] msd;
}

// calculate the spin polarization as a function of correlation time and
// Fermi energy. See Eq. (6) in [Phys. Rev. B 95, 041401(R) (2017)].
void find_spin_polarization(Model& model, Hamiltonian& H, Vector& random_state)
{
  Vector state(random_state);
  Vector state_sz(model.number_of_atoms);
  Vector inner_product_2(model.number_of_moments);

  real* inner_product_real;
  real* inner_product_imag;
  real* S;

  S = new real[model.number_of_energy_points];
  inner_product_real = new real[model.number_of_moments];
  inner_product_imag = new real[model.number_of_moments];

  std::ofstream output(model.input_dir + "/S.out", std::ios::app);
  if (!output.is_open()) {
    std::cout << "Error: cannot open " + model.input_dir + "/S.out" << std::endl;
    exit(1);
  }

  for (int m = 0; m < model.number_of_steps_correlation; ++m) {
    std::cout << "- calculating spin step " << m << std::endl;

    state_sz.apply_sz(state);

    find_moments_chebyshev(model, H, state, state_sz, inner_product_2);
    inner_product_2.copy_to_host(inner_product_real, inner_product_imag);
    apply_damping(model, inner_product_real, inner_product_imag);
    perform_chebyshev_summation(model, inner_product_real, inner_product_imag, S);

    for (int n = 0; n < model.number_of_energy_points; ++n) {
      output << S[n] << " ";
    }
    output << std::endl;

    if (m < model.number_of_steps_correlation - 1) {
      // update U^m |phi> to U^(m+1) |phi>
      real time_step_scaled = model.time_step[m] * model.energy_max;
      evolve(model, 1, time_step_scaled, H, state);
    }
  }

  output.close();

  delete[] inner_product_real;
  delete[] inner_product_imag;
  delete[] S;
}
