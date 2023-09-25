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
    The driver function of LSQT
------------------------------------------------------------------------------*/

#include "hamiltonian.cuh"
#include "lsqt.cuh"
#include "model.cuh"
#include "sigma.cuh"
#include "vector.cuh"
#include <iostream>

static void print_started_random_vector(int i)
{
  std::cout << std::endl;
  std::cout << "===========================================================";
  std::cout << std::endl;
  std::cout << "Started  simulation with random vector number " << i << std::endl;
  std::cout << std::endl;
}

static void print_finished_random_vector(int i)
{
  std::cout << std::endl;
  std::cout << "Finished simulation with random vector number " << i << std::endl;
  std::cout << "===========================================================";
  std::cout << std::endl << std::endl;
}

static void run_dos(Model& model, Hamiltonian& H, Vector& random_state)
{
  clock_t time_begin = clock();
  find_dos(model, H, random_state);
  clock_t time_finish = clock();
  real time_used = real(time_finish - time_begin) / CLOCKS_PER_SEC;
  std::cout << "- Time used for finding DOS = " << time_used << " s" << std::endl;
}

static void run_vac0(Model& model, Hamiltonian& H, Vector& random_state)
{
  clock_t time_begin = clock();
  find_vac0(model, H, random_state);
  clock_t time_finish = clock();
  real time_used = real(time_finish - time_begin) / CLOCKS_PER_SEC;
  std::cout << "- Time used for finding VAC0 = " << time_used << " s" << std::endl;
}

static void run_vac(Model& model, Hamiltonian& H, Vector& random_state)
{
  clock_t time_begin = clock();
  find_vac(model, H, random_state);
  clock_t time_finish = clock();
  real time_used = real(time_finish - time_begin) / CLOCKS_PER_SEC;
  std::cout << "- Time used for finding VAC = " << time_used << " s" << std::endl;
}

static void run_msd(Model& model, Hamiltonian& H, Vector& random_state)
{
  clock_t time_begin = clock();
  find_msd(model, H, random_state);
  clock_t time_finish = clock();
  real time_used = real(time_finish - time_begin) / CLOCKS_PER_SEC;
  std::cout << "- Time used for finding MSD = " << time_used << " s" << std::endl;
}

void LSQT::postprocess(std::string& input_directory)
{
  model.initialize();
  Hamiltonian H(model);
  Vector random_state(model.number_of_atoms);
  for (int i = 0; i < model.number_of_random_vectors; ++i) {
    print_started_random_vector(i);
    model.initialize_state(random_state);
    run_dos(model, H, random_state);
    run_vac0(model, H, random_state);
    run_vac(model, H, random_state);
    run_msd(model, H, random_state);
    print_finished_random_vector(i);
  }
}
