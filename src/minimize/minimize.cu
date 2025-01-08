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
The driver class for minimizers.
------------------------------------------------------------------------------*/

#include "force/force.cuh"
#include "minimize.cuh"
#include "minimizer_fire.cuh"
#include "minimizer_fire_box_change.cuh"
#include "minimizer_sd.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <cstring>
#include <memory>

void Minimize::parse_minimize(
  const char** param,
  int num_param,
  Force& force,
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& type,
  std::vector<Group>& group,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{

  int minimizer_type = 0;
  int number_of_steps = 0;
  double force_tolerance = 0.0;
  int box_change = 0;
  int hydrostatic_strain = 0;
  std::unique_ptr<Minimizer> minimizer;
  const int number_of_atoms = type.size();

  if (strcmp(param[1], "sd") == 0) {
    minimizer_type = 0;

    if (num_param != 4) {
      PRINT_INPUT_ERROR("minimize sd should have 2 parameters.");
    }

    if (!is_valid_real(param[2], &force_tolerance)) {
      PRINT_INPUT_ERROR("Force tolerance should be a number.");
    }

    if (!is_valid_int(param[3], &number_of_steps)) {
      PRINT_INPUT_ERROR("Number of steps should be an integer.");
    }
    if (number_of_steps <= 0) {
      PRINT_INPUT_ERROR("Number of steps should > 0.");
    }
  } else if (strcmp(param[1], "fire") == 0) {
    minimizer_type = 1;

    if (!((num_param >= 4) && (num_param <= 6))) {
      PRINT_INPUT_ERROR("minimize fire should have 2 to 4 parameters.");
    }

    if (!is_valid_real(param[2], &force_tolerance)) {
      PRINT_INPUT_ERROR("Force tolerance should be a number.");
    }

    if (!is_valid_int(param[3], &number_of_steps)) {
      PRINT_INPUT_ERROR("Number of steps should be an integer.");
    }
    if (number_of_steps <= 0) {
      PRINT_INPUT_ERROR("Number of steps should > 0.");
    }

    if (num_param >= 5) {
      if (!is_valid_int(param[4], &box_change)) {
        PRINT_INPUT_ERROR("Box_change should be an integer.");
      }
      if (!(box_change == 0 || box_change == 1)) {
        PRINT_INPUT_ERROR("Box_change should be 1 or 0.");
      }

      if (box_change == 1) {
        minimizer_type = 2;
      }
    }

    if (num_param >= 6) {
      if (!is_valid_int(param[5], &hydrostatic_strain)) {
        PRINT_INPUT_ERROR("Hydrostatic_strain should be an integer.");
      }
      if (!(hydrostatic_strain == 0 || hydrostatic_strain == 1)) {
        PRINT_INPUT_ERROR("Hydrostatic_strain should be 1 or 0.");
      }
    }
  } else {
    PRINT_INPUT_ERROR("Invalid minimizer.");
  }

  switch (minimizer_type) {
    case 0:
      printf("\nStart to do an energy minimization.\n");
      printf("    using the steepest descent method.\n");
      printf("    with fixed box.\n");
      printf("    with a force tolerance of %g eV/A.\n", force_tolerance);
      printf("    for maximally %d steps.\n", number_of_steps);

      minimizer.reset(new Minimizer_SD(number_of_atoms, number_of_steps, force_tolerance));

      minimizer->compute(
        force,
        box,
        position_per_atom,
        type,
        group,
        potential_per_atom,
        force_per_atom,
        virial_per_atom);

      break;
    case 1:
      printf("\nStart to do an energy minimization.\n");
      printf("    using the fast inertial relaxation engine (FIRE) method.\n");
      printf("    with fixed box.\n");
      printf("    with a force tolerance of %g eV/A.\n", force_tolerance);
      printf("    for maximally %d steps.\n", number_of_steps);

      minimizer.reset(new Minimizer_FIRE(number_of_atoms, number_of_steps, force_tolerance));

      minimizer->compute(
        force,
        box,
        position_per_atom,
        type,
        group,
        potential_per_atom,
        force_per_atom,
        virial_per_atom);

      break;
    case 2:
      printf("\nStart to do an energy minimization.\n");
      printf("    using the fast inertial relaxation engine (FIRE) method.\n");
      printf("    with variable box.\n");
      if (hydrostatic_strain == 1) {
        printf("    with hydrostatic pressure.\n");
      }
      printf("    with a force tolerance of %g eV/A.\n", force_tolerance);
      printf("    for maximally %d steps.\n", number_of_steps);

      minimizer.reset(new Minimizer_FIRE_Box_Change(
        number_of_atoms, number_of_steps, force_tolerance, hydrostatic_strain));

      minimizer->compute(
        force,
        box,
        position_per_atom,
        type,
        group,
        potential_per_atom,
        force_per_atom,
        virial_per_atom);
      break;
    default:
      PRINT_INPUT_ERROR("Invalid minimizer.");
      break;
  }
}
