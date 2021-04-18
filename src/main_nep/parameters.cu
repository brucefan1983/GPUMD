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

#include "parameters.cuh"
#include "utilities/error.cuh"

Parameters::Parameters(char* input_dir)
{
  print_line_1();
  printf("Started reading potential.in.\n");
  print_line_2();

  char file[200];
  strcpy(file, input_dir);
  strcat(file, "/potential.in");
  FILE* fid = my_fopen(file, "r");
  char name[20];
  int count = fscanf(fid, "%s%d%f", name, &num_neurons_2b, &rc_2b);
  PRINT_SCANF_ERROR(count, 3, "reading error for potential.in.");
  printf("two_body: number of neurons = %d, cutoff = %g A.\n", num_neurons_2b, rc_2b);
  count = fscanf(fid, "%s%d%f", name, &num_neurons_3b, &rc_3b);
  PRINT_SCANF_ERROR(count, 3, "reading error for potential.in.");
  printf("three_body: number of neurons = %d, cutoff = %g A.\n", num_neurons_3b, rc_3b);
  count = fscanf(fid, "%s%d%d%d", name, &num_neurons_mb, &n_max, &L_max);
  PRINT_SCANF_ERROR(count, 4, "reading error for potential.in.");
  printf("many_body: %d neurons, n_max = %d, l_max = %d.\n", num_neurons_mb, n_max, L_max);

  int number_of_variables_2b = 0;
  number_of_variables = 0;
  if (num_neurons_2b > 0) {
    number_of_variables_2b = num_neurons_2b * (num_neurons_2b + 3 + 1) + 1;
    number_of_variables += number_of_variables_2b;
  }
  printf("number of parameters to be optimized for 2-body part = %d.\n", number_of_variables_2b);

  int number_of_variables_3b = 0;
  if (num_neurons_3b > 0) {
    number_of_variables_3b = num_neurons_3b * (num_neurons_3b + 3 + 3) + 1;
    number_of_variables += number_of_variables_3b;
  }
  printf("number of parameters to be optimized for 3-body part = %d.\n", number_of_variables_3b);

  int number_of_variables_mb = 0;
  if (num_neurons_mb > 0) {
    number_of_variables_mb = num_neurons_mb * (num_neurons_mb + 3 + (n_max + 1) * (L_max + 1)) + 1;
    number_of_variables += number_of_variables_mb;
  }
  printf("number of parameters to be optimized for manybody part = %d.\n", number_of_variables_mb);
  printf("total number of parameters to be optimized = %d.\n", number_of_variables);

  count = fscanf(fid, "%s%f", name, &weight_force);
  PRINT_SCANF_ERROR(count, 2, "reading error for potential.in.");
  if (weight_force < 0) {
    PRINT_INPUT_ERROR("weight for force should >= 0.");
  }
  printf("weight for force is %g.\n", weight_force);

  count = fscanf(fid, "%s%f", name, &weight_energy);
  PRINT_SCANF_ERROR(count, 2, "reading error for potential.in.");
  if (weight_energy < 0) {
    PRINT_INPUT_ERROR("weight for energy should >= 0.");
  }
  printf("weight for energy is %g.\n", weight_energy);

  count = fscanf(fid, "%s%f", name, &weight_stress);
  PRINT_SCANF_ERROR(count, 2, "reading error for potential.in.");
  if (weight_stress < 0) {
    PRINT_INPUT_ERROR("weight for stress should >= 0.");
  }
  printf("weight for stress is %g.\n", weight_stress);

  count = fscanf(fid, "%s%d", name, &population_size);
  PRINT_SCANF_ERROR(count, 2, "reading error for potential.in.");
  if (population_size < 10) {
    PRINT_INPUT_ERROR("population_size should >= 10.");
  }
  printf("population_size is %d.\n", population_size);

  count = fscanf(fid, "%s%d", name, &maximum_generation);
  PRINT_SCANF_ERROR(count, 2, "reading error for potential.in.");
  if (maximum_generation < 1) {
    PRINT_INPUT_ERROR("maximum_generation should >= 1.");
  }
  printf("maximum_generation is %d.\n", maximum_generation);

  fclose(fid);
}
