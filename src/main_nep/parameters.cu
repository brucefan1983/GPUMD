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

  int count = fscanf(fid, "%s%f", name, &rc);
  PRINT_SCANF_ERROR(count, 2, "reading error for potential.in.");
  printf("cutoff = %g A.\n", rc);

  count = fscanf(fid, "%s%d", name, &num_neurons1);
  PRINT_SCANF_ERROR(count, 2, "reading error for potential.in.");
  printf("num_neurons1 = %d.\n", num_neurons1);

  count = fscanf(fid, "%s%d", name, &num_neurons2);
  PRINT_SCANF_ERROR(count, 2, "reading error for potential.in.");
  printf("num_neurons2 = %d.\n", num_neurons2);

  count = fscanf(fid, "%s%d", name, &n_max);
  PRINT_SCANF_ERROR(count, 2, "reading error for potential.in.");
  printf("n_max = %d.\n", n_max);

  count = fscanf(fid, "%s%d", name, &L_max);
  PRINT_SCANF_ERROR(count, 2, "reading error for potential.in.");
  printf("l_max = %d.\n", L_max);

  int dim = (n_max + 1) * (L_max + 1);
  number_of_variables = (dim + 1) * num_neurons1;           // w0 and b0
  number_of_variables += (num_neurons1 + 1) * num_neurons2; // w1 and b1
  number_of_variables += num_neurons2 + 1;                  // w2 and b2
  printf("number of parameters to be optimized = %d.\n", number_of_variables);

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
