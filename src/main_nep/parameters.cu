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
#include <cmath>

Parameters::Parameters(char* input_dir)
{
  print_line_1();
  printf("Started reading nep.in.\n");
  print_line_2();

  char file[200];
  strcpy(file, input_dir);
  strcat(file, "/nep.in");
  FILE* fid = my_fopen(file, "r");
  char name[20];

  int count = fscanf(fid, "%s%f%f", name, &rc_radial, &rc_angular);
  PRINT_SCANF_ERROR(count, 3, "reading error for cutoff.");
  printf("radial cutoff = %g A.\n", rc_radial);
  printf("angular cutoff = %g A.\n", rc_angular);
  if (rc_angular > rc_radial) {
    PRINT_INPUT_ERROR("angular cutoff should <= radial cutoff.");
  }
  if (rc_angular < 1.0f) {
    PRINT_INPUT_ERROR("angular cutoff should >= 1 A.");
  }
  if (rc_radial > 10.0f) {
    PRINT_INPUT_ERROR("radial cutoff should <= 10 A.");
  }

  count = fscanf(fid, "%s%d%d", name, &n_max_radial, &n_max_angular);
  PRINT_SCANF_ERROR(count, 3, "reading error for n_max.");
  printf("n_max_radial = %d.\n", n_max_radial);
  printf("n_max_angular = %d.\n", n_max_angular);
  if (n_max_radial < 0) {
    PRINT_INPUT_ERROR("n_max_radial should >= 0.");
  } else if (n_max_radial > 12) {
    PRINT_INPUT_ERROR("n_max_radial should <= 12.");
  }
  if (n_max_angular < 0) {
    PRINT_INPUT_ERROR("n_max_angular should >= 0.");
  } else if (n_max_angular > 12) {
    PRINT_INPUT_ERROR("n_max_angular should <= 12.");
  }

  count = fscanf(fid, "%s%d", name, &L_max);
  PRINT_SCANF_ERROR(count, 2, "reading error for l_max.");
  printf("l_max = %d.\n", L_max);
  if (L_max < 0) {
    PRINT_INPUT_ERROR("l_max should >= 0.");
  } else if (L_max > 6) {
    PRINT_INPUT_ERROR("l_max should <= 6.");
  }

  int dim = (n_max_radial + 1) + (n_max_angular + 1) * L_max;

  count = fscanf(fid, "%s%d", name, &num_neurons1);
  PRINT_SCANF_ERROR(count, 2, "reading error for ANN.");
  if (num_neurons1 < 1) {
    PRINT_INPUT_ERROR("num_neurons1 should >= 1.");
  } else if (num_neurons1 > 100) {
    PRINT_INPUT_ERROR("num_neurons1 should <= 100.");
  }
  num_neurons2 = 0; // use a single hidden layer currently

  printf("ANN = %d-%d-1.\n", dim, num_neurons1);

  number_of_variables = (dim + 1) * num_neurons1;
  number_of_variables += (num_neurons1 + 1) * num_neurons2;
  number_of_variables += (num_neurons2 == 0 ? num_neurons1 : num_neurons2) + 1;
  printf("number of parameters to be optimized = %d.\n", number_of_variables);

  batch_size = 100000; // use a single batch currently

  count = fscanf(fid, "%s%d", name, &population_size);
  PRINT_SCANF_ERROR(count, 2, "reading error for population_size.");
  printf("population_size = %d.\n", population_size);
  if (population_size < 10) {
    PRINT_INPUT_ERROR("population_size should >= 10.");
  } else if (population_size > 100) {
    PRINT_INPUT_ERROR("population_size should <= 100.");
  }

  count = fscanf(fid, "%s%d", name, &maximum_generation);
  PRINT_SCANF_ERROR(count, 2, "reading error for maximum_generation.");
  printf("maximum_generation = %d.\n", maximum_generation);
  if (maximum_generation < 100) {
    PRINT_INPUT_ERROR("maximum_generation should >= 100.");
  } else if (maximum_generation > 1000000) {
    PRINT_INPUT_ERROR("maximum_generation should <= 1000000.");
  }

  fclose(fid);
}
