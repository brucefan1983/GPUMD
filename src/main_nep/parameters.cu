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

  int count = fscanf(fid, "%s%d", name, &num_types);
  PRINT_SCANF_ERROR(count, 2, "reading error for num_types.");
  printf("num_types = %d.\n", num_types);
  if (num_types < 1 || num_types > 10) {
    PRINT_INPUT_ERROR("num_types should >=1 and <= 10.");
  }

  count = fscanf(fid, "%s%f%f", name, &rc_radial, &rc_angular);
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
  } else if (n_max_radial > 19) {
    PRINT_INPUT_ERROR("n_max_radial should <= 19.");
  }
  if (n_max_angular < 0) {
    PRINT_INPUT_ERROR("n_max_angular should >= 0.");
  } else if (n_max_angular > 19) {
    PRINT_INPUT_ERROR("n_max_angular should <= 19.");
  }

  count = fscanf(fid, "%s%d", name, &L_max);
  PRINT_SCANF_ERROR(count, 2, "reading error for l_max.");
  printf("l_max = %d.\n", L_max);
  if (L_max != 4) {
    PRINT_INPUT_ERROR("l_max should = 4.");
  }

  int dim = (n_max_radial + 1) + (n_max_angular + 1) * L_max;
  q_scaler.resize(dim, Memory_Type::managed);
  q_min.resize(dim, Memory_Type::managed);

  count = fscanf(fid, "%s%d", name, &num_neurons1);
  PRINT_SCANF_ERROR(count, 2, "reading error for ANN.");
  if (num_neurons1 < 1) {
    PRINT_INPUT_ERROR("num_neurons1 should >= 1.");
  } else if (num_neurons1 > 100) {
    PRINT_INPUT_ERROR("num_neurons1 should <= 100.");
  }

  printf("ANN = %d-%d-1.\n", dim, num_neurons1);

  number_of_variables_ann = (dim + 2) * num_neurons1 + 1;
  printf("number of neural network parameters to be optimized = %d.\n", number_of_variables_ann);
  int num_para_descriptor =
    (num_types == 1) ? 0 : num_types * num_types * (n_max_radial + n_max_angular + 2);
  printf("number of descriptor parameters to be optimized = %d.\n", num_para_descriptor);
  number_of_variables = number_of_variables_ann + num_para_descriptor;
  printf("total number of parameters to be optimized = %d.\n", number_of_variables);

  count = fscanf(fid, "%s%f%f", name, &L1_reg_para, &L2_reg_para);
  PRINT_SCANF_ERROR(count, 3, "reading error for regularization.");
  printf("regularization = %g, %g.\n", L1_reg_para, L2_reg_para);
  if (L1_reg_para < 0.0f) {
    PRINT_INPUT_ERROR("L1 regularization >= 0.");
  }
  if (L2_reg_para < 0.0f) {
    PRINT_INPUT_ERROR("L2 regularization >= 0.");
  }

  count = fscanf(fid, "%s%d", name, &batch_size);
  PRINT_SCANF_ERROR(count, 2, "reading error for batch_size.");
  printf("batch_size = %d.\n", batch_size);
  if (batch_size < 1) {
    PRINT_INPUT_ERROR("batch_size should >= 1.");
  }

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
  if (maximum_generation < 0) {
    PRINT_INPUT_ERROR("maximum_generation should >= 0.");
  } else if (maximum_generation > 10000000) {
    PRINT_INPUT_ERROR("maximum_generation should <= 10000000.");
  }

  fclose(fid);
}
