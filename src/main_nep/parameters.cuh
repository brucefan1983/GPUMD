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

#pragma once

class Parameters
{
public:
  Parameters(char* input_dir);
  int test_set_size = 0;       // number of configurations in the test set
  int batch_size = 0;          // number of configurations in one batch
  int population_size = 0;     // population size for SNES
  int maximum_generation = 0;  // maximum number of generations for SNES;
  int num_neurons1 = 0;        // number of nuerons in the 1st hidden layer
  int num_neurons2 = 0;        // number of nuerons in the 2nd hidden layer
  float rc = 0.0f;             // cutoff distance
  int n_max = 0;               // maximum order of the radial Chebyshev polynomials
  int L_max = 0;               // maximum order of the angular Legendre polynomials
  int number_of_variables = 0; // total number of parameters
  float L1_reg_para = 2.0e-2f; // good default
  float L2_reg_para = 2.0e-2f; // good default
};
