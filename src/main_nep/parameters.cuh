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

  int population_size = 80;        // population size for SNES
  int maximum_generation = 500000; // maximum number of generations for SNES;
  int num_neurons = 40;            // number of nuerons per layer
  float rc = 5.0f;                 // cutoff distance
  int n_max = 8;                   // maximum order of the radial Chebyshev polynomials
  int L_max = 8;                   // maximum order of the angular Legendre polynomials
  int number_of_variables = 0;     // total number of parameters
  float weight_force = 0.8f;       // good default
  float weight_energy = 0.1f;      // good default
  float weight_stress = 0.1f;      // good default
  float L1_reg_para = 5.0e-3f;     // good default
  float L2_reg_para = 5.0e-3f;     // good default
};
