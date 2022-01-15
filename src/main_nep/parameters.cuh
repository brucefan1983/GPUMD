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
#include "utilities/gpu_vector.cuh"
#include <string>
#include <vector>

#define MAX_NUM_TYPES 10

class Parameters
{
public:
  Parameters(char* input_dir);

  // parameters to be read in
  int batch_size;         // number of configurations in one batch
  int num_types;          // number of atom types
  int population_size;    // population size for SNES
  int maximum_generation; // maximum number of generations for SNES;
  int num_neurons1;       // number of nuerons in the 1st hidden layer (only one hidden layer)
  int n_max_radial;       // maximum order of the radial Chebyshev polynomials
  int n_max_angular;      // maximum order of the angular Chebyshev polynomials
  int L_max;              // maximum order of the angular Legendre polynomials
  float rc_radial;        // radial cutoff distance
  float rc_angular;       // angular cutoff distance
  float lambda_1;         // weight parameter for L1 regularization loss
  float lambda_2;         // weight parameter for L2 regularization loss
  float lambda_e;         // weight parameter for energy RMSE loss
  float lambda_f;         // weight parameter for force RMSE loss
  float lambda_v;         // weight parameter for virial RMSE loss
  float force_delta;      // a parameters used to modify the force loss

  // check if a parameter has been set:
  bool is_type_set;
  bool is_cutoff_set;
  bool is_n_max_set;
  bool is_l_max_set;
  bool is_neuron_set;
  bool is_lambda_1_set;
  bool is_lambda_2_set;
  bool is_lambda_e_set;
  bool is_lambda_f_set;
  bool is_lambda_v_set;
  bool is_batch_set;
  bool is_population_set;
  bool is_generation_set;
  bool is_type_weight_set;
  bool is_force_delta_set;

  // other parameters
  int dim;                            // dimension of the descriptor vector
  int dim_radial;                     // number of radial descriptor components
  int dim_angular;                    // number of angular descriptor components
  int number_of_variables;            // total number of parameters (NN and descriptor)
  int number_of_variables_ann;        // number of parameters in the NN only
  int number_of_variables_descriptor; // number of parameters in the descriptor only

  // some arrays
  GPU_Vector<float> type_weight_gpu;  // relative force weight for different atom types (GPU)
  std::vector<float> type_weight_cpu; // relative force weight for different atom types (CPU)
  GPU_Vector<float> q_scaler_gpu;     // used to scale some descriptor components (GPU)
  std::vector<float> q_scaler_cpu;    // used to scale some descriptor components (CPU)
  std::vector<std::string> elements;  // atom symbols

private:
  void set_default_parameters();
  void read_nep_in(char* input_dir);
  void calculate_parameters();
  void report_inputs();

  void parse_one_keyword(char** param, int num_param);
  void parse_type(char** param, int num_param);
  void parse_type_weight(char** param, int num_param);
  void parse_cutoff(char** param, int num_param);
  void parse_n_max(char** param, int num_param);
  void parse_l_max(char** param, int num_param);
  void parse_neuron(char** param, int num_param);
  void parse_lambda_1(char** param, int num_param);
  void parse_lambda_2(char** param, int num_param);
  void parse_lambda_e(char** param, int num_param);
  void parse_lambda_f(char** param, int num_param);
  void parse_lambda_v(char** param, int num_param);
  void parse_force_delta(char** param, int num_param);
  void parse_batch(char** param, int num_param);
  void parse_population(char** param, int num_param);
  void parse_generation(char** param, int num_param);
};
