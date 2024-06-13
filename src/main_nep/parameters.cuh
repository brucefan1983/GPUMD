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

class Parameters
{
public:
  Parameters();

  // parameters to be read in
  int version;            // nep version, can be 2 or 3
  int batch_size;         // number of configurations in one batch
  int use_full_batch;     // 1 for effective full-batch even though batch_size is not full-batch
  int num_types;          // number of atom types
  int population_size;    // population size for SNES
  int maximum_generation; // maximum number of generations for SNES;
  int num_neurons1;       // number of nuerons in the 1st hidden layer (only one hidden layer)
  int basis_size_radial;  // for nep3
  int basis_size_angular; // for nep3
  int n_max_radial;       // maximum order of the radial Chebyshev polynomials
  int n_max_angular;      // maximum order of the angular Chebyshev polynomials
  int L_max;              // maximum order of the 3body spherical harmonics
  int L_max_4body;        // maximum order of the 4body spherical harmonics
  int L_max_5body;        // maximum order of the 5body spherical harmonics
  float rc_radial;        // radial cutoff distance
  float rc_angular;       // angular cutoff distance
  float lambda_1;         // weight parameter for L1 regularization loss
  float lambda_2;         // weight parameter for L2 regularization loss
  float lambda_e;         // weight parameter for energy RMSE loss
  float lambda_f;         // weight parameter for force RMSE loss
  float lambda_v;         // weight parameter for virial RMSE loss
  float lambda_shear;     // extra weight parameter for shear virial
  float force_delta;      // a parameters used to modify the force loss
  bool enable_zbl;        // true for inlcuding the universal ZBL potential
  bool flexible_zbl;      // true for inlcuding the flexible ZBL potential
  float zbl_rc_inner;     // inner cutoff for the universal ZBL potential
  float zbl_rc_outer;     // outer cutoff for the universal ZBL potential
  int train_mode; // 0=potential, 1=dipole, 2=polarizability, 3=temperature-dependent free energy
  int prediction; // 0=no, 1=yes
  float initial_para;

  // check if a parameter has been set:
  bool is_train_mode_set;
  bool is_prediction_set;
  bool is_version_set;
  bool is_type_set;
  bool is_cutoff_set;
  bool is_n_max_set;
  bool is_basis_size_set;
  bool is_l_max_set;
  bool is_neuron_set;
  bool is_lambda_1_set;
  bool is_lambda_2_set;
  bool is_lambda_e_set;
  bool is_lambda_f_set;
  bool is_lambda_v_set;
  bool is_lambda_shear_set;
  bool is_batch_set;
  bool is_population_set;
  bool is_generation_set;
  bool is_type_weight_set;
  bool is_force_delta_set;
  bool is_zbl_set;

  // other parameters
  int dim;                            // dimension of the descriptor vector
  int dim_radial;                     // number of radial descriptor components
  int dim_angular;                    // number of angular descriptor components
  int number_of_variables;            // total number of parameters (NN and descriptor)
  int number_of_variables_ann;        // number of parameters in the ANN only
  int number_of_variables_descriptor; // number of parameters in the descriptor only

  // some arrays

  std::vector<float> type_weight_cpu; // relative force weight for different atom types (CPU)
  std::vector<float> q_scaler_cpu;    // used to scale some descriptor components (CPU)
  std::vector<std::string> elements;  // atom symbols
  std::vector<int> atomic_numbers;    // atomic numbers
  std::vector<float> zbl_para;        // parameters of zbl potential

  GPU_Vector<float> q_scaler_gpu[16]; // used to scale some descriptor components (GPU)

private:
  void set_default_parameters();
  void read_nep_in();
  void read_zbl_in();
  void calculate_parameters();
  void report_inputs();

  void parse_one_keyword(std::vector<std::string>& tokens);

  void parse_mode(const char** param, int num_param);
  void parse_prediction(const char** param, int num_param);
  void parse_version(const char** param, int num_param);
  void parse_type(const char** param, int num_param);
  void parse_type_weight(const char** param, int num_param);
  void parse_zbl(const char** param, int num_param);
  void parse_cutoff(const char** param, int num_param);
  void parse_n_max(const char** param, int num_param);
  void parse_basis_size(const char** param, int num_param);
  void parse_l_max(const char** param, int num_param);
  void parse_neuron(const char** param, int num_param);
  void parse_lambda_1(const char** param, int num_param);
  void parse_lambda_2(const char** param, int num_param);
  void parse_lambda_e(const char** param, int num_param);
  void parse_lambda_f(const char** param, int num_param);
  void parse_lambda_v(const char** param, int num_param);
  void parse_lambda_shear(const char** param, int num_param);
  void parse_force_delta(const char** param, int num_param);
  void parse_batch(const char** param, int num_param);
  void parse_population(const char** param, int num_param);
  void parse_generation(const char** param, int num_param);
  void parse_initial_para(const char** param, int num_param);
};
