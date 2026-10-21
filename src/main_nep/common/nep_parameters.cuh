/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/

#pragma once
#include "main_nep/parameters.cuh"
#include "main_nep/parameter_input.cuh"
#include "utilities/gpu_vector.cuh"
#include <string>
#include <vector>

// Configuration shared by the NEP descriptors, ANN, data and file helpers.
class NEP_Parameters : public Parameters
{
public:
  NEP_Parameters();
  virtual ~NEP_Parameters() = default;
  void read(const ParameterInput& input);
  void finish_reading(const ParameterInput& input, const char* model_name,
    const char* model_description = nullptr);
  virtual bool parse_keyword(const char** param, int num_param);

  // parameters to be read in
  int version;            // NEP version; only NEP4 is supported
  int num_neurons1;       // number of neurons in the first hidden layer
  int num_neurons2;       // number of neurons in the second hidden layer; 0 if unused
  int num_hidden_layers;  // number of hidden layers
  int basis_size_radial;
  int basis_size_angular;
  int n_max_radial;       // maximum order of the radial Chebyshev polynomials
  int n_max_angular;      // maximum order of the angular Chebyshev polynomials
  int L_max;              // maximum order of the three-body spherical harmonics
  int has_q_222;          // has q_222
  int has_q_1111;         // has q_1111
  int has_q_112;          // has q_112
  int has_q_123;          // has q_123
  int has_q_233;          // has q_233
  int has_q_134;          // has q_134
  float lambda_e;         // weight parameter for energy RMSE loss
  float lambda_f;         // weight parameter for force RMSE loss
  float lambda_v;         // weight parameter for virial RMSE loss
  float lambda_shear;     // extra weight parameter for shear virial
  float force_delta;      // a parameter used to modify the force loss
  bool enable_zbl;        // true for including the universal ZBL potential
  bool flexible_zbl;      // true for including the flexible ZBL potential
  float zbl_rc_inner;     // inner cutoff for the universal ZBL potential
  float zbl_rc_outer;     // outer cutoff for the universal ZBL potential
  bool use_typewise_cutoff_zbl;
  float typewise_cutoff_zbl_factor;
  int output_descriptor;
  int fine_tune = 0; // fine_tune option; 0=no, 1=yes
  int fine_tune_descriptor = 1; // fine-tune descriptor; 0=no, 1=yes
  std::string fine_tune_nep_txt = "";
  std::string fine_tune_nep_restart = "";
  bool import_q_scaler = false; // read q_scaler from the local nep.txt instead of recomputing it
  bool nep_compile = true; // runtime-specialize NEP-family training kernels

  // check if a parameter has been set:
  bool is_version_set;
  bool is_type_set;
  bool is_cutoff_set;
  bool is_n_max_set;
  bool is_basis_size_set;
  bool is_l_max_set;
  bool is_neuron_set;
  bool is_lambda_e_set;
  bool is_lambda_f_set;
  bool is_lambda_v_set;
  bool is_lambda_shear_set;
  bool is_type_weight_set;
  bool is_force_delta_set;
  bool is_zbl_set;
  bool is_use_typewise_cutoff_zbl_set;

  // other parameters
  int dim;                            // dimension of the descriptor vector
  int dim_radial;                     // number of radial descriptor components
  int dim_angular;                    // number of angular descriptor components
  std::string file_tag;
  int ann_count;
  int global_parameters;
  int number_of_variables_ann;        // number of parameters in the ANN only
  int number_of_variables_ann_1;      // number of parameters in the ANN for one element
  int number_of_variables_descriptor; // number of parameters in the descriptor only
  int number_of_nep_txt_header_lines = 0; // header length of the nep.txt file that has been checked

  // some arrays

  std::vector<float> type_weight_cpu; // relative force weight for different atom types (CPU)
  std::vector<float> q_scaler_cpu;    // used to scale some descriptor components (CPU)
  std::vector<std::string> elements;  // atom symbols
  std::vector<int> atomic_numbers;    // atomic numbers
  std::vector<float> zbl_para;        // parameters of zbl potential
  std::vector<float> rc_radial;       // radial cutoff distance
  std::vector<float> rc_angular;      // angular cutoff distance
  float rc_radial_max = 0.0f;         // maximal radial cutoff
  float rc_angular_max = 0.0f;        // maximal angular cutoff
  bool has_multiple_cutoffs = false;

  GPU_Vector<float> q_scaler_gpu[16]; // used to scale some descriptor components (GPU)
  GPU_Vector<float> q_scaler_max[16]; // used to scale some descriptor components (GPU)
  GPU_Vector<float> q_scaler_min[16]; // used to scale some descriptor components (GPU)

protected:
  virtual void report_model_inputs() const {}

private:
  void read_zbl_in();
  void initialize_q_scaler();
  void report_inputs(const ParameterInput& input, const char* model_name,
    const char* model_description);
  void parse_nep_compile(const char** param, int num_param);
  void parse_version(const char** param, int num_param);
  void parse_type(const char** param, int num_param);
  void parse_type_weight(const char** param, int num_param);
  void parse_zbl(const char** param, int num_param);
  void parse_force_delta(const char** param, int num_param);
  void parse_cutoff(const char** param, int num_param);
  void parse_n_max(const char** param, int num_param);
  void parse_basis_size(const char** param, int num_param);
  void parse_l_max(const char** param, int num_param);
  void parse_neuron(const char** param, int num_param);
  void parse_lambda_e(const char** param, int num_param);
  void parse_lambda_f(const char** param, int num_param);
  void parse_lambda_v(const char** param, int num_param);
  void parse_lambda_shear(const char** param, int num_param);
  void parse_use_typewise_cutoff_zbl(const char** param, int num_param);
  void parse_output_descriptor(const char** param, int num_param);
  void parse_fine_tune(const char** param, int num_param);
  void parse_import_q_scaler(const char** param, int num_param);
};

// Energy-difference options are shared only by energy models.
class NEP_Energy_Parameters : public NEP_Parameters
{
public:
  float lambda_d = 0.0f;
  bool is_lambda_d_set = false;
  bool has_ediff_combinations = false;
  bool parse_keyword(const char** param, int num_param) override;

protected:
  void report_model_inputs() const override;

private:
  void parse_lambda_d(const char** param, int num_param);
};
