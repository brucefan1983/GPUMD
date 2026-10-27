/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD and is distributed under LGPLv3 or later.

    Spin3 training shares its descriptor and derivative implementation with
    the production NEP_Spin runtime.
*/

#pragma once
#include "main_nep/common/nep_parameters.cuh"
#include "utilities/nep_spin_model_file.cuh"
class Spin_Parameters : public NEP_Energy_Parameters {
public:
  explicit Spin_Parameters(const ParameterInput& input);
  int spin_compress = 2, spin_order = 3, spin_soc = 1;
  int spin_basis_size[2] = {8, 0};
  int spin_l_max[3] = {2, 0, 0};
  float spin_cutoff = 8.0f;
  float lambda_m = 1.0f, lambda_tau = 0.0f, lambda_spin_response = 0.0f;
  float lambda_m_parallel = 0.0f, lambda_m_perp = 0.0f;
  bool split_mforce_loss = false;
  bool is_spin_type_set = false;
  bool spin_restart = false;
  int dim_struct = 0, dim_spin = 0;
  int number_of_variables_descriptor_spin = 0, number_of_variables_spin_projection = 0;
  std::vector<int> spin_type_active;
  std::vector<float> spin_cutoff_by_type;
  GPU_Vector<float> q_scaler_square_sum[16];
  GPU_Vector<unsigned long long> q_scaler_count[16];
private:
  bool parse_keyword(const char**, int) override;
  void report_model_inputs() const override;
  void read_zbl_in() override;
  std::vector<std::string> spin_type_names;
  void initialize_spin_checkpoint();
  void validate_spin_parameters();
  void calculate_spin_dimensions();
  bool is_lambda_m_set = false;
  bool is_lambda_m_parallel_set = false, is_lambda_m_perp_set = false;
  bool is_lambda_spin_response_set = false;
  bool is_lambda_tau_set = false;
  bool is_spin_basis_size_set = false;
  bool is_spin_compress_set = false;
  bool is_spin_cutoff_set = false;
  bool is_spin_l_max_set = false;
  bool is_spin_order_set = false;
  bool is_spin_soc_set = false;
  void parse_lambda_m(const char** param, int num_param);
  void parse_lambda_m_component(const char** param, int num_param, float&, bool&);
  void parse_lambda_tau(const char** param, int num_param);
  void parse_lambda_spin_response(const char** param, int num_param);
  void parse_spin_compress(const char** param, int num_param);
  void parse_spin_order(const char** param, int num_param);
  void parse_spin_soc(const char** param, int num_param);
  void parse_spin_basis_size(const char** param, int num_param);
  void parse_spin_l_max(const char** param, int num_param);
  void parse_spin_cutoff(const char** param, int num_param);
  void parse_spin_active_types(
  const char** param,
  int num_param,
  const char* keyword,
  std::vector<std::string>& names,
  bool& is_set);
};
