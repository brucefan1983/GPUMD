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
#include "main_nep/dataset.cuh"
#include "main_nep/potential.cuh"
#include "main_nep/common/nep_parameters.cuh"
#include "main_nep/common/nep_training.cuh"
#include "tnep_dataset.cuh"
#include "main_nep/common/nep_compiled_kernels.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_vector.cuh"
#include <memory>
class Dataset;

struct TNEP_Data {
  GPU_Vector<float> descriptors; // descriptors
  GPU_Vector<float> Fp;          // gradient of descriptors
  GPU_Vector<float> sum_fxyz;
  GPU_Vector<float> parameters; // parameters to be optimized
};

class TNEP_Parameters : public NEP_Parameters
{
public:
  int model_type = 1;
  int atomic_v = 0;
  bool is_atomic_v_set = false;
  bool parse_keyword(const char** param, int num_param) override;

protected:
  void report_model_inputs() const override;

private:
  void parse_atomic_v(const char** param, int num_param);
};

class TNEP : public Potential
{
public:
  struct ParaMB {
    float rc_radial[NUM_ELEMENTS];     // radial cutoff
    float rc_angular[NUM_ELEMENTS];    // angular cutoff
    int basis_size_radial = 0;
    int basis_size_angular = 0;
    int n_max_radial = 0;  // n_radial = 0, 1, 2, ..., n_max_radial
    int n_max_angular = 0; // n_angular = 0, 1, 2, ..., n_max_angular
    int L_max = 0;         // l = 1, 2, ..., L_max
    int dim_angular;
    int has_q_222;
    int has_q_1111;
    int has_q_112;
    int has_q_123;
    int has_q_233;
    int has_q_134;
    int num_L;
    int num_types = 0;
    int num_types_sq = 0;
    int num_c_radial = 0;
    int version = 4;
  };

  struct ANN {
    int dim = 0;                    // dimension of the descriptor
    int num_neurons1 = 0;           // number of neurons in the hidden layer
    int num_neurons2 = 0;           // number of neurons in the second hidden layer
    int num_hidden_layers = 0;      // number of hidden layers
    int num_para = 0;               // number of parameters
    int one_ann_no_bias = 0;        // number of parameters in the ANN without bias
    const float* wb[NUM_ELEMENTS];  // weight from the input layer to the hidden layer for dipole
    const float* b;                 // bias for the output layer for dipole
    // for the scalar part of polarizability
    const float* wb_pol[NUM_ELEMENTS]; // weight from the input layer to the hidden layer for polarizability
    const float* b_pol;                // bias for the output layer for polarizability
    // for elements in descriptor
    const float* c;
  };

  explicit TNEP(const ParameterInput& input);
  int prepare_data() override;
  void prepare_training() override;
  void evaluate(int batch, const float* population) override;
  void accumulate_fitness(float* fitness_total) override;
  void print_training_header() override;
  void report_error(const TrainingReport& report) override;
  void predict(float* parameters) override;
  Parameters& get_parameters() override { return *model_parameters; }

  void initialize_parameters(
    std::mt19937& rng,
    std::vector<float>& mu, std::vector<float>& sigma) override;
  void parameter_groups(std::vector<int>& groups) override;
  void read_model(float* parameters) override;

  void write_restart(
    const char* filename, const std::vector<float>& mu, const std::vector<float>& sigma) override;

private:
  void configure(TNEP_Parameters& para);
  void initialize(TNEP_Parameters& para, int N, int deviceCount);
  void find_force(
    TNEP_Parameters& para,
    const float* parameters,
    std::vector<Dataset>& dataset,
    bool calculate_q_scaler,
    int deviceCount);
  std::unique_ptr<TNEP_Parameters> model_parameters;
  std::unique_ptr<NEP_Training> training;
  std::vector<float> fitness_tensor;
  NEP_Batch_Data<TNEP_Atomic_Dataset> atomic_data;
  std::vector<float> get_rmse(Dataset& dataset, bool use_weight, int device_id);
  NEP_Neighbor neighbor[16];
  ParaMB paramb;
  ANN annmb[16];
  TNEP_Data nep_data[16];

  struct Compiled_Kernel
  {
    NEP_Compile library;
    NEP_Compiled_Kernels common;
    using AnnNepFunction =
      int (*)(int, const int*, const float*, const float*, const float*, double*, float*);
    using AnnTnepPolFunction =
      int (*)(int, const int*, const float*, const float*, const float*, float*, float*);
    AnnNepFunction ann_nep_ = nullptr;
    AnnTnepPolFunction ann_tnep_pol_ = nullptr;
    bool valid = false;
    explicit Compiled_Kernel(const NEP_Compile_Config& config);
    bool is_valid() const { return valid; }
    void launch_ann(
      int N,
      const int* type,
      const float* descriptors,
      const float* q_scaler,
      const float* parameters,
      double* pe,
      float* Fp);
    void launch_ann_tnep_pol(
      int N,
      const int* type,
      const float* descriptors,
      const float* q_scaler,
      const float* parameters,
      float* virial,
      float* Fp);
    void launch_force_tnep_radial(
      bool is_dipole,
      int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
      const float* x12, const float* y12, const float* z12,
      const float* parameters, const float* Fp,
      float* fx, float* fy, float* fz, float* virial);
    void launch_force_tnep_angular(
      bool is_dipole,
      int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
      const float* x12, const float* y12, const float* z12,
      const float* parameters, const float* Fp, const float* sum_fxyz,
      float* fx, float* fy, float* fz, float* virial);
  };
  std::unique_ptr<Compiled_Kernel> compiled_kernel_;
  void initialize_model(const TNEP_Parameters& para);
  void initialize_device(const TNEP_Parameters& para, int N, int device_id);
  void initialize_compiled_kernel(const TNEP_Parameters& para);
  void bind_parameters(const TNEP_Parameters& para, const float* parameters, ANN& ann);

  // Evaluation stages use the device selected by find_force().
  void compute_descriptors(
    const ANN& ann,
    const Dataset& dataset,
    const NEP_Neighbor& neighbor,
    TNEP_Data& data);
  void evaluate_ann(
    const ANN& ann,
    bool is_polarizability,
    Dataset& dataset,
    TNEP_Data& data,
    const float* q_scaler);
  void compute_tensor(
    const ANN& ann,
    bool is_dipole,
    Dataset& dataset,
    const NEP_Neighbor& neighbor,
    TNEP_Data& data);
};
