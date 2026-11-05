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
#include "main_nep/common/nep_charge_parameters.cuh"
#include "main_nep/common/nep_training.cuh"
#include "main_nep/common/nep_charge_dataset.cuh"
#include "main_nep/common/energy_difference.cuh"
#include "main_nep/common/nep_common.cuh"
#include "main_nep/common/nep_compiled_kernels.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_vector.cuh"
#include <memory>
class Dataset;

class NEP_Charge : public Potential
{
public:
  struct ParaMB {
    int charge_mode = 0;
    float rc_radial = 0.0f;     // radial cutoff
    float rc_angular = 0.0f;    // angular cutoff
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
    int num_para = 0;               // number of parameters
    const float* w0[NUM_ELEMENTS]; // weight from the input layer to the hidden layer
    const float* b0[NUM_ELEMENTS]; // bias for the hidden layer
    const float* w1[NUM_ELEMENTS]; // weight from the hidden layer to the output layer
    const float* sqrt_epsilon_inf; // sqrt(epsilon_inf) related to BEC
    const float* b1;               // bias for the output layer
    const float* c;                // for elements in descriptor
  };

  struct NEP_Charge_Data {
    GPU_Vector<float> descriptors;       // descriptors
    GPU_Vector<float> charge_derivative; // derivative of charge with respect to descriptor
    GPU_Vector<float> Fp;                // derivative of energy with respect to descriptor
    GPU_Vector<float> sum_fxyz;
    GPU_Vector<float> parameters; // parameters to be optimized
    GPU_Vector<float> kx;
    GPU_Vector<float> ky;
    GPU_Vector<float> kz;
    GPU_Vector<float> G;
    GPU_Vector<float> S_real;
    GPU_Vector<float> S_imag;
    GPU_Vector<float> D_real;
    GPU_Vector<int> kpoint_offset;
    const Dataset* kpoint_dataset = nullptr;
  };

  struct Charge_Para {
    float alpha = 0.5f; // 1 / (2 Angstrom)
    float alpha_factor = 1.0f; // 1 / (4 * alpha * alpha)
    float two_alpha_over_sqrt_pi = 0.564189583547756f;
  };

  explicit NEP_Charge(const ParameterInput& input);
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
  void configure(NEP_Charge_Parameters& para);
  void initialize(NEP_Charge_Parameters& para, int N, int Nc, int deviceCount);
  void find_force(
    NEP_Charge_Parameters& para,
    const float* parameters,
    std::vector<Dataset>& dataset,
    bool calculate_q_scaler,
    int deviceCount);
  std::unique_ptr<NEP_Charge_Parameters> model_parameters;
  std::unique_ptr<NEP_Training> training;
  NEP_Energy_Loss energy_loss;
  NEP_Batch_Data<NEP_Charge_Dataset> charge_data;
  EnergyDifference energy_difference;
  std::vector<float> fitness_charge;
  std::vector<float> fitness_bec;
  NEP_Neighbor neighbor[16];
  ParaMB paramb;
  ANN annmb[16];
  NEP_Charge_Data nep_data[16];
  NEP_ZBL zbl;
  Charge_Para charge_para;

  struct Compiled_Kernel
  {
    NEP_Compile library;
    NEP_Compiled_Kernels common;
    NEP_Compiled_BEC bec;
    using AnnChargeFunction =
      int (*)(int, const int*, const float*, const float*, const float*,
              double*, float*, float*, float*);
    AnnChargeFunction ann_charge_ = nullptr;
    bool valid = false;
    explicit Compiled_Kernel(const NEP_Compile_Config& config);
    bool is_valid() const { return valid; }
    void launch_ann_charge(
      int N,
      const int* type,
      const float* descriptors,
      const float* q_scaler,
      const float* parameters,
      double* pe,
      float* Fp,
      float* charge,
      float* charge_derivative);
    void launch_force_charge_radial(
      int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
      const float* x12, const float* y12, const float* z12,
      const float* parameters, const float* Fp,
      const float* charge_derivative, const float* D_real,
      float* fx, float* fy, float* fz, float* virial);
    void launch_force_charge_angular(
      int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
      const float* x12, const float* y12, const float* z12,
      const float* parameters, const float* Fp,
      const float* charge_derivative, const float* D_real,
      const float* sum_fxyz,
      float* fx, float* fy, float* fz, float* virial);
  };
  std::unique_ptr<Compiled_Kernel> compiled_kernel_;
  void initialize_model(const NEP_Charge_Parameters& para);
  void initialize_device(const NEP_Charge_Parameters& para, int N, int Nc, int device_id);
  void initialize_compiled_kernel(const NEP_Charge_Parameters& para);
  void bind_parameters(const float* parameters, ANN& ann);
  void prepare_kpoints(Dataset& dataset, int device_id);
  void compute_descriptors(
    const ANN& ann, const Dataset& data, const NEP_Neighbor& neighbors,
    NEP_Charge_Data& storage);
  void evaluate_ann(
    const ANN& ann, Dataset& data,
    NEP_Charge_Data& storage, const float* q_scaler);
  void compute_bec(
    const ANN& ann, Dataset& data, const NEP_Neighbor& neighbors,
    NEP_Charge_Data& storage);
  void compute_charge_long_range(
    Dataset& data, NEP_Charge_Data& storage);
  void compute_charge_real_space(
    Dataset& data, const NEP_Neighbor& neighbors,
    NEP_Charge_Data& storage);
  void compute_forces(
    const ANN& ann, Dataset& data, const NEP_Neighbor& neighbors,
    NEP_Charge_Data& storage);
};
