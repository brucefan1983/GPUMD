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
#include "main_nep/common/energy_difference.cuh"
#include "main_nep/common/nep_common.cuh"
#include "main_nep/common/nep_compiled_kernels.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_vector.cuh"
#include <memory>
class Dataset;

struct NEP_VDW_Data {
  GPU_Vector<float> descriptors; // descriptors
  GPU_Vector<float> Fp;          // gradient of descriptors
  GPU_Vector<float> C6;
  GPU_Vector<float> C6_derivative;
  GPU_Vector<float> D_C6;
  GPU_Vector<float> sum_fxyz;
  GPU_Vector<float> kx;
  GPU_Vector<float> ky;
  GPU_Vector<float> kz;
  GPU_Vector<float> G_vdw;
  GPU_Vector<float> G_vdw_virial;
  GPU_Vector<float> S_real;
  GPU_Vector<float> S_imag;
  GPU_Vector<int> kpoint_offset;
  const Dataset* kpoint_dataset = nullptr;
  GPU_Vector<float> parameters; // parameters to be optimized
};

class NEP_VDW : public Potential
{
public:
  struct ParaMB {
    float rc_radial[NUM_ELEMENTS];  // radial cutoff
    float rc_angular[NUM_ELEMENTS]; // angular cutoff
    float c6_ref_sqrt[NUM_ELEMENTS];
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
    int one_ann_no_bias = 0;        // number of parameters in the ANN without bias
    int num_para = 0;               // number of parameters
    const float* wb[NUM_ELEMENTS];  // weights and biases for the hidden layer
    const float* b;                 // bias for the output layer
    const float* c;                 // for elements in descriptor
  };

  struct VDW_Para {
    float alpha = 0.5f;
    float alpha_factor = 1.0f;
  };

  explicit NEP_VDW(const ParameterInput& input);
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
  void configure(NEP_Energy_Parameters& para);
  void initialize(NEP_Energy_Parameters& para, int N, int Nc, int deviceCount);
  void find_force(
    NEP_Energy_Parameters& para,
    const float* parameters,
    std::vector<Dataset>& dataset,
    bool calculate_q_scaler,
    int deviceCount);
  std::unique_ptr<NEP_Energy_Parameters> model_parameters;
  std::unique_ptr<NEP_Training> training;
  NEP_Energy_Loss energy_loss;
  EnergyDifference energy_difference;
  NEP_Neighbor neighbor[16];
  ParaMB paramb;
  ANN annmb[16];
  NEP_VDW_Data nep_data[16];
  NEP_ZBL zbl;
  VDW_Para vdw_para;

  struct Compiled_Kernel
  {
    NEP_Compile library;
    NEP_Compiled_Kernels common;
    using AnnChargeFunction =
      int (*)(int, const int*, const float*, const float*, const float*,
              double*, float*, float*, float*);
    AnnChargeFunction ann_vdw_ = nullptr;
    bool valid = false;
    explicit Compiled_Kernel(const NEP_Compile_Config& config);
    bool is_valid() const { return valid; }
    void launch_ann_vdw(
      int N,
      const int* type,
      const float* descriptors,
      const float* q_scaler,
      const float* parameters,
      double* pe,
      float* Fp,
      float* C6,
      float* C6_derivative);
    void launch_force_vdw_radial(
      int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
      const float* x12, const float* y12, const float* z12,
      const float* parameters, const float* Fp,
      const float* C6_derivative, const float* D_C6,
      float* fx, float* fy, float* fz, float* virial);
    void launch_force_vdw_angular(
      int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
      const float* x12, const float* y12, const float* z12,
      const float* parameters, const float* Fp,
      const float* C6_derivative, const float* D_C6,
      const float* sum_fxyz,
      float* fx, float* fy, float* fz, float* virial);
  };
  std::unique_ptr<Compiled_Kernel> compiled_kernel_;
  void initialize_model(const NEP_Energy_Parameters& para);
  void initialize_device(const NEP_Energy_Parameters& para, int N, int Nc, int device_id);
  void initialize_compiled_kernel(const NEP_Energy_Parameters& para);
  void bind_parameters(const float* parameters, ANN& ann);
  void prepare_kpoints(Dataset& dataset, int device_id);
  void compute_descriptors(
    const ANN& ann, const Dataset& data, const NEP_Neighbor& neighbors,
    NEP_VDW_Data& storage);
  void evaluate_ann(
    const ANN& ann, Dataset& data,
    NEP_VDW_Data& storage, const float* q_scaler);
  void compute_forces(
    const ANN& ann, Dataset& data, const NEP_Neighbor& neighbors,
    NEP_VDW_Data& storage);
};
