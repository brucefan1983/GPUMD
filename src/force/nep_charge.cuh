/*
    Copyright 2017 Zheyong Fan and GPUMD development team
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
#include "dftd3.cuh"
#include "potential.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_vector.cuh"
#include <random>

struct NEP_Charge_Data {
  GPU_Vector<float> f12x; // 3-body or manybody partial forces
  GPU_Vector<float> f12y; // 3-body or manybody partial forces
  GPU_Vector<float> f12z; // 3-body or manybody partial forces
  GPU_Vector<float> Fp;
  GPU_Vector<float> sum_fxyz;
  GPU_Vector<int> NN_radial;    // radial neighbor list
  GPU_Vector<int> NL_radial;    // radial neighbor list
  GPU_Vector<int> NN_angular;   // angular neighbor list
  GPU_Vector<int> NL_angular;   // angular neighbor list
  GPU_Vector<float> parameters; // parameters to be optimized
  GPU_Vector<int> cell_count;
  GPU_Vector<int> cell_count_sum;
  GPU_Vector<int> cell_contents;
  std::vector<int> cpu_NN_radial;
  std::vector<int> cpu_NN_angular;
#ifdef USE_TABLE
  GPU_Vector<float> gn_radial;   // tabulated gn_radial functions
  GPU_Vector<float> gnp_radial;  // tabulated gnp_radial functions
  GPU_Vector<float> gn_angular;  // tabulated gn_angular functions
  GPU_Vector<float> gnp_angular; // tabulated gnp_angular functions
#endif
  GPU_Vector<float> kx;
  GPU_Vector<float> ky;
  GPU_Vector<float> kz;
  GPU_Vector<float> G;
  GPU_Vector<float> S_real;
  GPU_Vector<float> S_imag;
  GPU_Vector<float> D_real;
  GPU_Vector<float> charge;
  GPU_Vector<float> charge_derivative;
};

class NEP_Charge : public Potential
{
public:
  using Potential::compute;

  NEP_Charge_Data nep_data;

  struct ParaMB {
    bool use_typewise_cutoff = false;
    bool use_typewise_cutoff_zbl = false;
    float typewise_cutoff_radial_factor = 0.0f;
    float typewise_cutoff_angular_factor = 0.0f;
    float typewise_cutoff_zbl_factor = 0.0f;
    float rc_radial = 0.0f;     // radial cutoff
    float rc_angular = 0.0f;    // angular cutoff
    float rcinv_radial = 0.0f;  // inverse of the radial cutoff
    float rcinv_angular = 0.0f; // inverse of the angular cutoff
    int MN_radial = 200;
    int MN_angular = 100;
    int n_max_radial = 0;  // n_radial = 0, 1, 2, ..., n_max_radial
    int n_max_angular = 0; // n_angular = 0, 1, 2, ..., n_max_angular
    int L_max = 0;         // l = 0, 1, 2, ..., L_max
    int dim_angular;
    int num_L;
    int basis_size_radial = 8;  // for nep3
    int basis_size_angular = 8; // for nep3
    int num_types_sq = 0;       // for nep3
    int num_c_radial = 0;       // for nep3
    int num_types = 0;
    float q_scaler[140];
    int atomic_numbers[NUM_ELEMENTS];
  };

  struct ANN {
    int dim = 0;                   // dimension of the descriptor
    int num_neurons1 = 0;          // number of neurons in the 1st hidden layer
    int num_para = 0;              // number of parameters
    int num_para_ann = 0;          // number of parameters for the ANN part
    const float* w0[NUM_ELEMENTS]; // weight from the input layer to the hidden layer
    const float* b0[NUM_ELEMENTS]; // bias for the hidden layer
    const float* w1[NUM_ELEMENTS]; // weight from the hidden layer to the output layer
    const float* sqrt_epsilon_inf; // sqrt(epsilon_inf) related to BEC
    const float* b1;               // bias for the output layer
    const float* c;
  };

  struct ZBL {
    bool enabled = false;
    bool flexibled = false;
    float rc_inner = 1.0f;
    float rc_outer = 2.0f;
    float para[550];
    int atomic_numbers[NUM_ELEMENTS];
    int num_types;
  };

  struct ExpandedBox {
    int num_cells[3];
    float h[18];
  };

  struct Charge_Para {
    int charge_mode = 0;
    int num_kpoints_max = 1;
    float alpha = 0.5f; // 1 / (2 Angstrom)
    float alpha_factor = 1.0f; // 1 / (4 * alpha * alpha)
    float two_alpha_over_sqrt_pi = 0.564189583547756f;
    float A;
    float B;
  };

  NEP_Charge(const char* file_potential, const int num_atoms);
  virtual ~NEP_Charge(void);
  virtual void compute(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);

  const GPU_Vector<int>& get_NN_radial_ptr();

  const GPU_Vector<int>& get_NL_radial_ptr();

private:
  ParaMB paramb;
  ANN annmb;
  ZBL zbl;
  ExpandedBox ebox;
  DFTD3 dftd3;
  Charge_Para charge_para;
  std::mt19937 rng;

  void update_potential(float* parameters, ANN& ann);
#ifdef USE_TABLE
  void construct_table(float* parameters);
#endif

  void compute_small_box(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);

  void compute_large_box(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);

  void find_k_and_G(const double* box);

  bool has_dftd3 = false;
  void initialize_dftd3();
};
