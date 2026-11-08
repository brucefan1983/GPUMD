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

/*----------------------------------------------------------------------------80
Charge NEP (qNEP) model of replica dynamics, a copy of NEP_Charge
(force/nep_charge/nep_charge.cuh); see replica_nep_charge_model.cu.
------------------------------------------------------------------------------*/

#pragma once
#include "force/potential.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_vector.cuh"
#include "replica_ewald.cuh"
#include "replica_pppm.cuh"

class RunInput;

struct NEP_Charge_Replica_Data {
  GPU_Vector<float> f12x; // 3-body or manybody partial forces
  GPU_Vector<float> f12y; // 3-body or manybody partial forces
  GPU_Vector<float> f12z; // 3-body or manybody partial forces
  GPU_Vector<float> Fp;
  GPU_Vector<float> sum_fxyz;
  GPU_Vector<double> energy_reference;
  GPU_Vector<int> NN_radial;    // radial neighbor list
  GPU_Vector<int> NL_radial;    // radial neighbor list
  GPU_Vector<int> NN_angular;   // angular neighbor list
  GPU_Vector<int> NL_angular;   // angular neighbor list
  GPU_Vector<float> parameters; // parameters to be optimized
  std::vector<int> cpu_NN_radial;
  std::vector<int> cpu_NN_angular;
  GPU_Vector<float> D_real;
  GPU_Vector<float> charge;
  GPU_Vector<float> charge_derivative;
};

class NEP_Charge_Replica_Model : public Potential
{
public:
  using Potential::compute;

  NEP_Charge_Replica_Data nep_data;

  struct ParaMB {
    int charge_mode = 0;
    bool use_typewise_cutoff_zbl = false;
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
    int has_q_222 = 0;
    int has_q_1111 = 0;
    int has_q_112 = 0;
    int has_q_123 = 0;
    int has_q_233 = 0;
    int has_q_134 = 0;
    int num_L;
    int basis_size_radial = 8;
    int basis_size_angular = 8;
    int num_types_sq = 0;
    int num_c_radial = 0;
    int num_types = 0;
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
    const float* c_type_pair;
    const float* q_scaler;
    const double* energy_reference = nullptr;
  };

  struct ZBL {
    bool enabled = false;
    float rc_inner = 1.0f;
    float rc_outer = 2.0f;
    int atomic_numbers[NUM_ELEMENTS];
  };

  struct ExpandedBox {
    int num_cells[3];
    float h[18];
  };

  struct Small_Box_Data {
    GPU_Vector<int> NN_radial;
    GPU_Vector<int> NL_radial;
    GPU_Vector<int> NN_angular;
    GPU_Vector<int> NL_angular;
    GPU_Vector<float> r12;
  } small_box_data;

  struct Charge_Para {
    int num_kpoints_max = 1;
    float alpha = 0.5f; // 1 / (2 Angstrom)
    float two_alpha_over_sqrt_pi = 0.564189583547756f;
    float A;
    float B;
  };

  virtual ~NEP_Charge_Replica_Model(void);
  virtual void compute(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);

  // Whether the box needs the small-box path; also fills the expanded box for it.
  static bool get_expanded_box(const double rc, const Box& box, ExpandedBox& ebox);

protected:
  // verbose = false skips the model and PPPM mesh information, e.g. for all but one of
  // several copies of the same model.
  NEP_Charge_Replica_Model(
    const char* file_potential,
    const int num_atoms,
    const RunInput& run_input,
    const bool verbose);

  // Stream for all kernel launches, set by the derived class (replica dynamics evaluates
  // one qNEP per replica stream).
  gpuStream_t stream_ = nullptr;
  int num_calls_large_box_ = 0;
  int num_calls_small_box_ = 0;

  // Global (candidate) neighbor list for the large-box path, provided by the derived class
  // on its stream.
  virtual void update_candidate_neighbors(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position_per_atom,
    const int*& NN,
    const int*& NL) = 0;

  ParaMB paramb;
  ANN annmb;
  ZBL zbl;
  ExpandedBox ebox;
  Charge_Para charge_para;
  Replica_Ewald ewald;
  Replica_PPPM pppm;

  void update_potential(float* parameters, ANN& ann);

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

  bool use_pppm = true; // use PPPM by default
  double pppm_spacing;
  void check_ewald_pppm(const RunInput& run_input);
};
