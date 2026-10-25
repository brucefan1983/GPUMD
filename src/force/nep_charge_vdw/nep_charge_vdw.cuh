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
#include "force/neighbor.cuh"
#include "force/potential.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_vector.cuh"
#include "force/ewald.cuh"
#include "force/pppm.cuh"

class RunInput;

struct NEP_Charge_VDW_Data {
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
  GPU_Vector<float> kx;
  GPU_Vector<float> ky;
  GPU_Vector<float> kz;
  GPU_Vector<float> G_vdw;
  GPU_Vector<float> G_vdw_virial;
  GPU_Vector<float> S_real;
  GPU_Vector<float> S_imag;
  GPU_Vector<float> D_real;
  GPU_Vector<float> charge;
  GPU_Vector<float> charge_derivative;
  GPU_Vector<float> C6;            // positive vdW amplitude sqrt(C6), as in training
  GPU_Vector<float> C6_derivative; // derivative of the amplitude with respect to descriptor
  GPU_Vector<float> D_C6;
  GPU_Vector<float> bec;               // BEC
};

class NEP_Charge_VDW : public Potential
{
public:
  using Potential::compute;

  NEP_Charge_VDW_Data nep_data;

  struct ParaMB {
    bool use_typewise_cutoff_zbl = false;
    bool has_charge = false;
    float typewise_cutoff_zbl_factor = 0.0f;
    float rc_radial = 0.0f;     // maximum radial cutoff
    float rc_angular = 0.0f;    // maximum angular cutoff
    float rc_radial_type[NUM_ELEMENTS];
    float rc_angular_type[NUM_ELEMENTS];
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
    const float* c;
    const float* c_type_pair;
    const float* q_scaler;
    const double* energy_reference = nullptr;
    float c6_ref_sqrt[NUM_ELEMENTS]; // reference amplitudes for the vdW ANN output
  };

  struct ZBL {
    bool enabled = false;
    bool flexible = false;
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

  struct Small_Box_Data {
    GPU_Vector<int> NN_radial;
    GPU_Vector<int> NL_radial;
    GPU_Vector<int> NN_angular;
    GPU_Vector<int> NL_angular;
    GPU_Vector<float> r12;
  } small_box_data;

  struct Charge_Para {
    float alpha = 0.5f; // 1 / (2 Angstrom)
    float two_alpha_over_sqrt_pi = 0.564189583547756f;
  };

  NEP_Charge_VDW(const char* file_potential, const int num_atoms, const RunInput& run_input);
  virtual ~NEP_Charge_VDW(void);
  virtual void compute(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);

  const GPU_Vector<int>& get_NN_radial_ptr();

  const GPU_Vector<int>& get_NL_radial_ptr();

  GPU_Vector<float>& get_charge_reference();

  GPU_Vector<float>& get_bec_reference();

private:
  ParaMB paramb;
  ANN annmb;
  ZBL zbl;
  ExpandedBox ebox;
  Charge_Para charge_para;
  Ewald ewald;
  PPPM pppm;
  NeighborManager neighbor_manager;

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

  int num_kpoints = 0;
  double kpoint_box[9] = {0.0};
  void find_k_and_G(const double* box);

  void find_force_vdw(
    const int N,
    const GPU_Vector<double>& position_per_atom,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom);

  bool need_bec = false;
  void check_need_bec(const RunInput& run_input);
  bool use_pppm = true; // use PPPM by default
  double pppm_spacing;
  void check_ewald_pppm(const RunInput& run_input);
};
