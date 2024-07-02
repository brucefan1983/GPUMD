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
#include "potential_cavity.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_vector.cuh"
#define PARAM_SIZE 90 // Decreased the maximum number of parameters slightly to make ANNMB fit in kernel calls (max 4096 bytes long)

struct NEP3_Data {
  GPU_Vector<double> f12x; // 3-body or manybody partial forces
  GPU_Vector<double> f12y; // 3-body or manybody partial forces
  GPU_Vector<double> f12z; // 3-body or manybody partial forces
  GPU_Vector<double> Fp;
  GPU_Vector<double> sum_fxyz;
  GPU_Vector<int> NN_radial;    // radial neighbor list
  GPU_Vector<int> NL_radial;    // radial neighbor list
  GPU_Vector<int> NN_angular;   // angular neighbor list
  GPU_Vector<int> NL_angular;   // angular neighbor list
  GPU_Vector<double> parameters; // parameters to be optimized
  GPU_Vector<int> cell_count;
  GPU_Vector<int> cell_count_sum;
  GPU_Vector<int> cell_contents;
  std::vector<int> cpu_NN_radial;
  std::vector<int> cpu_NN_angular;
#ifdef USE_TABLE
  GPU_Vector<double> gn_radial;   // tabulated gn_radial functions
  GPU_Vector<double> gnp_radial;  // tabulated gnp_radial functions
  GPU_Vector<double> gn_angular;  // tabulated gn_angular functions
  GPU_Vector<double> gnp_angular; // tabulated gnp_angular functions
#endif
};

class NEP3Cavity : public PotentialCavity
{
public:
  struct ParaMB {
    int version = 2; // NEP version, 2 for NEP2 and 3 for NEP3
    int model_type =
      0; // 0=potential, 1=dipole, 2=polarizability, 3=temperature-dependent free energy
    double rc_radial = 0.0;     // radial cutoff
    double rc_angular = 0.0;    // angular cutoff
    double rcinv_radial = 0.0;  // inverse of the radial cutoff
    double rcinv_angular = 0.0; // inverse of the angular cutoff
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
    double q_scaler[140];
  };

  struct ANN {
    int dim = 0;                 // dimension of the descriptor
    int num_neurons1 = 0;        // number of neurons in the 1st hidden layer
    int num_para = 0;            // number of parameters
    const double* w0[PARAM_SIZE]; // weight from the input layer to the hidden layer
    const double* b0[PARAM_SIZE]; // bias for the hidden layer
    const double* w1[PARAM_SIZE]; // weight from the hidden layer to the output layer
    const double* b1;             // bias for the output layer
    const double* c;
    // for the scalar part of polarizability
    const double* w0_pol[10];
    const double* b0_pol[10];
    const double* w1_pol[10];
    const double* b1_pol;
  };

  struct ZBL {
    bool enabled = false;
    bool flexibled = false;
    double rc_inner = 1.0;
    double rc_outer = 2.0;
    double para[550];
    double atomic_numbers[NUM_ELEMENTS];
    int num_types;
  };

  struct ExpandedBox {
    int num_cells[3];
    double h[18];
  };

  NEP3Cavity(const char* file_potential, const int num_atoms);
  virtual ~NEP3Cavity(void);
  virtual void compute(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);

  virtual void compute(
    const double temperature,
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);
  
  virtual void compute_jacobian(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial,
    GPU_Vector<int>& system_index);

private:
  ParaMB paramb;
  ANN annmb;
  ZBL zbl;
  NEP3_Data nep_data;
  ExpandedBox ebox;

  void update_potential(double* parameters, ANN& ann);
#ifdef USE_TABLE
  void construct_table(double* parameters);
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

  void compute_small_box(
    const double temperature,
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);

  void compute_large_box(
    const double temperature,
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);
  bool has_dftd3 = false;
};
