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
#include "potential.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_vector.cuh"
#include <stdio.h>
#include <vector>

// C, h-BN, TMD
#define MAX_TYPE_ILP_NEP 7
#define CUDA_MAX_NL_ILP_NEP_CBN 256   // neighs in different layer
#define CUDA_MAX_NL_ILP_NEP_TMD 2048  // neighs in different layer
#define MAX_ILP_NEIGHBOR_CBN 3 // neighs to calc normal
#define MAX_ILP_NEIGHBOR_TMD 6 // neighs to calc normal
#define MAX_BIG_ILP_NEIGHBOR_CBN 128



struct NEP3_Data {
  GPU_Vector<float> f12x; // 3-body or manybody partial forces
  GPU_Vector<float> f12y; // 3-body or manybody partial forces
  GPU_Vector<float> f12z; // 3-body or manybody partial forces
  GPU_Vector<float> Fp;
  GPU_Vector<float> sum_fxyz;
  GPU_Vector<int> NN_radial;    // radial neighbor list
  GPU_Vector<int> NL_radial;    // radial neighbor list
  GPU_Vector<int> NN_angular;   // angular neighbor list
  GPU_Vector<int> NL_angular;   // angular neighbor list
  std::vector<GPU_Vector<float>> parameters; // parameters to be optimized
  GPU_Vector<char> para_buffer_gpu;
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
};


struct ILP_Para {
  float rcutsq_ilp[MAX_TYPE_ILP_NEP][MAX_TYPE_ILP_NEP];
  float d[MAX_TYPE_ILP_NEP][MAX_TYPE_ILP_NEP];
  float d_Seff[MAX_TYPE_ILP_NEP][MAX_TYPE_ILP_NEP];      // d / S_R / r_eff
  float C_6[MAX_TYPE_ILP_NEP][MAX_TYPE_ILP_NEP];
  float z0[MAX_TYPE_ILP_NEP][MAX_TYPE_ILP_NEP];          // beta
  float lambda[MAX_TYPE_ILP_NEP][MAX_TYPE_ILP_NEP];      // alpha / beta
  float epsilon[MAX_TYPE_ILP_NEP][MAX_TYPE_ILP_NEP];
  float C[MAX_TYPE_ILP_NEP][MAX_TYPE_ILP_NEP];
  float delta2inv[MAX_TYPE_ILP_NEP][MAX_TYPE_ILP_NEP];   // 1 / delta ^ 2
  float S[MAX_TYPE_ILP_NEP][MAX_TYPE_ILP_NEP];           // scale
  float rcut_global[MAX_TYPE_ILP_NEP][MAX_TYPE_ILP_NEP];           // scale
};

struct ILP_Data {
  GPU_Vector<int> NN, NL;
  GPU_Vector<int> reduce_NL;
  GPU_Vector<int> big_ilp_NN, big_ilp_NL;
  GPU_Vector<int> ilp_NN, ilp_NL;
  GPU_Vector<int> cell_count;
  GPU_Vector<int> cell_count_sum;
  GPU_Vector<int> cell_contents;
  GPU_Vector<float> f12x;
  GPU_Vector<float> f12y;
  GPU_Vector<float> f12z;
  GPU_Vector<float> f12x_ilp_neigh;
  GPU_Vector<float> f12y_ilp_neigh;
  GPU_Vector<float> f12z_ilp_neigh;
};

// define paramb memory location
#define UTC             0     // use_typewise_cutoff
#define TCRF            1     // typewise_cutoff_radial_factor
#define TCAF            2     // typewise_cutoff_angular_factor
#define VERSION         3     // nep version
#define RCR             4     // rc_radial
#define RCA             5     // rc_angular     
#define RCIR            6     // rcinv_radial
#define RCIA            7     // rcinv_angular
#define MNR             8     // MN_radial
#define MNA             9     // MN_angular
#define NMAXR           10    // n_max_radial
#define NMAXA           11    // n_max_angular
#define LMAX            12    // L_max
#define DIMA            13    // dim_angular
#define NUML            14    // num_L
#define BSR             15    // basis_size_radial
#define BSA             16    // basis_size_angular
#define NTS             17    // num_types_sq
#define NCR             18    // num_c_radial
#define NT              19    // num_types
#define PTRQS           20    // pointer of q_scaler
#define PTRAN           (PTRQS + PTR_OFFSET)  // pointer of atomic number
#define H_PAR_OFFSET    (PTRAN + PTR_OFFSET)  // ptr offset of head of paramb            

// define annmb memory location
#define SIZEOF_POINTER  (UINTPTR_MAX / 255 % 255)  // size of pointer
#define SIZEOF_INT      4     // size of int
#define PTR_OFFSET      (SIZEOF_POINTER / SIZEOF_INT) // ptr offset by int ptr
#define ANNDIM          0     // ann dim
#define NNEUR           1     // num_neurous1
#define OUTB1           2     // bias for output layer
#define PTRC            3     // pointer of c
#define PTRW0           (PTRC + PTR_OFFSET)   // pointer of w0 of type0
#define PTRB0           (PTRW0 + PTR_OFFSET)  // pointer of b0 of type0
#define PTRW1           (PTRB0 + PTR_OFFSET)  // pointer of w1 of type0
#define H_ANN_OFFSET    (PTRW1 + PTR_OFFSET)  // ptr offset of head of annmb

#define INT_PTR(m)      ((int*) *((uintptr_t*) (m)))
#define FLT_PTR(m)      ((float*) *((uintptr_t*) (m)))

class ILP_NEP : public Potential
{
public:
  struct ParaMB {
    bool use_typewise_cutoff = false;
    float typewise_cutoff_radial_factor = 0.0f;
    float typewise_cutoff_angular_factor = 0.0f;
    int version = 4; // NEP version, 3 for NEP3 and 4 for NEP4
    int model_type =
      0; // 0=potential, 1=dipole, 2=polarizability, 3=temperature-dependent free energy
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
    const float* b1;               // bias for the output layer
    const float* c;
  };


  using Potential::compute;
  ILP_NEP(FILE*, FILE*, int, int);
  virtual ~ILP_NEP(void);
  virtual void compute(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);
  
  virtual void compute_ilp(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial,
    std::vector<Group> &group);
  

protected:
  ILP_Para ilp_para;
  ILP_Data ilp_data;

private:
  int num_nep = 0; // number of NEP file
  int max_MN_angular = 0;     // max of parambs[i].MN_angular
  int max_MN_radial = 0;      // max of parambs[i].MN_radial
  int max_n_max_angular = 0;  // max of parambs[i].n_max_angular
  int max_dim = 0;            // max of annmbs[i].dim
  std::vector<ParaMB> parambs;
  std::vector<ANN> annmbs;
  char* h_parambs;                  // pointer to the head of parambs in gpu
  char* h_annmbs;                   // pointer to the head of annmbs in gpu
  std::vector<int> nep_map_cpu;     // map nep group to nep parameters (cpu)
  GPU_Vector<int> nep_map;          // map nep group to nep parameters (gpu)
  std::vector<int> type_map_cpu;    // map ilp type to nep type (cpu)
  GPU_Vector<int> type_map;         // map ilp type to nep type (gpu)
  NEP3_Data nep_data;

  // two group methods for ilp and nep
  int ilp_group_method = 0;
  int nep_group_method = 0;

  // true if element type is in the sublayer
  bool sublayer_flag[MAX_TYPE_ILP_NEP] = {false};

  void update_potential(float* parameters, ParaMB& paramb, ANN& ann);
#ifdef USE_TABLE
  void construct_table(float* parameters);
#endif

  // void compute_nep(
  //   Box& box,
  //   const GPU_Vector<int>& type,
  //   const GPU_Vector<double>& position,
  //   GPU_Vector<double>& potential,
  //   GPU_Vector<double>& force,
  //   GPU_Vector<double>& virial);
};

static __constant__ float Tap_coeff[8];