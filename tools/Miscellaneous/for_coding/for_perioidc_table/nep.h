/*
    Copyright 2022 Zheyong Fan, Junjie Wang, Eric Lindgren
    This file is part of NEP_CPU.
    NEP_CPU is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    NEP_CPU is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with NEP_CPU.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once
#include <string>
#include <vector>

// #define USE_TABLE_FOR_RADIAL_FUNCTIONS

class NEP3
{
public:
  struct ParaMB {
    bool use_typewise_cutoff = false;
    bool use_typewise_cutoff_zbl = false;
    double typewise_cutoff_radial_factor = 2.5;
    double typewise_cutoff_angular_factor = 2.0;
    double typewise_cutoff_zbl_factor = 0.65;

    int model_type = 0; // 0=potential, 1=dipole, 2=polarizability
    int version = 4;
    double rc_radial = 0.0;
    double rc_angular = 0.0;
    double rcinv_radial = 0.0;
    double rcinv_angular = 0.0;
    int n_max_radial = 0;
    int n_max_angular = 0;
    int L_max = 0;
    int dim_angular;
    int num_L;
    int basis_size_radial = 8;
    int basis_size_angular = 8;
    int num_types_sq = 0;
    int num_c_radial = 0;
    int num_types = 0;
    double q_scaler[140];
    int atomic_numbers[94];
  };

  struct ANN {
    int dim = 0;
    int num_neurons1 = 0;
    int num_para = 0;
    int num_para_ann = 0;
    const double* w0[94];
    const double* b0[94];
    const double* w1[94];
    const double* b1;
    const double* c;
    // for the scalar part of polarizability
    const double* w0_pol[94];
    const double* b0_pol[94];
    const double* w1_pol[94];
    const double* b1_pol;
  };

  struct ZBL {
    bool enabled = false;
    bool flexibled = false;
    int num_types;
    double rc_inner = 1.0;
    double rc_outer = 2.0;
    double para[550];
  };

  struct DFTD3 {
    double s6 = 0.0;
    double s8 = 0.0;
    double a1 = 0.0;
    double a2 = 0.0;
    double rc_radial = 20.0;
    double rc_angular = 10.0;
    int atomic_number[94]; // H to Pu
    std::vector<double> cn;
    std::vector<double> dc6_sum;
    std::vector<double> dc8_sum;
  };

  NEP3();
  NEP3(const std::string& potential_filename);

  void init_from_file(const std::string& potential_filename, const bool is_rank_0);

  void update_type_map(const int ntype, int* type_map, char** elements);

  // type[num_atoms] should be integers 0, 1, ..., mapping to the atom types in nep.txt in order
  // box[9] is ordered as ax, bx, cx, ay, by, cy, az, bz, cz
  // position[num_atoms * 3] is ordered as x[num_atoms], y[num_atoms], z[num_atoms]
  // potential[num_atoms]
  // force[num_atoms * 3] is ordered as fx[num_atoms], fy[num_atoms], fz[num_atoms]
  // virial[num_atoms * 9] is ordered as v_xx[num_atoms], v_xy[num_atoms], v_xz[num_atoms],
  // v_yx[num_atoms], v_yy[num_atoms], v_yz[num_atoms], v_zx[num_atoms], v_zy[num_atoms],
  // v_zz[num_atoms]
  // descriptor[num_atoms * dim] is ordered as d0[num_atoms], d1[num_atoms], ...

  void compute(
    const std::vector<int>& type,
    const std::vector<double>& box,
    const std::vector<double>& position,
    std::vector<double>& potential,
    std::vector<double>& force,
    std::vector<double>& virial);

  void compute_with_dftd3(
    const std::string& xc,
    const double rc_potential,
    const double rc_coordination_number,
    const std::vector<int>& type,
    const std::vector<double>& box,
    const std::vector<double>& position,
    std::vector<double>& potential,
    std::vector<double>& force,
    std::vector<double>& virial);

  void compute_dftd3(
    const std::string& xc,
    const double rc_potential,
    const double rc_coordination_number,
    const std::vector<int>& type,
    const std::vector<double>& box,
    const std::vector<double>& position,
    std::vector<double>& potential,
    std::vector<double>& force,
    std::vector<double>& virial);

  void find_descriptor(
    const std::vector<int>& type,
    const std::vector<double>& box,
    const std::vector<double>& position,
    std::vector<double>& descriptor);

  void find_latent_space(
    const std::vector<int>& type,
    const std::vector<double>& box,
    const std::vector<double>& position,
    std::vector<double>& latent_space);

  void find_B_projection(
    const std::vector<int>& type,
    const std::vector<double>& box,
    const std::vector<double>& position,
    std::vector<double>& B_projection);

  void find_dipole(
    const std::vector<int>& type,
    const std::vector<double>& box,
    const std::vector<double>& position,
    std::vector<double>& dipole // 3 components, for the whole box
  );

  void find_polarizability(
    const std::vector<int>& type,
    const std::vector<double>& box,
    const std::vector<double>& position,
    std::vector<double>& polarizability // 6 components, for the whole box
  );

  void compute_for_lammps(
    int nlocal,              // atom->nlocal
    int inum,                // list->inum
    int* ilist,              // list->ilist
    int* numneigh,           // list->numneigh
    int** firstneigh,        // list->firstneigh
    int* type,               // atom->type
    int* type_map,           // map from atom type to element
    double** x,              // atom->x
    double& total_potential, // total potential energy for the current processor
    double total_virial[6],  // total virial for the current processor
    double* potential,       // eatom or nullptr
    double** f,              // atom->f
    double** virial          // cvatom or nullptr
  );

  int num_atoms = 0;
  int num_cells[3];
  double ebox[18];
  ParaMB paramb;
  ANN annmb;
  ZBL zbl;
  DFTD3 dftd3;
  std::vector<int> NN_radial, NL_radial, NN_angular, NL_angular;
  std::vector<double> r12;
  std::vector<double> Fp;
  std::vector<double> sum_fxyz;
  std::vector<double> parameters;
  std::vector<std::string> element_list;
  void update_potential(double* parameters, ANN& ann);
  void allocate_memory(const int N);

#ifdef USE_TABLE_FOR_RADIAL_FUNCTIONS
  std::vector<double> gn_radial;   // tabulated gn_radial functions
  std::vector<double> gnp_radial;  // tabulated gnp_radial functions
  std::vector<double> gn_angular;  // tabulated gn_angular functions
  std::vector<double> gnp_angular; // tabulated gnp_angular functions
  void construct_table(double* parameters);
#endif

  bool set_dftd3_para_one(
    const std::string& functional_input,
    const std::string& functional_library,
    const double s6,
    const double a1,
    const double s8,
    const double a2);
  void set_dftd3_para_all(
    const std::string& functional_input,
    const double rc_potential,
    const double rc_coordination_number);
};
