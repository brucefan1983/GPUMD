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

/*----------------------------------------------------------------------------80
The LINCS (linear constraint solver) bond-length constraint method:
Hess et al., J. Comput. Chem. 18, 1463 (1997).
Constraints are either detected from the initial structure using covalent
radii (no topology information needed) or read from an explicit constraint
file. Only for non-reactive systems.
------------------------------------------------------------------------------*/

#pragma once

#include "model/box.cuh"
#include "utilities/gpu_vector.cuh"
#include <string>
#include <vector>

class Atom;

class Lincs
{
public:
  // parse the "lincs" keyword in run.in
  void parse(const char** param, int num_param);

  // build the constraint data (done once, at the first run)
  void setup(Atom& atom, Box& box);

  // position constraint, called after integrate.compute1 (before force compute)
  void compute1(const double time_step, const Box& box, Atom& atom);

  // velocity constraint, called after integrate.compute2
  void compute2(const Box& box, Atom& atom);

  bool enabled = false;
  int number_of_constraints = 0;

private:
  int mode = 1;                // 1: only bonds involving H; 2: all detected bonds; 3: from file
  int expansion_order = 4;     // order of the Neumann series expansion
  int num_iterations = 1;      // number of re-linearization iterations per step
  double cutoff_factor = 1.25; // bond if r_ij < (r_cov_i + r_cov_j) * cutoff_factor
  std::string filename;        // constraint file for mode 3
  bool setup_done = false;

  // constraint data on the GPU
  GPU_Vector<int> atom_i;      // first atom of each constraint
  GPU_Vector<int> atom_j;      // second atom of each constraint
  GPU_Vector<double> target;   // target bond length d_k of each constraint
  GPU_Vector<double> bond_vec; // bond vector at the constrained positions (3*K)
  GPU_Vector<double> rhs;      // right-hand side of the linear system
  GPU_Vector<double> diag_inv; // inverse of the diagonal of the coupling matrix
  GPU_Vector<double> lagrange; // solution (Lagrange multipliers)
  GPU_Vector<double> matvec_a; // buffers for the Neumann series expansion
  GPU_Vector<double> matvec_b;

  // sparse coupling structure in CSR format: constraint k couples with the
  // constraints coupl_index[coupl_offset[k], coupl_offset[k+1]), which share
  // the atom coupl_atom with it, with sign coupl_sign (+1 or -1)
  GPU_Vector<int> coupl_offset;
  GPU_Vector<int> coupl_index;
  GPU_Vector<int> coupl_atom;
  GPU_Vector<double> coupl_sign;

  // detect bonds from the initial structure using covalent radii (modes 1 and 2)
  void detect_bonds(
    Atom& atom,
    const Box& box,
    const std::vector<double>& position,
    std::vector<int>& cpu_atom_i,
    std::vector<int>& cpu_atom_j,
    std::vector<double>& cpu_target);

  // read explicit constraints from the constraint file (mode 3)
  void read_constraints_from_file(
    Atom& atom,
    const Box& box,
    const std::vector<double>& position,
    std::vector<int>& cpu_atom_i,
    std::vector<int>& cpu_atom_j,
    std::vector<double>& cpu_target);

  void compute_bond_vectors(const Box& box, Atom& atom);
  void compute_diag_inv(Atom& atom);
  void compute_rhs(const Box& box, Atom& atom);
  void compute_velocity_rhs(Atom& atom);
  void solve_linear_system(Atom& atom);
  void apply_position_correction(const double time_step, Atom& atom);
  void apply_velocity_correction(Atom& atom);
};
