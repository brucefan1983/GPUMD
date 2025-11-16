#pragma once

#include <vector>

class Atom;
class Group;

/*
  Add_Spring: add harmonic spring forces to groups of atoms.

  Supported modes (first parameter after 'add_spring'):

    1) ghost_com
       add_spring ghost_com gm gid k xg yg zg R0 vx vy vz

       - A ghost COM at (xg, yg, zg) pulls the COM of group (gm, gid)
         with spring constant k and equilibrium length R0.
       - The ghost COM moves with velocity (vx, vy, vz) per MD step.

    2) ghost_atom
       add_spring ghost_atom gm gid k R0 vx vy vz

       - Each atom in group (gm, gid) is connected to a corresponding
         ghost atom by a spring with constant k and equilibrium length R0.
       - The initial ghost positions are taken as the unwrapped positions
         of the atoms at step = 0, and then translated with
         velocity (vx, vy, vz) per step.

    3) couple_com
       add_spring couple_com gm1 gid1 gm2 gid2 k R0

       - The centers of mass of group (gm1, gid1) and (gm2, gid2)
         are connected by a spring of constant k and equilibrium length R0.
*/

enum SpringMode {
  SPRING_GHOST_COM = 0,
  SPRING_GHOST_ATOM = 1,
  SPRING_COUPLE_COM = 2
};

class Add_Spring
{
public:
  void parse(const char** param, int num_param, const std::vector<Group>& groups);
  void compute(const int step, const std::vector<Group>& groups, Atom& atom);
  void finalize();

private:
  static const int MAX_SPRING_CALLS = 10;
  int num_calls_ = 0;

  SpringMode mode_[MAX_SPRING_CALLS];

  // group info
  int grouping_method1_[MAX_SPRING_CALLS];
  int group_id1_[MAX_SPRING_CALLS];
  int grouping_method2_[MAX_SPRING_CALLS];
  int group_id2_[MAX_SPRING_CALLS];

  // spring parameters
  double k_[MAX_SPRING_CALLS];
  double R0_[MAX_SPRING_CALLS];

  // ghost_com: origin and velocity
  double ghost_com_origin_[MAX_SPRING_CALLS][3];   // (xg0, yg0, zg0)
  double ghost_com_velocity_[MAX_SPRING_CALLS][3]; // (vx, vy, vz)

  // ghost_atom: per-atom anchor (device) and common velocity
  double* d_ghost_atom_pos_[MAX_SPRING_CALLS];   // device pointer, length = 3 * group_size
  int     ghost_atom_group_size_[MAX_SPRING_CALLS];
  double  ghost_atom_velocity_[MAX_SPRING_CALLS][3];

  // scratch device buffers for COM / energy
  double* d_tmp_vec3_  = nullptr; // length 3
  double* d_tmp_scalar_ = nullptr; // length 1

  // spring energy for each call (optional use)
  double spring_energy_[MAX_SPRING_CALLS];
};

