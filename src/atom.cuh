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




class Atom
{
public:
    int *NN; int *NL;             // global neighbor list
    int *NN_local; int *NL_local; // local neighbor list
    int *type;                    // atom type (for force)
    int *type_local;              // local atom type (for force)
    int *label;                   // group label 
    int *group_size;              // # atoms in each group
    int *group_size_sum;          // # atoms in all previous groups
    int *group_contents;          // atom indices sorted based on groups
    real *x0; real *y0; real *z0; // for determing when to update neighbor list
    real *mass;                   // per-atom mass
    real *x; real *y; real *z;    // per-atom position
    real *vx; real *vy; real *vz; // per-atom velocity
    real *fx; real *fy; real *fz; // per-atom force
    real *heat_per_atom;          // per-atom heat current
    real *virial_per_atom_x;      // per-atom virial
    real *virial_per_atom_y;
    real *virial_per_atom_z;
    real *potential_per_atom;     // per-atom potential energy
    real *box_matrix;       // box matrix
    real *box_matrix_inv;   // inverse box matrix
    real *box_length;       // box length in each direction
    real *thermo;           // some thermodynamic quantities

    int* cpu_type;
    int* cpu_type_local;
    int* cpu_type_size;
    int* cpu_label;
    int* cpu_group_size;
    int* cpu_group_size_sum;
    int* cpu_group_contents;

    real* cpu_mass;
    real* cpu_x;
    real* cpu_y;
    real* cpu_z;
    real* cpu_box_matrix;
    real* cpu_box_matrix_inv;
    real* cpu_box_length;
};




