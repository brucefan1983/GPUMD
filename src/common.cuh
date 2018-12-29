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




#ifndef COMMON_H
#define COMMON_H

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h> // seems to be needed in Windows




#ifdef USE_DP
    typedef double real;
    #define ZERO  0.0
    #define HALF  0.5
    #define ONE   1.0
    #define TWO   2.0
    #define THREE 3.0
    #define FOUR  4.0
    #define FIVE  5.0
    #define SIX   6.0

    #define K_B   8.617343e-5      // Boltzmann's constant  
    #define K_C   1.441959e+1      // electrostatic constant
    #define PI    3.14159265358979 // pi

    #define TIME_UNIT_CONVERSION     1.018051e+1
    #define PRESSURE_UNIT_CONVERSION 1.602177e+2  
    #define KAPPA_UNIT_CONVERSION    1.573769e+5
#else
    typedef float real;
    #define ZERO  0.0f
    #define HALF  0.5f
    #define ONE   1.0f
    #define TWO   2.0f
    #define THREE 3.0f
    #define FOUR  4.0f
    #define FIVE  5.0f
    #define SIX   6.0f

    #define K_B   8.617343e-5f  // Boltzmann's constant  
    #define K_C   1.441959e+1f  // electrostatic constant
    #define PI    3.141593f     // pi

    #define TIME_UNIT_CONVERSION     1.018051e+1f
    #define PRESSURE_UNIT_CONVERSION 1.602177e+2f  
    #define KAPPA_UNIT_CONVERSION    1.573769e+5f

#endif



/*----------------------------------------------------------------------------80
    Macro "functions":
------------------------------------------------------------------------------*/

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}



#define MY_MALLOC(p, t, n) p = (t *) malloc(sizeof(t) * (n));                  \
                           if(p == NULL)                                       \
                           {                                                   \
                               printf("Failed to allocate!\n");                \
                               exit(EXIT_FAILURE);                             \
                           }


#define MY_FREE(p) if(p != NULL)                       \
                   {                                   \
                       free(p);                        \
                       p = NULL;                       \
                   }                                   \
                   else                                \
                   {                                   \
                       printf("Try to free NULL!\n");  \
                       exit(EXIT_FAILURE);             \
                   }




// Parameters related to VAC (velocity auto-correlation function)
struct VAC
{
    int compute;         // 1 means you want to do this computation
    int sample_interval; // sample interval for velocity
    int Nc;              // number of correlation points
    real omega_max;    // maximal angular frequency for phonons
};




// Parameters related to HAC (heat current auto-correlation function)
struct HAC
{
    int compute;         // 1 means do this computation
    int sample_interval; // sample interval for heat current
    int Nc;              // number of correlation points
    int output_interval; // only output Nc/output_interval data
};





// Parameters related to the HNEMD method for computing thermal conductivity
struct HNEMD
{
    int compute;           // 1 means do this computation
    int output_interval;   // average the data every so many time steps
    real fe_x, fe_y, fe_z; // the driving "force" vector (in units of 1/A)
    real fe;               // magnitude of the driving "force" vector
};




// Parameters related to SHC (spectral heat current)
struct SHC
{
    int compute;         // 1 means do this computation
    int sample_interval; // sample interval for heat current
    int Nc;              // number of correlation points
    int M;               // number of time origins for one average 
    int number_of_pairs;    // number of atom pairs between block A and block B
    int number_of_sections; // fixed to 1; may be changed in a future version
    int block_A;         // record the heat flowing from block A
    int block_B;         // record the heat flowing into block B
};




// Parameters for NEMD method of thermal conductivity claculations
struct Heat
{
    int sample;             // 1 means sample the block temperatures
    int sample_interval;    // sample interval of temperature
};




// Parameters for stress-strain claculations (not finished)
struct Strain 
{
    int compute;           // 1 means you want to do this calculation
    int direction;         // 1=x; 2=y; 3=z
    real rate;             // strain rate (in units of A/ps)
};




// Parameters for neighbor list updating
struct Neighbor
{
    int MN;               // upper bound of # neighbors for one particle
    int update;           // 1 means you want to update the neighbor list
    real skin;            // skin distance 
    real rc;              // cutoff used when building the neighbor list
};




// Parameters in the code (in a mess)
struct Parameters 
{
    // a structure?
    int N;                // number of atoms
    int number_of_groups; // number of groups 
    int fixed_group;      // ID of the group in which the atoms will be fixed 
    int number_of_types;  // number of atom types 

    // a structure?
    int pbc_x;           // pbc_x = 1 means periodic in the x-direction
    int pbc_y;           // pbc_y = 1 means periodic in the y-direction
    int pbc_z;           // pbc_z = 1 means periodic in the z-direction

    // make a structure?
    int number_of_steps; // number of steps in a specific run
    real initial_temperature; // initial temperature for velocity
    real temperature1;
    real temperature2; 
    // time step in a specific run; default value is 1 fs
    real time_step = ONE / TIME_UNIT_CONVERSION;

    // some well defined sub-structures
    Neighbor neighbor;
    Heat heat;
    Strain strain;
    VAC vac;
    HAC hac;
    SHC shc;
    HNEMD hnemd;
};




// All the CPU data
struct CPU_Data
{
    int *NN; int *NL; int *fv_index;
    int *a_map; int *b_map;
    int *count_a; int *count_b;
    int *type; int *label; int *group_size; int *group_size_sum;
    int *type_local;              // local atom type (for force)
    int *type_size; // number of atoms for each type
    int *group_contents;          // atom indices sorted based on groups
    real *mass; real *x; real *y; real *z; real *vx; real *vy; real *vz; 
    real *fx; real *fy; real *fz;   
    real *heat_per_atom;    // per-atom heat current
    real *thermo; real *group_temp;
    real *box_matrix;       // box matrix
    real *box_matrix_inv;   // inverse box matrix
    real *box_length;       // box length in each direction
}; 





// All the GPU data
struct GPU_Data
{
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
    real *vx_all; real *vy_all; real *vz_all; // data used for VAC
    real *heat_all;                           // data used for HAC
    real *box_matrix;       // box matrix
    real *box_matrix_inv;   // inverse box matrix
    real *box_length;       // box length in each direction
    real *thermo;           // some thermodynamic quantities
    real *fv; real *fv_all; // for SHC calculations
    int *fv_index;  // for SHC calculations
    int *a_map; int *b_map;
	int *count_a; int *count_b;
};




#endif // #ifndef COMMON_H



