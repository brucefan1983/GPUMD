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




#ifdef USE_LDG
    #define LDG(a, n) __ldg(a + n)
#else
    #define LDG(a, n) a[n]
#endif




#define BLOCK_SIZE               128  // a good block size for most kernels
#define FILE_NAME_LENGTH         100  
#define NOSE_HOOVER_CHAIN_LENGTH 4    // This is a good choice
#define NUM_OF_HAC_COMPONENTS    7    // x-i, x-o, x-c, y-i, y-o, y-c, z
#define NUM_OF_HEAT_COMPONENTS   5    // x-i, x-o, y-i, y-o, z
#define DIM                      3    // Space Dimension


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
typedef struct
{
    int compute;         // 1 means you want to do this computation
    int sample_interval; // sample interval for velocity
    int Nc;              // number of correlation points
    real omega_max;    // maximal angular frequency for phonons
} VAC;




// Parameters related to HAC (heat current auto-correlation function)
typedef struct
{
    int compute;         // 1 means do this computation
    int sample_interval; // sample interval for heat current
    int Nc;              // number of correlation points
    int output_interval; // only output Nc/output_interval data
} HAC;




// Parameters related to SHC (spectral heat current)
typedef struct
{
    int compute;         // 1 means do this computation
    int sample_interval; // sample interval for heat current
    int Nc;              // number of correlation points
    int M;               // number of time origins for one average 
    int number_of_pairs;    // number of atom pairs between block A and block B
    int number_of_sections; // fixed to 1; may be changed in a future version
    int block_A;         // record the heat flowing from block A
    int block_B;         // record the heat flowing into block B
} SHC;




// Parameters for NEMD method of thermal conductivity claculations
typedef struct 
{
    int compute;            // 1 means you want to do this calculation
    int sample;             // 1 means sample the block temperatures
    int sample_interval;    // sample interval of temperature
    int source;             // group label of the source
    int sink;               // group label of the sink
    real delta_temperature; // relative temperature
} Heat;




// Parameters for stress-strain claculations (not finished)
typedef struct 
{
    int compute;           // 1 means you want to do this calculation
    int direction;         // 1=x; 2=y; 3=z
    real rate;             // strain rate (in units of A/ps)
} Strain;




// Parameters for neighbor list updating
typedef struct 
{
    int MN;               // upper bound of # neighbors for one particle
    int update;           // 1 means you want to update the neighbor list
    real skin;            // skin distance 
    real rc;              // cutoff used when building the neighbor list
} Neighbor;




// Parameters in the code (in a mess)
typedef struct 
{
    // a structure?
    int N;                // number of atoms
    int number_of_groups; // number of groups 
    int fixed_group;      // ID of the group in which the atoms will be fixed  

    // a structure?
    int pbc_x;           // pbc_x = 1 means periodic in the x-direction
    int pbc_y;           // pbc_y = 1 means periodic in the y-direction
    int pbc_z;           // pbc_z = 1 means periodic in the z-direction

    // Dump structure?
    int dump_thermo;    
    int dump_position;
    int dump_velocity;
    int dump_force;
    int dump_potential;
    int dump_virial;
    int sample_interval_thermo;
    int sample_interval_position;
    int sample_interval_velocity;
    int sample_interval_force;
    int sample_interval_potential;
    int sample_interval_virial;

    // nose hoover chain
    real mas_nhc1[NOSE_HOOVER_CHAIN_LENGTH];
    real pos_nhc1[NOSE_HOOVER_CHAIN_LENGTH];
    real vel_nhc1[NOSE_HOOVER_CHAIN_LENGTH];
    real mas_nhc2[NOSE_HOOVER_CHAIN_LENGTH];
    real pos_nhc2[NOSE_HOOVER_CHAIN_LENGTH];
    real vel_nhc2[NOSE_HOOVER_CHAIN_LENGTH];

    // make a structure?
    int number_of_steps; // number of steps in a specific run
    int ensemble;        // ensemble in a specific run
    real initial_temperature; // initial temperature for velocity
    real temperature;  // target temperature at a specific time 
    real temperature1;
    real temperature2;
    real pressure_x;   // target pressure at a specific time
    real pressure_y;   
    real pressure_z; 
    real temperature_coupling;
    real pressure_coupling;  
    // time step in a specific run; default value is 1 fs
    real time_step = ONE / TIME_UNIT_CONVERSION;

    // some well defined sub-structures
    Neighbor neighbor;
    Heat heat;
    Strain strain;
    VAC vac;
    HAC hac;
    SHC shc;
} Parameters;




// All the CPU data
typedef struct 
{
    int *NN; int *NL; int *fv_index; 
    int *type; int *label; int *group_size; int *group_size_sum;
    real *mass; real *x; real *y; real *z; real *vx; real *vy; real *vz; 
    real *fx; real *fy; real *fz;   
    real *thermo; real *group_temp;
    real *box_matrix;       // box matrix
    real *box_matrix_inv;   // inverse box matrix
    real *box_length;       // box length in each direction
} CPU_Data; 





// All the GPU data
typedef struct 
{
    int *NN; int *NL;             // global neighbor list
    int *NN_local; int *NL_local; // local neighbor list
    int *fv_index;                // for SHC calculations
    int *type;                    // atom type (for force)
    int *label;                   // group label 
    int *group_size;              // # atoms in each group
    int *group_size_sum;          // # atoms in all previous groups
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
    real *b; real *bp;      // for bond-order potentials
    real *fv; real *fv_all; // for SHC calculations
    real *f12x, *f12y, *f12z; // partial force for many-body potentials
} GPU_Data;





// files
typedef struct 
{
    FILE *fid_thermo;
    FILE *fid_position;
    FILE *fid_velocity;
    FILE *fid_force;
    FILE *fid_potential;
    FILE *fid_virial;
    char thermo[FILE_NAME_LENGTH];       
    char position[FILE_NAME_LENGTH];    
    char velocity[FILE_NAME_LENGTH];    
    char force[FILE_NAME_LENGTH]; 
    char potential[FILE_NAME_LENGTH];
    char virial[FILE_NAME_LENGTH];      
    char vac[FILE_NAME_LENGTH];        
    char hac[FILE_NAME_LENGTH];          
    char shc[FILE_NAME_LENGTH];         
    char temperature[FILE_NAME_LENGTH];  
    char run_in[FILE_NAME_LENGTH];      
    char xyz_in[FILE_NAME_LENGTH];      
    char potential_in[FILE_NAME_LENGTH];
} Files;




// Parameters for the 12-6 Lenneard-Jones potential
typedef struct 
{
    real s6e24;
    real s12e24;
    real s6e4;
    real s12e4;
    real cutoff_square;
} LJ;




// Parameters for the RI potential
typedef struct 
{
    real a11, b11, c11, qq11;
    real a22, b22, c22, qq22;
    real a12, b12, c12, qq12;
    real cutoff;
} RI;




// Parameters for the Tersoff potential
typedef struct
{
    real a, b, lambda, mu, beta, n, c, d, c2, d2, h, r1, r2;
    real pi_factor, one_plus_c2overd2, minus_half_over_n;
} Tersoff;




// Parameters for the Stillinger-Weber potential
typedef struct
{
    real epsilon, A, lambda, B, a, gamma, sigma, cos0; 
    real epsilon_times_A, epsilon_times_lambda, sigma_times_a;
} SW;




// Parameters for two-element the Stillinger-Weber potential
typedef struct
{
    // 2-body part
    real A[3], B[3], a[3], sigma[3], gamma[3], rc[3];
    // 3-body part
    real lambda[8], cos0[8];
} SW2;




// Parameters for the Vashishta potential
typedef struct
{
    real B[2], cos0[2], C, r0, rc; real v_rc[3], dv_rc[3];
    real H[3], qq[3], lambda_inv[3], D[3], xi_inv[3], W[3];
    int eta[3];
} Vashishta;




// Parameters for the Vashishta-table potential
typedef struct
{
    real B[2], cos0[2], C, r0, rc; real v_rc[3], dv_rc[3];
    real H[3], qq[3], lambda_inv[3], D[3], xi_inv[3], W[3];
    int eta[3];
    real rmin;
    real scale;
    int N;
    real *table;
} Vashishta_Table;


// Parameters for the EAM-Zhou-2004 potential
typedef struct
{
    real re, fe, rho_e, rho_s, rho_n, rho_0, alpha, beta, A, B, kappa, lambda;
    real Fn0, Fn1, Fn2, Fn3, F0, F1, F2, F3, eta, Fe;
    real rc; // chosen by the user?
} EAM;




// Parameters for the EAM-Dai-2006 potential
typedef struct
{
    real A, d, c, c0, c1, c2, c3, c4, B;
} FS;




// Collection of all the potentials
typedef struct
{
    int           type;
    real          rc;
    LJ            lj1;
    RI            ri;
    EAM           eam1;
    FS            fs;
    Tersoff       ters0;
    Tersoff       ters1;
    Tersoff       ters2;
    SW            sw;
    SW2           sw2;
    Vashishta     vas;
    Vashishta_Table vas_table;
} Force_Model;




/*----------------------------------------------------------------------------80
    Function declarations
------------------------------------------------------------------------------*/

FILE *my_fopen(const char *filename, const char *mode);
void print_error (const char *str);




#endif // #ifndef COMMON_H



