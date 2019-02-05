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


/*----------------------------------------------------------------------------80
The driver class calculating force and related quantities.
------------------------------------------------------------------------------*/


#include "force.cuh"
#include "atom.cuh"
#include "error.cuh"
#include "ldg.cuh"
#include "potential.cuh"
#include "tersoff.cuh"
#include "rebo_mos2.cuh"
#include "vashishta.cuh"
#include "tersoff1988.cuh"
#include "sw.cuh"
#include "pair.cuh"
#include "eam.cuh"
#include "measure.cuh"

#define BLOCK_SIZE 128


Force::Force(void)
{
    for (int m = 0; m < MAX_NUM_OF_POTENTIALS; m++)
    {
        potential[m] = NULL;
    }
    num_of_potentials = 0;
    interlayer_only = 0;
    rc_max = ZERO;
}


Force::~Force(void)
{
    for (int m = 0; m < num_of_potentials; m++)
    {
        delete potential[m];
        potential[m] = NULL;
    }

    if (interlayer_only) 
    {
        CHECK(cudaFree(layer_label));
    }
}


static void print_type_error(int number_of_types, int number_of_types_expected)
{
    if (number_of_types != number_of_types_expected)
    {
        print_error("number of types does not match potential file.\n");
    }
}

static int get_number_of_types(FILE *fid_potential)
{
    int num_of_types;
    int count = fscanf(fid_potential, "%d", &num_of_types);
    if (count != 1)
    {
        print_error("Number of types not defined for potential.\n");
    }
    return num_of_types;
}

void Force::initialize_one_potential(Atom* atom, int m)
{
    FILE *fid_potential = my_fopen(file_potential[m], "r");
    char potential_name[20];
    int count = fscanf(fid_potential, "%s", potential_name);
    if (count != 1) 
    {
        print_error("reading error for potential.in.\n");
    }

    // determine the potential
    if (strcmp(potential_name, "tersoff_1989_1") == 0)
    {
        potential[m] = new Tersoff2(fid_potential, atom, 1);
        print_type_error(atom->number_of_types, 1);
    }
    else if (strcmp(potential_name, "tersoff_1989_2") == 0)
    { 
        potential[m] = new Tersoff2(fid_potential, atom, 2);
        print_type_error(atom->number_of_types, 2);
    }
    else if (strcmp(potential_name, "tersoff_1988") == 0)
    {
        int num_of_types = get_number_of_types(fid_potential);
        print_type_error(atom->number_of_types, num_of_types);
        potential[m] = new Tersoff1988(fid_potential, atom, num_of_types);
    }
    else if (strcmp(potential_name, "sw_1985") == 0)
    {
        potential[m] = new SW2(fid_potential, atom, 1);
        print_type_error(atom->number_of_types, 1);
    }
    else if (strcmp(potential_name, "sw_1985_2") == 0)
    {
        potential[m] = new SW2(fid_potential, atom, 2);
        print_type_error(atom->number_of_types, 2);
    }
    else if (strcmp(potential_name, "sw_1985_3") == 0)
    {
        potential[m] = new SW2(fid_potential, atom, 3);
        print_type_error(atom->number_of_types, 3);
    }
    else if (strcmp(potential_name, "rebo_mos2") == 0)
    {
        potential[m] = new REBO_MOS(atom);
        print_type_error(atom->number_of_types, 2);
    }
    else if (strcmp(potential_name, "lj1") == 0)
    {
        potential[m] = new Pair(fid_potential, 1);
        print_type_error(atom->number_of_types, 1);
    }
    else if (strcmp(potential_name, "lj2") == 0)
    {
        potential[m] = new Pair(fid_potential, 2);
        print_type_error(atom->number_of_types, 2);
    }
    else if (strcmp(potential_name, "lj3") == 0)
    {
        potential[m] = new Pair(fid_potential, 3);
        print_type_error(atom->number_of_types, 3);
    }
    else if (strcmp(potential_name, "lj4") == 0)
    {
        potential[m] = new Pair(fid_potential, 4);
        print_type_error(atom->number_of_types, 4);
    }
    else if (strcmp(potential_name, "lj5") == 0)
    {
        potential[m] = new Pair(fid_potential, 5);
        print_type_error(atom->number_of_types, 5);
    }
    else if (strcmp(potential_name, "ri") == 0)
    {
        potential[m] = new Pair(fid_potential, 0);
        print_type_error(atom->number_of_types, 2);
    }
    else if (strcmp(potential_name, "eam_zhou_2004_1") == 0)
    {
        potential[m] = new EAM(fid_potential, atom, potential_name);
        print_type_error(atom->number_of_types, 1);
    }
    else if (strcmp(potential_name, "eam_dai_2006") == 0)
    {
        potential[m] = new EAM(fid_potential, atom, potential_name);
        print_type_error(atom->number_of_types, 1);
    }
    else if (strcmp(potential_name, "vashishta") == 0)
    {
        potential[m] = new Vashishta(fid_potential, atom, 0);
        print_type_error(atom->number_of_types, 2);
    }
    else if (strcmp(potential_name, "vashishta_table") == 0)
    {
        potential[m] = new Vashishta(fid_potential, atom, 1);
        print_type_error(atom->number_of_types, 2);
    }
    else
    {
        print_error("illegal potential model.\n");
    }

    potential[m]->N1 = 0;
    potential[m]->N2 = atom->N;

    fclose(fid_potential);
}


void Force::initialize_two_body_potential(Atom* atom)
{
    FILE *fid_potential = my_fopen(file_potential[0], "r");
    char potential_name[20];
    int count = fscanf(fid_potential, "%s", potential_name);
    if (count != 1) 
    {
        print_error("reading error for potential file.\n");
    }

    // determine the potential
    if (strcmp(potential_name, "lj1") == 0)
    {
        potential[0] = new Pair(fid_potential, 1);
        print_type_error(atom->number_of_types, 1);
    }
    else if (strcmp(potential_name, "lj2") == 0)
    {
        potential[0] = new Pair(fid_potential, 2);
        print_type_error(atom->number_of_types, 2);
    }
    else if (strcmp(potential_name, "lj3") == 0)
    {
        potential[0] = new Pair(fid_potential, 3);
        print_type_error(atom->number_of_types, 3);
    }
    else if (strcmp(potential_name, "lj4") == 0)
    {
        potential[0] = new Pair(fid_potential, 4);
        print_type_error(atom->number_of_types, 4);
    }
    else if (strcmp(potential_name, "lj5") == 0)
    {
        potential[0] = new Pair(fid_potential, 5);
        print_type_error(atom->number_of_types, 5);
    }
    else if (strcmp(potential_name, "ri") == 0)
    {
        potential[0] = new Pair(fid_potential, 0);
        print_type_error(atom->number_of_types, 2);
    }
    else
    {
        print_error("illegal two-body potential model.\n");
    }

    potential[0]->N1 = 0;
    potential[0]->N2 = atom->N;

    fclose(fid_potential);
}


void Force::initialize_many_body_potential
(Atom* atom, int m)
{
    FILE *fid_potential = my_fopen(file_potential[m], "r");
    char potential_name[20];
    int count = fscanf(fid_potential, "%s", potential_name);
    if (count != 1) 
    {
        print_error("reading error for potential file.\n");
    }

    int number_of_types = type_end[m] - type_begin[m] + 1;
    // determine the potential
    if (strcmp(potential_name, "tersoff_1989_1") == 0)
    {
        potential[m] = new Tersoff2(fid_potential, atom, 1);
        print_type_error(number_of_types, 1);
    }
    else if (strcmp(potential_name, "tersoff_1989_2") == 0)
    {
        potential[m] = new Tersoff2(fid_potential, atom, 2);
        print_type_error(number_of_types, 2);
    }
    else if (strcmp(potential_name, "tersoff_1988") == 0)
    {
        int num_of_types = get_number_of_types(fid_potential);
        print_type_error(number_of_types, num_of_types);
        potential[m] = new Tersoff1988(fid_potential, atom, num_of_types);
    }
    else if (strcmp(potential_name, "sw_1985") == 0)
    {
        potential[m] = new SW2(fid_potential, atom, 1);
        print_type_error(number_of_types, 1);
    }
    else if (strcmp(potential_name, "sw_1985_2") == 0)
    {
        potential[m] = new SW2(fid_potential, atom, 2);
        print_type_error(number_of_types, 2);
    }
    else if (strcmp(potential_name, "sw_1985_3") == 0)
    {
        potential[m] = new SW2(fid_potential, atom, 3);
        print_type_error(number_of_types, 3);
    }
    else if (strcmp(potential_name, "rebo_mos2") == 0)
    {
        potential[m] = new REBO_MOS(atom);
        print_type_error(number_of_types, 2);
    }
    else if (strcmp(potential_name, "eam_zhou_2004_1") == 0)
    {
        potential[m] = new EAM(fid_potential, atom, potential_name);
        print_type_error(number_of_types, 1);
    }
    else if (strcmp(potential_name, "eam_dai_2006") == 0)
    {
        potential[m] = new EAM(fid_potential, atom, potential_name);
        print_type_error(number_of_types, 1);
    }
    else if (strcmp(potential_name, "vashishta") == 0)
    {
        potential[m] = new Vashishta(fid_potential, atom, 0);
        print_type_error(number_of_types, 2);
    }
    else if (strcmp(potential_name, "vashishta_table") == 0)
    {
        potential[m] = new Vashishta(fid_potential, atom, 1);
        print_type_error(number_of_types, 2);
    }
    else
    {
        print_error("illegal many-body potential model.\n");
    }

    potential[m]->N1 = 0;
    potential[m]->N2 = 0;
    for (int n = 0; n < type_begin[m]; ++n)
    {
        potential[m]->N1 += atom->cpu_type_size[n];
    }
    for (int n = 0; n <= type_end[m]; ++n)
    {
        potential[m]->N2 += atom->cpu_type_size[n];
    }
    printf
    (
        "       applies to atoms [%d, %d) from type %d to type %d.\n",
        potential[m]->N1, potential[m]->N2, type_begin[m], type_end[m]
    );

    fclose(fid_potential);
}


void Force::initialize(char *input_dir, Atom *atom)
{
    // a single potential
    if (num_of_potentials == 1) 
    {
        initialize_one_potential(atom, 0);
        rc_max = potential[0]->rc;
    }
    else // hybrid potentials
    {
        // the two-body part
        initialize_two_body_potential(atom);
        rc_max = potential[0]->rc;

        // if the intralayer interactions are to be excluded
        if (interlayer_only)
        {
            int memory = sizeof(int) * atom->N;
            CHECK(cudaMalloc((void**)&layer_label, memory));
            CHECK(cudaMemcpy(layer_label, atom->cpu_layer_label, memory,
                cudaMemcpyHostToDevice));
        }

        // the many-body part
        for (int m = 1; m < num_of_potentials; m++)
        {
            initialize_many_body_potential(atom, m);
            if (rc_max < potential[m]->rc) rc_max = potential[m]->rc;

            // check the atom types in xyz.in
            for (int n = potential[m]->N1; n < potential[m]->N2; ++n)
            {
                if (atom->cpu_type[n] < type_begin[m] ||
                    atom->cpu_type[n] > type_end[m])
                {
                    printf("ERROR: type for potential # %d not from %d to %d.",
                        m, type_begin[m], type_end[m]);
                    exit(1);
                }

                // the local type always starts from 0
                atom->cpu_type_local[n] -= type_begin[m];
            }
        }

        // copy the local atom type to the GPU
        CHECK(cudaMemcpy(atom->type_local, atom->cpu_type_local,
            sizeof(int) * atom->N, cudaMemcpyHostToDevice));
    }
}


// Construct the local neighbor list from the global one (Kernel)
template<int check_layer_label, int check_type>
static __global__ void gpu_find_neighbor_local
(
    int pbc_x, int pbc_y, int pbc_z, int type_begin, int type_end, int *type,
    int N, int N1, int N2, real cutoff_square, real *box_length,
    int *NN, int *NL, int *NN_local, int *NL_local, int *layer_label,
#ifdef USE_LDG
    const real* __restrict__ x, 
    const real* __restrict__ y, 
    const real* __restrict__ z
#else
    real *x, real *y, real *z
#endif
)
{
    //<<<(N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>
    int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
    int count = 0;
    real lx = box_length[0];
    real ly = box_length[1];
    real lz = box_length[2];

    int layer_n1;

    if (n1 >= N1 && n1 < N2)
    {  
        int neighbor_number = NN[n1];

        if (check_layer_label) layer_n1 = layer_label[n1];

        real x1 = LDG(x, n1);   
        real y1 = LDG(y, n1);
        real z1 = LDG(z, n1);  
        for (int i1 = 0; i1 < neighbor_number; ++i1)
        {   
            int n2 = NL[n1 + N * i1];

            // exclude intralayer interactions if needed
            if (check_layer_label) 
            {
                if (layer_n1 == layer_label[n2]) continue;
            }

            // only include neighors with the correct types
            if (check_type)
            {
                int type_n2 = type[n2];
                if (type_n2 < type_begin || type_n2 > type_end) continue;
            }

            real x12  = LDG(x, n2) - x1;
            real y12  = LDG(y, n2) - y1;
            real z12  = LDG(z, n2) - z1;
            dev_apply_mic(pbc_x, pbc_y, pbc_z, x12, y12, z12, lx, ly, lz);
            real distance_square = x12 * x12 + y12 * y12 + z12 * z12;
            if (distance_square < cutoff_square)
            {        
                NL_local[count * N + n1] = n2;
                ++count;
            }
        }
        NN_local[n1] = count;
    }
}


// Construct the local neighbor list from the global one (Wrapper)
void Force::find_neighbor_local(Atom *atom, int m)
{
    int type1 = type_begin[m];
    int type2 = type_end[m];
    int N = atom->N;
    int N1 = potential[m]->N1;
    int N2 = potential[m]->N2;
    int grid_size = (N2 - N1 - 1) / BLOCK_SIZE + 1; 
    int pbc_x = atom->box.pbc_x;
    int pbc_y = atom->box.pbc_y;
    int pbc_z = atom->box.pbc_z;
    int *NN = atom->NN;
    int *NL = atom->NL;
    int *NN_local = atom->NN_local;
    int *NL_local = atom->NL_local;
    int *type = atom->type; // global type
    real rc2 = potential[m]->rc * potential[m]->rc;
    real *x = atom->x;
    real *y = atom->y;
    real *z = atom->z;
    real *box = atom->box.h;
      
    if (0 == m)
    {
        if (interlayer_only)
        {
            gpu_find_neighbor_local<1, 0><<<grid_size, BLOCK_SIZE>>>
            (
                pbc_x, pbc_y, pbc_z, type1, type2, type, N, N1, N2, 
                rc2, box, NN, NL, NN_local, NL_local, layer_label, x, y, z
            );
            CUDA_CHECK_KERNEL
        }
        else
        {
            gpu_find_neighbor_local<0, 0><<<grid_size, BLOCK_SIZE>>>
            (
                pbc_x, pbc_y, pbc_z, type1, type2, type, N, N1, N2, 
                rc2, box, NN, NL, NN_local, NL_local, layer_label, x, y, z
            );
            CUDA_CHECK_KERNEL
        }
    }
    else
    {
        gpu_find_neighbor_local<0, 1><<<grid_size, BLOCK_SIZE>>>
        (
            pbc_x, pbc_y, pbc_z, type1, type2, type, N, N1, N2, 
            rc2, box, NN, NL, NN_local, NL_local, layer_label, x, y, z
        );
        CUDA_CHECK_KERNEL
    }
}


static __device__ void warp_reduce(volatile real *s, int t) 
{
    s[t] += s[t + 32]; s[t] += s[t + 16]; s[t] += s[t + 8];
    s[t] += s[t + 4];  s[t] += s[t + 2];  s[t] += s[t + 1];
}


// get the total force
static __global__ void gpu_sum_force
(int N, real *g_fx, real *g_fy, real *g_fz, real *g_f)
{
    //<<<3, 1024>>>
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int number_of_patches = (N - 1) / 1024 + 1; 
    __shared__ real s_f[1024];
    s_f[tid] = ZERO;

    switch (bid)
    {
        case 0:
            for (int patch = 0; patch < number_of_patches; ++patch)
            {
                int n = tid + patch * 1024;
                if (n < N) s_f[tid] += g_fx[n];
            }
            break;
        case 1:
            for (int patch = 0; patch < number_of_patches; ++patch)
            {
                int n = tid + patch * 1024;
                if (n < N) s_f[tid] += g_fy[n];
            }
            break;
        case 2:
            for (int patch = 0; patch < number_of_patches; ++patch)
            {
                int n = tid + patch * 1024;
                if (n < N) s_f[tid] += g_fz[n];
            }
            break;
    }

    __syncthreads();
    if (tid < 512) s_f[tid] += s_f[tid + 512]; __syncthreads();
    if (tid < 256) s_f[tid] += s_f[tid + 256]; __syncthreads();
    if (tid < 128) s_f[tid] += s_f[tid + 128]; __syncthreads();
    if (tid <  64) s_f[tid] += s_f[tid + 64];  __syncthreads();
    if (tid <  32) warp_reduce(s_f, tid);
    if (tid ==  0) { g_f[bid] = s_f[0]; }
}


// correct the total force
static __global__ void gpu_correct_force
(int N, real *g_fx, real *g_fy, real *g_fz, real *g_f)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        g_fx[i] -= g_f[0] / N;
        g_fy[i] -= g_f[1] / N;
        g_fz[i] -= g_f[2] / N;
    }
}


static __global__ void initialize_properties
(
    int N, real *g_fx, real *g_fy, real *g_fz, real *g_pe,
    real *g_sx, real *g_sy, real *g_sz, real *g_h
)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (n1 < N)
    {
        g_fx[n1] = ZERO;
        g_fy[n1] = ZERO;
        g_fz[n1] = ZERO;
        g_sx[n1] = ZERO;
        g_sy[n1] = ZERO;
        g_sz[n1] = ZERO;
        g_pe[n1] = ZERO;
        g_h[n1 + 0 * N] = ZERO;
        g_h[n1 + 1 * N] = ZERO;
        g_h[n1 + 2 * N] = ZERO;
        g_h[n1 + 3 * N] = ZERO;
        g_h[n1 + 4 * N] = ZERO;
    }
}


static __global__ void initialize_shc_properties(int M, real *g_fv)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (n1 < M) { g_fv[n1] = ZERO; }
}


void Force::compute(Atom *atom, Measure* measure)
{
    initialize_properties<<<(atom->N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>
    (
        atom->N, atom->fx, atom->fy, atom->fz, atom->potential_per_atom,
        atom->virial_per_atom_x, atom->virial_per_atom_y,
        atom->virial_per_atom_z, atom->heat_per_atom
    );
    CUDA_CHECK_KERNEL

    if (measure->shc.compute)
    {
        int M = measure->shc.number_of_pairs * 12;
        initialize_shc_properties<<<(M - 1)/ BLOCK_SIZE + 1, BLOCK_SIZE>>>
        (M, measure->shc.fv);
        CUDA_CHECK_KERNEL
    }

    for (int m = 0; m < num_of_potentials; m++)
    {
        // first build a local neighbor list
        find_neighbor_local(atom, m);
        // and then calculate the forces and related quantities
        potential[m]->compute(atom, measure);
    }

    // correct the force when using the HNEMD method
    if (measure->hnemd.compute)
    {
        real *ftot; // total force vector of the system
        CHECK(cudaMalloc((void**)&ftot, sizeof(real) * 3));
        gpu_sum_force<<<3, 1024>>>
        (atom->N, atom->fx, atom->fy, atom->fz, ftot);
        CUDA_CHECK_KERNEL

        int grid_size = (atom->N - 1) / BLOCK_SIZE + 1;
        gpu_correct_force<<<grid_size, BLOCK_SIZE>>>
        (atom->N, atom->fx, atom->fy, atom->fz, ftot);
        CUDA_CHECK_KERNEL

        CHECK(cudaFree(ftot));
    }
}


