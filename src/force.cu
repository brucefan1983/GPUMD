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
#include "mic.cuh"
#include "potential.cuh"
#include "tersoff1989.cuh"
#include "rebo_mos2.cuh"
#include "vashishta.cuh"
#include "tersoff1988.cuh"
#include "tersoff_modc.cuh"
#include "tersoff_mini.cuh"
#include "sw.cuh"
#include "lj.cuh"
#include "ri.cuh"
#include "eam.cuh"
#include "fcp.cuh"
#include "measure.cuh"

#define BLOCK_SIZE 128


Force::Force(void)
{
    for (int m = 0; m < MAX_NUM_OF_POTENTIALS; m++)
    {
        potential[m] = NULL;
    }
    num_of_potentials = 0;
    rc_max = ZERO;
    group_method = -1;
}


Force::~Force(void)
{
    for (int m = 0; m < num_of_potentials; m++)
    {
        delete potential[m];
        potential[m] = NULL;
    }
    MY_FREE(manybody_participation);
}


int Force::get_number_of_types(FILE *fid_potential)
{
    int num_of_types;
    int count = fscanf(fid_potential, "%d", &num_of_types);
    if (count != 1)
    {
        print_error("Number of types not defined for potential.\n");
    }
    return num_of_types;
}

void Force::valdiate_potential_definitions()
{
    for (int i = 0; i < num_kind; i++)
    {
        if (potential_participation[i] == 0)
        {
            print_error("All atoms must participate in at least "
                    "one potential.\n");
        }
    }
}

void Force::initialize_potential(char* input_dir, Atom* atom, int m)
{
    FILE *fid_potential = my_fopen(file_potential[m], "r");
    char potential_name[20];
    int count = fscanf(fid_potential, "%s", potential_name);
    if (count != 1) 
    {
        print_error("reading error for potential file.\n");
    }

    int num_types = get_number_of_types(fid_potential);
    int potential_type = 0; // 0 - manybody, 1 - two-body
    // determine the potential
    if (strcmp(potential_name, "tersoff_1989") == 0)
    {
        potential[m] = new Tersoff1989(fid_potential, atom, num_types);
    }
    else if (strcmp(potential_name, "tersoff_1988") == 0)
    {
        potential[m] = new Tersoff1988(fid_potential, atom, num_types);
    }
    else if (strcmp(potential_name, "tersoff_modc") == 0)
    {
        potential[m] = new Tersoff_modc(fid_potential, atom, num_types);
    }
    else if (strcmp(potential_name, "tersoff_mini") == 0)
    {
        potential[m] = new Tersoff_mini(fid_potential, atom, num_types);
    }
    else if (strcmp(potential_name, "sw_1985") == 0)
    {
        potential[m] = new SW2(fid_potential, atom, num_types);
    }
    else if (strcmp(potential_name, "rebo_mos2") == 0)
    {
        potential[m] = new REBO_MOS(atom);
    }
    else if (strcmp(potential_name, "eam_zhou_2004") == 0)
    {
        potential[m] = new EAM(fid_potential, atom, potential_name);
    }
    else if (strcmp(potential_name, "eam_dai_2006") == 0)
    {
        potential[m] = new EAM(fid_potential, atom, potential_name);
    }
    else if (strcmp(potential_name, "vashishta") == 0)
    {
        potential[m] = new Vashishta(fid_potential, atom, 0);
    }
    else if (strcmp(potential_name, "vashishta_table") == 0)
    {
        potential[m] = new Vashishta(fid_potential, atom, 1);
    }
    else if (strcmp(potential_name, "fcp") == 0)
    {
        potential[m] = new FCP(fid_potential, input_dir, atom);
    }
    else if (strcmp(potential_name, "lj") == 0)
    {
        potential[m] = new LJ(fid_potential, num_types,
                participating_kinds, atom_end[m]-atom_begin[m]+1);
        potential_type = 1;
    }
    else if (strcmp(potential_name, "ri") == 0)
    {
        if (!kinds_are_contiguous()) // special case for RI
        {
            print_error("Defined types/groups for RI potential must be "
                    "contiguous and ascending.\n");
        }
        potential[m] = new RI(fid_potential);
        potential_type = 1;
    }
    else
    {
        print_error("illegal potential model.\n");
    }

    if (potential_type == 0)
    {
        if (atom_end[m] - atom_begin[m] + 1 > num_types)
        {
            print_error("Error: types/groups must be listed contiguously.\n");
        }
    }

    // check if manybody has sequential types (don't care for two-body)
    if (potential_type == 0 && !kinds_are_contiguous())
    {
        print_error("Defined types/groups for manybody potentials must be "
                            "contiguous and ascending.\n");
    }

    potential[m]->N1 = 0;
    potential[m]->N2 = 0;

    if (group_method > -1)
    {
        for (int n = 0; n < atom_begin[m]; ++n)
        {
            potential[m]->N1 += atom->group[group_method].cpu_size[n];
        }
        for (int n = 0; n <= atom_end[m]; ++n)
        {
            potential[m]->N2 += atom->group[group_method].cpu_size[n];
        }
    }
    else
    {
        for (int n = 0; n < atom_begin[m]; ++n)
        {
            potential[m]->N1 += atom->cpu_type_size[n];
        }
        for (int n = 0; n <= atom_end[m]; ++n)
        {
            potential[m]->N2 += atom->cpu_type_size[n];
        }
    }

    // definition bookkeeping
    for (int n1 = atom_begin[m]; n1 < atom_end[m]+1; n1++)
    {

        if (potential_type == 0 && manybody_participation[n1])
        {
            print_error("Only a single many-body potential "
                    "definition is allowed per atom type/group (depending "
                    "on parse_potential keyword).\n");
        }

        if (potential_type == 0)
        {
            manybody_participation[n1] = 1;
            potential_participation[n1]++;
        }
        else
        {
            if (kind_is_participating(n1, m))
                potential_participation[n1]++;
        }
    }

    if (group_method > -1)
    {
        printf
        (
            "       applies to participating atoms [%d, %d) from group %d to "
            "group %d.\n", potential[m]->N1, potential[m]->N2, atom_begin[m],
            atom_end[m]
        );
    }
    else
    {
        printf
        (
            "       applies to participating atoms [%d, %d) from type %d to "
            "type %d.\n", potential[m]->N1, potential[m]->N2, atom_begin[m],
            atom_end[m]
        );
    }


    fclose(fid_potential);
}

bool Force::kind_is_participating(int kind, int pot_idx)
{
    for (int i = 0; i < (int)participating_kinds.size(); i++)
    {
        if(kind == participating_kinds[i]) return true;
    }
    return false;
}

bool Force::kinds_are_contiguous()
{
    for (int i = 0; i < (int)participating_kinds.size()-1; i++)
    {
        if (participating_kinds[i] + 1 != participating_kinds[i+1])
            return false;
    }
    return true;
}

void Force::add_potential(char* input_dir, Atom *atom)
{
    int m = num_of_potentials-1;
    initialize_potential(input_dir, atom, m);
    if (rc_max < potential[m]->rc) rc_max = potential[m]->rc;

    // check the atom types in xyz.in
    for (int n = potential[m]->N1; n < potential[m]->N2; ++n)
    {
        int kind;
        if (group_method > -1) kind = atom->group[group_method].cpu_label[n];
        else kind = atom->cpu_type[n];

        if (kind < atom_begin[m] || kind > atom_end[m])
        {
            printf("ERROR: type for potential # %d not from %d to %d.",
                m, atom_begin[m], atom_end[m]);
            exit(1);
        }
    }
    atom->shift[m] = atom_begin[m];
    participating_kinds.clear(); // reset after every definition
}


// Construct the local neighbor list from the global one (Kernel)
static __global__ void gpu_find_neighbor_local
(
    int triclinic, int pbc_x, int pbc_y, int pbc_z, 
    int type_begin, int type_end, int *type,
    int *group, int group_method,
    int N, int N1, int N2, real cutoff_square, 
    const real* __restrict__ box,
    int *NN, int *NL, int *NN_local, int *NL_local,
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


    if (n1 >= N1 && n1 < N2)
    {  
        int neighbor_number = NN[n1];

        real x1 = LDG(x, n1);   
        real y1 = LDG(y, n1);
        real z1 = LDG(z, n1);  
        for (int i1 = 0; i1 < neighbor_number; ++i1)
        {   
            int n2 = NL[n1 + N * i1];

            // only include neighbors with the correct types
            int type_n2;
            if (group_method > -1)
                type_n2 = group[n2];
            else
                type_n2 = type[n2];
            if (type_n2 < type_begin || type_n2 > type_end) continue;

            real x12  = LDG(x, n2) - x1;
            real y12  = LDG(y, n2) - y1;
            real z12  = LDG(z, n2) - z1;
            dev_apply_mic(triclinic, pbc_x, pbc_y, pbc_z, box, x12, y12, z12);
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
    int type1 = atom_begin[m];
    int type2 = atom_end[m];
    int N = atom->N;
    int N1 = potential[m]->N1;
    int N2 = potential[m]->N2;
    int grid_size = (N2 - N1 - 1) / BLOCK_SIZE + 1; 
    int triclinic = atom->box.triclinic;
    int pbc_x = atom->box.pbc_x;
    int pbc_y = atom->box.pbc_y;
    int pbc_z = atom->box.pbc_z;
    int *group = atom->group[group_method].label;
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
      
    gpu_find_neighbor_local<<<grid_size, BLOCK_SIZE>>>
    (
        triclinic, pbc_x, pbc_y, pbc_z, type1, type2, type, group,
        group_method, N, N1, N2, rc2, box, NN, NL, NN_local, NL_local, x, y, z
    );
    CUDA_CHECK_KERNEL

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

    #pragma unroll
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (tid < offset) { s_f[tid] += s_f[tid + offset]; }
        __syncthreads();
    } 

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
    real *g_virial, real *g_h
)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (n1 < N)
    {
        g_fx[n1] = ZERO;
        g_fy[n1] = ZERO;
        g_fz[n1] = ZERO;
        g_pe[n1] = ZERO;
        g_virial[n1 + 0 * N] = ZERO;
        g_virial[n1 + 1 * N] = ZERO;
        g_virial[n1 + 2 * N] = ZERO;
        g_virial[n1 + 3 * N] = ZERO;
        g_virial[n1 + 4 * N] = ZERO;
        g_virial[n1 + 5 * N] = ZERO;
        g_virial[n1 + 6 * N] = ZERO;
        g_virial[n1 + 7 * N] = ZERO;
        g_virial[n1 + 8 * N] = ZERO;
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
        atom->virial_per_atom, atom->heat_per_atom
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
        potential[m]->compute(atom, measure, m);
    }

    // correct the force when using the HNEMD method or the FCP potential
#ifdef USE_FCP
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
#else
    if (measure->hnemd.compute || measure->hnema.compute)
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
#endif
}


