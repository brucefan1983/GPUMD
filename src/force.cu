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
#include "read_file.cuh"
#include <vector>

#define BLOCK_SIZE 128


Force::Force(void)
{
    for (int m = 0; m < MAX_NUM_OF_POTENTIALS; m++)
    {
        potential[m] = NULL;
    }
    num_of_potentials = 0;
    rc_max = 0.0;
    group_method = -1;
}


Force::~Force(void)
{
    for (int m = 0; m < num_of_potentials; m++)
    {
        delete potential[m];
        potential[m] = NULL;
    }
}


void Force::parse_potential_definition(char **param, int num_param, Atom *atom)
{
    // 'potential_definition' must be called before all 'potential' keywords
    if (num_of_potentials > 0)
    {
        PRINT_INPUT_ERROR("potential_definition must be called before all "
                "potential keywords.\n");
    }

    if (num_param != 2 && num_param != 3)
    {
        PRINT_INPUT_ERROR("potential_definition should have only 1 or 2 "
                "parameters.\n");
    }
    if (num_param == 2)
    {
        //default is to use type, check for deviations
        if(strcmp(param[1], "group") == 0)
        {
            PRINT_INPUT_ERROR("potential_definition must have "
                    "group_method listed.\n");
        }
        else if(strcmp(param[1], "type") != 0)
        {
            PRINT_INPUT_ERROR("potential_definition only accepts "
                    "'type' or 'group' kind.\n");
        }
    }
    if (num_param == 3)
    {
        if(strcmp(param[1], "group") != 0)
        {
            PRINT_INPUT_ERROR("potential_definition: kind must be 'group' if 2 "
                    "parameters are used.\n");

        }
        else if(!is_valid_int(param[2], &group_method))
        {
            PRINT_INPUT_ERROR("potential_definition: group_method should be an "
                    "integer.\n");
        }
        else if(group_method > MAX_NUMBER_OF_GROUPS)
        {
            PRINT_INPUT_ERROR("Specified group_method is too large (> 10).\n");
        }
    }
}


// a potential
void Force::parse_potential(char **param, int num_param)
{
    // check for at least the file path
    if (num_param < 3)
    {
        PRINT_INPUT_ERROR("potential should have at least 2 parameters.\n");
    }
    strcpy(file_potential[num_of_potentials], param[1]);

    //open file to check number of types used in potential
    char potential_name[20];
    FILE *fid_potential = my_fopen(file_potential[num_of_potentials], "r");
    int count = fscanf(fid_potential, "%s", potential_name);
    int num_types = get_number_of_types(fid_potential);
    fclose(fid_potential);

    if (num_param != num_types + 2)
    {
        PRINT_INPUT_ERROR("potential has incorrect number of types/groups defined.\n");
    }

    participating_kinds.resize(num_types);

    for (int i = 0; i < num_types; i++)
    {
        if(!is_valid_int(param[i+2], &participating_kinds[i]))
        {
            PRINT_INPUT_ERROR("type/groups should be an integer.\n");
        }
        if (i != 0 && participating_kinds[i] < participating_kinds[i-1])
        {
            PRINT_INPUT_ERROR("potential types/groups must be listed in "
                    "ascending order.\n");
        }
    }
    atom_begin[num_of_potentials] = participating_kinds[0];
    atom_end[num_of_potentials] = participating_kinds[num_types-1];

    num_of_potentials++;
}


void Force::initialize_participation_and_shift(Atom* atom)
{
    if (group_method > -1)
        num_kind = atom->group[group_method].number;
    else
        num_kind = atom->number_of_types;

    // initialize bookkeeping data structures
    manybody_participation.resize(num_kind, 0);
    potential_participation.resize(num_kind, 0);
    atom->shift.resize(MAX_NUM_OF_POTENTIALS, 0);
}

	
int Force::get_number_of_types(FILE *fid_potential)
{
    int num_of_types;
    int count = fscanf(fid_potential, "%d", &num_of_types);
    PRINT_SCANF_ERROR(count, 1, "Reading error for number of types.");

    return num_of_types;
}

void Force::valdiate_potential_definitions()
{
    for (int i = 0; i < num_kind; i++)
    {
        if (potential_participation[i] == 0)
        {
            PRINT_INPUT_ERROR("All atoms must participate in at least one potential.");
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
        PRINT_INPUT_ERROR("reading error for potential file.");
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
        potential[m] = new Vashishta(fid_potential, atom);
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
            PRINT_INPUT_ERROR("Defined types/groups for RI potential must be contiguous and ascending.\n");
        }
        potential[m] = new RI(fid_potential);
        potential_type = 1;
    }
    else
    {
        PRINT_INPUT_ERROR("illegal potential model.\n");
    }

    if (potential_type == 0)
    {
        if (atom_end[m] - atom_begin[m] + 1 > num_types)
        {
            PRINT_INPUT_ERROR("Error: types/groups must be listed contiguously.\n");
        }
    }

    // check if manybody has sequential types (don't care for two-body)
    if (potential_type == 0 && !kinds_are_contiguous())
    {
        PRINT_INPUT_ERROR("Defined types/groups for manybody potentials must be contiguous and ascending.\n");
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
            PRINT_INPUT_ERROR("Only a single many-body potential definition is allowed per atom type/group.");
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
            "    applies to participating atoms [%d, %d) from group %d to "
            "group %d.\n", potential[m]->N1, potential[m]->N2, atom_begin[m],
            atom_end[m]
        );
    }
    else
    {
        printf
        (
            "    applies to participating atoms [%d, %d) from type %d to "
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
    Box box, int type_begin, int type_end, int *type,
    int N, int N1, int N2, double cutoff_square, 
    int *NN, int *NL, int *NN_local, int *NL_local,
    const double* __restrict__ x, 
    const double* __restrict__ y, 
    const double* __restrict__ z
)
{
    //<<<(N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>
    int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
    int count = 0;

    if (n1 >= N1 && n1 < N2)
    {  
        int neighbor_number = NN[n1];

        double x1 = x[n1];
        double y1 = y[n1];
        double z1 = z[n1];
        for (int i1 = 0; i1 < neighbor_number; ++i1)
        {   
            int n2 = NL[n1 + N * i1];

            // only include neighbors with the correct types
            int type_n2 = type[n2];
            if (type_n2 < type_begin || type_n2 > type_end) continue;

            double x12  = x[n2] - x1;
            double y12  = y[n2] - y1;
            double z12  = z[n2] - z1;
            dev_apply_mic(box, x12, y12, z12);
            double distance_square = x12 * x12 + y12 * y12 + z12 * z12;
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
    int *NN = atom->neighbor.NN.data();
    int *NL = atom->neighbor.NL.data();
    int *NN_local = atom->neighbor.NN_local.data();
    int *NL_local = atom->neighbor.NL_local.data();
    int *type = (group_method >= 0) ? atom->group[group_method].label.data()
                                    : atom->type;
    double rc2 = potential[m]->rc * potential[m]->rc;
    double *x = atom->x;
    double *y = atom->y;
    double *z = atom->z;
      
    gpu_find_neighbor_local<<<grid_size, BLOCK_SIZE>>>
    (
        atom->box, type1, type2, type,
        N, N1, N2, rc2, NN, NL, NN_local, NL_local, x, y, z
    );
    CUDA_CHECK_KERNEL

}


static __global__ void gpu_add_driving_force
(
    int N,
    double fe_x, double fe_y, double fe_z,
    double *g_sxx, double *g_sxy, double *g_sxz,
    double *g_syx, double *g_syy, double *g_syz,
    double *g_szx, double *g_szy, double *g_szz,
    double *g_fx, double *g_fy, double *g_fz
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        g_fx[i] += fe_x * g_sxx[i] + fe_y * g_syx[i] + fe_z * g_szx[i];
        g_fy[i] += fe_x * g_sxy[i] + fe_y * g_syy[i] + fe_z * g_szy[i];
        g_fz[i] += fe_x * g_sxz[i] + fe_y * g_syz[i] + fe_z * g_szz[i];
    }
}


// get the total force
static __global__ void gpu_sum_force
(int N, double *g_fx, double *g_fy, double *g_fz, double *g_f)
{
    //<<<3, 1024>>>
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int number_of_patches = (N - 1) / 1024 + 1; 
    __shared__ double s_f[1024];
    double f = 0.0;

    switch (bid)
    {
        case 0:
            for (int patch = 0; patch < number_of_patches; ++patch)
            {
                int n = tid + patch * 1024;
                if (n < N) f += g_fx[n];
            }
            break;
        case 1:
            for (int patch = 0; patch < number_of_patches; ++patch)
            {
                int n = tid + patch * 1024;
                if (n < N) f += g_fy[n];
            }
            break;
        case 2:
            for (int patch = 0; patch < number_of_patches; ++patch)
            {
                int n = tid + patch * 1024;
                if (n < N) f += g_fz[n];
            }
            break;
    }
    s_f[tid] = f;
    __syncthreads();

    #pragma unroll
    for (int offset = blockDim.x >> 1; offset > 32; offset >>= 1)
    {
        if (tid < offset) { s_f[tid] += s_f[tid + offset]; }
        __syncthreads();
    } 
    for (int offset = 32; offset > 0; offset >>= 1)
    {
        if (tid < offset) { s_f[tid] += s_f[tid + offset]; }
        __syncwarp();
    }

    if (tid ==  0) { g_f[bid] = s_f[0]; }
}


// correct the total force
static __global__ void gpu_correct_force
(int N, double one_over_N, double *g_fx, double *g_fy, double *g_fz, double *g_f)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        g_fx[i] -= g_f[0] * one_over_N;
        g_fy[i] -= g_f[1] * one_over_N;
        g_fz[i] -= g_f[2] * one_over_N;
    }
}


static __global__ void initialize_properties
(
    int N, double *g_fx, double *g_fy, double *g_fz, double *g_pe, double *g_virial
)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (n1 < N)
    {
        g_fx[n1] = 0.0;
        g_fy[n1] = 0.0;
        g_fz[n1] = 0.0;
        g_pe[n1] = 0.0;
        g_virial[n1 + 0 * N] = 0.0;
        g_virial[n1 + 1 * N] = 0.0;
        g_virial[n1 + 2 * N] = 0.0;
        g_virial[n1 + 3 * N] = 0.0;
        g_virial[n1 + 4 * N] = 0.0;
        g_virial[n1 + 5 * N] = 0.0;
        g_virial[n1 + 6 * N] = 0.0;
        g_virial[n1 + 7 * N] = 0.0;
        g_virial[n1 + 8 * N] = 0.0;
    }
}


void Force::set_hnemd_parameters
(
    const bool compute_hnemd, 
    const double hnemd_fe_x, 
    const double hnemd_fe_y, 
    const double hnemd_fe_z
)
{
    compute_hnemd_ = compute_hnemd;
    if (compute_hnemd)
    {
        hnemd_fe_[0] = hnemd_fe_x;
        hnemd_fe_[1] = hnemd_fe_y;
        hnemd_fe_[2] = hnemd_fe_z;
    }
} 


void Force::compute(Atom *atom)
{
    initialize_properties<<<(atom->N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>
    (
        atom->N, atom->fx, atom->fy, atom->fz, atom->potential_per_atom,
        atom->virial_per_atom
    );
    CUDA_CHECK_KERNEL

    for (int m = 0; m < num_of_potentials; m++)
    {
        // first build a local neighbor list
#ifndef USE_FCP // the FCP does not use a neighbor list at all
        find_neighbor_local(atom, m);
#endif
        // and then calculate the forces and related quantities
        potential[m]->compute(atom, m);
    }

    if (compute_hnemd_)
    {
        int grid_size = (atom->N - 1) / BLOCK_SIZE + 1;

        // the virial tensor:
        // xx xy xz    0 3 4
        // yx yy yz    6 1 5
        // zx zy zz    7 8 2
        gpu_add_driving_force<<<grid_size, BLOCK_SIZE>>>
        (
            atom->N,
            hnemd_fe_[0], hnemd_fe_[1], hnemd_fe_[2],
            atom->virial_per_atom + 0 * atom->N,
            atom->virial_per_atom + 3 * atom->N,
            atom->virial_per_atom + 4 * atom->N,
            atom->virial_per_atom + 6 * atom->N,
            atom->virial_per_atom + 1 * atom->N,
            atom->virial_per_atom + 5 * atom->N,
            atom->virial_per_atom + 7 * atom->N,
            atom->virial_per_atom + 8 * atom->N,
            atom->virial_per_atom + 2 * atom->N,      
            atom->fx, atom->fy, atom->fz
        );

        GPU_Vector<double> ftot(3); // total force vector of the system

        gpu_sum_force<<<3, 1024>>>
        (atom->N, atom->fx, atom->fy, atom->fz, ftot.data());
        CUDA_CHECK_KERNEL

        gpu_correct_force<<<grid_size, BLOCK_SIZE>>>
        (atom->N, 1.0 / atom->N, atom->fx, atom->fy, atom->fz, ftot.data());
        CUDA_CHECK_KERNEL
    }

    // always correct the force when using the FCP potential
#ifdef USE_FCP
    if (!compute_hnemd_)
    {
        GPU_Vector<double> ftot(3); // total force vector of the system
        gpu_sum_force<<<3, 1024>>>
        (atom->N, atom->fx, atom->fy, atom->fz, ftot.data());
        CUDA_CHECK_KERNEL

        int grid_size = (atom->N - 1) / BLOCK_SIZE + 1;
        gpu_correct_force<<<grid_size, BLOCK_SIZE>>>
        (atom->N, 1.0 / atom->N, atom->fx, atom->fy, atom->fz, ftot.data());
        CUDA_CHECK_KERNEL
    }
#endif
}


