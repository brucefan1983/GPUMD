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



#include "common.cuh"
#include "initialize.cuh"
#include "neighbor.cuh"



// to be improved
static void initialize_files(char *input_dir, Files *files)
{ 
    // input files 
    strcpy(files->xyz_in, input_dir);
    strcpy(files->run_in, input_dir);

    strcat(files->xyz_in, "/xyz.in");
    strcat(files->run_in, "/run.in");

    // output files
    strcpy(files->vac, input_dir);
    strcpy(files->hac, input_dir);
    strcpy(files->shc, input_dir);
    strcpy(files->kappa, input_dir);
    strcpy(files->temperature, input_dir);

    strcat(files->vac, "/vac.out");
    strcat(files->hac, "/hac.out");
    strcat(files->shc, "/shc.out");
    strcat(files->kappa, "/kappa.out");
    strcat(files->temperature, "/temperature.out");
}


// Initialize the positions.
static void initialize_position
(Files *files, Parameters *para, CPU_Data *cpu_data)
{  
    printf("INFO:  read in initial positions and related parameters.\n");

    int count = 0;

    FILE *fid_xyz = my_fopen(files->xyz_in, "r"); 

    // the first line of the xyz.in file
#ifdef USE_DP
    count = fscanf
    (
        fid_xyz, "%d%d%lf", &(para->N), 
        &(para->neighbor.MN), &(para->neighbor.rc)
    );
#else
    count = fscanf
    (
        fid_xyz, "%d%d%f", &(para->N), 
        &(para->neighbor.MN), &(para->neighbor.rc)
    );
#endif
    if (count != 3)
    {
        printf("Error: reading error for xyz.in.\n");
    }

    printf("INPUT: number of atoms is %d.\n", para->N);

    printf("INPUT: maximum number of neighbors is %d.\n", para->neighbor.MN);

    printf
    ("INPUT: initial cutoff for neighbor list is %g A.\n", para->neighbor.rc);    

    // now we have enough information to allocate memroy for the major data
    MY_MALLOC(cpu_data->NN,         int, para->N);
    MY_MALLOC(cpu_data->NL,         int, para->N * para->neighbor.MN);
    MY_MALLOC(cpu_data->type,       int, para->N);
    MY_MALLOC(cpu_data->label,      int, para->N);
    MY_MALLOC(cpu_data->mass, real, para->N);
    MY_MALLOC(cpu_data->x,    real, para->N);
    MY_MALLOC(cpu_data->y,    real, para->N);
    MY_MALLOC(cpu_data->z,    real, para->N);
    MY_MALLOC(cpu_data->vx,   real, para->N);
    MY_MALLOC(cpu_data->vy,   real, para->N);
    MY_MALLOC(cpu_data->vz,   real, para->N);
    MY_MALLOC(cpu_data->fx,   real, para->N);
    MY_MALLOC(cpu_data->fy,   real, para->N);
    MY_MALLOC(cpu_data->fz,   real, para->N);
    MY_MALLOC(cpu_data->heat_per_atom, real, para->N * NUM_OF_HEAT_COMPONENTS);
    MY_MALLOC(cpu_data->thermo, real, 6);
    MY_MALLOC(cpu_data->box_length, real, DIM);
    MY_MALLOC(cpu_data->box_matrix, real, 9);
    MY_MALLOC(cpu_data->box_matrix_inv, real, 9);

#ifdef TRICLINIC

    // second line: boundary conditions
    count = fscanf
    (fid_xyz, "%d%d%d", &(para->pbc_x), &(para->pbc_y), &(para->pbc_z));
    if (count != 3)
    {
        printf("Error: reading error for xyz.in.\n");
    }

    // third line: triclinic box parameters
#if USE_DP   
    count = fscanf
    (
        fid_xyz, "%lf%lf%lf%lf%lf%lf%lf%lf%lf", 
        &cpu_data->box_matrix[0], 
        &cpu_data->box_matrix[1], 
        &cpu_data->box_matrix[2], 
        &cpu_data->box_matrix[3], 
        &cpu_data->box_matrix[4], 
        &cpu_data->box_matrix[5], 
        &cpu_data->box_matrix[6], 
        &cpu_data->box_matrix[7], 
        &cpu_data->box_matrix[8]
    ); 
#else
    fscanf
    (
        fid_xyz, "%f%f%f%f%f%f%f%f%f", 
        &cpu_data->box_matrix[0], 
        &cpu_data->box_matrix[1], 
        &cpu_data->box_matrix[2], 
        &cpu_data->box_matrix[3], 
        &cpu_data->box_matrix[4], 
        &cpu_data->box_matrix[5], 
        &cpu_data->box_matrix[6], 
        &cpu_data->box_matrix[7], 
        &cpu_data->box_matrix[8]
    );
#endif

    if (count != 9)
    {
        printf("Error: reading error for xyz.in.\n");
    }

    real volume = cpu_data->box_matrix[0]
                * cpu_data->box_matrix[4]
                * cpu_data->box_matrix[8] 
                + cpu_data->box_matrix[1]
                * cpu_data->box_matrix[5]
                * cpu_data->box_matrix[6] 
                + cpu_data->box_matrix[2]
                * cpu_data->box_matrix[3]
                * cpu_data->box_matrix[7]
                - cpu_data->box_matrix[2]
                * cpu_data->box_matrix[4]
                * cpu_data->box_matrix[6] 
                - cpu_data->box_matrix[1]
                * cpu_data->box_matrix[3]
                * cpu_data->box_matrix[8] 
                - cpu_data->box_matrix[0]
                * cpu_data->box_matrix[5]
                * cpu_data->box_matrix[7];

    cpu_data->box_matrix_inv[0] = cpu_data->box_matrix[4]
                                * cpu_data->box_matrix[8] 
                                - cpu_data->box_matrix[5]
                                * cpu_data->box_matrix[7];
    cpu_data->box_matrix_inv[1] = cpu_data->box_matrix[2]
                                * cpu_data->box_matrix[7] 
                                - cpu_data->box_matrix[1]
                                * cpu_data->box_matrix[8];
    cpu_data->box_matrix_inv[2] = cpu_data->box_matrix[1]
                                * cpu_data->box_matrix[5] 
                                - cpu_data->box_matrix[2]
                                * cpu_data->box_matrix[4];
    cpu_data->box_matrix_inv[3] = cpu_data->box_matrix[5]
                                * cpu_data->box_matrix[6] 
                                - cpu_data->box_matrix[3]
                                * cpu_data->box_matrix[8];
    cpu_data->box_matrix_inv[4] = cpu_data->box_matrix[0]
                                * cpu_data->box_matrix[8] 
                                - cpu_data->box_matrix[2]
                                * cpu_data->box_matrix[6];
    cpu_data->box_matrix_inv[5] = cpu_data->box_matrix[2]
                                * cpu_data->box_matrix[3] 
                                - cpu_data->box_matrix[0]
                                * cpu_data->box_matrix[5];
    cpu_data->box_matrix_inv[6] = cpu_data->box_matrix[3]
                                * cpu_data->box_matrix[7] 
                                - cpu_data->box_matrix[4]
                                * cpu_data->box_matrix[6];
    cpu_data->box_matrix_inv[7] = cpu_data->box_matrix[1]
                                * cpu_data->box_matrix[6] 
                                - cpu_data->box_matrix[0]
                                * cpu_data->box_matrix[7];
    cpu_data->box_matrix_inv[8] = cpu_data->box_matrix[0]
                                * cpu_data->box_matrix[4] 
                                - cpu_data->box_matrix[1]
                                * cpu_data->box_matrix[3];

    for (int n = 0; n < 9; n++) cpu_data->box_matrix_inv[n] /= volume;

#else // #ifdef TRICLINIC

    // the second line of the xyz.in file (boundary conditions and box size)
#ifdef USE_DP
    count = fscanf
    (
        fid_xyz, "%d%d%d%lf%lf%lf", 
        &(para->pbc_x), &(para->pbc_y), &(para->pbc_z),
        &(cpu_data->box_length[0]), 
        &(cpu_data->box_length[1]), 
        &(cpu_data->box_length[2])
    );
#else
    count = fscanf
    (
        fid_xyz, "%d%d%d%f%f%f", 
        &(para->pbc_x), &(para->pbc_y), &(para->pbc_z),
        &(cpu_data->box_length[0]), 
        &(cpu_data->box_length[1]), 
        &(cpu_data->box_length[2])
    );
#endif

    if (count != 6)
    {
        printf("Error: reading error for xyz.in.\n");
    }

#endif // #ifdef TRICLINIC

    if (para->pbc_x == 1)
    {
        printf("INPUT: use periodic boundary conditions along x.\n");
    }
    else if (para->pbc_x == 0)
    {
        printf("INPUT: use free boundary conditions along x.\n");
    }
    else
    {
        printf("Error: invalid boundary conditions along x.\n");
    }

    if (para->pbc_y == 1)
    {
        printf("INPUT: use periodic boundary conditions along y.\n");
    }
    else if (para->pbc_y == 0)
    {
        printf("INPUT: use free boundary conditions along y.\n");
    }
    else
    {
        printf("Error: invalid boundary conditions along y.\n");
    }

    if (para->pbc_z == 1)
    {
        printf("INPUT: use periodic boundary conditions along z.\n");
    }
    else if (para->pbc_z == 0)
    {
        printf("INPUT: use free boundary conditions along z.\n");
    }
    else
    {
        printf("Error: invalid boundary conditions along z.\n");
    }

    // the remaining lines in the xyz.in file (type, label, mass, and positions)
    int max_label = -1; // used to determine the number of groups
    for (int n = 0; n < para->N; n++) 
    {
#ifdef USE_DP
        count = fscanf
        (
            fid_xyz, "%d%d%lf%lf%lf%lf", 
            &(cpu_data->type[n]), &(cpu_data->label[n]), &(cpu_data->mass[n]),
            &(cpu_data->x[n]), &(cpu_data->y[n]), &(cpu_data->z[n])
        );
#else
        count = fscanf
        (
            fid_xyz, "%d%d%f%f%f%f", 
            &(cpu_data->type[n]), &(cpu_data->label[n]), &(cpu_data->mass[n]),
            &(cpu_data->x[n]), &(cpu_data->y[n]), &(cpu_data->z[n])
        );
#endif

        if (count != 6)
        {
            printf("Error: reading error for xyz.in.\n");
        }

        if (cpu_data->label[n] > max_label)
        {
            max_label = cpu_data->label[n];
        }
    }

    fclose(fid_xyz);

    // number of groups determined
    para->number_of_groups = max_label + 1;
    printf("INPUT: there are %d groups of atoms.\n", para->number_of_groups);

    // determine the number of atoms in each group
    MY_MALLOC(cpu_data->group_size, int, para->number_of_groups);
    MY_MALLOC(cpu_data->group_size_sum, int, para->number_of_groups);
    for (int m = 0; m < para->number_of_groups; m++)
    {
        cpu_data->group_size[m] = 0;
        cpu_data->group_size_sum[m] = 0;
    }
    for (int n = 0; n < para->N; n++) 
    {
        cpu_data->group_size[cpu_data->label[n]]++;
    }
    for (int m = 0; m < para->number_of_groups; m++)
    {
        printf("       %d atoms in group %d.\n", cpu_data->group_size[m], m);
    }   
    
    // calculate the number of atoms before a group
    for (int m = 1; m < para->number_of_groups; m++)
    {
        for (int n = 0; n < m; n++)
        {
            cpu_data->group_size_sum[m] += cpu_data->group_size[n];
        } 
    }
    printf("INFO:  positions and related parameters initialized.\n\n");
}




//allocate the major memory on the GPU
static void allocate_memory_gpu(Parameters *para, GPU_Data *gpu_data)
{
    // memory amount
    int m1 = sizeof(int) * para->N;
    int m2 = m1 * para->neighbor.MN;
    int m3 = sizeof(int) * para->number_of_groups;
    int m4 = sizeof(real) * para->N;
    int m5 = m4 * NUM_OF_HEAT_COMPONENTS;

    // for indexing
    CHECK(cudaMalloc((void**)&gpu_data->NN, m1)); 
    CHECK(cudaMalloc((void**)&gpu_data->NL, m2)); 
#ifndef FIXED_NL
    CHECK(cudaMalloc((void**)&gpu_data->NN_local, m1)); 
    CHECK(cudaMalloc((void**)&gpu_data->NL_local, m2));
#endif
    CHECK(cudaMalloc((void**)&gpu_data->type, m1));  
    CHECK(cudaMalloc((void**)&gpu_data->label, m1)); 
    CHECK(cudaMalloc((void**)&gpu_data->group_size, m3)); 
    CHECK(cudaMalloc((void**)&gpu_data->group_size_sum, m3));

    // for atoms
    CHECK(cudaMalloc((void**)&gpu_data->mass, m4));
    CHECK(cudaMalloc((void**)&gpu_data->x0,   m4));
    CHECK(cudaMalloc((void**)&gpu_data->y0,   m4));
    CHECK(cudaMalloc((void**)&gpu_data->z0,   m4));
    CHECK(cudaMalloc((void**)&gpu_data->x,    m4));
    CHECK(cudaMalloc((void**)&gpu_data->y,    m4));
    CHECK(cudaMalloc((void**)&gpu_data->z,    m4));
    CHECK(cudaMalloc((void**)&gpu_data->vx,   m4));
    CHECK(cudaMalloc((void**)&gpu_data->vy,   m4));
    CHECK(cudaMalloc((void**)&gpu_data->vz,   m4));
    CHECK(cudaMalloc((void**)&gpu_data->fx,   m4));
    CHECK(cudaMalloc((void**)&gpu_data->fy,   m4));
    CHECK(cudaMalloc((void**)&gpu_data->fz,   m4));

    CHECK(cudaMalloc((void**)&gpu_data->heat_per_atom, m5));

    // per-atom stress and potential energy, which are always needed
    CHECK(cudaMalloc((void**)&gpu_data->virial_per_atom_x,  m4));
    CHECK(cudaMalloc((void**)&gpu_data->virial_per_atom_y,  m4));
    CHECK(cudaMalloc((void**)&gpu_data->virial_per_atom_z,  m4));
    CHECK(cudaMalloc((void**)&gpu_data->potential_per_atom, m4));

    // box lengths
    CHECK(cudaMalloc((void**)&gpu_data->box_matrix,     sizeof(real) * 9));
    CHECK(cudaMalloc((void**)&gpu_data->box_matrix_inv, sizeof(real) * 9));
    CHECK(cudaMalloc((void**)&gpu_data->box_length, sizeof(real) * DIM));

    // 6 thermodynamic quantities
    CHECK(cudaMalloc((void**)&gpu_data->thermo, sizeof(real) * 6));

}


// copy some data from the CPU to the GPU
static void copy_from_cpu_to_gpu
(Parameters *para, CPU_Data *cpu_data, GPU_Data *gpu_data)
{
    int m1 = sizeof(int) * para->N;
    int m2 = sizeof(int) * para->number_of_groups;
    int m3 = sizeof(real) * para->N;
    int m4 = sizeof(real) * DIM;

    cudaMemcpy(gpu_data->type, cpu_data->type, m1, cudaMemcpyHostToDevice); 
    cudaMemcpy(gpu_data->label, cpu_data->label, m1, cudaMemcpyHostToDevice); 

    cudaMemcpy
    (gpu_data->group_size, cpu_data->group_size, m2, cudaMemcpyHostToDevice);
    cudaMemcpy
    (
        gpu_data->group_size_sum, cpu_data->group_size_sum, m2, 
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(gpu_data->mass, cpu_data->mass, m3, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_data->x, cpu_data->x, m3, cudaMemcpyHostToDevice); 
    cudaMemcpy(gpu_data->y, cpu_data->y, m3, cudaMemcpyHostToDevice); 
    cudaMemcpy(gpu_data->z, cpu_data->z, m3, cudaMemcpyHostToDevice);

    cudaMemcpy
    (
        gpu_data->box_matrix, cpu_data->box_matrix, 
        9 * sizeof(real), cudaMemcpyHostToDevice
    );
    cudaMemcpy
    (
        gpu_data->box_matrix_inv, cpu_data->box_matrix_inv, 
        9 * sizeof(real), cudaMemcpyHostToDevice
    );
    cudaMemcpy
    (gpu_data->box_length, cpu_data->box_length, m4, cudaMemcpyHostToDevice);
}



void initialize
(
    char *input_dir, Files *files, Parameters *para, 
    CPU_Data *cpu_data, GPU_Data *gpu_data
)
{
    // initialization on the CPU  
    initialize_files(input_dir, files);
    initialize_position(files, para, cpu_data);

    // initialization on the GPU
    allocate_memory_gpu(para, gpu_data);
    copy_from_cpu_to_gpu(para, cpu_data, gpu_data);

    // build the initial neighbor list
    int is_first = 1;
    find_neighbor(para, cpu_data, gpu_data, is_first);
}



