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




#include "atom.cuh"


#include "neighbor.cuh"
#include "memory.cuh"
#include "error.cuh"
#include "io.cuh"



#define DIM 3
#define NUM_OF_HEAT_COMPONENTS 5
#define FILE_NAME_LENGTH 200




void Atom::initialize_position(char *input_dir)
{  
    printf("---------------------------------------------------------------\n");
    printf("INFO:  read in initial positions and related parameters.\n");

    int count = 0;
    char file_xyz[FILE_NAME_LENGTH];
    strcpy(file_xyz, input_dir);
    strcat(file_xyz, "/xyz.in");
    FILE *fid_xyz = my_fopen(file_xyz, "r"); 

    // the first line of the xyz.in file
    double rc;
    count = fscanf(fid_xyz, "%d%d%lf", &N, &neighbor.MN, &rc);
    if (count != 3) print_error("reading error for line 1 of xyz.in.\n");
    neighbor.rc = rc;
    if (N < 1)
        print_error("number of atoms should >= 1\n");
    else
        printf("INPUT: number of atoms is %d.\n", N);
    
    if (neighbor.MN < 0)
        print_error("maximum number of neighbors should >= 0\n");
    else
        printf("INPUT: maximum number of neighbors is %d.\n",neighbor.MN);

    if (neighbor.rc < 0)
        print_error("initial cutoff for neighbor list should >= 0\n");
    else
        printf
        (
            "INPUT: initial cutoff for neighbor list is %g A.\n", 
            neighbor.rc
        );    

    // now we have enough information to allocate memroy for the major data
    MY_MALLOC(cpu_type,       int, N);
    MY_MALLOC(cpu_type_local, int, N);
    MY_MALLOC(cpu_label,      int, N);
    MY_MALLOC(cpu_mass, real, N);
    MY_MALLOC(cpu_x,    real, N);
    MY_MALLOC(cpu_y,    real, N);
    MY_MALLOC(cpu_z,    real, N);
    MY_MALLOC(cpu_box_length, real, 3);
    MY_MALLOC(cpu_box_matrix, real, 9);
    MY_MALLOC(cpu_box_matrix_inv, real, 9);

#ifdef TRICLINIC

    // second line: boundary conditions
    count = fscanf
    (fid_xyz, "%d%d%d", &(atom->pbc_x), &(atom->pbc_y), &(atom->pbc_z));
    if (count != 3) print_error("reading error for line 2 of xyz.in.\n");

    // third line: triclinic box parameters
    double box[9];   
    count = fscanf
    (
        fid_xyz, "%lf%lf%lf%lf%lf%lf%lf%lf%lf", &box[0], &box[1], &box[2], 
        &box[3], &box[4], &box[5], &box[6], &box[7], &box[8]
    ); 
    if (count != 9) print_error("reading error for line 3 of xyz.in.\n");
    for (int n = 0; n < 9; ++n) cpu_box_matrix[n] = box[n];

    real volume = cpu_box_matrix[0]
                * cpu_box_matrix[4]
                * cpu_box_matrix[8] 
                + cpu_box_matrix[1]
                * cpu_box_matrix[5]
                * cpu_box_matrix[6] 
                + cpu_box_matrix[2]
                * cpu_box_matrix[3]
                * cpu_box_matrix[7]
                - cpu_box_matrix[2]
                * cpu_box_matrix[4]
                * cpu_box_matrix[6] 
                - cpu_box_matrix[1]
                * cpu_box_matrix[3]
                * cpu_box_matrix[8] 
                - cpu_box_matrix[0]
                * cpu_box_matrix[5]
                * cpu_box_matrix[7];

    cpu_box_matrix_inv[0] = cpu_box_matrix[4]
                                * cpu_box_matrix[8] 
                                - cpu_box_matrix[5]
                                * cpu_box_matrix[7];
    cpu_box_matrix_inv[1] = cpu_box_matrix[2]
                                * cpu_box_matrix[7] 
                                - cpu_box_matrix[1]
                                * cpu_box_matrix[8];
    cpu_box_matrix_inv[2] = cpu_box_matrix[1]
                                * cpu_box_matrix[5] 
                                - cpu_box_matrix[2]
                                * cpu_box_matrix[4];
    cpu_box_matrix_inv[3] = cpu_box_matrix[5]
                                * cpu_box_matrix[6] 
                                - cpu_box_matrix[3]
                                * cpu_box_matrix[8];
    cpu_box_matrix_inv[4] = cpu_box_matrix[0]
                                * cpu_box_matrix[8] 
                                - cpu_box_matrix[2]
                                * cpu_box_matrix[6];
    cpu_box_matrix_inv[5] = cpu_box_matrix[2]
                                * cpu_box_matrix[3] 
                                - cpu_box_matrix[0]
                                * cpu_box_matrix[5];
    cpu_box_matrix_inv[6] = cpu_box_matrix[3]
                                * cpu_box_matrix[7] 
                                - cpu_box_matrix[4]
                                * cpu_box_matrix[6];
    cpu_box_matrix_inv[7] = cpu_box_matrix[1]
                                * cpu_box_matrix[6] 
                                - cpu_box_matrix[0]
                                * cpu_box_matrix[7];
    cpu_box_matrix_inv[8] = cpu_box_matrix[0]
                                * cpu_box_matrix[4] 
                                - cpu_box_matrix[1]
                                * cpu_box_matrix[3];

    for (int n = 0; n < 9; n++) cpu_box_matrix_inv[n] /= volume;

#else // #ifdef TRICLINIC

    // the second line of the xyz.in file (boundary conditions and box size)
    double lx, ly, lz;
    count = fscanf
    (
        fid_xyz, "%d%d%d%lf%lf%lf", 
        &pbc_x, &pbc_y, &pbc_z, &lx, &ly, &lz
    );
    if (count != 6) print_error("reading error for line 2 of xyz.in.\n");
    cpu_box_length[0] = lx;
    cpu_box_length[1] = ly;
    cpu_box_length[2] = lz;

#endif // #ifdef TRICLINIC

    if (pbc_x == 1)
        printf("INPUT: use periodic boundary conditions along x.\n");
    else if (pbc_x == 0)
        printf("INPUT: use     free boundary conditions along x.\n");
    else
        print_error("invalid boundary conditions along x.\n");

    if (pbc_y == 1)
        printf("INPUT: use periodic boundary conditions along y.\n");
    else if (pbc_y == 0)
        printf("INPUT: use     free boundary conditions along y.\n");
    else
        print_error("invalid boundary conditions along y.\n");

    if (pbc_z == 1)
        printf("INPUT: use periodic boundary conditions along z.\n");
    else if (pbc_z == 0)
        printf("INPUT: use     free boundary conditions along z.\n");
    else
        print_error("invalid boundary conditions along z.\n");

    // the remaining lines in the xyz.in file (type, label, mass, and positions)
    int max_label = -1; // used to determine the number of groups
    int max_type = -1; // used to determine the number of types
    for (int n = 0; n < N; n++) 
    {
        double mass, x, y, z;
        count = fscanf
        (
            fid_xyz, "%d%d%lf%lf%lf%lf", 
            &(cpu_type[n]), &(cpu_label[n]), &mass, &x, &y, &z
        );
        if (count != 6) print_error("reading error for xyz.in.\n");
        cpu_mass[n] = mass;
        cpu_x[n] = x;
        cpu_y[n] = y;
        cpu_z[n] = z;

        if (cpu_label[n] > max_label)
            max_label = cpu_label[n];

        if (cpu_type[n] > max_type)
            max_type = cpu_type[n];

        // copy
        cpu_type_local[n] = cpu_type[n];
    }

    fclose(fid_xyz);

    // number of groups determined
    number_of_groups = max_label + 1;
    if (number_of_groups == 1)
        printf("INPUT: there is only one group of atoms.\n");
    else
        printf("INPUT: there are %d groups of atoms.\n", number_of_groups);

    // determine the number of atoms in each group
    MY_MALLOC(cpu_group_size, int, number_of_groups);
    MY_MALLOC(cpu_group_size_sum, int, number_of_groups);
    for (int m = 0; m < number_of_groups; m++)
    {
        cpu_group_size[m] = 0;
        cpu_group_size_sum[m] = 0;
    }
    for (int n = 0; n < N; n++) 
        cpu_group_size[cpu_label[n]]++;
    for (int m = 0; m < number_of_groups; m++)
        printf("       %d atoms in group %d.\n", cpu_group_size[m], m);   
    
    // calculate the number of atoms before a group
    for (int m = 1; m < number_of_groups; m++)
        for (int n = 0; n < m; n++)
            cpu_group_size_sum[m] += cpu_group_size[n];

    // determine the atom indices from the first to the last group
    MY_MALLOC(cpu_group_contents, int, N);
    int *offset;
    MY_MALLOC(offset, int, number_of_groups);
    for (int m = 0; m < number_of_groups; m++) offset[m] = 0;
    for (int n = 0; n < N; n++) 
        for (int m = 0; m < number_of_groups; m++)
            if (cpu_label[n] == m)
            {
                cpu_group_contents[cpu_group_size_sum[m]+offset[m]] 
                    = n;
                offset[m]++;
            }
    MY_FREE(offset);

    // number of types determined
    number_of_types = max_type + 1;
    if (number_of_types == 1)
        printf("INPUT: there is only one atom type.\n");
    else
        printf("INPUT: there are %d atom types.\n", number_of_types);

    // determine the number of atoms in each type
    MY_MALLOC(cpu_type_size, int, number_of_types);
    for (int m = 0; m < number_of_types; m++)
        cpu_type_size[m] = 0;
    for (int n = 0; n < N; n++) 
        cpu_type_size[cpu_type[n]]++;
    for (int m = 0; m < number_of_types; m++)
        printf("       %d atoms of type %d.\n", cpu_type_size[m], m); 

    printf("INFO:  positions and related parameters initialized.\n");
    printf("---------------------------------------------------------------\n");
    printf("\n");
}




void Atom::allocate_memory_gpu(void)
{
    // memory amount
    int m1 = sizeof(int) * N;
    int m2 = m1 * neighbor.MN;
    int m3 = sizeof(int) * number_of_groups;
    int m4 = sizeof(real) * N;
    int m5 = m4 * NUM_OF_HEAT_COMPONENTS;

    // for indexing
    CHECK(cudaMalloc((void**)&NN, m1)); 
    CHECK(cudaMalloc((void**)&NL, m2)); 
#ifndef FIXED_NL
    CHECK(cudaMalloc((void**)&NN_local, m1)); 
    CHECK(cudaMalloc((void**)&NL_local, m2));
#endif
    CHECK(cudaMalloc((void**)&type, m1));  
    CHECK(cudaMalloc((void**)&type_local, m1));
    CHECK(cudaMalloc((void**)&label, m1)); 
    CHECK(cudaMalloc((void**)&group_size, m3)); 
    CHECK(cudaMalloc((void**)&group_size_sum, m3));
    CHECK(cudaMalloc((void**)&group_contents, m1));

    // for atoms
    CHECK(cudaMalloc((void**)&mass, m4));
    CHECK(cudaMalloc((void**)&x0,   m4));
    CHECK(cudaMalloc((void**)&y0,   m4));
    CHECK(cudaMalloc((void**)&z0,   m4));
    CHECK(cudaMalloc((void**)&x,    m4));
    CHECK(cudaMalloc((void**)&y,    m4));
    CHECK(cudaMalloc((void**)&z,    m4));
    CHECK(cudaMalloc((void**)&vx,   m4));
    CHECK(cudaMalloc((void**)&vy,   m4));
    CHECK(cudaMalloc((void**)&vz,   m4));
    CHECK(cudaMalloc((void**)&fx,   m4));
    CHECK(cudaMalloc((void**)&fy,   m4));
    CHECK(cudaMalloc((void**)&fz,   m4));

    CHECK(cudaMalloc((void**)&heat_per_atom, m5));

    // per-atom stress and potential energy, which are always needed
    CHECK(cudaMalloc((void**)&virial_per_atom_x,  m4));
    CHECK(cudaMalloc((void**)&virial_per_atom_y,  m4));
    CHECK(cudaMalloc((void**)&virial_per_atom_z,  m4));
    CHECK(cudaMalloc((void**)&potential_per_atom, m4));

    // box lengths
    CHECK(cudaMalloc((void**)&box_matrix,     sizeof(real) * 9));
    CHECK(cudaMalloc((void**)&box_matrix_inv, sizeof(real) * 9));
    CHECK(cudaMalloc((void**)&box_length, sizeof(real) * DIM));

    // 6 thermodynamic quantities
    CHECK(cudaMalloc((void**)&thermo, sizeof(real) * 6));

}




void Atom::copy_from_cpu_to_gpu(void)
{
    int m1 = sizeof(int) * N;
    int m2 = sizeof(int) * number_of_groups;
    int m3 = sizeof(real) * N;
    int m4 = sizeof(real) * DIM;

    cudaMemcpy(type, cpu_type, m1, cudaMemcpyHostToDevice); 
    cudaMemcpy
    (type_local, cpu_type, m1, cudaMemcpyHostToDevice);
    cudaMemcpy(label, cpu_label, m1, cudaMemcpyHostToDevice); 

    cudaMemcpy
    (group_size, cpu_group_size, m2, cudaMemcpyHostToDevice);
    cudaMemcpy
    (
        group_size_sum, cpu_group_size_sum, m2, 
        cudaMemcpyHostToDevice
    );
    cudaMemcpy
    (
        group_contents, cpu_group_contents, m1, 
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(mass, cpu_mass, m3, cudaMemcpyHostToDevice);
    cudaMemcpy(x, cpu_x, m3, cudaMemcpyHostToDevice); 
    cudaMemcpy(y, cpu_y, m3, cudaMemcpyHostToDevice); 
    cudaMemcpy(z, cpu_z, m3, cudaMemcpyHostToDevice);

    cudaMemcpy
    (
        box_matrix, cpu_box_matrix, 
        9 * sizeof(real), cudaMemcpyHostToDevice
    );
    cudaMemcpy
    (
        box_matrix_inv, cpu_box_matrix_inv, 
        9 * sizeof(real), cudaMemcpyHostToDevice
    );
    cudaMemcpy
    (box_length, cpu_box_length, m4, cudaMemcpyHostToDevice);
}




Atom::Atom(char *input_dir)
{ 
    initialize_position(input_dir);
    allocate_memory_gpu();
    copy_from_cpu_to_gpu();

    // build the initial neighbor list
    int is_first = 1;
    find_neighbor(this, is_first);
}




Atom::~Atom(void)
{
    // Free the memory allocated on the GPU
    CHECK(cudaFree(NN)); 
    CHECK(cudaFree(NL)); 
    CHECK(cudaFree(NN_local)); 
    CHECK(cudaFree(NL_local));
    CHECK(cudaFree(type));  
    CHECK(cudaFree(type_local));
    CHECK(cudaFree(label)); 
    CHECK(cudaFree(group_size)); 
    CHECK(cudaFree(group_size_sum));
    CHECK(cudaFree(group_contents));
    CHECK(cudaFree(mass));
    CHECK(cudaFree(x0));  
    CHECK(cudaFree(y0));  
    CHECK(cudaFree(z0));
    CHECK(cudaFree(x));  
    CHECK(cudaFree(y));  
    CHECK(cudaFree(z));
    CHECK(cudaFree(vx)); 
    CHECK(cudaFree(vy)); 
    CHECK(cudaFree(vz));
    CHECK(cudaFree(fx)); 
    CHECK(cudaFree(fy)); 
    CHECK(cudaFree(fz));
    CHECK(cudaFree(virial_per_atom_x));
    CHECK(cudaFree(virial_per_atom_y));
    CHECK(cudaFree(virial_per_atom_z));
    CHECK(cudaFree(potential_per_atom));
    CHECK(cudaFree(heat_per_atom));    
    //#ifdef TRICLINIC
    CHECK(cudaFree(box_matrix));
    CHECK(cudaFree(box_matrix_inv));
    //#else
    CHECK(cudaFree(box_length));
    //#endif
    CHECK(cudaFree(thermo));

    // Free the major memory allocated on the CPU
    MY_FREE(cpu_type);
    MY_FREE(cpu_type_local);
    MY_FREE(cpu_label);
    MY_FREE(cpu_group_size);
    MY_FREE(cpu_group_size_sum);
    MY_FREE(cpu_group_contents);
    MY_FREE(cpu_type_size);
    MY_FREE(cpu_mass);
    MY_FREE(cpu_x);
    MY_FREE(cpu_y);
    MY_FREE(cpu_z);
    MY_FREE(cpu_box_length);
    MY_FREE(cpu_box_matrix);
    MY_FREE(cpu_box_matrix_inv);
}




