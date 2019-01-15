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
The class defining the simulation model.
------------------------------------------------------------------------------*/




#include "atom.cuh"

#include "error.cuh"

#define DIM 3
#define NUM_OF_HEAT_COMPONENTS 5
#define FILE_NAME_LENGTH 200




Atom::Atom(char *input_dir)
{ 
    initialize_position(input_dir);
    allocate_memory_gpu();
    copy_from_cpu_to_gpu();
    find_neighbor(1);
}




Atom::~Atom(void)
{
    free_memory_cpu();
    free_memory_gpu();
}




void Atom::read_xyz_in_line_1(FILE* fid_xyz)
{
    double rc;
    int count = fscanf(fid_xyz, "%d%d%lf", &N, &neighbor.MN, &rc);
    if (count != 3) print_error("reading error for line 1 of xyz.in.\n");
    neighbor.rc = rc;
    if (N < 1)
        print_error("number of atoms should >= 1\n");
    else
        printf("Number of atoms is %d.\n", N);
    
    if (neighbor.MN < 1)
        print_error("maximum number of neighbors should >= 1\n");
    else
        printf("Maximum number of neighbors is %d.\n",neighbor.MN);

    if (neighbor.rc < 0)
        print_error("initial cutoff for neighbor list should >= 0\n");
    else
        printf("Initial cutoff for neighbor list is %g A.\n", neighbor.rc);
}  




void Atom::read_xyz_in_line_2(FILE* fid_xyz)
{
    MY_MALLOC(cpu_box_length, real, 3);

    double lx, ly, lz;
    int count = fscanf(fid_xyz, "%d%d%d%lf%lf%lf", &pbc_x, &pbc_y, &pbc_z,
        &lx, &ly, &lz);
    if (count != 6) print_error("reading error for line 2 of xyz.in.\n");
    cpu_box_length[0] = lx;
    cpu_box_length[1] = ly;
    cpu_box_length[2] = lz;

    if (pbc_x == 1)
        printf("Use periodic boundary conditions along x.\n");
    else if (pbc_x == 0)
        printf("Use     free boundary conditions along x.\n");
    else
        print_error("invalid boundary conditions along x.\n");

    if (pbc_y == 1)
        printf("Use periodic boundary conditions along y.\n");
    else if (pbc_y == 0)
        printf("Use     free boundary conditions along y.\n");
    else
        print_error("invalid boundary conditions along y.\n");

    if (pbc_z == 1)
        printf("Use periodic boundary conditions along z.\n");
    else if (pbc_z == 0)
        printf("Use     free boundary conditions along z.\n");
    else
        print_error("invalid boundary conditions along z.\n");
}




void Atom::read_xyz_in_line_3(FILE* fid_xyz)
{
    MY_MALLOC(cpu_type, int, N);
    MY_MALLOC(cpu_type_local, int, N);
    MY_MALLOC(group[0].cpu_label, int, N);
    MY_MALLOC(cpu_mass, real, N);
    MY_MALLOC(cpu_x, real, N);
    MY_MALLOC(cpu_y, real, N);
    MY_MALLOC(cpu_z, real, N);

    group[0].number = -1; number_of_types = -1;
    for (int n = 0; n < N; n++)
    {
        double mass, x, y, z;
        int count = fscanf(fid_xyz, "%d%d%lf%lf%lf%lf", 
            &(cpu_type[n]), &(group[0].cpu_label[n]), &mass, &x, &y, &z);
        if (count != 6) print_error("reading error for xyz.in.\n");
        cpu_mass[n] = mass; cpu_x[n] = x; cpu_y[n] = y; cpu_z[n] = z;
        if (group[0].cpu_label[n] > group[0].number) 
            group[0].number = group[0].cpu_label[n];
        if (cpu_type[n] > number_of_types) number_of_types = cpu_type[n];
        cpu_type_local[n] = cpu_type[n];
    }
    group[0].number++; number_of_types++;
}




void Atom::find_group_size(void)
{
    MY_MALLOC(group[0].cpu_size, int, group[0].number);
    MY_MALLOC(group[0].cpu_size_sum, int, group[0].number);
    MY_MALLOC(group[0].cpu_contents, int, N);
    if (group[0].number == 1)
        printf("There is only one group of atoms.\n");
    else
        printf("There are %d groups of atoms.\n", group[0].number);

    // determine the number of atoms in each group
    for (int m = 0; m < group[0].number; m++)
    {
        group[0].cpu_size[m] = 0;
        group[0].cpu_size_sum[m] = 0;
    }
    for (int n = 0; n < N; n++) group[0].cpu_size[group[0].cpu_label[n]]++;
    for (int m = 0; m < group[0].number; m++)
        printf("    %d atoms in group %d.\n", group[0].cpu_size[m], m);   
    
    // calculate the number of atoms before a group
    for (int m = 1; m < group[0].number; m++)
        for (int n = 0; n < m; n++)
            group[0].cpu_size_sum[m] += group[0].cpu_size[n];
}



void Atom::find_group_contents(void)
{
    // determine the atom indices from the first to the last group
    int *offset; MY_MALLOC(offset, int, group[0].number);
    for (int m = 0; m < group[0].number; m++) offset[m] = 0;
    for (int n = 0; n < N; n++) 
        for (int m = 0; m < group[0].number; m++)
            if (group[0].cpu_label[n] == m)
                group[0].cpu_contents[group[0].cpu_size_sum[m]+offset[m]++] = n;
    MY_FREE(offset);
}




void Atom::find_type_size(void)
{
    MY_MALLOC(cpu_type_size, int, number_of_types);
    if (number_of_types == 1)
        printf("There is only one atom type.\n");
    else
        printf("There are %d atom types.\n", number_of_types);
    // determine the number of atoms in each type
    for (int m = 0; m < number_of_types; m++) cpu_type_size[m] = 0;
    for (int n = 0; n < N; n++) cpu_type_size[cpu_type[n]]++;
    for (int m = 0; m < number_of_types; m++)
        printf("    %d atoms of type %d.\n", cpu_type_size[m], m);
}




void Atom::initialize_position(char *input_dir)
{
    print_line_1();
    printf("Started initializing positions and related parameters.\n");
    print_line_2();

    char file_xyz[FILE_NAME_LENGTH];
    strcpy(file_xyz, input_dir);
    strcat(file_xyz, "/xyz.in");
    FILE *fid_xyz = my_fopen(file_xyz, "r");

    read_xyz_in_line_1(fid_xyz);
    read_xyz_in_line_2(fid_xyz);
    read_xyz_in_line_3(fid_xyz);

    fclose(fid_xyz);

    find_group_size();
    find_group_contents();
    find_type_size();

    print_line_1();
    printf("Finished initializing positions and related parameters.\n");
    print_line_2();
}




void Atom::allocate_memory_gpu(void)
{
    // memory amount
    int m1 = sizeof(int) * N;
    int m2 = m1 * neighbor.MN;
    int m3 = sizeof(int) * group[0].number;
    int m4 = sizeof(real) * N;
    int m5 = m4 * NUM_OF_HEAT_COMPONENTS;

    // for indexing
    CHECK(cudaMalloc((void**)&NN, m1));
    CHECK(cudaMalloc((void**)&NL, m2));
    CHECK(cudaMalloc((void**)&NN_local, m1));
    CHECK(cudaMalloc((void**)&NL_local, m2));
    CHECK(cudaMalloc((void**)&type, m1));
    CHECK(cudaMalloc((void**)&type_local, m1));
    CHECK(cudaMalloc((void**)&group[0].label, m1));
    CHECK(cudaMalloc((void**)&group[0].size, m3));
    CHECK(cudaMalloc((void**)&group[0].size_sum, m3));
    CHECK(cudaMalloc((void**)&group[0].contents, m1));

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
    CHECK(cudaMalloc((void**)&virial_per_atom_x,  m4));
    CHECK(cudaMalloc((void**)&virial_per_atom_y,  m4));
    CHECK(cudaMalloc((void**)&virial_per_atom_z,  m4));
    CHECK(cudaMalloc((void**)&potential_per_atom, m4));
    CHECK(cudaMalloc((void**)&heat_per_atom,      m5));

    CHECK(cudaMalloc((void**)&box_length, sizeof(real) * DIM));
    CHECK(cudaMalloc((void**)&thermo, sizeof(real) * 6));
}




void Atom::copy_from_cpu_to_gpu(void)
{
    int m1 = sizeof(int) * N;
    int m2 = sizeof(int) * group[0].number;
    int m3 = sizeof(real) * N;
    int m4 = sizeof(real) * DIM;

    CHECK(cudaMemcpy(type, cpu_type, m1, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(type_local, cpu_type, m1, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(group[0].label, group[0].cpu_label, m1,
        cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(group[0].size, group[0].cpu_size, m2,
        cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(group[0].size_sum, group[0].cpu_size_sum, m2,
        cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(group[0].contents, group[0].cpu_contents, m1,
        cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(mass, cpu_mass, m3, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(x, cpu_x, m3, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(y, cpu_y, m3, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(z, cpu_z, m3, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(box_length, cpu_box_length, m4, cudaMemcpyHostToDevice));
}




void Atom::free_memory_cpu(void)
{
    MY_FREE(cpu_type);
    MY_FREE(cpu_type_local);
    MY_FREE(group[0].cpu_label);
    MY_FREE(group[0].cpu_size);
    MY_FREE(group[0].cpu_size_sum);
    MY_FREE(group[0].cpu_contents);
    MY_FREE(cpu_type_size);
    MY_FREE(cpu_mass);
    MY_FREE(cpu_x);
    MY_FREE(cpu_y);
    MY_FREE(cpu_z);
    MY_FREE(cpu_box_length);
}




void Atom::free_memory_gpu(void)
{
    CHECK(cudaFree(NN)); 
    CHECK(cudaFree(NL)); 
    CHECK(cudaFree(NN_local)); 
    CHECK(cudaFree(NL_local));
    CHECK(cudaFree(type));  
    CHECK(cudaFree(type_local));
    CHECK(cudaFree(group[0].label)); 
    CHECK(cudaFree(group[0].size)); 
    CHECK(cudaFree(group[0].size_sum));
    CHECK(cudaFree(group[0].contents));
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
    CHECK(cudaFree(box_length));
    CHECK(cudaFree(thermo));
}




