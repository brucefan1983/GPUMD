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
    int count = fscanf(fid_xyz, "%d%d%lf%d%d%d%d\n", &N, &neighbor.MN, &rc,
        &box.triclinic, &has_velocity_in_xyz, &has_layer_in_xyz,
        &num_of_grouping_methods);
    if (count != 7) print_error("Reading error for line 1 of xyz.in.\n");
    neighbor.rc = rc;
    if (N < 2)
        print_error("Number of atoms should >= 2\n");
    else
        printf("Number of atoms is %d.\n", N);
    if (neighbor.MN < 1)
        print_error("Maximum number of neighbors should >= 1\n");
    else
        printf("Maximum number of neighbors is %d.\n", neighbor.MN);
    if (neighbor.rc <= 0)
        print_error("Initial cutoff for neighbor list should > 0\n");
    else
        printf("Initial cutoff for neighbor list is %g A.\n", neighbor.rc);
    if (box.triclinic == 0)
    {
        printf("Use orthogonal box.\n");
        box.memory = sizeof(real) * 3;
    }
    else if (box.triclinic == 1)
    {
        printf("Use triclinic box.\n");
        box.memory = sizeof(real) * 9;
    }
    else
        print_error("Invalid box type.\n");
    if (has_velocity_in_xyz == 0)
        printf("Do not specify initial velocities here.\n");
    else
        printf("Specify initial velocities here.\n");
    if (has_layer_in_xyz == 0)
        printf("Do not specify layer indices here.\n");
    else
        printf("Specify layer indices here.\n");
    if (num_of_grouping_methods == 0)
        printf("Have no grouping method.\n");
    else if (num_of_grouping_methods > 0 && num_of_grouping_methods <= 2)
        printf("Have %d grouping method(s).\n", num_of_grouping_methods);
    else
        print_error("Number of grouping methods should be 1 or 2.\n");
}  


void Atom::read_xyz_in_line_2(FILE* fid_xyz)
{
    if (box.triclinic == 1)
    {
        MY_MALLOC(box.cpu_h, real, 18);
        double ax, ay, az, bx, by, bz, cx, cy, cz;
        int count = fscanf(fid_xyz, "%d%d%d",
            &box.pbc_x, &box.pbc_y, &box.pbc_z);
        if (count != 3) print_error("reading error for xyz.in.\n");
        count = fscanf(fid_xyz, "%lf%lf%lf", &ax, &ay, &az);
        if (count != 3) print_error("reading error for xyz.in.\n");
        count = fscanf(fid_xyz, "%lf%lf%lf", &bx, &by, &bz);
        if (count != 3) print_error("reading error for xyz.in.\n");
        count = fscanf(fid_xyz, "%lf%lf%lf", &cx, &cy, &cz);
        if (count != 3) print_error("reading error for xyz.in.\n");
        box.cpu_h[0] = ax; box.cpu_h[1] = ay; box.cpu_h[2] = az;
        box.cpu_h[3] = bx; box.cpu_h[4] = by; box.cpu_h[5] = bz;
        box.cpu_h[6] = cx; box.cpu_h[7] = cy; box.cpu_h[8] = cz;
        box.get_inverse();
    }
    else
    {
        MY_MALLOC(box.cpu_h, real, 6);
        double lx, ly, lz;
        int count = fscanf(fid_xyz, "%d%d%d%lf%lf%lf",
            &box.pbc_x, &box.pbc_y, &box.pbc_z, &lx, &ly, &lz);
        if (count != 6) print_error("reading error for line 2 of xyz.in.\n");
        box.cpu_h[0] = lx; box.cpu_h[1] = ly; box.cpu_h[2] = lz;
        box.cpu_h[3] = lx*0.5; box.cpu_h[4] = ly*0.5; box.cpu_h[5] = lz*0.5;
    }

    if (box.pbc_x == 1)
        printf("Use periodic boundary conditions along x.\n");
    else if (box.pbc_x == 0)
        printf("Use     free boundary conditions along x.\n");
    else
        print_error("invalid boundary conditions along x.\n");
    if (box.pbc_y == 1)
        printf("Use periodic boundary conditions along y.\n");
    else if (box.pbc_y == 0)
        printf("Use     free boundary conditions along y.\n");
    else
        print_error("invalid boundary conditions along y.\n");
    if (box.pbc_z == 1)
        printf("Use periodic boundary conditions along z.\n");
    else if (box.pbc_z == 0)
        printf("Use     free boundary conditions along z.\n");
    else
        print_error("invalid boundary conditions along z.\n");
}


void Atom::read_xyz_in_line_3(FILE* fid_xyz)
{
    MY_MALLOC(cpu_type, int, N);
    MY_MALLOC(cpu_type_local, int, N);
    MY_MALLOC(cpu_mass, real, N);
    MY_MALLOC(cpu_x, real, N);
    MY_MALLOC(cpu_y, real, N);
    MY_MALLOC(cpu_z, real, N);
    MY_MALLOC(cpu_vx, real, N);
    MY_MALLOC(cpu_vy, real, N);
    MY_MALLOC(cpu_vz, real, N);
    number_of_types = -1;
    for (int m = 0; m < num_of_grouping_methods; ++m)
    {
        MY_MALLOC(group[m].cpu_label, int, N);
        group[m].number = -1;
    }
    for (int n = 0; n < N; n++)
    {
        double mass, x, y, z;
        int count = fscanf(fid_xyz, "%d%lf%lf%lf%lf", 
            &(cpu_type[n]), &x, &y, &z, &mass);
        if (count != 5) { print_error("reading error for xyz.in.\n"); }
        cpu_mass[n] = mass; cpu_x[n] = x; cpu_y[n] = y; cpu_z[n] = z;
        cpu_type_local[n] = cpu_type[n];
        if (cpu_type[n] > number_of_types) { number_of_types = cpu_type[n]; }
        if (has_velocity_in_xyz)
        {
            double vx, vy, vz;
            count = fscanf(fid_xyz, "%lf%lf%lf", &vx, &vy, &vz);
            if (count != 3) { print_error("reading error for xyz.in.\n"); }
            cpu_vx[n] = vx; cpu_vy[n] = vy; cpu_vz[n] = vz;
        }
        if (has_layer_in_xyz)
        {
            count = fscanf(fid_xyz, "%d", &cpu_layer_label[n]);
            if (count != 1) { print_error("reading error for xyz.in.\n"); }
        }
        for (int m = 0; m < num_of_grouping_methods; ++m)
        {
            count = fscanf(fid_xyz, "%d", &group[m].cpu_label[n]);
            if (count != 1) { print_error("reading error for xyz.in.\n"); }
            if (group[m].cpu_label[n] > group[m].number)
            {
                group[m].number = group[m].cpu_label[n];
            }
        }
    }
    for (int m = 0; m < num_of_grouping_methods; ++m) { group[m].number++; }
    number_of_types++;
}


void Atom::find_group_size(int k)
{
    MY_MALLOC(group[k].cpu_size, int, group[k].number);
    MY_MALLOC(group[k].cpu_size_sum, int, group[k].number);
    MY_MALLOC(group[k].cpu_contents, int, N);
    if (group[k].number == 1)
        printf("There is only one group of atoms in grouping method %d.\n", k);
    else
        printf("There are %d groups of atoms in grouping method %d.\n",
            group[k].number, k);
    for (int m = 0; m < group[k].number; m++)
    {
        group[k].cpu_size[m] = 0;
        group[k].cpu_size_sum[m] = 0;
    }
    for (int n = 0; n < N; n++) group[k].cpu_size[group[k].cpu_label[n]]++;
    for (int m = 0; m < group[k].number; m++)
        printf("    %d atoms in group %d.\n", group[k].cpu_size[m], m);   
    for (int m = 1; m < group[k].number; m++)
        for (int n = 0; n < m; n++)
            group[k].cpu_size_sum[m] += group[k].cpu_size[n];
}


void Atom::find_group_contents(int k)
{
    // determine the atom indices from the first to the last group
    int *offset; MY_MALLOC(offset, int, group[k].number);
    for (int m = 0; m < group[k].number; m++) offset[m] = 0;
    for (int n = 0; n < N; n++) 
        for (int m = 0; m < group[k].number; m++)
            if (group[k].cpu_label[n] == m)
                group[k].cpu_contents[group[k].cpu_size_sum[m]+offset[m]++] = n;
    MY_FREE(offset);
}


void Atom::find_type_size(void)
{
    MY_MALLOC(cpu_type_size, int, number_of_types);
    if (number_of_types == 1)
        printf("There is only one atom type.\n");
    else
        printf("There are %d atom types.\n", number_of_types);
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
    for (int m = 0; m < num_of_grouping_methods; ++m)
    {
        find_group_size(m);
        find_group_contents(m);
    }
    find_type_size();
    print_line_1();
    printf("Finished initializing positions and related parameters.\n");
    print_line_2();
}


void Atom::allocate_memory_gpu(void)
{
    int m1 = sizeof(int) * N;
    int m2 = m1 * neighbor.MN;
    int m4 = sizeof(real) * N;
    int m5 = m4 * NUM_OF_HEAT_COMPONENTS;
    CHECK(cudaMalloc((void**)&NN, m1));
    CHECK(cudaMalloc((void**)&NL, m2));
    CHECK(cudaMalloc((void**)&NN_local, m1));
    CHECK(cudaMalloc((void**)&NL_local, m2));
    CHECK(cudaMalloc((void**)&type, m1));
    CHECK(cudaMalloc((void**)&type_local, m1));
    for (int m = 0; m < num_of_grouping_methods; ++m)
    {
        int m3 = sizeof(int) * group[m].number;
        CHECK(cudaMalloc((void**)&group[m].label, m1));
        CHECK(cudaMalloc((void**)&group[m].size, m3));
        CHECK(cudaMalloc((void**)&group[m].size_sum, m3));
        CHECK(cudaMalloc((void**)&group[m].contents, m1));
    }
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
    CHECK(cudaMalloc((void**)&thermo, sizeof(real) * 6));
    box.allocate_memory_gpu();
}


void Atom::copy_from_cpu_to_gpu(void)
{
    int m1 = sizeof(int) * N;
    int m3 = sizeof(real) * N;
    CHECK(cudaMemcpy(type, cpu_type, m1, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(type_local, cpu_type, m1, cudaMemcpyHostToDevice));
    for (int m = 0; m < num_of_grouping_methods; ++m)
    {
        int m2 = sizeof(int) * group[m].number;
        CHECK(cudaMemcpy(group[m].label, group[m].cpu_label, m1,
            cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(group[m].size, group[m].cpu_size, m2,
            cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(group[m].size_sum, group[m].cpu_size_sum, m2,
            cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(group[m].contents, group[m].cpu_contents, m1,
            cudaMemcpyHostToDevice));
    }
    CHECK(cudaMemcpy(mass, cpu_mass, m3, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(x, cpu_x, m3, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(y, cpu_y, m3, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(z, cpu_z, m3, cudaMemcpyHostToDevice));
    box.copy_from_cpu_to_gpu();
}


void Atom::free_memory_cpu(void)
{
    MY_FREE(cpu_type);
    MY_FREE(cpu_type_local);
    for (int m = 0; m < num_of_grouping_methods; ++m)
    {
        MY_FREE(group[m].cpu_label);
        MY_FREE(group[m].cpu_size);
        MY_FREE(group[m].cpu_size_sum);
        MY_FREE(group[m].cpu_contents);
    }
    MY_FREE(cpu_type_size);
    MY_FREE(cpu_mass);
    MY_FREE(cpu_x);
    MY_FREE(cpu_y);
    MY_FREE(cpu_z);
    MY_FREE(cpu_vx);
    MY_FREE(cpu_vy);
    MY_FREE(cpu_vz);
    box.free_memory_cpu();
}


void Atom::free_memory_gpu(void)
{
    CHECK(cudaFree(NN));
    CHECK(cudaFree(NL));
    CHECK(cudaFree(NN_local));
    CHECK(cudaFree(NL_local));
    CHECK(cudaFree(type));
    CHECK(cudaFree(type_local));
    for (int m = 0; m < num_of_grouping_methods; ++m)
    {
        CHECK(cudaFree(group[m].label));
        CHECK(cudaFree(group[m].size));
        CHECK(cudaFree(group[m].size_sum));
        CHECK(cudaFree(group[m].contents));
    }
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
    CHECK(cudaFree(thermo));
    box.free_memory_gpu();
}


