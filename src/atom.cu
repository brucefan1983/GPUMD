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
#include "read_file.cuh"
#include <vector>

const int NUM_OF_HEAT_COMPONENTS = 5;


Atom::Atom(char *input_dir)
{ 
    initialize_position(input_dir);
    allocate_memory_gpu();
    copy_from_cpu_to_gpu();
#ifndef USE_FCP // the FCP does not use a neighbor list at all
    find_neighbor(1);
#endif
}


Atom::~Atom(void)
{
    free_memory_gpu();
}


void Atom::read_xyz_in_line_1(FILE* fid_xyz)
{
    double rc;
    int count = fscanf
    (
        fid_xyz, "%d%d%lf%d%d%d\n", &N, &neighbor.MN, &rc, &box.triclinic, 
        &has_velocity_in_xyz, &num_of_grouping_methods
    );
    PRINT_SCANF_ERROR(count, 6, "Reading error for line 1 of xyz.in.");
    neighbor.rc = rc;

    if (N < 2)
    {
        PRINT_INPUT_ERROR("Number of atoms should >= 2.");
    }
    else
    {
        printf("Number of atoms is %d.\n", N);
    }

    if (neighbor.MN < 1)
    {
        PRINT_INPUT_ERROR("Maximum number of neighbors should >= 1.");
    }
    else if (neighbor.MN > 1024)
    {
        PRINT_INPUT_ERROR("Maximum number of neighbors should <= 1024.");
    }
    else
    {
        printf("Maximum number of neighbors is %d.\n", neighbor.MN);
    }

    if (neighbor.rc <= 0)
    {
        PRINT_INPUT_ERROR("Initial cutoff for neighbor list should > 0.");
    }
    else
    {
        printf("Initial cutoff for neighbor list is %g A.\n", neighbor.rc);
    }

    if (box.triclinic == 0)
    {
        printf("Use orthogonal box.\n");
    }
    else if (box.triclinic == 1)
    {
        printf("Use triclinic box.\n");
    }
    else
    {
        PRINT_INPUT_ERROR("Invalid box type.");
    }

    if (has_velocity_in_xyz == 0)
    {
        printf("Do not specify initial velocities here.\n");
    }
    else if (has_velocity_in_xyz == 1)
    {
        printf("Specify initial velocities here.\n");
    }
    else
    {
        PRINT_INPUT_ERROR("Invalid has_velocity flag.");
    }

    if (num_of_grouping_methods == 0)
    {
        printf("Have no grouping method.\n");
    }
    else if (num_of_grouping_methods > 0 && num_of_grouping_methods <= 10)
    {
        printf("Have %d grouping method(s).\n", num_of_grouping_methods);
    }
    else
    {
        PRINT_INPUT_ERROR("Number of grouping methods should be 1 to 10.");
    }
}  


void Atom::read_xyz_in_line_2(FILE* fid_xyz)
{
    if (box.triclinic == 1)
    {
        double ax, ay, az, bx, by, bz, cx, cy, cz;
        int count = fscanf
        (
            fid_xyz, "%d%d%d%lf%lf%lf%lf%lf%lf%lf%lf%lf",
            &box.pbc_x, &box.pbc_y, &box.pbc_z, &ax, &ay, &az, &bx, &by, &bz,
            &cx, &cy, &cz
        );
        PRINT_SCANF_ERROR(count, 12, "Reading error for line 2 of xyz.in.");

        box.cpu_h[0] = ax; box.cpu_h[3] = ay; box.cpu_h[6] = az;
        box.cpu_h[1] = bx; box.cpu_h[4] = by; box.cpu_h[7] = bz;
        box.cpu_h[2] = cx; box.cpu_h[5] = cy; box.cpu_h[8] = cz;
        box.get_inverse();

        printf("Box matrix h = [a, b, c] is\n");
        for (int d1 = 0; d1 < 3; ++d1)
        {
            for (int d2 = 0; d2 < 3; ++d2)
            {
                printf ("%20.10e", box.cpu_h[d1 * 3 + d2]);
            }
            printf("\n");
        }

        printf("Inverse box matrix g = inv(h) is\n");
        for (int d1 = 0; d1 < 3; ++d1)
        {
            for (int d2 = 0; d2 < 3; ++d2)
            {
                printf ("%20.10e", box.cpu_h[9 + d1 * 3 + d2]);
            }
            printf("\n");
        }
    }
    else
    {
        double lx, ly, lz;
        int count = fscanf
        (
            fid_xyz, "%d%d%d%lf%lf%lf",
            &box.pbc_x, &box.pbc_y, &box.pbc_z, &lx, &ly, &lz
        );
        PRINT_SCANF_ERROR(count, 6, "Reading error for line 2 of xyz.in.");

        if (lx < 0) { PRINT_INPUT_ERROR("Box length in x direction < 0."); }
        if (ly < 0) { PRINT_INPUT_ERROR("Box length in y direction < 0."); }
        if (lz < 0) { PRINT_INPUT_ERROR("Box length in z direction < 0."); }

        box.cpu_h[0] = lx; box.cpu_h[1] = ly; box.cpu_h[2] = lz;
        box.cpu_h[3] = lx*0.5; box.cpu_h[4] = ly*0.5; box.cpu_h[5] = lz*0.5;

        printf("Box lengths (lx, ly, lz) are\n");
        printf("%20.10e%20.10e%20.10e\n", lx, ly, lz);
    }

    if (box.pbc_x == 1)
    {
        printf("Use periodic boundary conditions along x.\n");
    }
    else if (box.pbc_x == 0)
    {
        printf("Use     free boundary conditions along x.\n");
    }
    else
    {
        PRINT_INPUT_ERROR("Invalid boundary conditions along x.");
    }

    if (box.pbc_y == 1)
    {
        printf("Use periodic boundary conditions along y.\n");
    }
    else if (box.pbc_y == 0)
    {
        printf("Use     free boundary conditions along y.\n");
    }
    else
    {
        PRINT_INPUT_ERROR("Invalid boundary conditions along y.");
    }

    if (box.pbc_z == 1)
    {
        printf("Use periodic boundary conditions along z.\n");
    }
    else if (box.pbc_z == 0)
    {
        printf("Use     free boundary conditions along z.\n");
    }
    else
    {
        PRINT_INPUT_ERROR("Invalid boundary conditions along z.");
    }
}


void Atom::read_xyz_in_line_3(FILE* fid_xyz)
{
    cpu_type.resize(N);
    cpu_mass.resize(N);
    cpu_x.resize(N);
    cpu_y.resize(N);
    cpu_z.resize(N);
    cpu_vx.resize(N);
    cpu_vy.resize(N);
    cpu_vz.resize(N);
    number_of_types = -1;

    for (int m = 0; m < num_of_grouping_methods; ++m)
    {
        group[m].cpu_label.resize(N);
        group[m].number = -1;
    }

    for (int n = 0; n < N; n++)
    {
        double mass, x, y, z;
        int count = fscanf
        (fid_xyz, "%d%lf%lf%lf%lf", &(cpu_type[n]), &x, &y, &z, &mass);
        PRINT_SCANF_ERROR(count, 5, "Reading error for xyz.in.");

        if (cpu_type[n] < 0 || cpu_type[n] >= N)
        {
            PRINT_INPUT_ERROR("Atom type should >= 0 and < N.");
        }

        if (mass <= 0)
        {
            PRINT_INPUT_ERROR("Atom mass should > 0.");
        }

        cpu_mass[n] = mass; cpu_x[n] = x; cpu_y[n] = y; cpu_z[n] = z;

        if (cpu_type[n] > number_of_types) { number_of_types = cpu_type[n]; }

        if (has_velocity_in_xyz)
        {
            double vx, vy, vz;
            count = fscanf(fid_xyz, "%lf%lf%lf", &vx, &vy, &vz);
            PRINT_SCANF_ERROR(count, 3, "Reading error for xyz.in.");
            cpu_vx[n] = vx; cpu_vy[n] = vy; cpu_vz[n] = vz;
        }

        for (int m = 0; m < num_of_grouping_methods; ++m)
        {
            count = fscanf(fid_xyz, "%d", &group[m].cpu_label[n]);
            PRINT_SCANF_ERROR(count, 1, "Reading error for xyz.in.");

            if (group[m].cpu_label[n] < 0 || group[m].cpu_label[n] >= N)
            {
                PRINT_INPUT_ERROR("Group label should >= 0 and < N.");
            }

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
    group[k].cpu_size.resize(group[k].number);
    group[k].cpu_size_sum.resize(group[k].number);
    group[k].cpu_contents.resize(N);

    if (group[k].number == 1)
    {
        printf("There is only one group of atoms in grouping method %d.\n", k);
    }
    else
    {
        printf
        (
            "There are %d groups of atoms in grouping method %d.\n",
            group[k].number, k
        );
    }

    for (int m = 0; m < group[k].number; m++)
    {
        group[k].cpu_size[m] = 0;
        group[k].cpu_size_sum[m] = 0;
    }

    for (int n = 0; n < N; n++) { group[k].cpu_size[group[k].cpu_label[n]]++; }

    for (int m = 0; m < group[k].number; m++)
    {
        printf("    %d atoms in group %d.\n", group[k].cpu_size[m], m);   
    }

    for (int m = 1; m < group[k].number; m++)
    {
        for (int n = 0; n < m; n++)
        {
            group[k].cpu_size_sum[m] += group[k].cpu_size[n];
        }
    }
}


// re-arrange the atoms from the first to the last group
void Atom::find_group_contents(int k)
{
    std::vector<int> offset(group[k].number);
    for (int m = 0; m < group[k].number; m++) { offset[m] = 0; }

    for (int n = 0; n < N; n++) 
    {
        for (int m = 0; m < group[k].number; m++)
        {
            if (group[k].cpu_label[n] == m)
            {
                group[k].cpu_contents[group[k].cpu_size_sum[m]+offset[m]++] = n;
            }
        }
    }
}


void Atom::find_type_size(void)
{
    cpu_type_size.resize(number_of_types);

    if (number_of_types == 1)
    {
        printf("There is only one atom type.\n");
    }
    else
    {
        printf("There are %d atom types.\n", number_of_types);
    }

    for (int m = 0; m < number_of_types; m++) { cpu_type_size[m] = 0; }
    for (int n = 0; n < N; n++) { cpu_type_size[cpu_type[n]]++; }
    for (int m = 0; m < number_of_types; m++)
    {
        printf("    %d atoms of type %d.\n", cpu_type_size[m], m);
    }
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
    int m4 = sizeof(double) * N;
    int m5 = m4 * NUM_OF_HEAT_COMPONENTS;
    CHECK(cudaMalloc((void**)&NN, m1));
    CHECK(cudaMalloc((void**)&NL, m2));
    CHECK(cudaMalloc((void**)&NN_local, m1));
    CHECK(cudaMalloc((void**)&NL_local, m2));

    CHECK(cudaMalloc((void**)&neighbor.cell_count, m1));
    CHECK(cudaMalloc((void**)&neighbor.cell_count_sum, m1));
    CHECK(cudaMalloc((void**)&neighbor.cell_contents, m1));

    CHECK(cudaMalloc((void**)&type, m1));
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
    CHECK(cudaMalloc((void**)&virial_per_atom,  m4 * 9));
    CHECK(cudaMalloc((void**)&potential_per_atom, m4));
    CHECK(cudaMalloc((void**)&heat_per_atom,      m5));
    CHECK(cudaMalloc((void**)&thermo, sizeof(double) * 6));
}


void Atom::copy_from_cpu_to_gpu(void)
{
    int m1 = sizeof(int) * N;
    int m3 = sizeof(double) * N;
    CHECK(cudaMemcpy(type, cpu_type.data(), m1, cudaMemcpyHostToDevice));
    for (int m = 0; m < num_of_grouping_methods; ++m)
    {
        int m2 = sizeof(int) * group[m].number;
        CHECK(cudaMemcpy(group[m].label, group[m].cpu_label.data(), m1,
            cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(group[m].size, group[m].cpu_size.data(), m2,
            cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(group[m].size_sum, group[m].cpu_size_sum.data(), m2,
            cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(group[m].contents, group[m].cpu_contents.data(), m1,
            cudaMemcpyHostToDevice));
    }
    CHECK(cudaMemcpy(mass, cpu_mass.data(), m3, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(x, cpu_x.data(), m3, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(y, cpu_y.data(), m3, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(z, cpu_z.data(), m3, cudaMemcpyHostToDevice));
}


void Atom::free_memory_gpu(void)
{
    CHECK(cudaFree(NN));
    CHECK(cudaFree(NL));
    CHECK(cudaFree(NN_local));
    CHECK(cudaFree(NL_local));

    CHECK(cudaFree(neighbor.cell_count));
    CHECK(cudaFree(neighbor.cell_count_sum));
    CHECK(cudaFree(neighbor.cell_contents));

    CHECK(cudaFree(type));
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
    CHECK(cudaFree(virial_per_atom));
    CHECK(cudaFree(potential_per_atom));
    CHECK(cudaFree(heat_per_atom));
    CHECK(cudaFree(thermo));
}


void Atom::parse_velocity(char **param, int num_param)
{
    if (num_param != 2)
    {
        PRINT_INPUT_ERROR("velocity should have 1 parameter.\n");
    }
    if (!is_valid_real(param[1], &initial_temperature))
    {
        PRINT_INPUT_ERROR("initial temperature should be a real number.\n");
    }
    if (initial_temperature <= 0.0)
    {
        PRINT_INPUT_ERROR("initial temperature should be a positive number.\n");
    }
}


void Atom::parse_time_step (char **param, int num_param)
{
    if (num_param != 2)
    {
        PRINT_INPUT_ERROR("time_step should have 1 parameter.\n");
    }
    if (!is_valid_real(param[1], &time_step))
    {
        PRINT_INPUT_ERROR("time_step should be a real number.\n");
    }
    printf("Time step for this run is %g fs.\n", time_step);
    time_step /= TIME_UNIT_CONVERSION;
}


void Atom::parse_run(char **param, int num_param)
{
    if (num_param != 2)
    {
        PRINT_INPUT_ERROR("run should have 1 parameter.\n");
    }
    if (!is_valid_int(param[1], &number_of_steps))
    {
        PRINT_INPUT_ERROR("number of steps should be an integer.\n");
    }
    printf("Run %d steps.\n", number_of_steps);
}


void Atom::parse_neighbor(char **param, int num_param, double force_rc_max)
{
    neighbor.update = 1;

    if (num_param != 2)
    {
        PRINT_INPUT_ERROR("neighbor should have 1 parameter.\n");
    }
    if (!is_valid_real(param[1], &neighbor.skin))
    {
        PRINT_INPUT_ERROR("neighbor list skin should be a number.\n");
    }
    printf("Build neighbor list with a skin of %g A.\n", neighbor.skin);

    // change the cutoff
    neighbor.rc = force_rc_max + neighbor.skin;
}


