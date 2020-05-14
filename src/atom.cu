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
#ifndef USE_FCP // the FCP does not use a neighbor list at all
    neighbor.find_neighbor(1, box, x.data(), y.data(), z.data());
#endif
}


void Atom::read_xyz_in_line_1(FILE* fid_xyz)
{
    int num_of_grouping_methods = 0;
    double rc;
    int count = fscanf
    (
        fid_xyz, "%d%d%lf%d%d%d\n", &N, &neighbor.MN, &rc, &box.triclinic, 
        &has_velocity_in_xyz, &num_of_grouping_methods
    );
    PRINT_SCANF_ERROR(count, 6, "Reading error for line 1 of xyz.in.");
    neighbor.rc = rc;
    group.resize(num_of_grouping_methods);

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

    for (int m = 0; m < group.size(); ++m)
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

        for (int m = 0; m < group.size(); ++m)
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

    for (int m = 0; m < group.size(); ++m) { group[m].number++; }

    number_of_types++;
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

    char file_xyz[200];
    strcpy(file_xyz, input_dir);
    strcat(file_xyz, "/xyz.in");
    FILE *fid_xyz = my_fopen(file_xyz, "r");

    read_xyz_in_line_1(fid_xyz);
    read_xyz_in_line_2(fid_xyz);
    read_xyz_in_line_3(fid_xyz);

    fclose(fid_xyz);

    for (int m = 0; m < group.size(); ++m)
    {
        group[m].find_size(N, m);
        group[m].find_contents(N);
    }

    find_type_size();

    print_line_1();
    printf("Finished initializing positions and related parameters.\n");
    print_line_2();
}


void Atom::allocate_memory_gpu(void)
{
    neighbor.NN.resize(N);
    neighbor.NL.resize(N * neighbor.MN);
    neighbor.NN_local.resize(N);
    neighbor.NL_local.resize(N * neighbor.MN);

    neighbor.cell_count.resize(N);
    neighbor.cell_count_sum.resize(N);
    neighbor.cell_contents.resize(N);

    type.resize(N);
    type.copy_from_host(cpu_type.data());
    for (int m = 0; m < group.size(); ++m)
    {
        group[m].label.resize(N);
        group[m].size.resize(group[m].number);
        group[m].size_sum.resize(group[m].number);
        group[m].contents.resize(N);
        group[m].label.copy_from_host(group[m].cpu_label.data());
        group[m].size.copy_from_host(group[m].cpu_size.data());
        group[m].size_sum.copy_from_host(group[m].cpu_size_sum.data());
        group[m].contents.copy_from_host(group[m].cpu_contents.data());
    }
    mass.resize(N);
    mass.copy_from_host(cpu_mass.data());
    neighbor.x0.resize(N);
    neighbor.y0.resize(N);
    neighbor.z0.resize(N);
    x.resize(N);
    y.resize(N);
    z.resize(N);
    x.copy_from_host(cpu_x.data());
    y.copy_from_host(cpu_y.data());
    z.copy_from_host(cpu_z.data());
    vx.resize(N);
    vy.resize(N);
    vz.resize(N);
    fx.resize(N);
    fy.resize(N);
    fz.resize(N);
    virial_per_atom.resize(N * 9);
    potential_per_atom.resize(N);
    heat_per_atom.resize(N * NUM_OF_HEAT_COMPONENTS);
    thermo.resize(6);
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


