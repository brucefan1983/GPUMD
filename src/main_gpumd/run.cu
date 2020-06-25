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
Run simulation according to the inputs in the run.in file.
------------------------------------------------------------------------------*/


#include "run.cuh"
#include "velocity.cuh"
#include "force/force.cuh"
#include "integrate/integrate.cuh"
#include "integrate/ensemble.cuh"
#include "measure/measure.cuh"
#include "model/read_xyz.cuh"
#include "model/neighbor.cuh"
#include "model/box.cuh"
#include "utilities/read_file.cuh"
#include "utilities/error.cuh"


Run::Run(char* input_dir)
{
    initialize_position
    (
        input_dir,
        N,
        has_velocity_in_xyz,
        number_of_types,
        box,
        neighbor,
        group,
        cpu_type,
        cpu_type_size,
        cpu_mass,
        cpu_position_per_atom,
        cpu_velocity_per_atom
    );

    allocate_memory_gpu
    (
        N,
        neighbor,
        group,
        cpu_type,
        cpu_mass,
        cpu_position_per_atom,
        type,
        mass,
        position_per_atom,
        velocity_per_atom,
        potential_per_atom,
        force_per_atom,
        virial_per_atom,
        heat_per_atom,
        thermo
    );

#ifndef USE_FCP // the FCP does not use a neighbor list at all
    neighbor.find_neighbor
    (
        1,
        box,
        position_per_atom
    );
#endif

    execute_run_in(input_dir);
}


// set some default values after each run
void Run::initialize_run()
{
    neighbor.update = 0;
    neighbor.number_of_updates = 0;
    integrate.fixed_group = -1; // no group has an index of -1
    integrate.deform_x = 0;
    integrate.deform_y = 0;
    integrate.deform_z = 0;
    measure.compute.compute_temperature  = 0;
    measure.compute.compute_potential    = 0;
    measure.compute.compute_force        = 0;
    measure.compute.compute_virial       = 0;
    measure.compute.compute_jp           = 0;
    measure.compute.compute_jk           = 0;
    measure.shc.compute    = 0;
    measure.vac.compute_dos= 0;
    measure.vac.compute_sdc= 0;
    measure.modal_analysis.compute   = 0;
    measure.modal_analysis.method   = NO_METHOD;
    measure.vac.grouping_method = -1;
    measure.vac.group = -1;
    measure.vac.num_dos_points = -1;
    measure.hac.compute    = 0;
    measure.hnemd.compute  = 0;
    measure.dump_thermo    = 0;
    measure.dump_velocity  = 0;
    measure.dump_restart   = 0;

    /*
     * Delete dump_pos if it exists. Ensure that dump_pos is NULL in case
     * it isn't set in parse. If we don't set to NULL, then we may end up
     * deleting some random address, corrupting memory.
     */
    if (measure.dump_pos)
    {
    	delete measure.dump_pos;
    }
    measure.dump_pos = NULL;
}


// run a number of steps for a given set of inputs
void Run::process_run(char *input_dir)
{
    integrate.initialize(N, time_step, group);

    measure.initialize
    (
        input_dir,
        number_of_steps,
        time_step,
        group,
        cpu_type_size,
        mass
    );

    clock_t time_begin = clock();

    for (int step = 0; step < number_of_steps; ++step)
    {
        global_time += time_step;
		
#ifndef USE_FCP // the FCP does not use a neighbor list at all
        if (neighbor.update)
        {
            neighbor.find_neighbor
            (
                0,
                box,
                position_per_atom
            );
        }
#endif

        integrate.compute1
        (
            time_step,
            double(step) / number_of_steps,
            group,
            mass,
            potential_per_atom,
            force_per_atom,
            virial_per_atom,
            box,
            position_per_atom,
            velocity_per_atom,
            thermo
        );

        force.compute
        (
            box,
            position_per_atom,
            type,
            group,
            neighbor,
            potential_per_atom,
            force_per_atom,
            virial_per_atom
        );

        integrate.compute2
        (
            time_step,
            double(step) / number_of_steps,
            group,
            mass,
            potential_per_atom,
            force_per_atom,
            virial_per_atom,
            box,
            position_per_atom,
            velocity_per_atom,
            thermo
        );

        measure.process
        (
            input_dir,
            number_of_steps,
            step,
            integrate.fixed_group,
            global_time,
            integrate.temperature2,
            integrate.ensemble->energy_transferred,
            cpu_type,
            box,
            neighbor,
            group,
            thermo,
            mass,
            cpu_mass,
            position_per_atom,
            cpu_position_per_atom,
            velocity_per_atom,
            cpu_velocity_per_atom,
            potential_per_atom,
            force_per_atom,
            virial_per_atom,
            heat_per_atom
        );

        int base = (10 <= number_of_steps) ? (number_of_steps / 10) : 1;
        if (0 == (step + 1) % base)
        {
            printf("    %d steps completed.\n", step + 1);
        }
    }

    print_line_1();
    clock_t time_finish = clock();
    double time_used = (time_finish - time_begin) / (double) CLOCKS_PER_SEC;
    printf("Number of neighbor list updates = %d.\n",
        neighbor.number_of_updates);
    printf("Time used for this run = %g s.\n", time_used);
    double run_speed = N * (number_of_steps / time_used);
    printf("Speed of this run = %g atom*step/second.\n", run_speed);
    print_line_2();

    measure.finalize
    (
        input_dir,
        number_of_steps,
        time_step,
        integrate.temperature2,
        box.get_volume()
    );
}


static void print_start(int check)
{
    print_line_1();
    if (check) { printf("Started checking the inputs in run.in.\n"); }
    else { printf("Started executing the commands in run.in.\n"); }
    print_line_2();
}


static void print_finish(int check)
{
    print_line_1();
    if (check) { printf("Finished checking the inputs in run.in.\n"); }
    else { printf("Finished executing the commands in run.in.\n"); }
    print_line_2();
}


void Run::execute_run_in(char* input_dir)
{
    char file_run[200];
    strcpy(file_run, input_dir);
    strcat(file_run, "/run.in");
    char *input = get_file_contents(file_run);
    char *input_ptr = input; // Keep the pointer in order to free later
    const int max_num_param = 10; // never use more than 9 parameters
    int num_param;
    char *param[max_num_param];

    force.initialize_participation_and_shift(group, number_of_types);
    force.num_of_potentials = 0;

    initialize_run();

    print_start(false);

    while (input_ptr)
    {
        input_ptr = row_find_param(input_ptr, param, &num_param);

        if (num_param == 0) { continue; }
        is_potential_definition = false;
        is_potential = false;
        is_velocity = false;
        is_run = false;

        parse_one_keyword(param, num_param);
		
        if (is_potential_definition)
        {
            force.initialize_participation_and_shift(group, number_of_types);
        }

        if (is_potential)
        {
            force.add_potential
            (
                input_dir,
                box,
                neighbor,
                group,
                cpu_type,
                cpu_type_size
            );
        }

        if (is_velocity)
        {
            Velocity velocity;
            velocity.initialize
            (
                has_velocity_in_xyz,
                initial_temperature,
                cpu_mass,
                cpu_position_per_atom,
                cpu_velocity_per_atom,
                velocity_per_atom
            );
        }

        if (is_run)
        {
            force.valdiate_potential_definitions();
            bool compute_hnemd = measure.hnemd.compute ||
            (
                measure.modal_analysis.compute &&
                measure.modal_analysis.method == HNEMA_METHOD
            );
            force.set_hnemd_parameters
            (
                compute_hnemd, measure.hnemd.fe_x, measure.hnemd.fe_y,
                measure.hnemd.fe_z
            );
            process_run(input_dir);
            initialize_run();
        }
    }

    print_finish(false);

    free(input); // Free the input file contents
}


void Run::parse_one_keyword(char** param, int num_param)
{
    if (strcmp(param[0], "potential_definition") == 0)
    {
        is_potential_definition = true;
        force.parse_potential_definition(param, num_param);
    }
    else if (strcmp(param[0], "potential") == 0)
    {
        is_potential = true;
        force.parse_potential(param, num_param);
    }
    else if (strcmp(param[0], "velocity") == 0)
    {
        is_velocity = true;
        parse_velocity(param, num_param);
    }
    else if (strcmp(param[0], "ensemble") == 0)
    {
        integrate.parse_ensemble(param, num_param, group);
    }
    else if (strcmp(param[0], "time_step") == 0)
    {
        parse_time_step(param, num_param);
    }
    else if (strcmp(param[0], "neighbor") == 0)
    {
        parse_neighbor(param, num_param);
    }
    else if (strcmp(param[0], "dump_thermo") == 0)
    {
        measure.parse_dump_thermo(param, num_param);
    }
    else if (strcmp(param[0], "dump_position") == 0)
    {
        measure.parse_dump_position(param, num_param);
    }
    else if (strcmp(param[0], "dump_restart") == 0)
    {
        measure.parse_dump_restart(param, num_param);
    }
    else if (strcmp(param[0], "dump_velocity") == 0)
    {
        measure.parse_dump_velocity(param, num_param);
    }
    else if (strcmp(param[0], "compute_dos") == 0)
    {
        measure.parse_compute_dos(param, num_param, group.data());
    }
    else if (strcmp(param[0], "compute_sdc") == 0)
    {
        measure.parse_compute_sdc(param, num_param, group.data());
    }
    else if (strcmp(param[0], "compute_hac") == 0)
    {
        measure.parse_compute_hac(param, num_param);
    }
    else if (strcmp(param[0], "compute_hnemd") == 0)
    {
        measure.parse_compute_hnemd(param, num_param);
    }
    else if (strcmp(param[0], "compute_shc") == 0)
    {
        measure.parse_compute_shc(param, num_param, group);
    }
    else if (strcmp(param[0], "compute_gkma") == 0)
    {
        measure.parse_compute_gkma(param, num_param, number_of_types);
    }
    else if (strcmp(param[0], "compute_hnema") == 0)
    {
        measure.parse_compute_hnema(param, num_param, number_of_types);
    }
    else if (strcmp(param[0], "deform") == 0)
    {
        integrate.parse_deform(param, num_param);
    }
    else if (strcmp(param[0], "compute") == 0)
    {
        measure.parse_compute(param, num_param, group);
    }
    else if (strcmp(param[0], "fix") == 0)
    {
        integrate.parse_fix(param, num_param, group);
    }
    else if (strcmp(param[0], "run") == 0)
    {
        is_run = true;
        parse_run(param, num_param);
    }
    else
    {
        PRINT_KEYWORD_ERROR(param[0]);
    }
}


void Run::parse_velocity(char **param, int num_param)
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


void Run::parse_time_step (char **param, int num_param)
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


void Run::parse_run(char **param, int num_param)
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


void Run::parse_neighbor(char **param, int num_param)
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
    neighbor.rc = force.rc_max + neighbor.skin;
}


