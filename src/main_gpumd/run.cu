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
#include "model/atom.cuh"
#include "model/neighbor.cuh"
#include "model/box.cuh"
#include "utilities/read_file.cuh"
#include "utilities/error.cuh"


Run::Run
(
    char* input_dir, Atom* atom, Force* force,
    Integrate* integrate, Measure* measure
)
{
    run(input_dir, atom, force, integrate, measure, 1);
    force->initialize_participation_and_shift
    (
        atom->group,
        atom->number_of_types
    );
    run(input_dir, atom, force, integrate, measure, 0);
}


Run::~Run(void)
{
    // nothing
}


// set some default values after each run
void Run::initialize_run(Atom* atom, Integrate* integrate, Measure* measure)
{
    atom->neighbor.update = 0;
    atom->neighbor.number_of_updates = 0;
    integrate->fixed_group = -1; // no group has an index of -1
    integrate->deform_x = 0;
    integrate->deform_y = 0;
    integrate->deform_z = 0;
    measure->compute.compute_temperature  = 0;
    measure->compute.compute_potential    = 0;
    measure->compute.compute_force        = 0;
    measure->compute.compute_virial       = 0;
    measure->compute.compute_jp           = 0;
    measure->compute.compute_jk           = 0;
    measure->shc.compute    = 0;
    measure->vac.compute_dos= 0;
    measure->vac.compute_sdc= 0;
    measure->modal_analysis.compute   = 0;
    measure->modal_analysis.method   = NO_METHOD;
    measure->vac.grouping_method = -1;
    measure->vac.group = -1;
    measure->vac.num_dos_points = -1;
    measure->hac.compute    = 0;
    measure->hnemd.compute  = 0;
    measure->dump_thermo    = 0;
    measure->dump_velocity  = 0;
    measure->dump_restart   = 0;

    /*
     * Delete dump_pos if it exists. Ensure that dump_pos is NULL in case
     * it isn't set in parse. If we don't set to NULL, then we may end up
     * deleting some random address, corrupting memory.
     */
    if (measure->dump_pos)
    {
    	delete measure->dump_pos;
    }
    measure->dump_pos = NULL;

}


void Run::print_velocity_and_potential_error(void)
{
    if (0 == number_of_times_potential)
    {
        PRINT_INPUT_ERROR("No 'potential' keyword before run.");
    }

    if (0 == number_of_times_velocity)
    {
        PRINT_INPUT_ERROR("No 'velocity' keyword before run.");
    }
}


void Run::print_velocity_error(void)
{
    if (1 < number_of_times_velocity)
    {
        PRINT_INPUT_ERROR("Multiple 'velocity' keywords.");
    }
}


static void check_run_parameters
(Atom *atom, Integrate *integrate, Measure *measure)
{
    if (atom->box.triclinic == 1 && integrate->type == 11)
    {
        PRINT_INPUT_ERROR("Cannot use triclinic box with NPT ensemble.");
    }
}


static void print_finished_steps(int step, int number_of_steps)
{
    int base = (10 <= number_of_steps) ? (number_of_steps / 10) : 1;
    if (0 == (step + 1) % base)
    {
        printf("    %d steps completed.\n", step + 1);
    }
}


static void print_time_and_speed(clock_t time_begin, Atom* atom)
{
    print_line_1();
    clock_t time_finish = clock();
    double time_used = (time_finish - time_begin) / (double) CLOCKS_PER_SEC;
    printf("Number of neighbor list updates = %d.\n",
        atom->neighbor.number_of_updates);
    printf("Time used for this run = %g s.\n", time_used);
    double run_speed = atom->N * (atom->number_of_steps / time_used);
    printf("Speed of this run = %g atom*step/second.\n", run_speed);
    print_line_2();
}


// run a number of steps for a given set of inputs
static void process_run
(
    char *input_dir, Atom *atom, Force *force, Integrate *integrate,
    Measure *measure
)
{
    integrate->initialize(atom->N, atom->time_step, atom->group);

    measure->initialize
    (
        input_dir,
        atom->number_of_steps,
        atom->time_step,
        atom->group,
        atom->cpu_type_size,
        atom->mass
    );

    clock_t time_begin = clock();

    for (int step = 0; step < atom->number_of_steps; ++step)
    {
        atom->step = step;
        atom->global_time += atom->time_step;
		
#ifndef USE_FCP // the FCP does not use a neighbor list at all
        if (atom->neighbor.update)
        {
            atom->neighbor.find_neighbor
            (
                0,
                atom->box,
                atom->position_per_atom
            );
        }
#endif

        integrate->compute1
        (
            atom->time_step,
            double(step) / atom->number_of_steps,
            atom->group,
            atom->mass,
            atom->potential_per_atom,
            atom->force_per_atom,
            atom->virial_per_atom,
            atom->box,
            atom->position_per_atom,
            atom->velocity_per_atom,
            atom->thermo
        );

        force->compute
        (
            atom->box,
            atom->position_per_atom,
            atom->type,
            atom->group,
            atom->neighbor,
            atom->potential_per_atom,
            atom->force_per_atom,
            atom->virial_per_atom
        );

        integrate->compute2
        (
            atom->time_step,
            double(step) / atom->number_of_steps,
            atom->group,
            atom->mass,
            atom->potential_per_atom,
            atom->force_per_atom,
            atom->virial_per_atom,
            atom->box,
            atom->position_per_atom,
            atom->velocity_per_atom,
            atom->thermo
        );

        measure->process
        (
            input_dir,
            atom->number_of_steps,
            step,
            integrate->fixed_group,
            atom->global_time,
            integrate->temperature2,
            integrate->ensemble->energy_transferred,
            atom->cpu_type,
            atom->box,
            atom->neighbor,
            atom->group,
            atom->thermo,
            atom->mass,
            atom->cpu_mass,
            atom->position_per_atom,
            atom->cpu_position_per_atom,
            atom->velocity_per_atom,
            atom->cpu_velocity_per_atom,
            atom->potential_per_atom,
            atom->force_per_atom,
            atom->virial_per_atom,
            atom->heat_per_atom
        );
        print_finished_steps(step, atom->number_of_steps);
    }

    print_time_and_speed(time_begin, atom);

    measure->finalize
    (
        input_dir,
        atom->number_of_steps,
        atom->time_step,
        integrate->temperature2,
        atom->box.get_volume()
    );

    integrate->finalize();
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


// do something when the keyword is "potential"
void Run::add_potential
(
    char* input_dir, int check, Atom* atom, Force* force, Measure* measure
)
{
    if (!is_potential) { return; }
    if (check) { number_of_times_potential++; }
    else
    {
        force->add_potential
        (
            input_dir,
            atom->box,
            atom->neighbor,
            atom->group,
            atom->cpu_type,
            atom->cpu_type_size
        );
    }
}


// do something when the keyword is "velocity"
void Run::check_velocity(int check, Atom* atom)
{
    if (!is_velocity) { return; }
    if (check)
    {
        number_of_times_velocity++;
    }
    else
    {
        Velocity velocity;
        velocity.initialize
        (
            atom->has_velocity_in_xyz,
            atom->initial_temperature,
            atom->cpu_mass,
            atom->cpu_position_per_atom,
            atom->cpu_velocity_per_atom,
            atom->velocity_per_atom
        );
    }
}


// do something when the keyword is "run"
void Run::check_run
(
    char* input_dir, int check, Atom* atom,
    Force* force, Integrate* integrate, Measure* measure
)
{
    if (!is_run) { return; }
    if (check)
    {
        print_velocity_and_potential_error();
        check_run_parameters(atom, integrate, measure);
    }
    else
    {
        force->valdiate_potential_definitions();
        bool compute_hnemd = measure->hnemd.compute ||
            (
                measure->modal_analysis.compute &&
                measure->modal_analysis.method == HNEMA_METHOD
            );
        force->set_hnemd_parameters
        (
            compute_hnemd, measure->hnemd.fe_x, measure->hnemd.fe_y, 
            measure->hnemd.fe_z
        );
        process_run(input_dir, atom, force, integrate, measure);
    }
    initialize_run(atom, integrate, measure);
}


// Read and process the inputs from the "run.in" file
void Run::run
(
    char *input_dir, Atom *atom, Force *force, Integrate *integrate,
    Measure *measure, int check
)
{
    char file_run[200];
    strcpy(file_run, input_dir);
    strcat(file_run, "/run.in");
    char *input = get_file_contents(file_run);
    char *input_ptr = input; // Keep the pointer in order to free later
    const int max_num_param = 10; // never use more than 9 parameters
    int num_param;
    char *param[max_num_param];

    force->num_of_potentials = 0;

    initialize_run(atom, integrate, measure); // set some default values

    print_start(check);

    while (input_ptr)
    {
        input_ptr = row_find_param(input_ptr, param, &num_param);

        if (num_param == 0) { continue; } 

        is_potential = false;
        is_velocity = false;
        is_run = false;

        parse(param, num_param, atom, force, integrate, measure);
        add_potential(input_dir, check, atom, force, measure);
        check_velocity(check, atom);
        check_run(input_dir, check, atom, force, integrate, measure);
    }

    print_velocity_error();

    print_finish(check);

    free(input); // Free the input file contents
}


void Run::parse
(
    char **param, int num_param, Atom* atom,
    Force *force, Integrate *integrate, Measure *measure
)
{
    if (strcmp(param[0], "potential_definition") == 0)
    {
        force->parse_potential_definition(param, num_param);
    }
    else if (strcmp(param[0], "potential") == 0)
    {
        is_potential = true;
        force->parse_potential(param, num_param);
    }
    else if (strcmp(param[0], "velocity") == 0)
    {
        is_velocity = true;
        atom->parse_velocity(param, num_param);
    }
    else if (strcmp(param[0], "ensemble") == 0)
    {
        integrate->parse_ensemble(param, num_param, atom->group);
    }
    else if (strcmp(param[0], "time_step") == 0)
    {
        atom->parse_time_step(param, num_param);
    }
    else if (strcmp(param[0], "neighbor") == 0)
    {
        atom->parse_neighbor(param, num_param, force->rc_max);
    }
    else if (strcmp(param[0], "dump_thermo") == 0)
    {
        measure->parse_dump_thermo(param, num_param);
    }
    else if (strcmp(param[0], "dump_position") == 0)
    {
        measure->parse_dump_position(param, num_param);
    }
    else if (strcmp(param[0], "dump_restart") == 0)
    {
        measure->parse_dump_restart(param, num_param);
    }
    else if (strcmp(param[0], "dump_velocity") == 0)
    {
        measure->parse_dump_velocity(param, num_param);
    }
    else if (strcmp(param[0], "compute_dos") == 0)
    {
        measure->parse_compute_dos(param, num_param, atom->group.data());
    }
    else if (strcmp(param[0], "compute_sdc") == 0)
    {
        measure->parse_compute_sdc(param, num_param, atom->group.data());
    }
    else if (strcmp(param[0], "compute_hac") == 0)
    {
        measure->parse_compute_hac(param, num_param);
    }
    else if (strcmp(param[0], "compute_hnemd") == 0)
    {
        measure->parse_compute_hnemd(param, num_param);
    }
    else if (strcmp(param[0], "compute_shc") == 0)
    {
        measure->parse_compute_shc(param, num_param, atom->group);
    }
    else if (strcmp(param[0], "compute_gkma") == 0)
    {
        measure->parse_compute_gkma(param, num_param, atom->number_of_types);
    }
    else if (strcmp(param[0], "compute_hnema") == 0)
    {
        measure->parse_compute_hnema(param, num_param, atom->number_of_types);
    }
    else if (strcmp(param[0], "deform") == 0)
    {
        integrate->parse_deform(param, num_param);
    }
    else if (strcmp(param[0], "compute") == 0)
    {
        measure->parse_compute(param, num_param, atom->group);
    }
    else if (strcmp(param[0], "fix") == 0)
    {
        integrate->parse_fix(param, num_param, atom->group);
    }
    else if (strcmp(param[0], "run") == 0)
    {
        is_run = true;
        atom->parse_run(param, num_param);
    }
    else
    {
        PRINT_KEYWORD_ERROR(param[0]);
    }
}


