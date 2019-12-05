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
#include "force.cuh"
#include "validate.cuh"
#include "integrate.cuh"
#include "ensemble.cuh"
#include "measure.cuh"
#include "atom.cuh"
#include "read_file.cuh"
#include "parse.cuh"
#include "error.cuh"


Run::Run
(
    char* input_dir, Atom* atom, Force* force,
    Integrate* integrate, Measure* measure
)
{
    run(input_dir, atom, force, integrate, measure, 1);

    if (force->group_method > -1)
        force->num_kind = atom->group[force->group_method].number;
    else
        force->num_kind = atom->number_of_types;

    // initialize bookkeeping data structures
    ZEROS(force->manybody_participation, int, force->num_kind);
    ZEROS(force->potential_participation, int, force->num_kind);
    ZEROS(atom->shift, int, MAX_NUM_OF_POTENTIALS);

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
    measure->gkma.compute   = 0;
    measure->hnema.compute  = 0;
    measure->vac.grouping_method = -1;
    measure->vac.group		= -1;
    measure->dos.num_dos_points = -1;
    measure->hac.compute    = 0;
    measure->hnemd.compute  = 0;
    measure->dump_thermo    = 0;
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
    real time_used = (time_finish - time_begin) / (real) CLOCKS_PER_SEC;
    printf("Number of neighbor list updates = %d.\n",
        atom->neighbor.number_of_updates);
    printf("Time used for this run = %g s.\n", time_used);
    real run_speed = atom->N * (atom->number_of_steps / time_used);
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
    integrate->initialize(atom);
    measure->initialize(input_dir, atom);
    clock_t time_begin = clock();

    for (int step = 0; step < atom->number_of_steps; ++step)
    {
        atom->step = step;
        atom->global_time += atom->time_step;

        if (atom->neighbor.update) { atom->find_neighbor(0); }

        integrate->compute(atom, force, measure);
        measure->process(input_dir, atom, integrate, step);
        print_finished_steps(step, atom->number_of_steps);
    }

    print_time_and_speed(time_begin, atom);
    measure->finalize(input_dir, atom, integrate);
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
        force->add_potential(input_dir, atom);
    }
}


// do something when the keyword is "velocity"
void Run::check_velocity(int check, Atom* atom)
{
    if (!is_velocity) { return; }
    if (check) { number_of_times_velocity++; }
    else { atom->initialize_velocity(); }
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

    MY_FREE(input); // Free the input file contents
}


void Run::parse
(
    char **param, int num_param, Atom* atom,
    Force *force, Integrate *integrate, Measure *measure
)
{
    if (strcmp(param[0], "potential_definition") == 0)
    {
        parse_potential_definition(param, num_param, atom, force);
    }
    else if (strcmp(param[0], "potential") == 0)
    {
        is_potential = true;
        parse_potential(param, num_param, force);
    }
    else if (strcmp(param[0], "velocity") == 0)
    {
        is_velocity = true;
        parse_velocity(param, num_param, atom);
    }
    else if (strcmp(param[0], "ensemble") == 0)
    {
        integrate->parse_ensemble(param, num_param, atom);
    }
    else if (strcmp(param[0], "time_step") == 0)
    {
        parse_time_step(param, num_param, atom);
    }
    else if (strcmp(param[0], "neighbor") == 0)
    {
        parse_neighbor(param, num_param, atom, force);
    }
    else if (strcmp(param[0], "dump_thermo") == 0)
    {
        measure->parse_dump_thermo(param, num_param);
    }
    else if (strcmp(param[0], "dump_position") == 0)
    {
        measure->parse_dump_position(param, num_param, atom);
    }
    else if (strcmp(param[0], "dump_restart") == 0)
    {
        measure->parse_dump_restart(param, num_param);
    }
    else if (strcmp(param[0], "compute_dos") == 0)
    {
        measure->parse_compute_dos(param, num_param, atom->group);
    }
    else if (strcmp(param[0], "compute_sdc") == 0)
    {
        measure->parse_compute_sdc(param, num_param, atom->group);
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
        measure->parse_compute_shc(param, num_param, atom);
    }
    else if (strcmp(param[0], "compute_gkma") == 0)
    {
        measure->parse_compute_gkma(param, num_param, atom);
    }
    else if (strcmp(param[0], "compute_hnema") == 0)
    {
        measure->parse_compute_hnema(param, num_param, atom);
    }
    else if (strcmp(param[0], "deform") == 0)
    {
        integrate->parse_deform(param, num_param);
    }
    else if (strcmp(param[0], "compute") == 0)
    {
        measure->parse_compute(param, num_param, atom);
    }
    else if (strcmp(param[0], "fix") == 0)
    {
        integrate->parse_fix(param, num_param, atom);
    }
    else if (strcmp(param[0], "run") == 0)
    {
        is_run = true;
        parse_run(param, num_param, atom);
    }
    else
    {
        PRINT_KEYWORD_ERROR(param[0]);
    }
}


