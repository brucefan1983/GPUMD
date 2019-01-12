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




#include "gpumd.cuh"

#include "force.cuh"
#include "validate.cuh"
#include "integrate.cuh"
#include "ensemble.cuh"
#include "measure.cuh"
#include "atom.cuh"
#include "error.cuh"




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
        if (atom->neighbor.update) { atom->find_neighbor(0); }

        // set the current temperature;
        if (integrate->ensemble->type >= 1 && integrate->ensemble->type <= 20)
        {
            integrate->ensemble->temperature = atom->temperature1 
                + (atom->temperature2 - atom->temperature1)
                * real(step) / atom->number_of_steps;   
        }

        integrate->compute(atom, force, measure);
        measure->process(input_dir, atom, integrate, step);

        if (atom->number_of_steps >= 10)
        {
            if ((step + 1) % (atom->number_of_steps / 10) == 0)
            {
                printf("    %d steps completed.\n", step + 1);
            }
        }
    }

    print_line_1();
    clock_t time_finish = clock();
    real time_used = (time_finish - time_begin) / (real) CLOCKS_PER_SEC;
    printf("Time used for this run = %g s.\n", time_used);
    real run_speed = atom->N * (atom->number_of_steps / time_used);
    printf("Speed of this run = %g atom*step/second.\n", run_speed);
    print_line_2();

    measure->finalize(input_dir, atom, integrate);
    integrate->finalize();
}




#ifdef FORCE
static void print_initial_force(char* input_dir, Atom* atom)
{

    int m = sizeof(real) * atom->N;
    real *cpu_fx; MY_MALLOC(cpu_fx, real, atom->N);
    real *cpu_fy; MY_MALLOC(cpu_fy, real, atom->N);
    real *cpu_fz; MY_MALLOC(cpu_fz, real, atom->N);
    CHECK(cudaMemcpy(cpu_fx, atom->fx, m, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(cpu_fy, atom->fy, m, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(cpu_fz, atom->fz, m, cudaMemcpyDeviceToHost));
    char file_force[200];
    strcpy(file_force, input_dir);
    strcat(file_force, "/f.out");
    FILE *fid_force = my_fopen(file_force, "w");
    for (int n = 0; n < atom->N; n++)
    {
        fprintf(fid_force, "%20.10e%20.10e%20.10e\n", 
            cpu_fx[n], cpu_fy[n], cpu_fz[n]);
    }
    fflush(fid_force);
    fclose(fid_force);
    MY_FREE(cpu_fx);
    MY_FREE(cpu_fy);
    MY_FREE(cpu_fz);
}
#endif




// Read and process the inputs from the "run.in" file
void GPUMD::run
(
    char *input_dir, Atom *atom, Force *force, Integrate *integrate,
    Measure *measure
)
{
    char file_run[200];
    strcpy(file_run, input_dir);
    strcat(file_run, "/run.in");
    char *input = get_file_contents(file_run);
    char *input_ptr = input; // Keep the pointer in order to free later

    // Iterate the rows
    const int max_num_param = 10; // never use more than 9 parameters
    int num_param;
    char *param[max_num_param];

    initialize_run(atom, measure); // set some default values

    print_line_1();
    printf("Started executing the commands in run.in.\n");
    print_line_2();

    while (input_ptr)
    {
        // get one line from the input file
        input_ptr = row_find_param(input_ptr, param, &num_param);
        if (num_param == 0) { continue; } 

        // set default values
        int is_potential = 0;
        int is_velocity = 0;
        int is_run = 0;

        // parse a line of the input file 
        parse(param, num_param, atom, force, integrate, measure,
            &is_potential, &is_velocity, &is_run);

        // check for some special keywords
        if (is_potential)
        {
            force->initialize(input_dir, atom);
            force->compute(atom, measure);
#ifdef FORCE
            print_initial_force(input_dir, atom);
#endif
        }
        if (is_velocity) { atom->initialize_velocity(); }
        if (is_run)
        {
            process_run(input_dir, atom, force, integrate, measure);
            initialize_run(atom, measure); // change back to the default
        }
    }

    print_line_1();
    printf("Finished executing the commands in run.in.\n");
    print_line_2();

    MY_FREE(input); // Free the input file contents
}




