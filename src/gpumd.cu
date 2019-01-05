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




#include "gpumd.cuh"

#include "force.cuh"
#include "validate.cuh"
#include "integrate.cuh"
#include "ensemble.cuh"
#include "measure.cuh"
#include "atom.cuh"
#include "error.cuh"

#include <ctype.h>




GPUMD::GPUMD(char *input_dir)
{
    Atom        atom(input_dir);
    Force       force;
    Integrate   integrate;
    Measure     measure(input_dir);
    run(input_dir, &atom, &force, &integrate, &measure);
}




GPUMD::~GPUMD(void)
{
    // nothing
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
        if (atom->neighbor.update) { atom->find_neighbor(0); }

        // set the current temperature;
        if (integrate->ensemble->type >= 1 && integrate->ensemble->type <= 20)
        {
            integrate->ensemble->temperature = atom->temperature1 
                + (atom->temperature2 - atom->temperature1)
                * real(step) / atom->number_of_steps;   
        }

        integrate->compute(atom, force, measure);
        measure->compute(input_dir, atom, integrate, step);

        if (atom->number_of_steps >= 10)
        {
            if ((step + 1) % (atom->number_of_steps / 10) == 0)
            {
                printf("INFO:  %d steps completed.\n", step + 1);
            }
        }
    }

    printf("INFO:  This run is completed.\n\n");
    clock_t time_finish = clock();
    real time_used = (time_finish - time_begin) / (real) CLOCKS_PER_SEC;
    printf("INFO:  Time used for this run = %g s.\n", time_used);
    real run_speed = atom->N * (atom->number_of_steps / time_used);
    printf("INFO:  Speed of this run = %g atom*step/second.\n\n", run_speed);

    measure->finalize(input_dir, atom, integrate);
    integrate->finalize();
}




// set some default values after each run
static void initialize_run(Atom* atom, Measure* measure)
{
    atom->neighbor.update = 0;
    measure->heat.sample     = 0;
    measure->shc.compute     = 0;
    measure->vac.compute     = 0;
    measure->hac.compute     = 0;
    measure->hnemd.compute   = 0;
    atom->fixed_group     = -1; // no group has an index of -1
}




// Read the input file to memory
static char *get_file_contents (char *filename)
{
    char *contents;
    int contents_size;
    FILE *in = my_fopen(filename, "r");

    // Find file size
    fseek(in, 0, SEEK_END);
    contents_size = ftell(in);
    rewind(in);

    MY_MALLOC(contents, char, contents_size + 1);
    int size_read_in = fread(contents, sizeof(char), contents_size, in);
    if (size_read_in != contents_size)
    {
        print_error ("File size mismatch.");
    }

    fclose(in);
    contents[contents_size] = '\0'; // Assures proper null termination

    return contents;
}




// Parse a single row
static char *row_find_param (char *s, char *param[], int *num_param)
{
    *num_param = 0;
    int start_new_word = 1, comment_found = 0;
    if (s == NULL) return NULL;

    while(*s)
    {
        if(*s == '\n')
        {
            *s = '\0';
            return s + sizeof(char);
        }
        else if (comment_found)
        {
            // Do nothing
        }
        else if (*s == '#')
        {
            *s = '\0';
            comment_found = 1;
        }
        else if(isspace(*s))
        {
            *s = '\0';
            start_new_word = 1;
        }
        else if (start_new_word)
        {
            param[*num_param] = s;
            ++(*num_param);
            start_new_word = 0;
        }
        ++s;
    }
    return NULL;
}



#define FORCE
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
    char file_force[FILE_NAME_LENGTH];
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
    char file_run[FILE_NAME_LENGTH];
    strcpy(file_run, input_dir);
    strcat(file_run, "/run.in");
    char *input = get_file_contents(file_run);
    char *input_ptr = input; // Keep the pointer in order to free later

    // Iterate the rows
    const int max_num_param = 10; // never use more than 9 parameters
    int num_param;
    char *param[max_num_param];

    initialize_run(atom, measure); // set some default values

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

    MY_FREE(input); // Free the input file contents
}




