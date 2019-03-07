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
The driver class for gafit calculations
------------------------------------------------------------------------------*/


#include "gapot.cuh"
#include "atom.cuh"
#include "force.cuh"
#include "measure.cuh"
#include "ga.cuh"
#include "parse.cuh"
#include "read_file.cuh"
#include "error.cuh"
#include <errno.h>


GApot::GApot(char* input_dir)
{
    Atom atom(input_dir);
    Force force;
    Measure measure(input_dir);
    GA ga;
    compute(input_dir, &atom, &force, &measure, &ga);
}


GApot::~GApot(void)
{
    // nothing
}


void GApot::compute
(char* input_dir, Atom* atom, Force* force, Measure* measure, GA* ga)
{
    char file_run[200];
    strcpy(file_run, input_dir);
    strcat(file_run, "/gapot.in");
    char *input = get_file_contents(file_run);
    char *input_ptr = input; // Keep the pointer in order to free later
    const int max_num_param = 10; // never use more than 9 parameters
    int num_param;
    char *param[max_num_param];
    while (input_ptr)
    {
        input_ptr = row_find_param(input_ptr, param, &num_param);
        if (num_param == 0) { continue; } 
        parse(param, num_param, force, ga);
    }
    MY_FREE(input); // Free the input file contents
    force->initialize(input_dir, atom);
    ga->compute(input_dir);
}


void GApot::parse(char **param, int num_param, Force *force, GA* ga)
{
    if (strcmp(param[0], "potential") == 0)
    {
        parse_potential(param, num_param, force);
    }
    else if (strcmp(param[0], "potentials") == 0)
    {
        parse_potentials(param, num_param, force);
    }
    else
    {
        printf("Error: '%s' is invalid keyword.\n", param[0]);
        exit(1);
    }
}


