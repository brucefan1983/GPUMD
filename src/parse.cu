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
Parse the commands in run.in.
------------------------------------------------------------------------------*/


#include "run.cuh"
#include "atom.cuh"
#include "error.cuh"
#include "force.cuh"
#include "hessian.cuh"
#include "read_file.cuh"


void parse_velocity(char **param, int num_param, Atom *atom)
{
    if (num_param != 2)
    {
        PRINT_INPUT_ERROR("velocity should have 1 parameter.\n");
    }
    if (!is_valid_real(param[1], &atom->initial_temperature))
    {
        PRINT_INPUT_ERROR("initial temperature should be a real number.\n");
    }
    if (atom->initial_temperature <= 0.0)
    {
        PRINT_INPUT_ERROR("initial temperature should be a positive number.\n");
    }
}


void parse_time_step (char **param,  int num_param, Atom* atom)
{
    if (num_param != 2)
    {
        PRINT_INPUT_ERROR("time_step should have 1 parameter.\n");
    }
    if (!is_valid_real(param[1], &atom->time_step))
    {
        PRINT_INPUT_ERROR("time_step should be a real number.\n");
    }
    printf("Time step for this run is %g fs.\n", atom->time_step);
    atom->time_step /= TIME_UNIT_CONVERSION;
}


void parse_run(char **param,  int num_param, Atom* atom)
{
    if (num_param != 2)
    {
        PRINT_INPUT_ERROR("run should have 1 parameter.\n");
    }
    if (!is_valid_int(param[1], &atom->number_of_steps))
    {
        PRINT_INPUT_ERROR("number of steps should be an integer.\n");
    }
    printf("Run %d steps.\n", atom->number_of_steps);
}


void parse_cutoff(char **param, int num_param, Hessian* hessian)
{
    if (num_param != 2)
    {
        PRINT_INPUT_ERROR("cutoff should have 1 parameter.\n");
    }
    if (!is_valid_real(param[1], &hessian->cutoff))
    {
        PRINT_INPUT_ERROR("cutoff for hessian should be a number.\n");
    }
    if (hessian->cutoff <= 0)
    {
        PRINT_INPUT_ERROR("cutoff for hessian should be positive.\n");
    }
    printf("Cutoff distance for hessian = %g A.\n", hessian->cutoff);
}


void parse_delta(char **param, int num_param, Hessian* hessian)
{
    if (num_param != 2)
    {
        PRINT_INPUT_ERROR("compute_hessian should have 1 parameter.\n");
    }
    if (!is_valid_real(param[1], &hessian->dx))
    {
        PRINT_INPUT_ERROR("displacement for hessian should be a number.\n");
    }
    if (hessian->dx <= 0)
    {
        PRINT_INPUT_ERROR("displacement for hessian should be positive.\n");
    }
    printf("Displacement for hessian = %g A.\n", hessian->dx);
}


