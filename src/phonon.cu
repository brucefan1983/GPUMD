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
The driver class for phonon calculations
------------------------------------------------------------------------------*/


#include "phonon.cuh"
#include "atom.cuh"
#include "force.cuh"
#include "measure.cuh"
#include "hessian.cuh"
#include "read_file.cuh"
#include "error.cuh"
#include <errno.h>


Phonon::Phonon(char* input_dir)
{
    Atom atom(input_dir);
    Force force;
    Measure measure(input_dir);
    Hessian hessian;
    compute(input_dir, &atom, &force, &measure, &hessian);
}


Phonon::~Phonon(void)
{
    // nothing
}


void Phonon::compute
(char* input_dir, Atom* atom, Force* force, Measure* measure, Hessian* hessian)
{
    char file_run[200];
    strcpy(file_run, input_dir);
    strcat(file_run, "/phonon.in");
    char *input = get_file_contents(file_run);
    char *input_ptr = input; // Keep the pointer in order to free later
    const int max_num_param = 10; // never use more than 9 parameters
    int num_param;
    char *param[max_num_param];
    while (input_ptr)
    {
        input_ptr = row_find_param(input_ptr, param, &num_param);
        if (num_param == 0) { continue; } 
        parse(param, num_param, force, hessian);
    }
    MY_FREE(input); // Free the input file contents
    force->initialize(input_dir, atom);
    hessian->compute(input_dir, atom, force, measure);
}


static int is_valid_int (const char *s, int *result)
{
    if (s == NULL || *s == '\0') { return 0; }
    char *p;
    errno = 0;
    *result = (int) strtol (s, &p, 0);
    if (errno != 0 || s == p || *p != 0) { return 0; }
    else {return 1; }
}


static int is_valid_real (const char *s, real *result)
{
    if (s == NULL || *s == '\0') { return 0; }
    char *p;
    errno = 0;
    *result = strtod (s, &p);
    if (errno != 0 || s == p || *p != 0) { return 0; }
    else { return 1; }
}


// a single potential
static void parse_potential(char **param, int num_param, Force *force)
{
    if (num_param != 2)
    {
        print_error("potential should have 1 parameter.\n");
    }
    strcpy(force->file_potential[0], param[1]);
    force->num_of_potentials = 1;
}


// multiple potentials
static void parse_potentials(char **param, int num_param, Force *force)
{
    if (num_param == 6)
    {
        force->num_of_potentials = 2;
    }
    else if (num_param == 9)
    {
        force->num_of_potentials = 3;
    }
    else
    {
        print_error("potentials should have 5 or 8 parameters.\n");
    }

    // two-body part
    strcpy(force->file_potential[0], param[1]);
    if (!is_valid_int(param[2], &force->interlayer_only))
    {
        print_error("interlayer_only should be an integer.\n");
    }

    // the first many-body part
    strcpy(force->file_potential[1], param[3]);
    if (!is_valid_int(param[4], &force->type_begin[1]))
    {
        print_error("type_begin should be an integer.\n");
    }
    if (!is_valid_int(param[5], &force->type_end[1]))
    {
        print_error("type_end should be an integer.\n");
    }

    // the second many-body part
    if (force->num_of_potentials > 2)
    {
        strcpy(force->file_potential[2], param[6]);
        if (!is_valid_int(param[7], &force->type_begin[2]))
        {
            print_error("type_begin should be an integer.\n");
        }
        if (!is_valid_int(param[8], &force->type_end[2]))
        {
            print_error("type_end should be an integer.\n");
        }
    }
}


static void parse_cutoff(char **param, int num_param, Hessian* hessian)
{
    hessian->yes = 1;
    if (num_param != 2)
    {
        print_error("cutoff should have 1 parameter.\n");
    }
    if (!is_valid_real(param[1], &hessian->cutoff))
    {
        print_error("cutoff for hessian should be a number.\n");
    }
    if (hessian->cutoff <= 0)
    {
        print_error("cutoff for hessian should be positive.\n");
    }
    printf("Cutoff distance for hessian = %g A.\n", hessian->cutoff);
}


static void parse_delta(char **param, int num_param, Hessian* hessian)
{
    if (num_param != 2)
    {
        print_error("compute_hessian should have 1 parameter.\n");
    }
    if (!is_valid_real(param[1], &hessian->dx))
    {
        print_error("displacement for hessian should be a number.\n");
    }
    if (hessian->dx <= 0)
    {
        print_error("displacement for hessian should be positive.\n");
    }
    printf("Displacement for hessian = %g A.\n", hessian->dx);
}


void Phonon::parse(char **param, int num_param, Force *force, Hessian* hessian)
{
    if (strcmp(param[0], "potential") == 0)
    {
        parse_potential(param, num_param, force);
    }
    else if (strcmp(param[0], "potentials") == 0)
    {
        parse_potentials(param, num_param, force);
    }
    else if (strcmp(param[0], "cutoff") == 0)
    {
        parse_cutoff(param, num_param, hessian);
    }
    else if (strcmp(param[0], "delta") == 0)
    {
        parse_delta(param, num_param, hessian);
    }
    else
    {
        printf("Error: '%s' is invalid keyword.\n", param[0]);
        exit(1);
    }
}


