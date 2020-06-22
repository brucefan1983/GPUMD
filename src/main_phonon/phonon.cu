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
#include "hessian.cuh"
#include "model/atom.cuh"
#include "force/force.cuh"
#include "utilities/read_file.cuh"
#include "utilities/error.cuh"
#include <errno.h>


Phonon::Phonon(char* input_dir)
{
    Atom atom;

    initialize_position
    (
        input_dir,
        atom.N,
        atom.has_velocity_in_xyz,
        atom.number_of_types,
        atom.box,
        atom.neighbor,
        atom.group,
        atom.cpu_type,
        atom.cpu_type_size,
        atom.cpu_mass,
        atom.cpu_position_per_atom,
        atom.cpu_velocity_per_atom
    );

    atom.allocate_memory_gpu();

#ifndef USE_FCP // the FCP does not use a neighbor list at all
    atom.neighbor.find_neighbor
    (
        1,
        atom.box,
        atom.position_per_atom
    );
#endif

    Force force;
    Hessian hessian;

    compute(input_dir, &atom, &force, &hessian, 1);

    force.initialize_participation_and_shift
    (
        atom.group,
        atom.number_of_types
    );

    compute(input_dir, &atom, &force, &hessian, 0);
}


void Phonon::compute
(
    char* input_dir, Atom* atom, Force* force,
    Hessian* hessian, int check
)
{
    char file_run[200];
    strcpy(file_run, input_dir);
    strcat(file_run, "/phonon.in");
    char *input = get_file_contents(file_run);
    char *input_ptr = input; // Keep the pointer in order to free later
    const int max_num_param = 10; // never use more than 9 parameters
    int num_param;
    force->num_of_potentials = 0;
    char *param[max_num_param];
    while (input_ptr)
    {
        int is_potential = 0;
        input_ptr = row_find_param(input_ptr, param, &num_param);
        if (num_param == 0) { continue; } 
        parse(param, num_param, atom, force, hessian, &is_potential);
        if (!check && is_potential)
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
    free(input); // Free the input file contents
    if (!check)
    {
        hessian->compute
        (
            input_dir,
            force,
            atom->box,
            atom->cpu_position_per_atom,
            atom->position_per_atom,
            atom->type,
            atom->group,
            atom->neighbor,
            atom->potential_per_atom,
            atom->force_per_atom,
            atom->virial_per_atom
        );
    }
}


void Phonon::parse
(
    char **param, int num_param, Atom* atom,
    Force *force, Hessian* hessian, int* is_potential
)
{
    if (strcmp(param[0], "potential_definition") == 0)
    {
        force->parse_potential_definition(param, num_param);
    }
    if (strcmp(param[0], "potential") == 0)
    {
        *is_potential = 1;
        force->parse_potential(param, num_param);
    }
    else if (strcmp(param[0], "cutoff") == 0)
    {
        hessian->parse_cutoff(param, num_param);
    }
    else if (strcmp(param[0], "delta") == 0)
    {
        hessian->parse_delta(param, num_param);
    }
    else
    {
        PRINT_INPUT_ERROR("Invalid keyword.\n");
    }
}


