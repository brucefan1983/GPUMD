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


void parse_potential_definition
(char **param, int num_param, Atom *atom, Force *force)
{
    // 'potential_definition' must be called before all 'potential' keywords
    if (force->num_of_potentials > 0)
    {
        print_error("potential_definition must be called before all "
                "potential keywords.\n");
    }

    if (num_param != 2 && num_param != 3)
    {
        print_error("potential_definition should have only 1 or 2 "
                "parameters.\n");
    }
    if (num_param == 2)
    {
        //default is to use type, check for deviations
        if(strcmp(param[1], "group") == 0)
        {
            print_error("potential_definition must have "
                    "group_method listed.\n");
        }
        else if(strcmp(param[1], "type") != 0)
        {
            print_error("potential_definition only accepts "
                    "'type' or 'group' kind.\n");
        }
    }
    if (num_param == 3)
    {
        if(strcmp(param[1], "group") != 0)
        {
            print_error("potential_definition: kind must be 'group' if 2 "
                    "parameters are used.\n");

        }
        else if(!is_valid_int(param[2], &force->group_method))
        {
            print_error("potential_definition: group_method should be an "
                    "integer.\n");
        }
        else if(force->group_method > MAX_NUMBER_OF_GROUPS)
        {
            print_error("Specified group_method is too large (> 10).\n");
        }
    }
}

// a potential
void parse_potential(char **param, int num_param, Force *force)
{
    // check for at least the file path
    if (num_param < 3)
    {
        print_error("potential should have at least 2 parameters.\n");
    }
    strcpy(force->file_potential[force->num_of_potentials], param[1]);

    //open file to check number of types used in potential
    char potential_name[20];
    FILE *fid_potential = my_fopen(
            force->file_potential[force->num_of_potentials], "r");
    int count = fscanf(fid_potential, "%s", potential_name);
    int num_types = force->get_number_of_types(fid_potential);
    fclose(fid_potential);

    if (num_param != num_types + 2)
    {
        print_error("potential has incorrect number of types/groups defined.\n");
    }

    force->participating_kinds.resize(num_types);

    for (int i = 0; i < num_types; i++)
    {
        if(!is_valid_int(param[i+2], &force->participating_kinds[i]))
        {
            print_error("type/groups should be an integer.\n");
        }
        if (i != 0 &&
            force->participating_kinds[i] < force->participating_kinds[i-1])
        {
            print_error("potential types/groups must be listed in "
                    "ascending order.\n");
        }
    }
    force->atom_begin[force->num_of_potentials] =
            force->participating_kinds[0];
    force->atom_end[force->num_of_potentials] =
            force->participating_kinds[num_types-1];

    force->num_of_potentials++;

}


void parse_velocity(char **param, int num_param, Atom *atom)
{
    if (num_param != 2)
    {
        print_error("velocity should have 1 parameter.\n");
    }
    if (!is_valid_real(param[1], &atom->initial_temperature))
    {
        print_error("initial temperature should be a real number.\n");
    }
    if (atom->initial_temperature <= 0.0)
    {
        print_error("initial temperature should be a positive number.\n");
    }
}


void parse_time_step (char **param,  int num_param, Atom* atom)
{
    if (num_param != 2)
    {
        print_error("time_step should have 1 parameter.\n");
    }
    if (!is_valid_real(param[1], &atom->time_step))
    {
        print_error("time_step should be a real number.\n");
    }
    printf("Time step for this run is %g fs.\n", atom->time_step);
    atom->time_step /= TIME_UNIT_CONVERSION;
}


void parse_neighbor
(char **param,  int num_param, Atom* atom, Force *force)
{
    atom->neighbor.update = 1;

    if (num_param != 2)
    {
        print_error("neighbor should have 1 parameter.\n");
    }
    if (!is_valid_real(param[1], &atom->neighbor.skin))
    {
        print_error("neighbor list skin should be a number.\n");
    }
    printf("Build neighbor list with a skin of %g A.\n", atom->neighbor.skin);

    // change the cutoff
    atom->neighbor.rc = force->rc_max + atom->neighbor.skin;
}


void parse_run(char **param,  int num_param, Atom* atom)
{
    if (num_param != 2)
    {
        print_error("run should have 1 parameter.\n");
    }
    if (!is_valid_int(param[1], &atom->number_of_steps))
    {
        print_error("number of steps should be an integer.\n");
    }
    printf("Run %d steps.\n", atom->number_of_steps);
}


void parse_cutoff(char **param, int num_param, Hessian* hessian)
{
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


void parse_delta(char **param, int num_param, Hessian* hessian)
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


