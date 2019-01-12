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
#include "integrate.cuh"
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

    check_run(input_dir, &atom, &force, &integrate, &measure);
    run(input_dir, &atom, &force, &integrate, &measure);
}




GPUMD::~GPUMD(void)
{
    // nothing
}




// set some default values after each run
void GPUMD::initialize_run(Atom* atom, Measure* measure)
{
    // modify
    atom->neighbor.update = 0;
    atom->fixed_group     = -1; // no group has an index of -1
    atom->deform_x = 0;
    atom->deform_y = 0;
    atom->deform_z = 0;

    // measure
    measure->compute.compute_force = 0;
    measure->compute.compute_temperature = 0;
    measure->shc.compute    = 0;
    measure->vac.compute    = 0;
    measure->hac.compute    = 0;
    measure->hnemd.compute  = 0;
    measure->dump_thermo    = 0;
    measure->dump_position  = 0;
    measure->dump_velocity  = 0;
    measure->dump_force     = 0;
    measure->dump_potential = 0;
    measure->dump_virial    = 0;
    measure->dump_heat      = 0;
}




// Read the input file to memory
char* GPUMD::get_file_contents (char *filename)
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
char* GPUMD::row_find_param (char *s, char *param[], int *num_param)
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




