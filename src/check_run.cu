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

#include "error.cuh"




static void print_velocity_and_potential_error_1
(int number_of_times_potential, int number_of_times_velocity)
{
    if (0 == number_of_times_potential)
    {
        print_error("No 'potential(s)' keyword before run.\n");
    }
    else if (1 < number_of_times_potential)
    {
        print_error("Multiple 'potential(s)' keywords before run.\n");
    }

    if (0 == number_of_times_velocity)
    {
        print_error("No 'velocity' keyword before run.\n");
    }
    else if (1 < number_of_times_velocity)
    {
        print_error("Multiple 'velocity' keywords before run.\n");
    }
}




static void print_velocity_and_potential_error_2
(int number_of_times_potential, int number_of_times_velocity)
{
    if (1 < number_of_times_potential)
    {
        print_error("Multiple 'potential(s)' keywords.\n");
    }
    if (1 < number_of_times_velocity)
    {
        print_error("Multiple 'velocity' keywords.\n");
    }
}




void GPUMD::check_run
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

    const int max_num_param = 10; // never use more than 9 parameters
    int num_param;
    char *param[max_num_param];

    initialize_run(atom, measure); // set some default values

    print_line_1();
    printf("Started checking the inputs in run.in.\n");
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
        if (is_potential) { number_of_times_potential++; }
        if (is_velocity) { number_of_times_velocity++; }
        if (is_run)
        {
            print_velocity_and_potential_error_1
            (number_of_times_potential, number_of_times_velocity);
            initialize_run(atom, measure); // change back to the default
        }
    }
    print_velocity_and_potential_error_2
    (number_of_times_potential, number_of_times_velocity);

    print_line_1();
    printf("Finished checking the inputs in run.in.\n");
    print_line_2();

    MY_FREE(input); // Free the input file contents
}




