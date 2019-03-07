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


#include "gapot.cuh"
#include "error.cuh"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

void print_welcome_information(void);
int get_number_of_input_directories(void);


int main(int argc, char *argv[])
{
    print_welcome_information();
    int number_of_inputs = get_number_of_input_directories();
    for (int n = 0; n < number_of_inputs; ++n)
    {
        char input_directory[200];
        int count = scanf("%s", input_directory);
        if (count != 1)
        {
            print_error("reading error for input directory.\n");
        }
        print_line_1();
        printf("Run simulation for '%s'.\n", input_directory);
        print_line_2();
        clock_t time_begin = clock();
        GApot gapot(input_directory);
        clock_t time_finish = clock();
        double time_used = (time_finish - time_begin) / double(CLOCKS_PER_SEC);
        print_line_1();
        printf("Time used for '%s' = %f s.\n", input_directory, time_used);
        print_line_2();
    }
    print_line_1();
    printf("Finished running gapot.\n");
    print_line_2();
    return EXIT_SUCCESS;
}


void print_welcome_information(void)
{
    printf("\n");
    printf("***************************************************************\n");
    printf("*                 Welcome to use GPUMD                        *\n");
    printf("*     (Graphics Processing Units Molecular Dynamics)          *\n");
    printf("*                      Version 2.3                            *\n");
    printf("*              This is the gapot excutable                    *\n");
    printf("* Authors:                                                    *\n");
    printf("*     Zheyong Fan <brucenju@gmail.com>                        *\n");
    printf("*     Ville Vierimaa                                          *\n");
    printf("*     Mikko Ervasti                                           *\n");
    printf("*     Alexander J. Gabourie                                   *\n");
    printf("*     Ari Harju                                               *\n");
    printf("***************************************************************\n");
    printf("\n");
}


int get_number_of_input_directories(void)
{
    int number_of_inputs;
    int count = scanf("%d", &number_of_inputs);
    if (count != 1)
    {
        print_error("reading error for number of inputs.\n");
    }
    return number_of_inputs;
}


