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

#include <stdlib.h>
#include <stdio.h>
#include <time.h>




int main(int argc, char *argv[])
{
    printf("\n");
    printf("***************************************************************\n");
    printf("*                 Welcome to use GPUMD                        *\n");
    printf("*     (Graphics Processing Units Molecular Dynamics)          *\n");
    printf("*                      Version 2.1                            *\n");
    printf("* Authors:                                                    *\n");
    printf("*     Zheyong Fan <brucenju@gmail.com>                        *\n");
    printf("*     Ville Vierimaa                                          *\n");
    printf("*     Mikko Ervasti                                           *\n");
    printf("*     Alexander J. Gabourie                                   *\n");
    printf("*     Ari Harju                                               *\n");
    printf("***************************************************************\n");
    printf("\n");

    print_line_1();
    printf("Compiling options:\n");
    print_line_2();

#ifdef DEBUG
    printf("DEBUG is on: Use a fixed PRNG seed for differnt runs.\n");
#else
    srand(time(NULL));
    printf("DEBUG is off: Use differnt PRNG seeds for differnt runs.\n");
#endif

#ifndef USE_SP
    printf("USE_SP is off: Use double-precision version.\n");
#else
    printf("USE_SP is on: Use single-precision version.\n");
#endif

#ifdef MOS2_JIANG
    printf("MOS2_JIANG is on: Special verison for Ke Xu.\n");
#endif

#ifdef MURTY_ATWATER
    printf("MURTY_ATWATER is on: Special verison for Qi You.\n");
#endif

#ifdef ZHEN_LI
    printf("ZHEN_LI is on: Special verison for Zhen Li.\n");
#endif

#ifdef CBN
    printf("CBN is on: Special verison for Haikuan Dong.\n");
#endif

#ifdef HEAT_CURRENT
    printf("HEAT_CURRENT is on: Special verison for Davide Donadio.\n");
#endif

    // get the number of input directories
    int number_of_inputs;
    char input_directory[200];

    int count = scanf("%d", &number_of_inputs);
    if (count != 1)
    {
        printf("Error: reading error for number of inputs.\n");
        exit(1);
    }

    // Run GPUMD for the input directories one by one
    for (int n = 0; n < number_of_inputs; ++n)
    {
        count = scanf("%s", input_directory);
        if (count != 1)
        {
            printf("Error: reading error for input directory.\n");
            exit(1);
        }

        print_line_1();
        printf("Run simulation for '%s'.\n", input_directory);
        print_line_2();

        clock_t time_begin = clock();

        // Run GPUMD for "input_directory"
        GPUMD gpumd(input_directory);

        clock_t time_finish = clock();

        double time_used = (time_finish - time_begin) / double(CLOCKS_PER_SEC);

        print_line_1();
        printf("Time used for '%s' = %f s.\n", input_directory, time_used);
        print_line_2();
    }

    print_line_1();
    printf("Finished running GPUMD.\n");
    print_line_2();

    return EXIT_SUCCESS;
}




