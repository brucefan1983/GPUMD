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




#include <stdio.h>
#include <time.h>

#include "gpumd.h"




int main(int argc, char *argv[])
{
    printf("\n");
    printf("***************************************************************\n");
    printf("*                 Welcome to use GPUMD                        *\n");
    printf("*     (Graphics Processing Units Molecular Dynamics)          *\n");
    printf("***************************************************************\n");
    printf("\n");
    
    printf("\n");
    printf("===============================================================\n");
    printf("INFO:  Compiled with the following options:\n");
    #ifdef DEBUG
        printf("       Debug mode is activated.\n");
    #else
        srand(time(NULL));
        printf("       Debug mode is not activated.\n");
    #endif
    #ifdef USE_DP
        printf("       Use double precision.\n");
    #else
        printf("       Use single precision.\n");
    #endif
    printf("===============================================================\n");
    printf("\n");
    
    // get the number of input directories
    int number_of_inputs;
    char input_directory[100];

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

        printf("\n");
        printf("===========================================================\n");
        printf("Run simulation for '%s'.\n", input_directory); 
        printf("===========================================================\n");
        printf("\n");

        clock_t time_begin = clock();

        //  Run GPUMD for "input_directory"
        gpumd(input_directory);

        clock_t time_finish = clock();

        double time_used = (time_finish - time_begin) / double(CLOCKS_PER_SEC);

        printf("\n");
        printf("===========================================================\n");
        printf("Time used for '%s' = %f s.\n", input_directory, time_used); 
        printf("===========================================================\n");
        printf("\n");
    }

    return EXIT_SUCCESS;
}



