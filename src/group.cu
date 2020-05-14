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
The class defining the grouping methods
------------------------------------------------------------------------------*/


#include "group.cuh"
#include <vector>


void Group::find_size(const int N, const int k)
{
    cpu_size.resize(number);
    cpu_size_sum.resize(number);
    cpu_contents.resize(N);

    if (number == 1)
    {
        printf("There is only one group of atoms in grouping method %d.\n", k);
    }
    else
    {
        printf
        (
            "There are %d groups of atoms in grouping method %d.\n",
            number, k
        );
    }

    for (int m = 0; m < number; m++)
    {
        cpu_size[m] = 0;
        cpu_size_sum[m] = 0;
    }

    for (int n = 0; n < N; n++) { cpu_size[cpu_label[n]]++; }

    for (int m = 0; m < number; m++)
    {
        printf("    %d atoms in group %d.\n", cpu_size[m], m);
    }

    for (int m = 1; m < number; m++)
    {
        for (int n = 0; n < m; n++)
        {
            cpu_size_sum[m] += cpu_size[n];
        }
    }
}


// re-arrange the atoms from the first to the last group
void Group::find_contents(const int N)
{
    std::vector<int> offset(number);
    for (int m = 0; m < number; m++) { offset[m] = 0; }

    for (int n = 0; n < N; n++)
    {
        for (int m = 0; m < number; m++)
        {
            if (cpu_label[n] == m)
            {
                cpu_contents[cpu_size_sum[m] + offset[m]++] = n;
            }
        }
    }
}


