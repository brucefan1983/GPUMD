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
The class defining the simulation box.
------------------------------------------------------------------------------*/


#include "box.cuh"
#include "error.cuh"


void Box::allocate_memory_gpu(void)
{
    CHECK(cudaMalloc((void**)&pbc, sizeof(int) * 4)); // 3 + 1
    if (triclinic)
    {
        CHECK(cudaMalloc((void**)&h, memory * 2)); // 9 * 2
    }
    else
    {
        CHECK(cudaMalloc((void**)&h, memory)); // 3
    }
}


void Box::copy_from_cpu_to_gpu(void)
{
    // copy boundary conditions and box type
    int* cpu_pbc; MY_MALLOC(cpu_pbc, int, 4);
    cpu_pbc[0] = pbc_x;
    cpu_pbc[1] = pbc_y;
    cpu_pbc[2] = pbc_z;
    cpu_pbc[3] = triclinic;
    CHECK(cudaMemcpy(pbc, cpu_pbc, sizeof(int) * 4, cudaMemcpyHostToDevice));
    MY_FREE(cpu_pbc);
    // copy box
    CHECK(cudaMemcpy(h, cpu_h, memory, cudaMemcpyHostToDevice));
    // copy inverse box
    if (triclinic)
    {
        CHECK(cudaMemcpy(h + 9, cpu_g, memory, cudaMemcpyHostToDevice));
    }
}


void Box::free_memory_cpu(void)
{
    MY_FREE(cpu_h);
    if (triclinic)
    {
        MY_FREE(cpu_g);
    }
}


void Box::free_memory_gpu(void)
{
    CHECK(cudaFree(pbc));
    CHECK(cudaFree(h));
}


void Box::update_cpu_h(void)
{
    CHECK(cudaMemcpy(cpu_h, h, memory, cudaMemcpyDeviceToHost));
}


real Box::get_volume(void)
{
    real volume;
    if (triclinic)
    {
        volume = cpu_h[0] * (cpu_h[4]*cpu_h[8] - cpu_h[5]*cpu_h[7])
               + cpu_h[1] * (cpu_h[5]*cpu_h[6] - cpu_h[3]*cpu_h[8])
               + cpu_h[2] * (cpu_h[3]*cpu_h[7] - cpu_h[4]*cpu_h[6]);
    }
    else
    {
        volume = cpu_h[0] * cpu_h[1] * cpu_h[2];
    }
    return volume;
}


