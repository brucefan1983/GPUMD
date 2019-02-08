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
The class defining the simulation 
------------------------------------------------------------------------------*/


#include "box.cuh"
#include "error.cuh"


void Box::allocate_memory_gpu(void)
{
    CHECK(cudaMalloc((void**)&h, memory * 2));
}


void Box::copy_from_cpu_to_gpu(void)
{
    CHECK(cudaMemcpy(h, cpu_h, memory*2, cudaMemcpyHostToDevice));
}


void Box::free_memory_cpu(void)
{
    MY_FREE(cpu_h);
}


void Box::free_memory_gpu(void)
{
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


void Box::get_inverse(void)
{
    cpu_h[9]  = cpu_h[4]*cpu_h[8] - cpu_h[5]*cpu_h[7];
    cpu_h[10] = cpu_h[2]*cpu_h[7] - cpu_h[1]*cpu_h[8];
    cpu_h[11] = cpu_h[1]*cpu_h[5] - cpu_h[2]*cpu_h[4];
    cpu_h[12] = cpu_h[5]*cpu_h[6] - cpu_h[3]*cpu_h[8];
    cpu_h[13] = cpu_h[0]*cpu_h[8] - cpu_h[2]*cpu_h[6];
    cpu_h[14] = cpu_h[2]*cpu_h[3] - cpu_h[0]*cpu_h[5];
    cpu_h[15] = cpu_h[3]*cpu_h[7] - cpu_h[4]*cpu_h[6];
    cpu_h[16] = cpu_h[1]*cpu_h[6] - cpu_h[0]*cpu_h[7];
    cpu_h[17] = cpu_h[0]*cpu_h[4] - cpu_h[1]*cpu_h[3];
    real volume = get_volume();
    for (int n = 9; n < 18; n++)
    {
        cpu_h[n] /= volume;
    }
}


