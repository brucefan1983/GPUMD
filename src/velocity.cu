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



#include "common.h"

#include "velocity.h"



// Initialize velocities
void initialize_velocity
(
    int number_of_particles, real temperature_prescribed, 
    real *m, real *vx, real *vy, real *vz
)
{
    real momentum_average[3] = {0.0, 0.0, 0.0};
    for (int n = 0; n < number_of_particles; ++n)
    { 
        vx[n] = -1.0 + (rand() * 2.0) / RAND_MAX; 
        vy[n] = -1.0 + (rand() * 2.0) / RAND_MAX; 
        vz[n] = -1.0 + (rand() * 2.0) / RAND_MAX;    
        
        momentum_average[0] += m[n] * vx[n] / number_of_particles;
        momentum_average[1] += m[n] * vy[n] / number_of_particles;
        momentum_average[2] += m[n] * vz[n] / number_of_particles;
    } 

    // zero the total momentum
    for (int n = 0; n < number_of_particles; ++n) 
    { 
        vx[n] -= momentum_average[0] / m[n];
        vy[n] -= momentum_average[1] / m[n];
        vz[n] -= momentum_average[2] / m[n]; 
    }

    // scale the velocities
    real temperature = 0.0;
    for (int n = 0; n < number_of_particles; ++n) 
    {
        real v2 = vx[n] * vx[n] + vy[n] * vy[n] + vz[n] * vz[n];     
        temperature += m[n] * v2; 
    }

    temperature /= 3.0 * K_B * number_of_particles;
    real scale_factor = sqrt(temperature_prescribed / temperature);
    for (int n = 0; n < number_of_particles; ++n)
    { 
        vx[n] *= scale_factor;
        vy[n] *= scale_factor;
        #ifdef USE_2D
            vz[n] = 0.0; // for 2D  simulation
        #else
            vz[n] *= scale_factor;
        #endif
    }
}




//initialize the velocities according to the input initial temperature
void process_velocity(Parameters *para, CPU_Data *cpu_data, GPU_Data *gpu_data)
{
    int N = para->N;
    int M = sizeof(real) * N; 

    initialize_velocity
    (
        para->N, para->initial_temperature, cpu_data->mass, 
        cpu_data->vx, cpu_data->vy, cpu_data->vz
    );

    CHECK(cudaMemcpy(gpu_data->vx, cpu_data->vx, M, cudaMemcpyHostToDevice)); 
    CHECK(cudaMemcpy(gpu_data->vy, cpu_data->vy, M, cudaMemcpyHostToDevice)); 
    CHECK(cudaMemcpy(gpu_data->vz, cpu_data->vz, M, cudaMemcpyHostToDevice));

    printf("INFO : velocities are initialized.\n\n");
}




