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




#include "common.cuh"
#include "velocity.cuh"
#include "atom.cuh"
#include "memory.cuh"
#include "error.cuh"

#ifdef USE_DP
    #define K_B   8.617343e-5
#else
    #define K_B   8.617343e-5f
#endif




// Initialize velocities
void initialize_velocity
(
    int number_of_particles, real temperature_prescribed, 
    real *m, real *x, real *y, real *z, real *vx, real *vy, real *vz
)
{
    // random velocities
    for (int n = 0; n < number_of_particles; ++n)
    { 
        vx[n] = -1.0 + (rand() * 2.0) / RAND_MAX; 
        vy[n] = -1.0 + (rand() * 2.0) / RAND_MAX; 
        vz[n] = -1.0 + (rand() * 2.0) / RAND_MAX;    
    }
    
    // linear momentum
    real p[3] = {0.0, 0.0, 0.0};
    for (int n = 0; n < number_of_particles; ++n)
    {       
        p[0] += m[n] * vx[n] / number_of_particles;
        p[1] += m[n] * vy[n] / number_of_particles;
        p[2] += m[n] * vz[n] / number_of_particles;
    } 

    // zero the linear momentum
    for (int n = 0; n < number_of_particles; ++n) 
    { 
        vx[n] -= p[0] / m[n];
        vy[n] -= p[1] / m[n];
        vz[n] -= p[2] / m[n]; 
    }
    
    // center of mass position
    real r0[3] = {0, 0, 0};
    real mass_total = 0;
    for (int i = 0; i < number_of_particles; i++)
    {
        real mass = m[i];
        mass_total += mass;
        r0[0] += x[i] * mass;
        r0[1] += y[i] * mass;
        r0[2] += z[i] * mass;
    }
    r0[0] /= mass_total;
    r0[1] /= mass_total;
    r0[2] /= mass_total;

    // angular momentum 
    real L[3] = {0, 0, 0};
    for (int i = 0; i < number_of_particles; i++)
    {
        real mass = m[i];
        real dx = x[i] - r0[0];
        real dy = y[i] - r0[1];
        real dz = z[i] - r0[2];
        L[0] += mass * (dy * vz[i] - dz * vy[i]);
        L[1] += mass * (dz * vx[i] - dx * vz[i]);
        L[2] += mass * (dx * vy[i] - dy * vx[i]);
    }
 
    // moment of inertia
    real I[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    for (int i = 0; i < number_of_particles; i++)
    {
        real mass = m[i];
        real dx = x[i] - r0[0];
        real dy = y[i] - r0[1];
        real dz = z[i] - r0[2];
        I[0][0] += mass * (dy*dy + dz*dz);
        I[1][1] += mass * (dx*dx + dz*dz);
        I[2][2] += mass * (dx*dx + dy*dy);
        I[0][1] -= mass * dx*dy;
        I[1][2] -= mass * dy*dz;
        I[0][2] -= mass * dx*dz;
    }
    I[1][0] = I[0][1];
    I[2][1] = I[1][2];
    I[2][0] = I[0][2];

    // inverse of I
    real inverse[3][3];
    inverse[0][0] =   I[1][1]*I[2][2] - I[1][2]*I[2][1];
    inverse[0][1] = -(I[0][1]*I[2][2] - I[0][2]*I[2][1]);
    inverse[0][2] =   I[0][1]*I[1][2] - I[0][2]*I[1][1];

    inverse[1][0] = -(I[1][0]*I[2][2] - I[1][2]*I[2][0]);
    inverse[1][1] =   I[0][0]*I[2][2] - I[0][2]*I[2][0];
    inverse[1][2] = -(I[0][0]*I[1][2] - I[0][2]*I[1][0]);

    inverse[2][0] =   I[1][0]*I[2][1] - I[1][1]*I[2][0];
    inverse[2][1] = -(I[0][0]*I[2][1] - I[0][1]*I[2][0]);
    inverse[2][2] =   I[0][0]*I[1][1] - I[0][1]*I[1][0];

    real determinant = I[0][0]*I[1][1]*I[2][2] + I[0][1]*I[1][2]*I[2][0] +
                       I[0][2]*I[1][0]*I[2][1] - I[0][0]*I[1][2]*I[2][1] -
                       I[0][1]*I[1][0]*I[2][2] - I[2][0]*I[1][1]*I[0][2];

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            inverse[i][j] /= determinant;
        }
    }

    // angular velocity w = inv(I) * L, because L = I * w
    real w[3];
    w[0] = inverse[0][0] * L[0] + inverse[0][1] * L[1] + inverse[0][2] * L[2];
    w[1] = inverse[1][0] * L[0] + inverse[1][1] * L[1] + inverse[1][2] * L[2];
    w[2] = inverse[2][0] * L[0] + inverse[2][1] * L[1] + inverse[2][2] * L[2];
    
    // zero the angular momentum: v = v - w x r
    for (int i = 0; i < number_of_particles; i++)
    {
        real dx = x[i] - r0[0];
        real dy = y[i] - r0[1];
        real dz = z[i] - r0[2];
        vx[i] -= w[1] * dz - w[2] * dy;
        vy[i] -= w[2] * dx - w[0] * dz;
        vz[i] -= w[0] * dy - w[1] * dx;
    }  

    // instant temperature
    real temperature = 0.0;
    for (int n = 0; n < number_of_particles; ++n) 
    {
        real v2 = vx[n] * vx[n] + vy[n] * vy[n] + vz[n] * vz[n];     
        temperature += m[n] * v2; 
    }
    temperature /= 3.0 * K_B * number_of_particles;
    
    // scale the velocities
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
void process_velocity(Parameters *para, Atom *atom)
{
    int N = para->N;
    int M = sizeof(real) * N; 

    real* mass = atom->cpu_mass;
    real* x = atom->cpu_x;
    real* y = atom->cpu_y;
    real* z = atom->cpu_z;

    real* vx;
    real* vy;
    real* vz;
    MY_MALLOC(vx, real, N);
    MY_MALLOC(vy, real, N);
    MY_MALLOC(vz, real, N);

    initialize_velocity
    (
        para->N, para->initial_temperature, mass, 
        x, y, z, vx, vy, vz
    );

    CHECK(cudaMemcpy(atom->vx, vx, M, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(atom->vy, vy, M, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(atom->vz, vz, M, cudaMemcpyHostToDevice));
    MY_FREE(vx);
    MY_FREE(vy);
    MY_FREE(vz);

    printf("INFO : velocities are initialized.\n\n");
}



