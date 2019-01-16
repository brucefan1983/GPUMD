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
Initialize the velocities of the system:
    total linear momentum is zero
    total angular momentum is zero
If DEBUG is on in the makefile, the velocities are the same from run to run.
If DEBUG is off, the velocities are different in different runs.
------------------------------------------------------------------------------*/




#include "atom.cuh"
#include "error.cuh"




void Atom::scale_velocity(void)
{
    real temperature = 0.0;
    for (int n = 0; n < N; ++n) 
    {
        real v2 = cpu_vx[n]*cpu_vx[n]+cpu_vy[n]*cpu_vy[n]+cpu_vz[n]*cpu_vz[n];
        temperature += cpu_mass[n] * v2;
    }
    temperature /= 3.0 * K_B * N;

    // scale the velocities
    real scale_factor = sqrt(initial_temperature / temperature);
    for (int n = 0; n < N; ++n)
    {
        cpu_vx[n] *= scale_factor;
        cpu_vy[n] *= scale_factor;
        cpu_vz[n] *= scale_factor;
    }
}




void Atom::initialize_velocity_cpu(void)
{
    // random velocities
    for (int n = 0; n < N; ++n)
    {
        cpu_vx[n] = -1.0 + (rand() * 2.0) / RAND_MAX; 
        cpu_vy[n] = -1.0 + (rand() * 2.0) / RAND_MAX; 
        cpu_vz[n] = -1.0 + (rand() * 2.0) / RAND_MAX;    
    }
    
    // linear momentum
    real p[3] = {0.0, 0.0, 0.0};
    for (int n = 0; n < N; ++n)
    {       
        p[0] += cpu_mass[n] * cpu_vx[n] / N;
        p[1] += cpu_mass[n] * cpu_vy[n] / N;
        p[2] += cpu_mass[n] * cpu_vz[n] / N;
    }

    // zero the linear momentum
    for (int n = 0; n < N; ++n) 
    { 
        cpu_vx[n] -= p[0] / cpu_mass[n];
        cpu_vy[n] -= p[1] / cpu_mass[n];
        cpu_vz[n] -= p[2] / cpu_mass[n]; 
    }

    // center of mass position
    real r0[3] = {0, 0, 0};
    real mass_total = 0;
    for (int i = 0; i < N; i++)
    {
        real mass = cpu_mass[i];
        mass_total += mass;
        r0[0] += cpu_x[i] * mass;
        r0[1] += cpu_y[i] * mass;
        r0[2] += cpu_z[i] * mass;
    }
    r0[0] /= mass_total;
    r0[1] /= mass_total;
    r0[2] /= mass_total;

    // angular momentum 
    real L[3] = {0, 0, 0};
    for (int i = 0; i < N; i++)
    {
        real mass = cpu_mass[i];
        real dx = cpu_x[i] - r0[0];
        real dy = cpu_y[i] - r0[1];
        real dz = cpu_z[i] - r0[2];
        L[0] += mass * (dy * cpu_vz[i] - dz * cpu_vy[i]);
        L[1] += mass * (dz * cpu_vx[i] - dx * cpu_vz[i]);
        L[2] += mass * (dx * cpu_vy[i] - dy * cpu_vx[i]);
    }

    // moment of inertia
    real I[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    for (int i = 0; i < N; i++)
    {
        real mass = cpu_mass[i];
        real dx = cpu_x[i] - r0[0];
        real dy = cpu_y[i] - r0[1];
        real dz = cpu_z[i] - r0[2];
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
    for (int i = 0; i < N; i++)
    {
        real dx = cpu_x[i] - r0[0];
        real dy = cpu_y[i] - r0[1];
        real dz = cpu_z[i] - r0[2];
        cpu_vx[i] -= w[1] * dz - w[2] * dy;
        cpu_vy[i] -= w[2] * dx - w[0] * dz;
        cpu_vz[i] -= w[0] * dy - w[1] * dx;
    }
}




void Atom::initialize_velocity(void)
{
    if (has_velocity_in_xyz == 0) { initialize_velocity_cpu(); }
    scale_velocity();

    int M = sizeof(real) * N;
    CHECK(cudaMemcpy(vx, cpu_vx, M, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(vy, cpu_vy, M, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(vz, cpu_vz, M, cudaMemcpyHostToDevice));

    printf("Initialized velocities with T = %g K.\n", initial_temperature);
}




