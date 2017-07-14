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
#include "dump.h"


// dump thermodynamic properties
static void gpu_sample_thermo
(
    FILE *fid, Parameters *para, CPU_Data *cpu_data, 
    real *gpu_thermo, real *gpu_box_length
)
{

    // copy data from GPU to CPU
    real *thermo = cpu_data->thermo;
    real *box_length = cpu_data->box_length;
    int m1 = sizeof(real) * 6;
    int m2 = sizeof(real) * DIM;
    CHECK(cudaMemcpy(thermo, gpu_thermo, m1, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(box_length, gpu_box_length, m2, cudaMemcpyDeviceToHost));

    // system energies
    real energy_system_kin = (HALF * DIM) * para->N * K_B * thermo[0];
    real energy_system_pot = thermo[1];
    real energy_system_total = energy_system_kin + energy_system_pot; 

    if (para->ensemble == 3)
    {
        // energy of the Nose-Hoover chain thermostat
        real kT = K_B * para->temperature; 
        real energy_nhc = kT * (DIM * para->N) * para->pos_nhc1[0];
        for (int m = 1; m < NOSE_HOOVER_CHAIN_LENGTH; m++)
        {
            energy_nhc += kT * para->pos_nhc1[m];
        }
        for (int m = 0; m < NOSE_HOOVER_CHAIN_LENGTH; m++)
        { 
            energy_nhc += 0.5 * para->vel_nhc1[m] 
                        * para->vel_nhc1[m] / para->mas_nhc1[m];
        }
        fprintf
        (
            fid, "%20.10e%20.10e%20.10e", thermo[0], 
            energy_system_total, energy_nhc
        );
    }
    else
    {
        fprintf
        (
            fid, "%20.10e%20.10e%20.10e", thermo[0], 
            energy_system_kin, energy_system_pot
        );
    }    

    fprintf // presure (x, y, z)
    (
        fid, "%20.10e%20.10e%20.10e", 
        thermo[2] * PRESSURE_UNIT_CONVERSION, 
        thermo[3] * PRESSURE_UNIT_CONVERSION, 
        thermo[4] * PRESSURE_UNIT_CONVERSION
    ); 

    // box length (x, y, z)
    fprintf
    (
        fid, "%20.10e%20.10e%20.10e\n", 
        box_length[0], box_length[1], box_length[2]
    ); 

    fflush(fid);
}


// dump thermodynamic properties (A wrapper function)
void dump_thermos
(FILE *fid, Parameters *para, CPU_Data *cpu_data, GPU_Data *gpu_data, int step)
{
    if (para->dump_thermo)
    {
        if ((step + 1) % para->sample_interval_thermo == 0)
        {
            gpu_sample_thermo
            (fid, para, cpu_data, gpu_data->thermo, gpu_data->box_length);
        }
    }
}


static void gpu_dump_3(int N, FILE *fid, real *a, real *b, real *c)
{
    real *cpu_a, *cpu_b, *cpu_c;
    MY_MALLOC(cpu_a, real, N);
    MY_MALLOC(cpu_b, real, N);
    MY_MALLOC(cpu_c, real, N);
    CHECK(cudaMemcpy(cpu_a, a, sizeof(real) * N, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(cpu_b, b, sizeof(real) * N, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(cpu_c, c, sizeof(real) * N, cudaMemcpyDeviceToHost));

    for (int n = 0; n < N; n++)
    {
        fprintf(fid, "%20.10e%20.10e%20.10e\n", cpu_a[n], cpu_b[n], cpu_c[n]);
    }
    fflush(fid);

    MY_FREE(cpu_a);
    MY_FREE(cpu_b);
    MY_FREE(cpu_c);
}


void dump_positions
(FILE *fid, Parameters *para, CPU_Data *cpu_data, GPU_Data *gpu_data, int step)
{
    if (para->dump_position)
    {
        if ((step + 1) % para->sample_interval_position == 0)
        {
            gpu_dump_3(para->N, fid, gpu_data->x, gpu_data->y, gpu_data->z);
        }
    }
}


void dump_velocities
(FILE *fid, Parameters *para, CPU_Data *cpu_data, GPU_Data *gpu_data, int step)
{
    if (para->dump_velocity)
    {
        if ((step + 1) % para->sample_interval_velocity == 0)
        {
            gpu_dump_3(para->N, fid, gpu_data->vx, gpu_data->vy, gpu_data->vz);
        }
    }
}


void dump_forces
(FILE *fid, Parameters *para, CPU_Data *cpu_data, GPU_Data *gpu_data, int step)
{
    if (para->dump_force)
    {
        if ((step + 1) % para->sample_interval_force == 0)
        {
            gpu_dump_3(para->N, fid, gpu_data->fx, gpu_data->fy, gpu_data->fz);
        }
    }
}


void dump_virial
(FILE *fid, Parameters *para, CPU_Data *cpu_data, GPU_Data *gpu_data, int step)
{
    if (para->dump_virial)
    {
        if ((step + 1) % para->sample_interval_virial == 0)
        {
            gpu_dump_3
            (
                para->N, fid, gpu_data->virial_per_atom_x, 
                gpu_data->virial_per_atom_y, gpu_data->virial_per_atom_z
            );
        }
    }
}


static void gpu_dump_1(int N, FILE *fid, real *a)
{
    real *cpu_a;
    MY_MALLOC(cpu_a, real, N);
    CHECK(cudaMemcpy(cpu_a, a, sizeof(real) * N, cudaMemcpyDeviceToHost));
    for (int n = 0; n < N; n++)
    {
        fprintf(fid, "%20.10e\n", cpu_a[n]);
    }
    fflush(fid);
    MY_FREE(cpu_a);
}


void dump_potential
(FILE *fid, Parameters *para, CPU_Data *cpu_data, GPU_Data *gpu_data, int step)
{
    if (para->dump_potential)
    {
        if ((step + 1) % para->sample_interval_potential == 0)
        {
            gpu_dump_1(para->N, fid, gpu_data->potential_per_atom);
        }
    }
}




