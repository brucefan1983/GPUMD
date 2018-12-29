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
#include "integrate.cuh"
#include "ensemble.cuh"
#include "measure.cuh"
#include "heat.cuh"
#include "vac.cuh"
#include "hac.cuh"
#include "shc.cuh"
#include "hnemd_kappa.cuh"

#define DIM 3
#define NUM_OF_HEAT_COMPONENTS 5




static FILE *my_fopen(const char *filename, const char *mode)
{
    FILE *fid = fopen(filename, mode);
    if (fid == NULL) 
    {
        printf ("Failed to open %s!\n", filename);
        printf ("%s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    return fid;
}




Measure::Measure(char *input_dir)
{
    dump_thermo = 0;
    dump_position = 0;
    dump_velocity = 0;
    dump_force = 0;
    dump_potential = 0;
    dump_virial = 0;
    dump_heat = 0;

    strcpy(file_thermo, input_dir);
    strcpy(file_position, input_dir);
    strcpy(file_velocity, input_dir);
    strcpy(file_force, input_dir);
    strcpy(file_potential, input_dir);
    strcpy(file_virial, input_dir);
    strcpy(file_heat, input_dir);

    strcat(file_thermo, "/thermo.out");
    strcat(file_position, "/xyz.out");
    strcat(file_velocity, "/v.out");
    strcat(file_force, "/f.out");
    strcat(file_potential, "/potential.out");
    strcat(file_virial, "/virial.out");
    strcat(file_heat, "/heat.out");
}




Measure::~Measure(void)
{
    // nothing
}




void Measure::initialize
(Parameters *para, CPU_Data *cpu_data, GPU_Data *gpu_data)
{
    if (dump_thermo)    {fid_thermo   = my_fopen(file_thermo,   "a");}
    if (dump_position)  {fid_position = my_fopen(file_position, "a");}
    if (dump_velocity)  {fid_velocity = my_fopen(file_velocity, "a");}
    if (dump_force)     {fid_force    = my_fopen(file_force,    "a");}
    if (dump_potential) {fid_potential= my_fopen(file_potential,"a");}
    if (dump_virial)    {fid_virial   = my_fopen(file_virial,   "a");}
    if (dump_heat)      {fid_heat     = my_fopen(file_heat,     "a");}

    preprocess_vac(para,  cpu_data, gpu_data);
    preprocess_hac(para,  cpu_data, gpu_data);  
    preprocess_shc(para,  cpu_data, gpu_data); 
    preprocess_heat(para, cpu_data);      
    preprocess_hnemd_kappa(para, cpu_data, gpu_data);  
}




void Measure::finalize
(
    char *input_dir, Parameters *para, CPU_Data *cpu_data, GPU_Data *gpu_data, 
    Integrate *integrate
)
{
    if (dump_thermo)    {fclose(fid_thermo);    dump_thermo    = 0;}
    if (dump_position)  {fclose(fid_position);  dump_position  = 0;}
    if (dump_velocity)  {fclose(fid_velocity);  dump_velocity  = 0;}
    if (dump_force)     {fclose(fid_force);     dump_force     = 0;}
    if (dump_potential) {fclose(fid_potential); dump_potential = 0;}
    if (dump_virial)    {fclose(fid_virial);    dump_virial    = 0;}
    if (dump_heat)      {fclose(fid_heat);      dump_heat      = 0;}

    postprocess_vac(input_dir, para, cpu_data, gpu_data);
    postprocess_hac(input_dir, para, cpu_data, gpu_data, integrate);
    postprocess_shc(para, cpu_data, gpu_data);
    postprocess_heat(input_dir, para, cpu_data, integrate);
    postprocess_hnemd_kappa(para, cpu_data, gpu_data);
}




// dump thermodynamic properties
static void gpu_sample_thermo
(
    FILE *fid, Parameters *para, CPU_Data *cpu_data, 
    real *gpu_thermo, real *gpu_box_length, Ensemble *ensemble
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

    if (ensemble->type == 2)
    {
        // energy of the Nose-Hoover chain thermostat
        real kT = K_B * ensemble->temperature; 
        real energy_nhc = kT * (DIM * para->N) * ensemble->pos_nhc1[0];
        for (int m = 1; m < NOSE_HOOVER_CHAIN_LENGTH; m++)
        {
            energy_nhc += kT * ensemble->pos_nhc1[m];
        }
        for (int m = 0; m < NOSE_HOOVER_CHAIN_LENGTH; m++)
        { 
            energy_nhc += 0.5 * ensemble->vel_nhc1[m] 
                        * ensemble->vel_nhc1[m] / ensemble->mas_nhc1[m];
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
void Measure::dump_thermos
(
    FILE *fid, Parameters *para, CPU_Data *cpu_data, GPU_Data *gpu_data, 
    Integrate *integrate, int step
)
{
    if (dump_thermo)
    {
        if ((step + 1) % sample_interval_thermo == 0)
        {
            gpu_sample_thermo
            (
                fid, para, cpu_data, gpu_data->thermo, gpu_data->box_length, 
                integrate->ensemble
            );
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




void Measure::dump_positions
(FILE *fid, Parameters *para, CPU_Data *cpu_data, GPU_Data *gpu_data, int step)
{
    if (dump_position)
    {
        if ((step + 1) % sample_interval_position == 0)
        {
            gpu_dump_3(para->N, fid, gpu_data->x, gpu_data->y, gpu_data->z);
        }
    }
}




void Measure::dump_velocities
(FILE *fid, Parameters *para, CPU_Data *cpu_data, GPU_Data *gpu_data, int step)
{
    if (dump_velocity)
    {
        if ((step + 1) % sample_interval_velocity == 0)
        {
            gpu_dump_3(para->N, fid, gpu_data->vx, gpu_data->vy, gpu_data->vz);
        }
    }
}




void Measure::dump_forces
(FILE *fid, Parameters *para, CPU_Data *cpu_data, GPU_Data *gpu_data, int step)
{
    if (dump_force)
    {
        if ((step + 1) % sample_interval_force == 0)
        {
            gpu_dump_3(para->N, fid, gpu_data->fx, gpu_data->fy, gpu_data->fz);
        }
    }
}




void Measure::dump_virials
(FILE *fid, Parameters *para, CPU_Data *cpu_data, GPU_Data *gpu_data, int step)
{
    if (dump_virial)
    {
        if ((step + 1) % sample_interval_virial == 0)
        {
            gpu_dump_3
            (
                para->N, fid, gpu_data->virial_per_atom_x, 
                gpu_data->virial_per_atom_y, gpu_data->virial_per_atom_z
            );
        }
    }
}




static real get_volume(real *box_gpu)
{
    real *box_cpu;
    MY_MALLOC(box_cpu, real, 3);
    cudaMemcpy(box_cpu, box_gpu, sizeof(real) * 3, cudaMemcpyDeviceToHost);
    real volume = box_cpu[0] * box_cpu[1] * box_cpu[2];
    MY_FREE(box_cpu);
    return volume;
}




void Measure::dump_heats
(FILE *fid, Parameters *para, CPU_Data *cpu_data, GPU_Data *gpu_data, int step)
{
    if (dump_heat)
    {
        if (step == 0)
        {
            for (int n = 0; n < para->N * NUM_OF_HEAT_COMPONENTS; n++)
            {
                cpu_data->heat_per_atom[n] = ZERO;
            }
        }

        if ((step + 1) % sample_interval_heat == 0)
        {
            real *heat_cpu;
            MY_MALLOC(heat_cpu, real, para->N * NUM_OF_HEAT_COMPONENTS);
            cudaMemcpy
            (
                heat_cpu, gpu_data->heat_per_atom, 
                sizeof(real) * para->N * NUM_OF_HEAT_COMPONENTS, 
                cudaMemcpyDeviceToHost
            );
            for (int n = 0; n < para->N * NUM_OF_HEAT_COMPONENTS; n++)
            {
                cpu_data->heat_per_atom[n] += heat_cpu[n];
            }

            if ((step + 1) == para->number_of_steps)
            {
                int num = para->number_of_steps / sample_interval_heat;
                real volume = get_volume(gpu_data->box_length) / para->N;
                real factor = 10 * KAPPA_UNIT_CONVERSION / (num * volume);
                for (int n = 0; n < para->N; n++)
                {
                    for (int k = 0; k < NUM_OF_HEAT_COMPONENTS; k++)
                    {
                        // output per-atom heat flux in units of GW/m^2
                        real tmp = cpu_data->heat_per_atom[k * para->N + n];
                        fprintf(fid, "%25.15f", tmp * factor);
                    }
                    fprintf(fid, "\n");
                }
                fflush(fid);
            }
            MY_FREE(heat_cpu);
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




void Measure::dump_potentials
(FILE *fid, Parameters *para, CPU_Data *cpu_data, GPU_Data *gpu_data, int step)
{
    if (dump_potential)
    {
        if ((step + 1) % sample_interval_potential == 0)
        {
            gpu_dump_1(para->N, fid, gpu_data->potential_per_atom);
        }
    }
}




void Measure::compute
(
    char *input_dir, Parameters *para, CPU_Data *cpu_data, GPU_Data *gpu_data, 
    Integrate *integrate, int step
)
{
    dump_thermos(fid_thermo, para, cpu_data, gpu_data, integrate, step);
    dump_positions(fid_position, para, cpu_data, gpu_data, step);
    dump_velocities(fid_velocity, para, cpu_data, gpu_data, step);
    dump_forces(fid_force, para, cpu_data, gpu_data, step);
    dump_potentials(fid_potential, para, cpu_data, gpu_data, step);
    dump_virials(fid_virial, para, cpu_data, gpu_data, step);
    dump_heats(fid_heat, para, cpu_data, gpu_data, step);

    sample_vac(step, para, cpu_data, gpu_data);
    sample_hac(step, input_dir, para, cpu_data, gpu_data);
    sample_block_temperature(step, para, cpu_data, gpu_data, integrate);
    process_shc(step, input_dir, para, cpu_data, gpu_data);
    process_hnemd_kappa(step, input_dir, para, cpu_data, gpu_data, integrate); 
}




