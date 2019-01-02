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




#include "measure.cuh"

#include "integrate.cuh"
#include "ensemble.cuh"
#include "atom.cuh"
#include "memory.cuh"
#include "error.cuh"
#include "io.cuh"


#define DIM 3
#define NUM_OF_HEAT_COMPONENTS 5
#ifdef USE_DP
    #define K_B   8.617343e-5
    #define PRESSURE_UNIT_CONVERSION 1.602177e+2
#else
    #define K_B   8.617343e-5f
    #define PRESSURE_UNIT_CONVERSION 1.602177e+2f
#endif




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




void Measure::initialize(Atom *atom)
{
    if (dump_thermo)    {fid_thermo   = my_fopen(file_thermo,   "a");}
    if (dump_position)  {fid_position = my_fopen(file_position, "a");}
    if (dump_velocity)  {fid_velocity = my_fopen(file_velocity, "a");}
    if (dump_force)     {fid_force    = my_fopen(file_force,    "a");}
    if (dump_potential) {fid_potential= my_fopen(file_potential,"a");}
    if (dump_virial)    {fid_virial   = my_fopen(file_virial,   "a");}
    if (dump_heat)      {fid_heat     = my_fopen(file_heat,     "a");}

    vac.preprocess_vac(atom);
    hac.preprocess_hac(atom);
    shc.preprocess_shc(atom);
    heat.preprocess_heat(atom);
    hnemd.preprocess_hnemd_kappa(atom);
}




void Measure::finalize
(char *input_dir, Atom *atom, Integrate *integrate)
{
    if (dump_thermo)    {fclose(fid_thermo);    dump_thermo    = 0;}
    if (dump_position)  {fclose(fid_position);  dump_position  = 0;}
    if (dump_velocity)  {fclose(fid_velocity);  dump_velocity  = 0;}
    if (dump_force)     {fclose(fid_force);     dump_force     = 0;}
    if (dump_potential) {fclose(fid_potential); dump_potential = 0;}
    if (dump_virial)    {fclose(fid_virial);    dump_virial    = 0;}
    if (dump_heat)      {fclose(fid_heat);      dump_heat      = 0;}

    vac.postprocess_vac(input_dir, atom);
    hac.postprocess_hac(input_dir, atom, integrate);
    shc.postprocess_shc();
    heat.postprocess_heat(input_dir, atom, integrate);
    hnemd.postprocess_hnemd_kappa(atom);
}




// dump thermodynamic properties
static void gpu_sample_thermo
(
    FILE *fid, Atom* atom,
    real *gpu_thermo, real *gpu_box_length, Ensemble *ensemble
)
{

    // copy data from GPU to CPU
    real *thermo;
    MY_MALLOC(thermo, real, 6);
    real *box_length;
    MY_MALLOC(box_length, real, DIM);
    int m1 = sizeof(real) * 6;
    int m2 = sizeof(real) * DIM;
    CHECK(cudaMemcpy(thermo, gpu_thermo, m1, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(box_length, gpu_box_length, m2, cudaMemcpyDeviceToHost));

    // system energies
    real energy_system_kin = (0.5 * DIM) * atom->N * K_B * thermo[0];
    real energy_system_pot = thermo[1];
    real energy_system_total = energy_system_kin + energy_system_pot; 

    if (ensemble->type == 2)
    {
        // energy of the Nose-Hoover chain thermostat
        real kT = K_B * ensemble->temperature; 
        real energy_nhc = kT * (DIM * atom->N) * ensemble->pos_nhc1[0];
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
    MY_FREE(thermo);
    MY_FREE(box_length);
}




// dump thermodynamic properties (A wrapper function)
void Measure::dump_thermos
(FILE *fid, Atom *atom, Integrate *integrate, int step)
{
    if (dump_thermo)
    {
        if ((step + 1) % sample_interval_thermo == 0)
        {
            gpu_sample_thermo
            (fid, atom, atom->thermo, atom->box_length, integrate->ensemble);
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




void Measure::dump_positions(FILE *fid, Atom *atom, int step)
{
    if (dump_position)
    {
        if ((step + 1) % sample_interval_position == 0)
        {
            gpu_dump_3(atom->N, fid, atom->x, atom->y, atom->z);
        }
    }
}




void Measure::dump_velocities(FILE *fid, Atom *atom, int step)
{
    if (dump_velocity)
    {
        if ((step + 1) % sample_interval_velocity == 0)
        {
            gpu_dump_3(atom->N, fid, atom->vx, atom->vy, atom->vz);
        }
    }
}




void Measure::dump_forces(FILE *fid, Atom *atom, int step)
{
    if (dump_force)
    {
        if ((step + 1) % sample_interval_force == 0)
        {
            gpu_dump_3(atom->N, fid, atom->fx, atom->fy, atom->fz);
        }
    }
}




void Measure::dump_virials(FILE *fid, Atom *atom, int step)
{
    if (dump_virial)
    {
        if ((step + 1) % sample_interval_virial == 0)
        {
            gpu_dump_3
            (
                atom->N, fid, atom->virial_per_atom_x, 
                atom->virial_per_atom_y, atom->virial_per_atom_z
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




void Measure::dump_potentials(FILE *fid, Atom *atom, int step)
{
    if (dump_potential)
    {
        if ((step + 1) % sample_interval_potential == 0)
        {
            gpu_dump_1(atom->N, fid, atom->potential_per_atom);
        }
    }
}




void Measure::dump_heats(FILE *fid, Atom *atom, int step)
{
    if (dump_heat)
    {
        if ((step + 1) % sample_interval_heat == 0)
        {
            gpu_dump_1
            (atom->N * NUM_OF_HEAT_COMPONENTS, fid, atom->heat_per_atom);
        }
    }
}




void Measure::compute
(
    char *input_dir, Atom *atom, 
    Integrate *integrate, int step
)
{
    dump_thermos(fid_thermo, atom, integrate, step);
    dump_positions(fid_position, atom, step);
    dump_velocities(fid_velocity, atom, step);
    dump_forces(fid_force, atom, step);
    dump_potentials(fid_potential, atom, step);
    dump_virials(fid_virial, atom, step);
    dump_heats(fid_heat, atom, step);

    vac.sample_vac(step, atom);
    hac.sample_hac(step, input_dir, atom);
    heat.sample_block_temperature(step, atom, integrate);
    shc.process_shc(step, input_dir, atom);
    hnemd.process_hnemd_kappa(step, input_dir, atom, integrate);
}




