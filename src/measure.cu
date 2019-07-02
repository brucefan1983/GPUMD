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
The driver class dealing with measurement.
------------------------------------------------------------------------------*/


#include "measure.cuh"
#include "integrate.cuh"
#include "ensemble.cuh"
#include "atom.cuh"
#include "error.cuh"

#define DIM 3
#define NUM_OF_HEAT_COMPONENTS 5
#define NUM_OF_PROPERTIES      5 


Measure::Measure(char *input_dir)
{
    dump_thermo = 0;
    dump_restart = 0;
    dump_velocity = 0;
    dump_force = 0;
    dump_potential = 0;
    dump_virial = 0;
    dump_heat = 0;
    dump_pos = NULL; // to avoid deleting random memory in run
    strcpy(file_thermo, input_dir);
    strcpy(file_restart, input_dir);
    strcpy(file_velocity, input_dir);
    strcpy(file_force, input_dir);
    strcpy(file_potential, input_dir);
    strcpy(file_virial, input_dir);
    strcpy(file_heat, input_dir);
    strcat(file_thermo, "/thermo.out");
    strcat(file_restart, "/restart.out");
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


void Measure::initialize(char* input_dir, Atom *atom)
{
    if (dump_thermo)    {fid_thermo   = my_fopen(file_thermo,   "a");}
    if (dump_velocity)  {fid_velocity = my_fopen(file_velocity, "a");}
    if (dump_force)     {fid_force    = my_fopen(file_force,    "a");}
    if (dump_potential) {fid_potential= my_fopen(file_potential,"a");}
    if (dump_virial)    {fid_virial   = my_fopen(file_virial,   "a");}
    if (dump_heat)      {fid_heat     = my_fopen(file_heat,     "a");}
    if (dump_pos)       {dump_pos->initialize(input_dir);}
    vac.preprocess(atom);
    dos.preprocess(atom, &vac);
    hac.preprocess(atom);
    shc.preprocess(atom);
    compute.preprocess(input_dir, atom);
    hnemd.preprocess(atom);
}


void Measure::finalize
(char *input_dir, Atom *atom, Integrate *integrate)
{
    if (dump_thermo)    {fclose(fid_thermo);    dump_thermo    = 0;}
    if (dump_restart)   {                       dump_restart   = 0;}
    if (dump_velocity)  {fclose(fid_velocity);  dump_velocity  = 0;}
    if (dump_force)     {fclose(fid_force);     dump_force     = 0;}
    if (dump_potential) {fclose(fid_potential); dump_potential = 0;}
    if (dump_virial)    {fclose(fid_virial);    dump_virial    = 0;}
    if (dump_heat)      {fclose(fid_heat);      dump_heat      = 0;}
    if (dump_pos)       {dump_pos->finalize();}
    vac.postprocess(input_dir, atom, &dos, &sdc);
    hac.postprocess(input_dir, atom, integrate);
    shc.postprocess();
    compute.postprocess(atom, integrate);
    hnemd.postprocess(atom);
}


void Measure::dump_thermos(FILE *fid, Atom *atom, int step)
{
    if (!dump_thermo) return;
    if ((step + 1) % sample_interval_thermo != 0) return;
    real *thermo; MY_MALLOC(thermo, real, NUM_OF_PROPERTIES);
    int m1 = sizeof(real) * NUM_OF_PROPERTIES;
    CHECK(cudaMemcpy(thermo, atom->thermo, m1, cudaMemcpyDeviceToHost));
    int N_fixed = (atom->fixed_group == -1) ? 0 :
        atom->group[0].cpu_size[atom->fixed_group];
    real energy_kin = (0.5 * DIM) * (atom->N - N_fixed) * K_B * thermo[0];
    fprintf(fid, "%20.10e%20.10e%20.10e%20.10e%20.10e%20.10e", thermo[0],
        energy_kin, thermo[1], thermo[2]*PRESSURE_UNIT_CONVERSION,
        thermo[3]*PRESSURE_UNIT_CONVERSION, thermo[4]*PRESSURE_UNIT_CONVERSION);
    int number_of_box_variables = atom->box.triclinic ? 9 : 3;
    for (int m = 0; m < number_of_box_variables; ++m)
    {
        fprintf(fid, "%20.10e", atom->box.cpu_h[m]);
    }
    fprintf(fid, "\n"); fflush(fid); MY_FREE(thermo);
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
        fprintf(fid, "%g %g %g\n", cpu_a[n], cpu_b[n], cpu_c[n]);
    }
    fflush(fid);
    MY_FREE(cpu_a); MY_FREE(cpu_b); MY_FREE(cpu_c);
}


void Measure::dump_restarts(Atom *atom, int step)
{
    if (!dump_restart) return;
    if ((step + 1) % sample_interval_restart != 0) return;
    int memory = sizeof(real) * atom->N;
    CHECK(cudaMemcpy(atom->cpu_x, atom->x, memory, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(atom->cpu_y, atom->y, memory, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(atom->cpu_z, atom->z, memory, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(atom->cpu_vx, atom->vx, memory, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(atom->cpu_vy, atom->vy, memory, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(atom->cpu_vz, atom->vz, memory, cudaMemcpyDeviceToHost));
    fid_restart = my_fopen(file_restart, "w"); 
    fprintf(fid_restart, "%d %d %g %d %d %d %d\n", atom->N, atom->neighbor.MN,
        atom->neighbor.rc, atom->box.triclinic, 1,
        atom->num_of_grouping_methods);
    if (atom->box.triclinic == 0)
    {
        fprintf(fid_restart, "%d %d %d %g %g %g\n", atom->box.pbc_x,
            atom->box.pbc_y, atom->box.pbc_z, atom->box.cpu_h[0],
            atom->box.cpu_h[1], atom->box.cpu_h[2]);
    }
    else
    {
        fprintf(fid_restart, "%d %d %d\n", atom->box.pbc_x,
            atom->box.pbc_y, atom->box.pbc_z);
        for (int d1 = 0; d1 < 3; ++d1)
        {
            for (int d2 = 0; d2 < 3; ++d2)
            {
                fprintf(fid_restart, "%g ", atom->box.cpu_h[d1 * 3 + d2]);
            }
            fprintf(fid_restart, "\n");
        }
    }
    for (int n = 0; n < atom->N; n++)
    {
        fprintf(fid_restart, "%d %g %g %g %g %g %g %g ", atom->cpu_type[n],
            atom->cpu_x[n], atom->cpu_y[n], atom->cpu_z[n], atom->cpu_mass[n],
            atom->cpu_vx[n], atom->cpu_vy[n], atom->cpu_vz[n]);
        for (int m = 0; m < atom->num_of_grouping_methods; ++m)
        {
            fprintf(fid_restart, "%d ", atom->group[m].cpu_label[n]);
        }
        fprintf(fid_restart, "\n");
    }
    fflush(fid_restart);
    fclose(fid_restart);
}


void Measure::dump_velocities(FILE *fid, Atom *atom, int step)
{
    if (!dump_velocity) return;
    if ((step + 1) % sample_interval_velocity != 0) return;
    gpu_dump_3(atom->N, fid, atom->vx, atom->vy, atom->vz);
}


void Measure::dump_forces(FILE *fid, Atom *atom, int step)
{
    if (!dump_force) return;
    if ((step + 1) % sample_interval_force != 0) return;
    gpu_dump_3(atom->N, fid, atom->fx, atom->fy, atom->fz);
}


void Measure::dump_virials(FILE *fid, Atom *atom, int step)
{
    if (!dump_virial) return;
    if ((step + 1) % sample_interval_virial != 0) return;
    gpu_dump_3(atom->N, fid, atom->virial_per_atom_x, atom->virial_per_atom_y,
        atom->virial_per_atom_z);
}


static void gpu_dump_1(int N, FILE *fid, real *a)
{
    real *cpu_a; MY_MALLOC(cpu_a, real, N);
    CHECK(cudaMemcpy(cpu_a, a, sizeof(real) * N, cudaMemcpyDeviceToHost));
    for (int n = 0; n < N; n++) { fprintf(fid, "%g\n", cpu_a[n]); }
    fflush(fid); MY_FREE(cpu_a);
}


void Measure::dump_potentials(FILE *fid, Atom *atom, int step)
{
    if (!dump_potential) return;
    if ((step + 1) % sample_interval_potential != 0) return;
    gpu_dump_1(atom->N, fid, atom->potential_per_atom);
}


void Measure::dump_heats(FILE *fid, Atom *atom, int step)
{
    if (!dump_heat) return;
    if ((step + 1) % sample_interval_heat != 0) return;
    gpu_dump_1(atom->N * NUM_OF_HEAT_COMPONENTS, fid, atom->heat_per_atom);
}


void Measure::process
(char *input_dir, Atom *atom, Integrate *integrate, int step)
{
    dump_thermos(fid_thermo, atom, step);
    dump_restarts(atom, step);
    dump_velocities(fid_velocity, atom, step);
    dump_forces(fid_force, atom, step);
    dump_potentials(fid_potential, atom, step);
    dump_virials(fid_virial, atom, step);
    dump_heats(fid_heat, atom, step);
    compute.process(step, atom, integrate);
    vac.process(step, atom);
    hac.process(step, input_dir, atom);
    shc.process(step, input_dir, atom);
    hnemd.process(step, input_dir, atom, integrate);
    dump_pos->dump(atom, step);
}


