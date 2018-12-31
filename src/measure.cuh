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




#ifndef MEASURE_H
#define MEASURE_H

#include "vac.cuh"
#include "hac.cuh"
#include "shc.cuh"
#include "hnemd_kappa.cuh"
#include "heat.cuh"
class Integrate;



#define FILE_NAME_LENGTH      200




class Measure
{
public:
    Measure(char *input_dir);
    ~Measure(void);
    void initialize(Parameters*, CPU_Data*, GPU_Data*);
    void finalize(char*, Parameters*, CPU_Data*, GPU_Data*, Integrate*);
    void compute(char*, Parameters*, CPU_Data*, GPU_Data*, Integrate*, int);
    int dump_thermo; 
    int dump_position;
    int dump_velocity;
    int dump_force;
    int dump_potential;
    int dump_virial;
    int dump_heat;
    int sample_interval_thermo;
    int sample_interval_position;
    int sample_interval_velocity;
    int sample_interval_force;
    int sample_interval_potential;
    int sample_interval_virial;
    int sample_interval_heat;
    FILE *fid_thermo;
    FILE *fid_position;
    FILE *fid_velocity;
    FILE *fid_force;
    FILE *fid_potential;
    FILE *fid_virial;
    FILE *fid_heat;
    char file_thermo[FILE_NAME_LENGTH];       
    char file_position[FILE_NAME_LENGTH];    
    char file_velocity[FILE_NAME_LENGTH];    
    char file_force[FILE_NAME_LENGTH]; 
    char file_potential[FILE_NAME_LENGTH];
    char file_virial[FILE_NAME_LENGTH];    
    char file_heat[FILE_NAME_LENGTH];
    VAC vac;
    HAC hac;
    SHC shc;
    HNEMD hnemd;
    Heat heat;
protected:
    void dump_thermos(FILE*, Parameters*, CPU_Data*, GPU_Data*, Integrate*, int);
    void dump_positions(FILE*, Parameters*, CPU_Data*, GPU_Data*, int);
    void dump_velocities(FILE*, Parameters*, CPU_Data*, GPU_Data*, int);
    void dump_forces(FILE*, Parameters*, CPU_Data*, GPU_Data*, int);
    void dump_potentials(FILE*, Parameters*, CPU_Data*, GPU_Data*, int);
    void dump_virials(FILE*, Parameters*, CPU_Data*, GPU_Data*, int);
    void dump_heats(FILE*, Parameters*, CPU_Data*, GPU_Data*, int);
};




#endif




