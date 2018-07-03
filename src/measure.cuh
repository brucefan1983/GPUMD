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
class Integrate;




class Measure
{
public:
    Measure(void);
    ~Measure(void);
    void initialize(Files*);
    void finalize(Files*);
    void compute(Files*, Parameters*, CPU_Data*, GPU_Data*, Integrate*, int);
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




