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


#ifndef DUMP_H
#define DUMP_H

void dump_thermos
(FILE *fid, Parameters *para, CPU_Data *cpu_data, GPU_Data *gpu_data, int step);

void dump_positions
(FILE *fid, Parameters *para, CPU_Data *cpu_data, GPU_Data *gpu_data, int step);

void dump_velocities
(FILE *fid, Parameters *para, CPU_Data *cpu_data, GPU_Data *gpu_data, int step);

void dump_forces
(FILE *fid, Parameters *para, CPU_Data *cpu_data, GPU_Data *gpu_data, int step);

void dump_potential
(FILE *fid, Parameters *para, CPU_Data *cpu_data, GPU_Data *gpu_data, int step);

void dump_virial
(FILE *fid, Parameters *para, CPU_Data *cpu_data, GPU_Data *gpu_data, int step);

void dump_heat
(FILE *fid, Parameters *para, CPU_Data *cpu_data, GPU_Data *gpu_data, int step);

#endif
