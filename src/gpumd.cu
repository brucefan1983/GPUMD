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
#include "force.cuh"
#include "integrate.cuh"
#include "gpumd.cuh"
#include "measure.cuh"
#include "initialize.cuh"
#include "finalize.cuh"
#include "run.cuh" 




/*----------------------------------------------------------------------------80
    This is the driver function of GPUMD.
------------------------------------------------------------------------------*/

void gpumd(char *input_dir)
{ 
    // Data structures:
    Parameters  para;
    CPU_Data    cpu_data;
    GPU_Data    gpu_data;
    Force       force;
    Integrate   integrate;
    Measure     measure(input_dir);

    // initialize:
    initialize(input_dir, &para, &cpu_data, &gpu_data);
    
    // run 
    run_md(input_dir, &para, &cpu_data, &gpu_data, &force, &integrate, &measure);

    // finilize
    finalize(&cpu_data, &gpu_data);
}




