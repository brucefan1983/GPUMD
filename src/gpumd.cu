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
#include "gpumd.h"

#include "initialize.h"
#include "finalize.h"
#include "run.h" 




/*----------------------------------------------------------------------------80
    This is the driver function of GPUMD.
------------------------------------------------------------------------------*/

void gpumd(char *input_dir)
{ 
    // Data structures:
    Parameters  para;
    Files       files;
    Force_Model force_model;
    CPU_Data    cpu_data;
    GPU_Data    gpu_data;

    // initialize:
    initialize(input_dir, &files, &para, &cpu_data, &gpu_data);
    
    // run 
    run_md(&files, &force_model, &para, &cpu_data, &gpu_data);

    // finilize
    finalize(&force_model, &cpu_data, &gpu_data);
}



