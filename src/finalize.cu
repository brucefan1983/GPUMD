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
#include "finalize.h"




void finalize(Force_Model *force_model, CPU_Data *cpu_data, GPU_Data *gpu_data)
{
    // Free the memory allocated on the GPU
    CHECK(cudaFree(gpu_data->NN)); 
    CHECK(cudaFree(gpu_data->NL)); 
#ifndef FIXED_NL
    CHECK(cudaFree(gpu_data->NN_local)); 
    CHECK(cudaFree(gpu_data->NL_local));
#endif
    CHECK(cudaFree(gpu_data->type));  
    CHECK(cudaFree(gpu_data->label)); 
    CHECK(cudaFree(gpu_data->group_size)); 
    CHECK(cudaFree(gpu_data->group_size_sum));
    CHECK(cudaFree(gpu_data->mass));
    CHECK(cudaFree(gpu_data->x0));  
    CHECK(cudaFree(gpu_data->y0));  
    CHECK(cudaFree(gpu_data->z0));
    CHECK(cudaFree(gpu_data->x));  
    CHECK(cudaFree(gpu_data->y));  
    CHECK(cudaFree(gpu_data->z));
    CHECK(cudaFree(gpu_data->vx)); 
    CHECK(cudaFree(gpu_data->vy)); 
    CHECK(cudaFree(gpu_data->vz));
    CHECK(cudaFree(gpu_data->fx)); 
    CHECK(cudaFree(gpu_data->fy)); 
    CHECK(cudaFree(gpu_data->fz));
    CHECK(cudaFree(gpu_data->virial_per_atom_x));
    CHECK(cudaFree(gpu_data->virial_per_atom_y));
    CHECK(cudaFree(gpu_data->virial_per_atom_z));
    CHECK(cudaFree(gpu_data->potential_per_atom));
    CHECK(cudaFree(gpu_data->heat_per_atom));    
    //#ifdef TRICLINIC
    CHECK(cudaFree(gpu_data->box_matrix));
    CHECK(cudaFree(gpu_data->box_matrix_inv));
    //#else
    CHECK(cudaFree(gpu_data->box_length));
    //#endif
    CHECK(cudaFree(gpu_data->thermo));

    // only for Tersoff-type potentials
    if (force_model->type >= 40 && force_model->type < 50)
    {
        CHECK(cudaFree(gpu_data->b));
        CHECK(cudaFree(gpu_data->bp));
    }

    // for the Vashishta-table potential
    if (force_model->type == 34)
    {
        CHECK(cudaFree(force_model->vas_table.table));
    }

    // Free the major memory allocated on the CPU
    MY_FREE(cpu_data->NN);
    MY_FREE(cpu_data->NL);
    MY_FREE(cpu_data->type);
    MY_FREE(cpu_data->label);
    MY_FREE(cpu_data->group_size);
    MY_FREE(cpu_data->group_size_sum);
    MY_FREE(cpu_data->mass);
    MY_FREE(cpu_data->x);
    MY_FREE(cpu_data->y);
    MY_FREE(cpu_data->z);
    MY_FREE(cpu_data->vx);
    MY_FREE(cpu_data->vy);
    MY_FREE(cpu_data->vz);
    MY_FREE(cpu_data->fx);
    MY_FREE(cpu_data->fy);
    MY_FREE(cpu_data->fz);  
    MY_FREE(cpu_data->thermo);
    MY_FREE(cpu_data->box_length);
    MY_FREE(cpu_data->box_matrix);
    MY_FREE(cpu_data->box_matrix_inv);
}



