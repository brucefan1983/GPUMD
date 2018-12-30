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


#ifndef SHC_H
#define SHC_H



class SHC
{
public:
    int compute = 0;
    int sample_interval; // sample interval for heat current
    int Nc;              // number of correlation points
    int M;               // number of time origins for one average 
    int number_of_pairs;    // number of atom pairs between block A and block B
    int number_of_sections; // fixed to 1; may be changed in a future version
    int block_A;         // record the heat flowing from block A
    int block_B;         // record the heat flowing into block B

    void preprocess_shc(Parameters*, CPU_Data*, GPU_Data*);
    void process_shc(int step, char *, Parameters*, CPU_Data*, GPU_Data*);
    void postprocess_shc(Parameters*, CPU_Data*, GPU_Data*);

    void build_fv_table
    (Parameters *para, CPU_Data *cpu_data, GPU_Data *gpu_data);
    void find_k_time
    (char *input_dir, Parameters *para, CPU_Data *cpu_data,GPU_Data *gpu_data);
};




#endif




