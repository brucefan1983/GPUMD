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


#pragma once
#include "common.cuh"



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
    int count_a;
    int count_b;

    real* fv;
    real* fv_all;
    int* fv_index;
    int *a_map;
    int *b_map;

    void preprocess_shc(Atom*);
    void process_shc(int step, char *, Atom*);
    void postprocess_shc(void);

    void build_fv_table(Atom*, int*, int*, int*, int*, int*);
    void find_k_time(char *input_dir, Atom *atom);
};




