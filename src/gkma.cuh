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
#include "error.cuh"
#include "mic.cuh"

class GKMA
{
public:
    int compute = 0;
    int sample_interval;// number of time steps per heat current computation
    int output_interval;// number of time steps to output average heat current
    int first_mode;     // first mode to consider
    int last_mode;      // last mode to consider
    int bin_size;       // number of modes per bin
    real f_bin_size;    // freq. range per bin (THz)
    int f_flag;         // 0 -> modes, 1 -> freq.
    int num_modes;      // total number of modes to consider
    int atom_begin;     // Beginning atom group/type
    int atom_end;       // End atom group/type

    real* eig;          // eigenvectors
    real* xdotn;        // per-atom modal velocity
    real* xdot;         // modal velocities
    real* jmn;          // per-atom modal heat current
    real* jm;           // total modal heat current
    real* bin_out;      // modal binning structure
    int* bin_count;     // Number of modes per bin when f_flag=1
    int* bin_sum;       // Running sum from bin_count


    char eig_file_position[FILE_NAME_LENGTH];
    char gkma_file_position[FILE_NAME_LENGTH];

    void preprocess(char*, Atom*);
    void process(int, Atom*);
    void postprocess();

private:
    int samples_per_output;// samples to be averaged for output
    int num_bins;          // number of bins to output
    int N1;                // Atom starting index
    int N2;                // Atom ending index
    int num_participating; // Number of particles participating

    void compute_gkma_heat(Atom*);
    void setN(Atom*);

};
