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
GPUMD Contributing author: Alexander Gabourie (Stanford University)
------------------------------------------------------------------------------*/

#pragma once
#include "common.cuh"
#include "mic.cuh"
#include "atom.cuh"
#include "integrate.cuh"
#include "ensemble.cuh"
#include <fstream>
#include <string>
#include <iostream>
#include <sstream>

#define NO_METHOD -1
#define GKMA_METHOD 0
#define HNEMA_METHOD 1

class MODAL_ANALYSIS
{
public:
    // Bookkeeping variables
    int compute = 0;
    int method = NO_METHOD; // Method to compute
    int output_interval;    // number of times steps to output average heat current
    int sample_interval;    // steps per heat current computation
    int first_mode;         // first mode to consider
    int last_mode;          // last mode to consider
    int bin_size;           // number of modes per bin
    double f_bin_size;        // freq. range per bin (THz)
    int f_flag;             // 0 -> modes, 1 -> freq.
    int num_modes;          // total number of modes to consider
    int atom_begin;         // Beginning atom group/type
    int atom_end;           // End atom group/type

    // Data variables
    float* eig_x;            // eigenvectors x
    float* eig_y;            // eigenvectors y
    float* eig_z;            // eigenvectors z

    float* xdot_x;           // modal velocities
    float* xdot_y;
    float* xdot_z;

    float* vim_x;            // real velocity mode projection
    float* vim_y;
    float* vim_z;

    float* vib_x;            // real velocity binned
    float* vib_y;
    float* vib_z;

    float* sqrtmass;         // precalculated mass values
    float* rsqrtmass;

    float* jmx;
    float* jmy;
    float* jmz;

    float* smx;               // stress by by square root mass
    float* smy;
    float* smz;

    float* jtmp;             // placeholder for intermediate

    //    float* jmn;              // per-atom modal heat current
//    float* jm;               // total modal heat current
    float* bin_out;          // modal binning structure
    int* bin_count;          // Number of modes per bin when f_flag=1
    int* bin_sum;            // Running sum from bin_count

    char eig_file_position[FILE_NAME_LENGTH];
    char output_file_position[FILE_NAME_LENGTH];

    void preprocess(char*, Atom*);
    void process(int, Atom*, Integrate*, double);
    void postprocess();

private:
    int samples_per_output;  // samples to be averaged for output
    int num_bins;            // number of bins to output
    int N1;                  // Atom starting index
    int N2;                  // Atom ending index
    int num_participating;   // Number of particles participating
    int num_heat_stored;     // Number of stored heat current elements
    float* mv_x;             // sqrt(mass)*velocity intermediate variable
    float* mv_y;
    float* mv_z;

    void compute_heat(Atom*);
    void setN(Atom*);
    void set_eigmode(int, std::ifstream&, float*);

};
