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
#include "gpu_vector.cuh"


class Atom;


class VAC
{
public:

    // NOTE: (compute_dos && compute_sdc) == 1 yields failure
    int compute_dos;          // 1 means mass-weighted VAC computed
    int compute_sdc;          // 1 means VAC computed
    int sample_interval;      // sample interval for velocity
    int grouping_method = -1; // grouping method to use, -1 means none set
    int group = -1;           // group to compute, -1 means none set
    int Nc;                   // number of correlation points
    int num_dos_points = -1;  // points to use for DOS output, -1 means not set
    double omega_max;           // maximal angular frequency for phonons

    void preprocess(Atom*);
    void process(const int, Atom*);
    void postprocess(const char*, Atom*);

private:

    int N;                    // number of atoms for computation
    int num_time_origins;     // number of time origins
    double dt;                  // time interval in natural units
    double dt_in_ps;            // time interval in units of ps
    void find_dos(const char *, Atom *);
    void find_sdc(const char *, Atom *);
    GPU_Vector<double> mass;
    GPU_Vector<double> vx, vy, vz;
    GPU_Vector<double> vac_x, vac_y, vac_z;
};


