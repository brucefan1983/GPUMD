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
#include "common.cuh"

class Atom;


class SHC
{
public:
    int compute = 0;       // 0 = not computing shc; 1 = computing shc
    int group_method = -1; // -1 means not using a group method
    int group_id = 0;      // calculating SHC for atoms in group id
    int sample_interval;   // sample interval for heat current
    int Nc;                // number of correlation points
    int direction;         // transport direction: 0=x; 1=y; 2=z
    void preprocess(Atom*);
    void process(int, Atom*);
    void postprocess(char*);
private:
    int num_time_origins;  // number of time origins for ensemble average
    int group_size;        // number of atoms in group_id
    GPU_Vector<double> vx, vy, vz; // Nc frames of velocity data
    GPU_Vector<double> sx, sy, sz; // one frame of virial data
    GPU_Vector<double> ki, ko;     // The correlation functions Ki(t) and Ko(t)
    void find_shc(char*, Atom*, int);
};


