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
#include "potential.cuh"
#include <vector>

#define MAX_TYPE 5 // do you want to have more than 5 atom types?


struct LJ_Para
{
    real s6e4[MAX_TYPE][MAX_TYPE];
    real s12e4[MAX_TYPE][MAX_TYPE];
    real cutoff_square[MAX_TYPE][MAX_TYPE];
};


struct RI_Para
{
    real a11, b11, c11, qq11;
    real a22, b22, c22, qq22;
    real a12, b12, c12, qq12;
    real v_rc, dv_rc; // potential and its derivative at the cutoff distance
    real cutoff;
};


class Pair : public Potential
{
public:   
    Pair(FILE*, int potential_model, const vector<int>, int);
    virtual ~Pair(void);
    virtual void compute(Atom*, Measure*, int);
    void initialize_lj(FILE *fid, int, const vector<int>, int);
    void initialize_ri(FILE *fid);
protected:
    int      potential_model; 
    LJ_Para  lj_para;
    RI_Para  ri_para;
    bool pair_participating(int, int, const vector<int>);
};


