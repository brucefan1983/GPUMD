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
#include <stdio.h>

//TODO allow a much larger MAX_TYPE -> need to allocate GPU memory
#define MAX_TYPE 10 // == max number of potentials


struct LJ_Para
{
    double s6e4[MAX_TYPE][MAX_TYPE];
    double s12e4[MAX_TYPE][MAX_TYPE];
    double cutoff_square[MAX_TYPE][MAX_TYPE];
};

class LJ : public Potential
{
public:
    LJ(FILE*, int, const std::vector<int>, int);
    virtual ~LJ(void);
    virtual void compute
    (
        const int type_shift,
        const Box& box,
        const Neighbor& neighbor,
        const GPU_Vector<int>& type,
        const GPU_Vector<double>& position,
        GPU_Vector<double>& potential,
        GPU_Vector<double>& force,
        GPU_Vector<double>& virial
    );
    void initialize_lj(FILE *fid, int, const std::vector<int>, int);

protected:
    LJ_Para  lj_para;
    bool pair_participating(int, int, const std::vector<int>);
};


