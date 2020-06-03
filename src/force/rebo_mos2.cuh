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
#include "utilities/gpu_vector.cuh"


struct REBO_MOS_Data
{
    GPU_Vector<double> b;     // bond-order function
    GPU_Vector<double> bp;
    GPU_Vector<double> p;     // coordination function
    GPU_Vector<double> pp;
    GPU_Vector<double> f12x;  // partial forces
    GPU_Vector<double> f12y;
    GPU_Vector<double> f12z;
    GPU_Vector<int> NN_short; // for many-body part
    GPU_Vector<int> NL_short; // for many-body part
};


class REBO_MOS : public Potential
{
public:   
    REBO_MOS(const Neighbor& neighbor);
    virtual ~REBO_MOS(void);
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
protected:
    REBO_MOS_Data rebo_mos_data;
};


