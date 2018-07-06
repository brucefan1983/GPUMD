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




#ifndef REBO_MOS2_H
#define REBO_MOS2_H




#include "potential.cuh"




struct REBO_MOS_Data
{
    real *b;     // bond-order function
    real *bp;
    real *p;     // coordination function
    real *pp;
    real *f12x;  // partial forces
    real *f12y;
    real *f12z;
    int *NN_short; // for many-body part
    int *NL_short; // for many-body part
};




class REBO_MOS : public Potential
{
public:   
    REBO_MOS(Parameters*);
    virtual ~REBO_MOS(void);
    virtual void compute(Parameters*, GPU_Data*);
protected:
    REBO_MOS_Data rebo_mos_data;
};




#endif


