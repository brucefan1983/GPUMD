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


#ifndef LJ1_H
#define LJ1_H




#include "potential.cuh"




struct LJ1_Para
{
    real s6e24;
    real s12e24;
    real s6e4;
    real s12e4;
    real cutoff_square;
};




// to be changed
struct RI_Para
{
    real a11, b11, c11, qq11;
    real a22, b22, c22, qq22;
    real a12, b12, c12, qq12;
    real cutoff;
};




class Pair : public Potential
{
public:   
    Pair(FILE*, Parameters*, int potential_model);
    virtual ~Pair(void);
    virtual void compute(Parameters*, GPU_Data*);
    void initialize_lj1(FILE *fid);
    void initialize_ri(FILE *fid);
protected:
    int      potential_model; 
    LJ1_Para lj1_para;
    RI_Para  ri_para;
};



#endif




