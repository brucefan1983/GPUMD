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




struct Vashishta_Para
{
    real B[2], cos0[2], C, r0, rc; real v_rc[3], dv_rc[3];
    real H[3], qq[3], lambda_inv[3], D[3], xi_inv[3], W[3];
    int eta[3];
    real rmin;
    real scale;
    int N;
};




struct Vashishta_Data
{
    real *table; // for the two-body part
    real *f12x;  // partial forces
    real *f12y;
    real *f12z;
    int *NN_short; // for three-body part
    int *NL_short; // for three-body part
};




class Vashishta : public Potential
{
public:   
    Vashishta(FILE*, Parameters*, Atom*, int use_table);  
    virtual ~Vashishta(void);
    virtual void compute(Parameters*, Atom*, Measure*);
    void initialize_0(FILE*);
    void initialize_1(FILE*);
protected:
    int            use_table;
    Vashishta_Para vashishta_para;
    Vashishta_Data vashishta_data;
};




