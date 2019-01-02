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




struct Tersoff2_Parameters
{
    real a, b, lambda, mu, beta, n, c, d, c2, d2, h, r1, r2;
    real pi_factor, one_plus_c2overd2, minus_half_over_n;
};




struct Tersoff2_Data
{
    real *b;     // bond orders
    real *bp;    // derivative of bond orders
    real *f12x;  // partial forces
    real *f12y;
    real *f12z;
};




class Tersoff2 : public Potential
{
public:   
    Tersoff2(FILE*, Parameters*, int sum_of_types);  
    virtual ~Tersoff2(void);
    virtual void compute(Parameters*, Atom*, Measure*);
protected:
    Tersoff2_Parameters ters0;
    Tersoff2_Parameters ters1;
    Tersoff2_Parameters ters2;
    Tersoff2_Data tersoff_data;
};




