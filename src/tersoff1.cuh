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




#ifndef TERSOFF1_H
#define TERSOFF1_H



#include "potential.cuh"




struct Tersoff_Parameters
{
    real a, b, lambda, mu, beta, n, c, d, c2, d2, h, r1, r2;
    real pi_factor, one_plus_c2overd2, minus_half_over_n;
};




struct Tersoff_Data
{
    real *b;     // bond orders
    real *bp;    // derivative of bond orders
    real *f12x;  // partial forces
    real *f12y;
    real *f12z;
};




class Tersoff1 : public Potential
{
public:   
    Tersoff1(FILE*, Parameters*);  
    virtual ~Tersoff1(void);
    virtual void compute(Parameters*, GPU_Data*);
protected:
    Tersoff_Parameters ters0;
    Tersoff_Data tersoff_data;
};




#endif




