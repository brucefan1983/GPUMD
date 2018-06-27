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


#ifndef SW1_H
#define SW1_H



#include "potential.cuh"



struct SW1_Para
{
    real epsilon, A, lambda, B, a, gamma, sigma, cos0; 
    real epsilon_times_A, epsilon_times_lambda, sigma_times_a;
};




struct SW1_Data
{
    real *f12x;  // partial forces
    real *f12y;
    real *f12z;
};




class SW1 : public Potential
{
public:   
    SW1(FILE*, Parameters*);  
    virtual ~SW1(void);
    virtual void compute(Parameters*, GPU_Data*);
protected:
    SW1_Para sw1_para;
    SW1_Data sw1_data;
};




#endif


