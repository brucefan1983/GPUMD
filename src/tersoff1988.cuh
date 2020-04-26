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
#include <stdio.h>


struct Tersoff1988_Data
{
    double *b;     // bond orders
    double *bp;    // derivative of bond orders
    double *f12x;  // partial forces
    double *f12y;
    double *f12z;
};




class Tersoff1988 : public Potential
{
public:   
    Tersoff1988(FILE*, Atom*, int sum_of_types);
    virtual ~Tersoff1988(void);
    virtual void compute(Atom*, Measure*, int);
protected:
    int num_types;
    double *ters;
    Tersoff1988_Data tersoff_data;
};




