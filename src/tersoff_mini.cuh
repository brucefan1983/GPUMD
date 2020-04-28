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


struct Tersoff_mini_Data
{
    double *b;     // bond orders
    double *bp;    // derivative of bond orders
    double *f12x;  // partial forces
    double *f12y;
    double *f12z;
};


struct Tersoff_mini_Para
{
    double a[3];
    double b[3];
    double lambda[3];
    double mu[3];
    double beta[3];
    double n[3];
    double h[3];
    double r1[3];
    double r2[3];
    double pi_factor[3];
    double minus_half_over_n[3];
};


class Tersoff_mini : public Potential
{
public:   
    Tersoff_mini(FILE*, Atom*, int);
    virtual ~Tersoff_mini(void);
    virtual void compute(Atom*, int);
protected:
    int num_types; // number of atom tpyes
    Tersoff_mini_Data tersoff_mini_data;
    Tersoff_mini_Para para;
};


