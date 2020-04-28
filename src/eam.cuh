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


struct EAM2004Zhou
{
    double re, fe, rho_e, rho_s, rho_n, rho_0, alpha, beta, A, B, kappa, lambda;
    double Fn0, Fn1, Fn2, Fn3, F0, F1, F2, F3, eta, Fe;
    double rc; // chosen by the user?
};


struct EAM2006Dai
{
    double A, d, c, c0, c1, c2, c3, c4, B, rc;
};


struct EAM_Data
{
    double *Fp;    // derivative of the density functional
};


class EAM : public Potential
{
public:   
    EAM(FILE*, Atom*, char*);  
    virtual ~EAM(void);
    virtual void compute(Atom*, int);
    void initialize_eam2004zhou(FILE*);
    void initialize_eam2006dai(FILE*);
protected:
    int          potential_model; 
    EAM2004Zhou  eam2004zhou;
    EAM2006Dai   eam2006dai;
    EAM_Data     eam_data;
};


