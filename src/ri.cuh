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

struct RI_Para
{
    double a11, b11, c11, qq11;
    double a22, b22, c22, qq22;
    double a12, b12, c12, qq12;
    double v_rc, dv_rc; // potential and its derivative at the cutoff distance
    double cutoff;
};

class RI : public Potential
{
public:
    RI(FILE*);
    virtual ~RI(void);
    virtual void compute(Atom*, Measure*, int);
    void initialize_ri(FILE *fid);
protected:
    RI_Para  ri_para;
};
