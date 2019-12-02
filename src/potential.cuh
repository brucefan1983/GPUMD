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
#include "common.cuh"


class Potential
{
public:
    int N1; int N2;
    real rc; // maximum cutoff distance
    Potential(void);
    virtual ~Potential(void);
    virtual void compute(Atom*, Measure*, int) = 0;

protected:
    //int compute_j   = 0; // 1 for computing heat current
    int compute_gkma = 0; // 1 for computing gkma
    //int compute_hnemd = 0; // 1 for computing hnemd or hnema
    void find_properties_many_body
    (Atom*, Measure*, int*, int*, real*, real*, real*);
};


