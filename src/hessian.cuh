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


class Hessian
{
public:
    int yes = 0;      // 1 means calculating the hessian
    real* H;          // the 3N x 3N hessian
    real dx = 0.005;  // displacement in units of A

    Hessian(Atom*, Force*, Measure*);
    ~Hessian(void);

protected:
    real *f_positive; // F_i(+)
    real *f_negative; // F_i(-)
    real dx2 = 0.010; // dx * 2

    void find_H(Atom*, Force*, Measure*);
    void find_H12(int, int, Atom*, Force*, Measure*, real*);
    void shift_atom(int, int, int, Atom*);
    void get_f(int, int, int, int, Atom*, Force*, Measure*, real*);
};


