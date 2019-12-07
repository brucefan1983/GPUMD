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
    real dx = 0.005;
    real cutoff = 4.0;
    void compute(char*, Atom*, Force*, Measure*);
    void parse_cutoff(char**, int);
    void parse_delta(char**, int);
protected:
    int num_basis;
    int num_kpoints;
    real cutoff_square;
    int* basis;
    int* label;
    real* mass;
    real* kpoints;
    real* H;
    real* DR;
    real* DI;
    void shift_atom(real, int, int, Atom*);
    void get_f(real, int, int, int, Atom*, Force*, Measure*, real*);
    void read_basis(char*, int N);
    void read_kpoints(char*);
    void initialize(char*, int);
    void finalize(void);
    void find_H(Atom*, Force*, Measure*);
    void find_H12(real, int, int, Atom*, Force*, Measure*, real*);
    bool is_too_far(int, int, Atom*);
    void find_D(char*, Atom*);
    void find_eigenvectors(char*, Atom*);
    void output_D(char*);
    void find_omega(FILE*, int);
    void find_omega_batch(FILE*);
};


