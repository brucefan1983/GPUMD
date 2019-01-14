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


class VAC
{
public:
    int compute;         // 1 means you want to do this computation
    int sample_interval; // sample interval for velocity
    int Nc;              // number of correlation points
    real omega_max;    // maximal angular frequency for phonons
    void preprocess(Atom*);
    void process(int step, Atom*);
    void postprocess(char*, Atom*);

private:
    void find_vac_rdc_dos(char *input_dir, Atom *atom);
    real *vx_all;
    real *vy_all;
    real *vz_all;
};




