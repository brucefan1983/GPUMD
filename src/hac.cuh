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


class HAC
{
public:
    int compute = 0;
    int sample_interval; // sample interval for heat current
    int Nc;              // number of correlation points
    int output_interval; // only output Nc/output_interval data

    void preprocess(Atom*);
    void process(int, char*, Atom*);
    void postprocess(char*, Atom*, Integrate*);

private:
    void find_hac_kappa(char*, Atom*, Integrate*);
    real *heat_all;
};


