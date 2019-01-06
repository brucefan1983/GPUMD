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




class Compute
{
public:
    int compute_temp = 0;
    int interval_temp;

    void preprocess(char*, Atom*);
    void postprocess(Atom* atom, Integrate*);
    void process(int, Atom*, Integrate*);

private:
    FILE *fid_temp;
    real* group_temp;
};




