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
#include "ensemble.cuh"
#include "gpu_vector.cuh"
#include <curand_kernel.h>


class Ensemble_LAN : public Ensemble
{
public:
    Ensemble_LAN(int, int, int, double, double);   
    Ensemble_LAN(int, int, int, int, int, int, int, int, double, double, double); 
    virtual ~Ensemble_LAN(void);
    virtual void compute1(Atom*);
    virtual void compute2(Atom*);
protected:
    int N_source, N_sink, offset_source, offset_sink;
    double c1, c2, c2_source, c2_sink;
    GPU_Vector<curandState> curand_states;
    GPU_Vector<curandState> curand_states_source;
    GPU_Vector<curandState> curand_states_sink;
    void integrate_nvt_lan_half(Atom*);
    void integrate_heat_lan_half(Atom*);
};


