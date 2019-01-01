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
class Force;
class Measure;
class Atom;


class Ensemble_BDP : public Ensemble
{
public:
    Ensemble_BDP(int, real, real);   
    Ensemble_BDP(int, int, int, real, real, real); 
    virtual ~Ensemble_BDP(void);
    virtual void compute(Parameters*, Atom*, Force*, Measure*);
protected:
    void integrate_nvt_bdp(Parameters*, Atom*, Force*, Measure*);
    void integrate_heat_bdp(Parameters*, Atom*, Force*, Measure*);
};




