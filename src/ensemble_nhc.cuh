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


class Ensemble_NHC : public Ensemble
{
public:
    Ensemble_NHC(int, int, int, double, double, double);   
    Ensemble_NHC(int, int, int, int, int, int, double, double, double, double); 
    virtual ~Ensemble_NHC(void);
    virtual void compute(Atom*, Force*);
protected:
    void integrate_nvt_nhc(Atom*, Force*);
    void integrate_heat_nhc(Atom*, Force*);
};


