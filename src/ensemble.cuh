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

#define NOSE_HOOVER_CHAIN_LENGTH 4




class Ensemble 
{
public:
    Ensemble(void);      
    virtual ~Ensemble(void);
    virtual void compute(Atom*, Force*, Measure*) = 0;
    int type;          // ensemble type in a specific run
    int source;
    int sink;
    real temperature;  // target temperature at a specific time 
    real delta_temperature;
    real pressure_x;   // target pressure at a specific time
    real pressure_y;   
    real pressure_z; 
    real temperature_coupling;
    real pressure_coupling;  

    real energy_transferred[2]; // energy transferred from system to heat baths
    
    real mas_nhc1[NOSE_HOOVER_CHAIN_LENGTH];
    real pos_nhc1[NOSE_HOOVER_CHAIN_LENGTH];
    real vel_nhc1[NOSE_HOOVER_CHAIN_LENGTH];
    real mas_nhc2[NOSE_HOOVER_CHAIN_LENGTH];
    real pos_nhc2[NOSE_HOOVER_CHAIN_LENGTH];
    real vel_nhc2[NOSE_HOOVER_CHAIN_LENGTH];

protected:
    void velocity_verlet_1(Atom*);
    void velocity_verlet_2(Atom*);
    void find_thermo(Atom*);
};




