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

class Atom; 


#define NOSE_HOOVER_CHAIN_LENGTH 4


class Ensemble 
{
public:
    Ensemble(void);      
    virtual ~Ensemble(void);
    virtual void compute1(Atom*) = 0;
    virtual void compute2(Atom*) = 0;
    int type;          // ensemble type in a specific run
    int source;
    int sink;
    int fixed_group;   // ID of the group in which the atoms will be fixed 
    double temperature;  // target temperature at a specific time 
    double delta_temperature;
    double pressure_x;   // target pressure at a specific time
    double pressure_y;   
    double pressure_z; 
    double temperature_coupling;
    double pressure_coupling;  
    int deform_x = 0;
    int deform_y = 0;
    int deform_z = 0;
    double deform_rate;

    double energy_transferred[2]; // energy transferred from system to heat baths
    
    double mas_nhc1[NOSE_HOOVER_CHAIN_LENGTH];
    double pos_nhc1[NOSE_HOOVER_CHAIN_LENGTH];
    double vel_nhc1[NOSE_HOOVER_CHAIN_LENGTH];
    double mas_nhc2[NOSE_HOOVER_CHAIN_LENGTH];
    double pos_nhc2[NOSE_HOOVER_CHAIN_LENGTH];
    double vel_nhc2[NOSE_HOOVER_CHAIN_LENGTH];

protected:
    void velocity_verlet(const bool is_step1, Atom*);
    void find_thermo(Atom*);
    void scale_velocity_global(Atom* atom, double);
    void find_vc_and_ke(Atom*, double*, double*, double*, double*);
    void scale_velocity_local(Atom*, double, double, double*, double*, double*, double*);
};


