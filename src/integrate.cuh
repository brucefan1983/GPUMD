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

#include "group.cuh"
#include <vector>


class Ensemble;
class Atom;


class Integrate 
{
public:
    Ensemble *ensemble; 
    Integrate(void);
    ~Integrate(void);
 
    void initialize
    (
        const int number_of_atoms,
        const double time_step,
        const std::vector<Group>& group
    );

    void finalize(void);
    void compute1(Atom*);
    void compute2(Atom*);

    // get inputs from run.in
    void parse_ensemble(char **param, int num_param, std::vector<Group>& group);
    void parse_deform(char**, int);
    void parse_fix(char**, int, std::vector<Group>& group);

    // these data will be used to initialize ensemble
    int type;          // ensemble type in a specific run
    int source;
    int sink;
    int fixed_group;   // ID of the group in which the atoms will be fixed 
    double temperature;  // target temperature at a specific time 
    double temperature1; // target initial temperature for a run
    double temperature2; // target final temperature for a run
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
};


