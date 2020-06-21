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

#ifdef USE_NETCDF

#pragma once
#ifndef DUMP_NETCDF_H
#define DUMP_NETCDF_H

#include "dump_pos.cuh"
#include "netcdf.h"

class DUMP_NETCDF: public DUMP_POS
{
public:
    void initialize(char*);
    void finalize();

    void dump
    (
        const int step,
        const double global_time,
        const Box& box,
        const std::vector<int>& cpu_type,
        GPU_Vector<double>& position_per_atom,
        std::vector<double>& cpu_position_per_atom
    );

    DUMP_NETCDF();
    ~DUMP_NETCDF(){}

private:
    int ncid; // NetCDF ID
    static bool append;

    // dimensions
    int frame_dim;
    int spatial_dim;
    int atom_dim;
    int cell_spatial_dim;
    int cell_angular_dim;
    int label_dim;

    // label variables
    int spatial_var;
    int cell_spatial_var;
    int cell_angular_var;

    // data variables
    int time_var;
    int cell_lengths_var;
    int cell_angles_var;
    int coordinates_var;
    int type_var;

    size_t lenp; // frame number

    void open_file(int frame_in_run);
    void write(Atom *atom);

};

#endif //DUMP_NETCDF

#endif
