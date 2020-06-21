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
#ifndef DUMP_XYZ_H
#define DUMP_XYZ_H

#include "dump_pos.cuh"

class DUMP_XYZ: public DUMP_POS
{
public:
    char precision_str[25];
    FILE *fid_position;
    virtual void initialize(char* input_dir, const int number_of_atoms);
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

    DUMP_XYZ();
    ~DUMP_XYZ(){};
};


#endif // DUMP_XYZ
