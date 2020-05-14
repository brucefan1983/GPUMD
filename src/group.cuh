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
#include "gpu_vector.cuh"
#include <vector>


class Group
{
public:
    int number;             // number of groups
    // GPU data
    GPU_Vector<int> label;             // atom label
    GPU_Vector<int> size;              // # atoms in each group
    GPU_Vector<int> size_sum;          // # atoms in all previous groups
    GPU_Vector<int> contents;          // atom indices sorted based on groups
    // CPU data corresponding to the above GPU data
    std::vector<int> cpu_label;
    std::vector<int> cpu_size;
    std::vector<int> cpu_size_sum;
    std::vector<int> cpu_contents;

    void find_size(const int N, const int k);
    void find_contents(const int N);

};


