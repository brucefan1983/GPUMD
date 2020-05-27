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
#include "potential.cuh"
#include "gpu_vector.cuh"
#include <stdio.h>


struct FCP_Data
{
    GPU_Vector<int> i2, j2, index2;
    GPU_Vector<int> i3, j3, k3, index3;
    GPU_Vector<int> i4, j4, k4, l4, index4;
    GPU_Vector<int> i5, j5, k5, l5, m5, index5;
    GPU_Vector<int> i6, j6, k6, l6, m6, n6, index6;
    GPU_Vector<float> u, utot, r0, pfv, xij2, yij2, zij2, xij3, yij3, zij3;
    GPU_Vector<float> phi2, phi3, phi4, phi5, phi6;
    GPU_Vector<float> weight4, weight5, weight6;
};


class FCP : public Potential
{
public:   
    FCP(FILE* fid, char *input_dir, const int N, const Box& box);
    virtual ~FCP(void);
    virtual void compute
    (
        const int type_shift,
        const Box& box,
        const Neighbor& neighbor,
        const GPU_Vector<int>& type,
        const GPU_Vector<double>& position,
        GPU_Vector<double>& potential,
        GPU_Vector<double>& force,
        GPU_Vector<double>& virial
    );
protected:
    int order, number2, number3, number4, number5, number6;
    char file_path[200];
    FCP_Data fcp_data;
    void read_r0(const int N);
    void read_fc2(const int N, const Box& box);
    void read_fc3(const int N, const Box& box);
    void read_fc4(const int N);
    void read_fc5(const int N);
    void read_fc6(const int N);
};


