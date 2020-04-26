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
#include <stdio.h>


struct FCP_Data
{
    int *ia2, *jb2;
    int *ia3, *jb3, *kc3;
    int *ia4, *jb4, *kc4, *ld4;
    int *ia5, *jb5, *kc5, *ld5, *me5;
    int *ia6, *jb6, *kc6, *ld6, *me6, *nf6;
    float *u, *r0, *pfv, *xij2, *yij2, *zij2;
    float *phi2, *phi3, *phi4, *phi5, *phi6;
};


class FCP : public Potential
{
public:   
    FCP(FILE* fid, char *input_dir, Atom *atom);  
    virtual ~FCP(void);
    virtual void compute(Atom*, Measure*, int);
protected:
    int order, number2, number3, number4, number5, number6;
    char file_path[FILE_NAME_LENGTH];
    FCP_Data fcp_data;
    void read_r0(Atom *atom);
    void read_fc2(Atom *atom);
    void read_fc3(Atom *atom);
    void read_fc4(Atom *atom);
    void read_fc5(Atom *atom);
    void read_fc6(Atom *atom);
};


