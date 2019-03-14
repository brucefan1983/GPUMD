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

#ifndef DOS_H
#define DOS_H

#pragma once
#include "common.cuh"
#include "vac.cuh"

class VAC; //forward declaration

class DOS
{
public:
	int num_dos_points = -1; // points to use for DOS output, -1 means not set
    real omega_max;    // maximal angular frequency for phonons
    void preprocess(Atom*, VAC*);
    void process(char*, Atom*, VAC*);

private:

};

#endif //DOS_H
