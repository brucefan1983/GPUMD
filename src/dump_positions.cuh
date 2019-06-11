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
#ifndef DUMP_POS_H
#define DUMP_POS_H

#include "common.cuh"

#define FILE_NAME_LENGTH      200

class DUMP_POS
{
public:
	int output_pos; // 0 = No output (default), 1 = Output
	int interval;  // output interval
	int format; // 0 = xyz (GPUMD default), 1 = netcdf
	int precision; // 0 = normal precision, 1 = high precision
	char precision_str[13] = "%d %g %g %g\n";
	FILE *fid_position;
    char file_position[FILE_NAME_LENGTH];
	void initialize(char*);
	void finalize(void);
	void dump(Atom *atom, int step);

private:
	void dump_xyz(Atom *atom, int step);
};


#endif //DUMP_POS
