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


const int FILE_NAME_LENGTH = 200;
#define GPUMD_VERSION "2.5"

#define ZERO  0.0
#define HALF  0.5
#define ONE   1.0
#define TWO   2.0
#define THREE 3.0
#define FOUR  4.0
#define FIVE  5.0
#define SIX   6.0
#define PI    3.14159265358979
#define K_B   8.617343e-5                      // Boltzmann's constant
#define K_C   14.399645                        // 1/(4*PI*epsilon_0)
const double PRESSURE_UNIT_CONVERSION = 1.602177e+2;   // from natural to GPa
const double TIME_UNIT_CONVERSION     = 1.018051e+1;   // from natural to fs
const double KAPPA_UNIT_CONVERSION    = 1.573769e+5;   // from natural to W/mK

