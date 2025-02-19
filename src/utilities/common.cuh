/*
    Copyright 2017 Zheyong Fan and GPUMD development team
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

const int MAX_NUM_BEADS = 128;
const int NUM_ELEMENTS = 94;
#define PI 3.14159265358979
#define HBAR 6.465412e-2                             // Planck's constant
#define K_B 8.617343e-5                              // Boltzmann's constant
#define K_C 14.399645                                // 1/(4*PI*epsilon_0)
#define K_C_SP 14.399645f                            // 1/(4*PI*epsilon_0)
const double PRESSURE_UNIT_CONVERSION = 1.602177e+2; // from natural to GPa
const double TIME_UNIT_CONVERSION = 1.018051e+1;     // from natural to fs
const double KAPPA_UNIT_CONVERSION = 1.573769e+5;    // from natural to W/mK
