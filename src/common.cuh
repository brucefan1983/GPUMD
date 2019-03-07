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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda.h> // seems to be needed in Windows

#define LDG(a, n) __ldg(a + n)


// See section 3.1 in the manual for the unit conventions
typedef double real;
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
#define K_C   1.441959e+1                      // 1/(4*PI*epsilon_0)
#define PRESSURE_UNIT_CONVERSION 1.602177e+2   // from natural to GPa
#define TIME_UNIT_CONVERSION     1.018051e+1   // from natural to fs
#define KAPPA_UNIT_CONVERSION    1.573769e+5   // from natural to W/mK


class Atom;
class Potential;
class Force;
class Measure;
class Integrate;
class Ensemble;
class Hessian;
class GA;


