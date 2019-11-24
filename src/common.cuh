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
#include <vector>

#define LDG(a, n) __ldg(a + n)

const int FILE_NAME_LENGTH = 200;
#define GPUMD_VERSION "2.5"

typedef double real;
const real ZERO  = 0.0;
const real HALF  = 0.5;
const real ONE   = 1.0;
const real TWO   = 2.0;
const real THREE = 3.0;
const real FOUR  = 4.0;
const real FIVE  = 5.0;
const real SIX   = 6.0;
const real PI    = 3.14159265358979;
const real K_B   = 8.617343e-5;                      // Boltzmann's constant
const real K_C   = 14.399645;                        // 1/(4*PI*epsilon_0)
const real PRESSURE_UNIT_CONVERSION = 1.602177e+2;   // from natural to GPa
const real TIME_UNIT_CONVERSION     = 1.018051e+1;   // from natural to fs
const real KAPPA_UNIT_CONVERSION    = 1.573769e+5;   // from natural to W/mK


class Atom;
class Potential;
class Force;
class Measure;
class Integrate;
class Ensemble;
class Hessian;


