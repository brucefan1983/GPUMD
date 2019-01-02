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




#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

struct Neighbor;
class Atom;
class Potential;
class Force;
class Measure;
class Integrate;
class Ensemble;





