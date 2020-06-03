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


/*----------------------------------------------------------------------------80
The GPUMD class, which is used by the main function.
------------------------------------------------------------------------------*/


#include "gpumd.cuh"
#include "run.cuh"
#include "model/atom.cuh"
#include "force/force.cuh"
#include "integrate/integrate.cuh"
#include "measure/measure.cuh"


GPUMD::GPUMD(char* input_dir)
{
    Atom atom(input_dir);

    atom.allocate_memory_gpu();

#ifndef USE_FCP // the FCP does not use a neighbor list at all
    atom.neighbor.find_neighbor
    (
        1,
        atom.box,
        atom.position_per_atom
    );
#endif

    Force force;
    Integrate integrate;
    Measure measure(input_dir);
    Run run(input_dir, &atom, &force, &integrate, &measure);
}


