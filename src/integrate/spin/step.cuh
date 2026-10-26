/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/

#pragma once
#include <vector>

class Atom;
class Box;
class Force;
class Group;
class Integrate;
template <typename T> class GPU_Vector;

// Refresh the corrected endpoint while retaining the force modifier sampled
// at the predictor. The saved predictor force is scratch storage for its delta.
void refresh_spin_endpoint(
  Force& force, Integrate& integrate, Box& box, Atom& atom, std::vector<Group>& group,
  GPU_Vector<double>& thermo, GPU_Vector<double>& spin_predictor_base_force);
