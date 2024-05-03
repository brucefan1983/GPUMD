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
The class defining the simulation box.
------------------------------------------------------------------------------*/

#include "atom.cuh"
#include "utilities/error.cuh"
#include <cmath>

int Atom::number_of_type(std::string& symbol)
{
  int sum = 0;
  for (int i = 0; i < number_of_atoms; i++) {
    if (cpu_atom_symbol[i] == symbol)
      sum++;
  }
  return sum;
}