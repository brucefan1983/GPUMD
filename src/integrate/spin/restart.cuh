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
#include <cstdio>
#include <string>
#include <vector>

class Atom;
class Integrate;

struct Spin_Lattice_Restart
{
  std::string mttk, nhc; // Pending endpoint, consumed once by the Spin lattice driver.
  double mttk_time = -1.0, nhc_time = -1.0; // Internal time units.
};

void read_spin_restart(const std::vector<std::string>& tokens, Atom& atom);
// Writes metadata and returns whether moments require checkpoint precision.
bool write_spin_restart(FILE* fid, const Atom& atom, const Integrate& integrate, double global_time);
