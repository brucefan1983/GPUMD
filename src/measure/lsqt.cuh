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
class Atom;
class Box;

class LSQT
{
public:
  void preprocess(Atom& atom);
  void postprocess(Atom& atom, Box& box);

private:
  int N;             // number of atoms
  int direction = 1; // transport direction
  int Nm = 1000;     // number of moments
  int Ne = 1001;     // number of energy points
  int Nt = 10;
  double Em = 10.1; // maximum energy
  double dt = 1.6;  // TODO (this is 1.6 * hbar/eV, which is about 1 fs)

  GPU_Vector<int> cell_count;
  GPU_Vector<int> cell_count_sum;
  GPU_Vector<int> cell_contents;
  GPU_Vector<int> NN;
  GPU_Vector<int> NL;

  GPU_Vector<double> xx;
  GPU_Vector<double> Hr;
  GPU_Vector<double> Hi;
  GPU_Vector<double> U;
  GPU_Vector<double> sr;
  GPU_Vector<double> si;
  GPU_Vector<double> sxr;
  GPU_Vector<double> sxi;
  GPU_Vector<double> scr;
  GPU_Vector<double> sci;

  std::vector<double> E;
  std::vector<double> dos;
  std::vector<double> velocity;
  std::vector<double> msd;
  std::vector<double> sigma;
};