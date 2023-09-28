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
  void parse(const char** param, const int num_param);
  void preprocess(Atom& atom, int number_of_steps, double time_step);
  void process(Atom& atom, Box& box, const int step);
  void postprocess();
  void find_dos_and_velocity(Atom& atom, Box& box);
  void find_sigma(Atom& atom, Box& box, const int step);

private:
  bool compute = false;
  int number_of_atoms;
  int transport_direction;
  int number_of_moments;
  int number_of_energy_points;
  int number_of_steps;
  double maximum_energy;
  double time_step;

  GPU_Vector<int> cell_count;
  GPU_Vector<int> cell_count_sum;
  GPU_Vector<int> cell_contents;
  GPU_Vector<int> NN;
  GPU_Vector<int> NL;

  GPU_Vector<double> xx;
  GPU_Vector<double> Hr;
  GPU_Vector<double> Hi;
  GPU_Vector<double> U;

  GPU_Vector<double> slr;
  GPU_Vector<double> sli;
  GPU_Vector<double> srr;
  GPU_Vector<double> sri;
  GPU_Vector<double> scr;
  GPU_Vector<double> sci;

  std::vector<double> E;
  std::vector<double> sigma;
};