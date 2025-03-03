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
#include "property.cuh"
class Atom;
class Box;

// choose between the two models
#define USE_GRAPHENE_TB
#ifdef USE_GRAPHENE_TB
#define number_of_orbitals_per_atom 1
#else
#define number_of_orbitals_per_atom 4
#endif

class LSQT : public Property
{
public:
#ifndef USE_GRAPHENE_TB
  struct TB {
    double onsite[number_of_orbitals_per_atom] = {-2.99, 3.71, 3.71, 3.71};
    double v_sss = -5.0;
    double v_sps = 4.7;
    double v_pps = 5.5;
    double v_ppp = -1.55;
    double nc = 6.5;
    double rc = 2.18;
    double r0 = 1.536329;
  };
#endif

  LSQT(const char** param, const int num_param);
  void parse(const char** param, const int num_param);
  virtual void preprocess(
    const int number_of_steps,
    const double time_step,
    Integrate& integrate,
    std::vector<Group>& group,
    Atom& atom,
    Box& box,
    Force& force);

  virtual void process(
      const int number_of_steps,
      int step,
      const int fixed_group,
      const int move_group,
      const double global_time,
      const double temperature,
      Integrate& integrate,
      Box& box,
      std::vector<Group>& group,
      GPU_Vector<double>& thermo,
      Atom& atom,
      Force& force);

  virtual void postprocess(
    Atom& atom,
    Box& box,
    Integrate& integrate,
    const int number_of_steps,
    const double time_step,
    const double temperature);
  void find_dos_and_velocity(Atom& atom, Box& box);
  void find_sigma(Atom& atom, Box& box, const int step);

private:
#ifndef USE_GRAPHENE_TB
  TB tb;
#endif
  bool compute = false;
  int number_of_atoms;
  int number_of_orbitals;
  int transport_direction;
  int number_of_moments;
  int number_of_energy_points;
  int number_of_steps;
  double maximum_energy;
  double time_step;

  GPU_Vector<int> cell_count;
  GPU_Vector<int> cell_count_sum;
  GPU_Vector<int> cell_contents;
  GPU_Vector<int> NN_atom;
  GPU_Vector<int> NL_atom;
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