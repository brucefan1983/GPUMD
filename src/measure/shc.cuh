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
#include "utilities/gpu_vector.cuh"
class Group;

class SHC : public Property
{
public:
  int group_num;         // number of groups
  int compute = 0;       // 0 = not computing shc; 1 = computing shc
  int group_method = -1; // -1 means not using a group method
  int group_id = 0;      // calculating SHC for atoms in group id
  int sample_interval;   // sample interval for heat current
  int Nc;                // number of correlation points
  int direction;         // transport direction: 0=x; 1=y; 2=z
  int num_omega;         // number of frequency points
  double max_omega;      // maximum angular frequency

  SHC(const char**, int, const std::vector<Group>& group);
  
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

  void parse(const char**, int, const std::vector<Group>& group);
  void find_shc(const double dt_in_ps, const double d_omega);
  void average_k();

private:
  int num_time_origins;                        // number of time origins for ensemble average
  int group_size;                              // number of atoms in group_id
  GPU_Vector<double> vx, vy, vz;               // Nc frames of velocity data
  GPU_Vector<double> sx, sy, sz;               // Nc frames of virial data
  GPU_Vector<double> ki_negative, ko_negative; // The correlation functions K(t) with t < 0
  GPU_Vector<double> ki_positive, ko_positive; // The correlation functions K(t) with t > 0
  std::vector<double> ki, ko;                  // The correlation functions K(t) with all t
  std::vector<double> shc_i, shc_o;            // The SHC(omega)
};
