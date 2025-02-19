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
#include "ensemble_mttk.cuh"

class Ensemble_NPHug : public Ensemble_MTTK
{
public:
  Ensemble_NPHug(const char** params, int num_params);
  Ensemble_NPHug(void);
  virtual ~Ensemble_NPHug(void);

  double p0, v0, e0, e_current, v_current, p_nphug_current;
  bool p0_given = false;
  bool v0_given = false;
  bool e0_given = false;
  double thermo_info[8];
  int uniaxial_compress;
  double dhugo;

  void get_target_temp();
  void get_thermo();
  void init_mttk();
};