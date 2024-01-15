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
#include "ensemble_mttk.cuh"

class Ensemble_nphug : public Ensemble_MTTK
{
  Ensemble_nphug(const char** params, int num_params);
  Ensemble_nphug(void);
  virtual ~Ensemble_nphug(void);

  double p0, v0, e0;
  int uniaxial_compress;

  void get_target_temp();
  double find_current_energy();
  double compute_hugoniot();
  void init();
};