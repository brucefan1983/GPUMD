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
The abstract base class (ABC) for the MC_Ensemble classes.
------------------------------------------------------------------------------*/

#include "mc_ensemble.cuh"
#include "utilities/common.cuh"
#include <chrono>

MC_Ensemble::MC_Ensemble(void)
{
  const int n_max = 16000;
  const int m_max = 100;
  NN_radial.resize(n_max);
  NN_angular.resize(n_max);
  NL_radial.resize(n_max * m_max);
  NL_angular.resize(n_max * m_max);
  type_before.resize(n_max);
  type_after.resize(n_max);
  x12_radial.resize(n_max * m_max);
  y12_radial.resize(n_max * m_max);
  z12_radial.resize(n_max * m_max);
  x12_angular.resize(n_max * m_max);
  y12_angular.resize(n_max * m_max);
  z12_angular.resize(n_max * m_max);
  pe_before.resize(n_max);
  pe_after.resize(n_max);

  // TODO
  nep_energy.initialize("../examples/11_NEP_potential_PbTe/nep.txt");

#ifdef DEBUG
  rng = std::mt19937(13579);
#else
  rng = std::mt19937(std::chrono::system_clock::now().time_since_epoch().count());
#endif
}

MC_Ensemble::~MC_Ensemble(void)
{
  // nothing now
}
