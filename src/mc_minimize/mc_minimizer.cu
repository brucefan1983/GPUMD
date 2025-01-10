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

#include "mc_minimizer.cuh"
#include <iostream>
#include <fstream>
#include <chrono>

MC_Minimizer::MC_Minimizer(const char** param, int num_param)
{
  mc_output.open("mc_minimize.out", std::ios::app);
  mc_output << "# ";
  for (int n = 0; n < num_param; ++n) {
    mc_output << param[n] << " ";
  }
  mc_output << "\n";
  if (strcmp(param[1], "local") == 0) {
    mc_output << "Step" << "\t" << "Maximum displacement" << "\t" << "Average displacement" 
    << "\t" << "Energy before" << "\t" << "Energy after" << "\t" <<  "Accept ratio" << std::endl;
  } else if (strcmp(param[1], "global") == 0) {
    mc_output << "Step" << "\t" << "Energy before" << "\t" << "Energy after" << "\t" <<  "Accept ratio" << std::endl;
  }



#ifdef DEBUG
  rng = std::mt19937(13579);
#else
  rng = std::mt19937(std::chrono::system_clock::now().time_since_epoch().count());
#endif
}

MC_Minimizer::~MC_Minimizer()
{
    //default destructor
}