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
#include "utilities/error.cuh"
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

// Step-indexed correlations require a single physical time interval per MD step.
class UniformTimeStep
{
public:
  void reset() { initialized_ = false; }

  // Returns true when the first actual integration step establishes the interval.
  bool observe(const double time_step, const char* command)
  {
    if (!std::isfinite(time_step) || time_step <= 0.0) {
      const std::string message = std::string(command) + " requires a finite, positive time step.";
      PRINT_INPUT_ERROR(message.c_str());
    }
    if (!initialized_) {
      time_step_ = time_step;
      initialized_ = true;
      return true;
    }
    const double tolerance = 32.0 * std::numeric_limits<double>::epsilon() *
                             std::max(std::abs(time_step_), std::abs(time_step));
    if (std::abs(time_step - time_step_) > tolerance) {
      const std::string message = std::string(command) +
        " requires a constant actual time step within each run. "
        "The adaptive time step changed; use a fixed time_step for this calculation.";
      PRINT_INPUT_ERROR(message.c_str());
    }
    return false;
  }

private:
  bool initialized_ = false;
  double time_step_ = 0.0;
};
