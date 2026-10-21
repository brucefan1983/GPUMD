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
#include "nep_parameters.cuh"

class NEP_Charge_Parameters : public NEP_Energy_Parameters
{
public:
  float lambda_q = 0.1f;
  float lambda_z = 0.5f;
  bool has_bec = false;
  int charge_mode = 0;
  int flip_charge = 0;
  bool parse_keyword(const char** param, int num_param) override;

protected:
  void report_model_inputs() const override;

private:
  void parse_lambda_q(const char** param, int num_param);
  void parse_lambda_z(const char** param, int num_param);
};
