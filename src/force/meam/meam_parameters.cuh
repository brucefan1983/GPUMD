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

#include <vector>

struct MeamElement {
  double beta[4];
  double t[3];
  double density_scale;
  double embedding_scale;
  double equilibrium_distance;
  double reference_density;
  double reference_shape[3];
  int coordination;
  int density_function;
};

struct MeamScreen {
  double minimum;
  double inverse_width;
};

struct MeamSpline {
  double value, slope, quadratic, cubic;
};

struct MeamSettings {
  double cutoff = 4.0;
  double smoothing_width = 0.1;
  double gamma_smoothing = 99.0;
  double inverse_table_step;
  int type_count;
  int alloy_rule = 0;
  bool mixed_reference = false;
  bool dynamic_background = false;
  bool linear_negative_embedding = false;
};

struct MeamParameters {
  MeamSettings settings;
  std::vector<MeamElement> elements;
  std::vector<double> screening_bounds;
  std::vector<MeamScreen> screening;
  std::vector<MeamSpline> pair_tables;
};

constexpr int MEAM_TABLE_POINTS = 1000;

MeamParameters read_meam_parameters(const char* filename, int number_of_types);
