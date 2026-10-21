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

#include "nep_model.cuh"
#include "nep_parameters.cuh"
#include "utilities/error.cuh"

void calculate_nep_model_parameters(
  NEP_Parameters& para, int extra_inputs, int extra_outputs, int global_parameters, int ann_count)
{
  para.ann_count = ann_count;
  para.global_parameters = global_parameters;
  para.dim_radial = para.n_max_radial + 1;                // 2-body descriptors q^i_n
  para.dim_angular = (para.n_max_angular + 1) * para.L_max; // 3-body descriptors q^i_nl
  if (para.has_q_222) {
    para.dim_angular += para.n_max_angular + 1;
  }
  if (para.has_q_1111) {
    para.dim_angular += para.n_max_angular + 1;
  }
  if (para.has_q_112) {
    para.dim_angular += para.n_max_angular + 1;
  }
  if (para.has_q_123) {
    para.dim_angular += para.n_max_angular + 1;
  }
  if (para.has_q_233) {
    para.dim_angular += para.n_max_angular + 1;
  }
  if (para.has_q_134) {
    para.dim_angular += para.n_max_angular + 1;
  }

  if (para.dim_angular > 90) {
    PRINT_INPUT_ERROR("Number of angular descriptors should not exceed 90.");
  }

  para.dim = para.dim_radial + para.dim_angular;
  para.dim += extra_inputs;

  if (para.num_hidden_layers == 2) {
    para.number_of_variables_ann_1 =
      (para.dim + 1) * para.num_neurons1 + (para.num_neurons1 + 2) * para.num_neurons2;
  } else {
    para.number_of_variables_ann_1 = (para.dim + 2) * para.num_neurons1;
  }
  para.number_of_variables_ann_1 += extra_outputs * para.num_neurons1;
  para.number_of_variables_ann =
    para.number_of_variables_ann_1 * para.num_types + global_parameters;

#ifdef USE_CJ
  para.number_of_variables_descriptor =
    para.num_types *
    (para.dim_radial * (para.basis_size_radial + 1) +
     (para.n_max_angular + 1) * (para.basis_size_angular + 1));
#else
  para.number_of_variables_descriptor =
    para.num_types * para.num_types *
    (para.dim_radial * (para.basis_size_radial + 1) +
     (para.n_max_angular + 1) * (para.basis_size_angular + 1));
#endif

  para.number_of_variables = para.number_of_variables_ann + para.number_of_variables_descriptor;
  para.number_of_variables += (ann_count - 1) * para.number_of_variables_ann;
}
