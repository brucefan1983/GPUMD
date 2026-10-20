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

#include "nep_init.cuh"
#include "nep_io.cuh"
#include "nep_parameters.cuh"

static void initialize_nep_mu_and_sigma_fine_tune(
  const NEP_Parameters& para, std::vector<float>& mu, std::vector<float>& sigma, int global_parameters)
{
  // This map is needed because the foundation model misses 5 elements between H-Pu
  const int element_map[94] = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,
    20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,
    40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,
    60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,
    80,81,82,0,0,0,0,0,83,84,85,86,87,88
  };
  // read in the whole foundation file first
  const int NUM89 = 89;
  const int num_ann = NUM89 * para.number_of_variables_ann_1 + global_parameters;
#ifdef USE_CJ
  const int num_cnk_radial = NUM89 * (para.n_max_radial + 1) * (para.basis_size_radial + 1);
  const int num_cnk_angular = NUM89 * (para.n_max_angular + 1) * (para.basis_size_angular + 1);
#else
  const int num_cnk_radial = NUM89 * NUM89 * (para.n_max_radial + 1) * (para.basis_size_radial + 1);
  const int num_cnk_angular = NUM89 * NUM89 * (para.n_max_angular + 1) * (para.basis_size_angular + 1);
#endif
  const int num_tot = num_ann + num_cnk_radial + num_cnk_angular;
  std::vector<float> restart_mu(num_tot);
  std::vector<float> restart_sigma(num_tot);

  read_nep_foundation_restart(para.fine_tune_nep_restart, restart_mu, restart_sigma);

  // get the required part
  int count = 0;
  for (int i = 0; i < para.num_types; ++ i) {
    int element_index = element_map[para.atomic_numbers[i] - 1];
    for (int j = 0; j < para.number_of_variables_ann_1; ++j) {
      mu[count] = restart_mu[element_index * para.number_of_variables_ann_1 + j];
      sigma[count] = restart_sigma[element_index * para.number_of_variables_ann_1 + j];
      ++count;
    }
  }
  count += global_parameters; // the global parameters

#ifdef USE_CJ

  // radial descriptors
  for (int t2 = 0; t2 < para.num_types; ++t2) {
    int element_index_2 = element_map[para.atomic_numbers[t2] - 1];
    for (int n = 0; n <= para.n_max_radial; ++n) {
      for (int k = 0; k <= para.basis_size_radial; ++k) {
        int nk = n * (para.basis_size_radial + 1) + k;
        mu[count] = restart_mu[nk * NUM89 + element_index_2 + num_ann];
        if (para.fine_tune_descriptor) {
          sigma[count] = restart_sigma[nk * NUM89 + element_index_2 + num_ann];
        } else {
          sigma[count] = 0.0f;
        }
        ++count;
      }
    }
  }

  // angular descriptors
  for (int t2 = 0; t2 < para.num_types; ++t2) {
    int element_index_2 = element_map[para.atomic_numbers[t2] - 1];
    for (int n = 0; n <= para.n_max_angular; ++n) {
      for (int k = 0; k <= para.basis_size_angular; ++k) {
        int nk = n * (para.basis_size_angular + 1) + k;
        mu[count] = restart_mu[nk * NUM89 + element_index_2 + num_ann + num_cnk_radial];
        if (para.fine_tune_descriptor) {
          sigma[count] = restart_sigma[nk * NUM89 + element_index_2 + num_ann + num_cnk_radial];
        } else {
          sigma[count] = 0.0f;
        }
        ++count;
      }
    }
  }

#else

  // radial descriptors
  for (int t1 = 0; t1 < para.num_types; ++t1) {
    for (int t2 = 0; t2 < para.num_types; ++t2) {
      int element_index_1 = element_map[para.atomic_numbers[t1] - 1];
      int element_index_2 = element_map[para.atomic_numbers[t2] - 1];
      int t12 = element_index_1 * NUM89 + element_index_2;
      for (int n = 0; n <= para.n_max_radial; ++n) {
        for (int k = 0; k <= para.basis_size_radial; ++k) {
          int nk = n * (para.basis_size_radial + 1) + k;
          mu[count] = restart_mu[nk * NUM89 * NUM89 + t12 + num_ann];
          if (para.fine_tune_descriptor) {
            sigma[count] = restart_sigma[nk * NUM89 * NUM89 + t12 + num_ann];
          } else {
            sigma[count] = 0.0f;
          }
          ++count;
        }
      }
    }
  }

  // angular descriptors
  for (int t1 = 0; t1 < para.num_types; ++t1) {
    for (int t2 = 0; t2 < para.num_types; ++t2) {
      int element_index_1 = element_map[para.atomic_numbers[t1] - 1];
      int element_index_2 = element_map[para.atomic_numbers[t2] - 1];
      int t12 = element_index_1 * NUM89 + element_index_2;
      for (int n = 0; n <= para.n_max_angular; ++n) {
        for (int k = 0; k <= para.basis_size_angular; ++k) {
          int nk = n * (para.basis_size_angular + 1) + k;
          mu[count] = restart_mu[nk * NUM89 * NUM89 + t12 + num_ann + num_cnk_radial];
          if (para.fine_tune_descriptor) {
            sigma[count] = restart_sigma[nk * NUM89 * NUM89 + t12 + num_ann + num_cnk_radial];
          } else {
            sigma[count] = 0.0f;
          }
          ++count;
        }
      }
    }
  }

#endif
}

NEP_Initialization initialize_nep_mu_and_sigma(
  const NEP_Parameters& para,
  std::mt19937& rng,
  std::vector<float>& mu,
  std::vector<float>& sigma,
  int global_parameters)
{
  if (para.fine_tune) {
    initialize_nep_mu_and_sigma_fine_tune(para, mu, sigma, global_parameters);
    return NEP_Initialization::FINE_TUNE;
  }

  if (read_nep_restart("nep.restart", para, mu, sigma)) {
    return NEP_Initialization::RESTART;
  }

  std::uniform_real_distribution<float> r1(0, 1);
  for (int n = 0; n < para.number_of_variables; ++n) {
    mu[n] = (r1(rng) - 0.5f) * 2.0f;
    sigma[n] = para.sigma0;
  }
  return NEP_Initialization::RANDOM;
}

void find_nep_type_of_variable(
  const NEP_Parameters& para, std::vector<int>& type_of_variable, int global_parameters, int ann_count)
{
  type_of_variable.assign(para.number_of_variables, para.num_types);
  int offset = 0;

  // NN part
  int num_ann = ann_count;
  for (int ann = 0; ann < num_ann; ++ann) {
    for (int t = 0; t < para.num_types; ++t) {
      for (int n = 0; n < para.number_of_variables_ann_1; ++n) {
        type_of_variable[n + offset] = t;
      }
      offset += para.number_of_variables_ann_1;
    }
    offset += global_parameters; // the bias
  }

  // descriptor part
#ifdef USE_CJ
  const int num_radial_basis = (para.n_max_radial + 1) * (para.basis_size_radial + 1);
  for (int t1 = 0; t1 < para.num_types; ++t1) {
    for (int basis = 0; basis < num_radial_basis; ++basis) {
      type_of_variable[t1 * num_radial_basis + basis + offset] = t1;
    }
  }
  offset += num_radial_basis * para.num_types;
  const int num_angular_basis = (para.n_max_angular + 1) * (para.basis_size_angular + 1);
  for (int t1 = 0; t1 < para.num_types; ++t1) {
    for (int basis = 0; basis < num_angular_basis; ++basis) {
      type_of_variable[t1 * num_angular_basis + basis + offset] = t1;
    }
  }
#else
  const int num_radial_basis = (para.n_max_radial + 1) * (para.basis_size_radial + 1);
  for (int t1 = 0; t1 < para.num_types; ++t1) {
    for (int t2 = 0; t2 < para.num_types; ++t2) {
      int t12 = t1 * para.num_types + t2;
      for (int basis = 0; basis < num_radial_basis; ++basis) {
        type_of_variable[t12 * num_radial_basis + basis + offset] = t1;
      }
    }
  }
  offset += num_radial_basis * para.num_types * para.num_types;
  const int num_angular_basis = (para.n_max_angular + 1) * (para.basis_size_angular + 1);
  for (int t1 = 0; t1 < para.num_types; ++t1) {
    for (int t2 = 0; t2 < para.num_types; ++t2) {
      int t12 = t1 * para.num_types + t2;
      for (int basis = 0; basis < num_angular_basis; ++basis) {
        type_of_variable[t12 * num_angular_basis + basis + offset] = t1;
      }
    }
  }
#endif
}
