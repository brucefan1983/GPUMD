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

#include "integrate/integrate.cuh"
#include "integrate/ensemble_fixed_lattice.cuh"
#include "lattice.cuh"
#include "model/atom.cuh"
#include "utilities/common.cuh"
#include "utilities/read_file.cuh"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>

namespace
{
bool is_tspin_option(const char* value)
{
  return strcmp(value, "lattice") == 0 ||
    strcmp(value, "mass_factor") == 0 || strcmp(value, "seed") == 0;
}

bool is_langevin_spin_option(const char* value)
{
  return strcmp(value, "lattice") == 0 || strcmp(value, "alpha") == 0 ||
    strcmp(value, "damping") == 0 || strcmp(value, "mobility") == 0 ||
    strcmp(value, "gamma") == 0 || strcmp(value, "stemp") == 0 ||
    strcmp(value, "seed") == 0;
}

void print_spin_mass_factors(
  const Atom& atom,
  const std::vector<double>& mass_factor_by_type)
{
  for (std::size_t type = 0; type < mass_factor_by_type.size(); ++type) {
    for (int atom_index = 0; atom_index < atom.number_of_atoms; ++atom_index) {
      if (atom.cpu_type[atom_index] == static_cast<int>(type)) {
        printf(
          "    spin mass factor for %s is %g.\n",
          atom.cpu_atom_symbol[atom_index].c_str(),
          mass_factor_by_type[type]);
        break;
      }
    }
  }
}
} // namespace

void Integrate::parse_spin_ensemble(
  const std::vector<std::string>& tokens, const Atom& atom,
  const Box& box, const std::vector<Group>& group)
{
  const int num_param = tokens.size();
  std::vector<const char*> args;
  for (const auto& token : tokens) args.push_back(token.c_str());
  const char** param = args.data();
  const bool spin_only = tokens[1] == "nve_sib" || tokens[1] == "nve_glsd";
  spin_lattice_enabled = !spin_only;
  spin_mass_factor_by_type.assign(atom.cpu_type_size.size(), 1.0);
  spin_seed = 12345;
  spin_sib_alpha = 0.0;
  spin_dynamics_gamma = 2.0 * 1000.0 / (HBAR * TIME_UNIT_CONVERSION);
  spin_dynamics_temperature = spin_only ? -1.0 : 0.0;
  spin_dynamics_seed = 12345;
  spin_glsd_damping = spin_glsd_mobility = 0.0;
  const bool npt = tokens[1] == "npt_tspin" || tokens[1] == "npt_sib" || tokens[1] == "npt_glsd";
  int spin_option_start = spin_only ? 2 : 5;
  if (npt) {
    spin_option_start = num_param;
    for (int i = 2; i < num_param; ++i) {
      if (use_spin_tspin ? is_tspin_option(param[i]) : is_langevin_spin_option(param[i])) {
        spin_option_start = i;
        break;
      }
    }
  }
  if (num_param < spin_option_start) PRINT_INPUT_ERROR("Missing spin ensemble parameters.");
  if (use_spin_tspin) {
    bool has_lattice = false;
    bool has_mass_factor = false;
    bool has_seed = false;
    int index = spin_option_start;
    while (index < num_param) {
      if (strcmp(param[index], "lattice") == 0) {
        if (index + 1 >= num_param) {
          PRINT_INPUT_ERROR("TSPIN lattice requires on or off.");
        }
        if (strcmp(param[1], "nvt_tspin") != 0) {
          PRINT_INPUT_ERROR("TSPIN lattice is only valid for nvt_tspin.");
        }
        if (has_lattice) {
          PRINT_INPUT_ERROR("TSPIN lattice cannot be repeated.");
        }
        has_lattice = true;
        if (strcmp(param[index + 1], "on") == 0) {
          spin_lattice_enabled = true;
        } else if (strcmp(param[index + 1], "off") == 0) {
          spin_lattice_enabled = false;
        } else {
          PRINT_INPUT_ERROR("TSPIN lattice should be on or off.");
        }
        index += 2;
      } else if (strcmp(param[index], "mass_factor") == 0) {
        if (has_mass_factor) {
          PRINT_INPUT_ERROR("TSPIN mass_factor cannot be repeated.");
        }
        has_mass_factor = true;
        if (index + 1 >= num_param) {
          PRINT_INPUT_ERROR("TSPIN mass_factor requires a value or element-value pairs.");
        }
        double common_factor = 0.0;
        if (is_valid_real(param[index + 1], &common_factor)) {
          if (!std::isfinite(common_factor) || common_factor <= 0.0) {
            PRINT_INPUT_ERROR("Scalar TSPIN mass_factor should be finite and > 0.");
          }
          std::fill(
            spin_mass_factor_by_type.begin(),
            spin_mass_factor_by_type.end(),
            common_factor);
          index += 2;
        } else {
          std::vector<int> configured(atom.cpu_type_size.size(), 0);
          ++index;
          while (index < num_param && !is_tspin_option(param[index])) {
            if (index + 1 >= num_param || is_tspin_option(param[index + 1])) {
              PRINT_INPUT_ERROR(
                "Element-wise TSPIN mass_factor requires element-value pairs.");
            }
            int type = -1;
            for (int atom_index = 0; atom_index < atom.number_of_atoms; ++atom_index) {
              if (atom.cpu_atom_symbol[atom_index] == param[index]) {
                type = atom.cpu_type[atom_index];
                break;
              }
            }
            if (type < 0) {
              PRINT_INPUT_ERROR("Unknown element in TSPIN mass_factor.");
            }
            if (configured[type] != 0) {
              PRINT_INPUT_ERROR("Element in TSPIN mass_factor cannot be repeated.");
            }
            double factor = 0.0;
            if (!is_valid_real(param[index + 1], &factor) ||
                !std::isfinite(factor) || factor < 0.0) {
              PRINT_INPUT_ERROR(
                "Element-wise TSPIN mass_factor should be finite and >= 0.");
            }
            spin_mass_factor_by_type[type] = factor;
            configured[type] = 1;
            index += 2;
          }
          for (int atom_index = 0; atom_index < atom.number_of_atoms; ++atom_index) {
            if (configured[atom.cpu_type[atom_index]] == 0) {
              PRINT_INPUT_ERROR(
                "Element-wise TSPIN mass_factor must list every element in model.xyz.");
            }
          }
        }
      } else if (strcmp(param[index], "seed") == 0) {
        if (index + 1 >= num_param) {
          PRINT_INPUT_ERROR("TSPIN seed requires a value.");
        }
        if (has_seed) {
          PRINT_INPUT_ERROR("TSPIN seed cannot be repeated.");
        }
        has_seed = true;
        if (!is_valid_int(param[index + 1], &spin_seed) || spin_seed <= 0) {
          PRINT_INPUT_ERROR("TSPIN seed should be a positive integer.");
        }
        index += 2;
      } else {
        PRINT_INPUT_ERROR("Unknown TSPIN optional keyword.");
      }
    }
  }

  if (use_spin_sib || use_spin_glsd) {
    bool has_lattice = false;
    bool has_damping = false;
    bool has_alpha = false;
    bool has_gamma = false;
    bool has_spin_temperature = false;
    bool has_seed = false;
    int index = spin_option_start;
    while (index < num_param) {
      if (index + 1 >= num_param) {
        PRINT_INPUT_ERROR("SIB/GLSD optional keywords require a value.");
      }
      if (strcmp(param[index], "lattice") == 0) {
        if (strcmp(param[1], "nvt_sib") != 0 && strcmp(param[1], "nvt_glsd") != 0) {
          PRINT_INPUT_ERROR("lattice is only valid for nvt_sib or nvt_glsd.");
        }
        if (has_lattice) {
          PRINT_INPUT_ERROR("SIB/GLSD lattice cannot be repeated.");
        }
        has_lattice = true;
        if (strcmp(param[index + 1], "on") == 0) {
          spin_lattice_enabled = true;
        } else if (strcmp(param[index + 1], "off") == 0) {
          spin_lattice_enabled = false;
        } else {
          PRINT_INPUT_ERROR("SIB/GLSD lattice should be on or off.");
        }
      } else if (strcmp(param[index], "damping") == 0) {
        if (!use_spin_glsd || has_damping || !is_valid_real(param[index + 1], &spin_glsd_damping) ||
            !std::isfinite(spin_glsd_damping) || spin_glsd_damping < 0.0) {
          PRINT_INPUT_ERROR("GLSD requires finite dimensionless damping >= 0 exactly once; SIB does not accept damping.");
        }
        has_damping = true;
      } else if (strcmp(param[index], "mobility") == 0) {
        PRINT_INPUT_ERROR("mobility is not supported; GLSD uses dimensionless damping = hbar * gamma_s_prime.");
      } else if (strcmp(param[index], "alpha") == 0) {
        if (use_spin_glsd || has_alpha) {
          PRINT_INPUT_ERROR("alpha belongs only to SIB and cannot be repeated.");
        }
        has_alpha = true;
        if (!is_valid_real(param[index + 1], &spin_sib_alpha) ||
            !std::isfinite(spin_sib_alpha) || spin_sib_alpha < 0.0) {
          PRINT_INPUT_ERROR("SIB alpha should be finite and >= 0.");
        }
      } else if (strcmp(param[index], "gamma") == 0) {
        if (has_gamma) {
          PRINT_INPUT_ERROR("SIB/GLSD gamma cannot be repeated.");
        }
        has_gamma = true;
        if (!is_valid_real(param[index + 1], &spin_dynamics_gamma) ||
            !std::isfinite(spin_dynamics_gamma) || spin_dynamics_gamma <= 0.0) {
          PRINT_INPUT_ERROR("SIB/GLSD gamma should be finite and > 0.");
        }
      } else if (strcmp(param[index], "stemp") == 0) {
        if (has_spin_temperature) {
          PRINT_INPUT_ERROR("SIB/GLSD stemp cannot be repeated.");
        }
        has_spin_temperature = true;
        if (!is_valid_real(param[index + 1], &spin_dynamics_temperature) ||
            !std::isfinite(spin_dynamics_temperature) ||
            (spin_dynamics_temperature < 0.0 && spin_dynamics_temperature != -1.0)) {
          PRINT_INPUT_ERROR("SIB/GLSD stemp should be -1, 0, or a positive temperature.");
        }
      } else if (strcmp(param[index], "seed") == 0) {
        if (has_seed) {
          PRINT_INPUT_ERROR("SIB/GLSD seed cannot be repeated.");
        }
        has_seed = true;
        if (!is_valid_int(param[index + 1], &spin_dynamics_seed) || spin_dynamics_seed <= 0) {
          PRINT_INPUT_ERROR("SIB/GLSD seed should be a positive integer.");
        }
      } else {
        PRINT_INPUT_ERROR("Unknown SIB/GLSD optional keyword.");
      }
      index += 2;
    }
    if (use_spin_glsd) {
      if (!has_damping) PRINT_INPUT_ERROR("GLSD requires explicit dimensionless damping.");
      // gamma = g_L / hbar, eta = hbar * gamma_s_prime, hence lambda = eta * gamma^2 * hbar.
      // Convert after parsing all options so keyword order cannot affect the model.
      const double hbar_ps = HBAR * TIME_UNIT_CONVERSION / 1000.0;
      spin_glsd_mobility = spin_glsd_damping * spin_dynamics_gamma * (spin_dynamics_gamma * hbar_ps);
      if (!std::isfinite(spin_glsd_mobility) ||
          (spin_glsd_damping > 0.0 && spin_glsd_mobility == 0.0)) {
        PRINT_INPUT_ERROR("GLSD damping/gamma conversion exceeds the representable mobility range.");
      }
    }
    if (spin_only && spin_dynamics_temperature == 0.0) {
      PRINT_INPUT_ERROR(
        "SIB/GLSD stemp 0 requires a temperature-controlled ensemble; use stemp -1 "
        "or a positive temperature with nve_sib/nve_glsd.");
    }
  }


  std::vector<std::string> lattice(tokens.begin(), tokens.begin() + spin_option_start);
  if (npt) {
    lattice[1] = "npt_mttk";
    auto mttk = std::make_unique<Spin_Lattice_MTTK>(lattice);
    temperature1 = mttk->t_start;
    temperature2 = mttk->t_stop;
    temperature = temperature1;
    temperature_coupling = mttk->get_temperature_period();
    type = EnsembleType::MTTK;
    ensemble_ = std::move(mttk);
  } else if (spin_only) {
    temperature = temperature1 = temperature2 = 0.0;
    type = EnsembleType::NVE;
    ensemble_ = std::make_unique<Ensemble_Fixed_Lattice>(type);
  } else {
    lattice[1] = "nvt_nhc";
    auto nhc = std::make_unique<Spin_Lattice_NHC>(lattice, group);
    temperature1 = nhc->get_temperature1();
    temperature2 = nhc->get_temperature2();
    temperature = temperature1;
    temperature_coupling = get_double_from_token(tokens[4], __FILE__, __LINE__);
    type = EnsembleType::NVT_NHC;
    if (spin_lattice_enabled) ensemble_ = std::move(nhc);
    else ensemble_ = std::make_unique<Ensemble_Fixed_Lattice>(type);
  }
  if (use_spin_tspin) {
    printf("Integrate spins with TSPIN.\n");
    print_spin_mass_factors(atom, spin_mass_factor_by_type);
  } else {
    if (use_spin_glsd) {
      printf("Integrate variable-magnitude spins with Ma-Dudarev GLSD (Heun).\n");
      printf("    damping = %.17g (dimensionless hbar * gamma_s_prime).\n", spin_glsd_damping);
      printf("    converted mobility = %.17g muB^2/(eV ps); Cartesian Boltzmann measure.\n", spin_glsd_mobility);
    } else {
      printf("Integrate spins with the semi-implicit B (SIB) method.\n");
      printf("    alpha = %g.\n", spin_sib_alpha);
    }
    printf("    gamma = %g (eV ps)^-1, stemp = %g K, seed = %d.\n",
      spin_dynamics_gamma, spin_dynamics_temperature, spin_dynamics_seed);
  }
}
