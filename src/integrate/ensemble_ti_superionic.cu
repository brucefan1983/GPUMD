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

#include "ensemble_ti_superionic.cuh"
#include "uf_reference.cuh"
#include "utilities/gpu_macro.cuh"
#include <cstdlib>
#include <cstring>
#include <math.h>

namespace
{

bool is_superionic_keyword(const char* token)
{
  return strcmp(token, "temp") == 0 || strcmp(token, "tperiod") == 0 ||
         strcmp(token, "tequil") == 0 || strcmp(token, "tswitch") == 0 ||
         strcmp(token, "press") == 0 || strcmp(token, "spring") == 0 ||
         strcmp(token, "uf") == 0;
}

int stage_number(SuperionicStage stage)
{
  return static_cast<int>(stage);
}

const char* csv_filename(SuperionicStage stage)
{
  return stage == SuperionicStage::stage1 ? "ti_superionic_stage1.csv"
                                          : "ti_superionic_stage2.csv";
}

const char* yaml_filename(SuperionicStage stage)
{
  return stage == SuperionicStage::stage1 ? "ti_superionic_stage1.yaml"
                                          : "ti_superionic_stage2.yaml";
}

bool is_nep_small_box(const Potential* potential, const Box& box)
{
  if (potential->nep_model_type < 0) {
    return false;
  }

  const double volume = box.get_volume();
  const double limit = 2.5 * (potential->rc + 1.0);
  return (box.pbc_x && volume / box.get_area(0) <= limit) ||
         (box.pbc_y && volume / box.get_area(1) <= limit) ||
         (box.pbc_z && volume / box.get_area(2) <= limit);
}

static __global__ void gpu_zero_superionic_arrays(
  const int N,
  double* einstein,
  double* uf_self,
  double* uf_cross,
  double* aux_fx,
  double* aux_fy,
  double* aux_fz,
  double* cross_fx,
  double* cross_fy,
  double* cross_fz)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    einstein[i] = 0.0;
    uf_self[i] = 0.0;
    uf_cross[i] = 0.0;
    aux_fx[i] = 0.0;
    aux_fy[i] = 0.0;
    aux_fz[i] = 0.0;
    cross_fx[i] = 0.0;
    cross_fy[i] = 0.0;
    cross_fz[i] = 0.0;
  }
}

static __global__ void gpu_find_superionic_spring(
  const int N,
  Box box,
  const double* k,
  const double* x,
  const double* y,
  const double* z,
  const double* x0,
  const double* y0,
  const double* z0,
  double* einstein,
  double* aux_fx,
  double* aux_fy,
  double* aux_fz)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    double dx = x[i] - x0[i];
    double dy = y[i] - y0[i];
    double dz = z[i] - z0[i];
    apply_mic(box, dx, dy, dz);
    const double spring_k = k[i];
    einstein[i] = 0.5 * spring_k * (dx * dx + dy * dy + dz * dz);
    aux_fx[i] += -spring_k * dx;
    aux_fy[i] += -spring_k * dy;
    aux_fz[i] += -spring_k * dz;
  }
}

static __global__ void gpu_find_superionic_uf(
  const int N,
  const int num_types,
  Box box,
  const double beta,
  const int* type,
  const double* x,
  const double* y,
  const double* z,
  const double* uf_p,
  const double* uf_sigma_sqrd,
  const int* uf_kind,
  const int* NN,
  const int* NL,
  double* uf_self,
  double* uf_cross,
  double* aux_fx,
  double* aux_fy,
  double* aux_fz,
  double* cross_fx,
  double* cross_fy,
  double* cross_fz)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    const double x1 = x[i];
    const double y1 = y[i];
    const double z1 = z[i];
    const int type_i = type[i];
    double local_uf_self = 0.0;
    double local_uf_cross = 0.0;
    double local_aux_fx = 0.0;
    double local_aux_fy = 0.0;
    double local_aux_fz = 0.0;
    double local_cross_fx = 0.0;
    double local_cross_fy = 0.0;
    double local_cross_fz = 0.0;

    for (int i1 = 0; i1 < NN[i]; ++i1) {
      const int j = NL[i + N * i1];
      const int type_j = type[j];
      const int pair = type_i * num_types + type_j;
      const int kind = uf_kind[pair];
      if (kind == 0) {
        continue;
      }

      double dx = x[j] - x1;
      double dy = y[j] - y1;
      double dz = z[j] - z1;
      apply_mic(box, dx, dy, dz);
      const double r2 = dx * dx + dy * dy + dz * dz;
      const double p = uf_p[pair];
      const double sigma2 = uf_sigma_sqrd[pair];
      const double pair_energy = -p / beta * log(1.0 - exp(-r2 / sigma2));
      const double factor = -2.0 * p / (beta * sigma2 * (exp(r2 / sigma2) - 1.0));
      const double fx = dx * factor;
      const double fy = dy * factor;
      const double fz = dz * factor;

      if (kind == 1) {
        local_uf_self += 0.5 * pair_energy;
        local_aux_fx += fx;
        local_aux_fy += fy;
        local_aux_fz += fz;
      } else if (kind == 2) {
        local_uf_cross += 0.5 * pair_energy;
        local_cross_fx += fx;
        local_cross_fy += fy;
        local_cross_fz += fz;
      }
    }

    uf_self[i] += local_uf_self;
    uf_cross[i] += local_uf_cross;
    aux_fx[i] += local_aux_fx;
    aux_fy[i] += local_aux_fy;
    aux_fz[i] += local_aux_fz;
    cross_fx[i] += local_cross_fx;
    cross_fy[i] += local_cross_fy;
    cross_fz[i] += local_cross_fz;
  }
}

static __global__ void gpu_apply_superionic_stage1(
  const int N,
  const double lambda,
  const double* aux_fx,
  const double* aux_fy,
  const double* aux_fz,
  const double* cross_fx,
  const double* cross_fy,
  const double* cross_fz,
  double* fx,
  double* fy,
  double* fz)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    fx[i] = aux_fx[i] + lambda * cross_fx[i];
    fy[i] = aux_fy[i] + lambda * cross_fy[i];
    fz[i] = aux_fz[i] + lambda * cross_fz[i];
  }
}

static __global__ void gpu_add_cross_to_aux(
  const int N,
  const double* cross_fx,
  const double* cross_fy,
  const double* cross_fz,
  double* aux_fx,
  double* aux_fy,
  double* aux_fz)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    aux_fx[i] += cross_fx[i];
    aux_fy[i] += cross_fy[i];
    aux_fz[i] += cross_fz[i];
  }
}

static __global__ void gpu_apply_superionic_stage2(
  const int N,
  const double lambda,
  const double* aux_fx,
  const double* aux_fy,
  const double* aux_fz,
  double* fx,
  double* fy,
  double* fz)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    fx[i] = (1.0 - lambda) * aux_fx[i] + lambda * fx[i];
    fy[i] = (1.0 - lambda) * aux_fy[i] + lambda * fy[i];
    fz[i] = (1.0 - lambda) * aux_fz[i] + lambda * fz[i];
  }
}

static __global__ void gpu_sum_array(const int N, double* data)
{
  const int tid = threadIdx.x;
  const int number_of_patches = (N - 1) / blockDim.x + 1;
  __shared__ double s_data[1024];
  s_data[tid] = 0.0;

  for (int patch = 0; patch < number_of_patches; ++patch) {
    const int n = tid + patch * blockDim.x;
    if (n < N) {
      s_data[tid] += data[n];
    }
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_data[tid] += s_data[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    data[0] = s_data[0];
  }
}

} // namespace

Ensemble_TI_Superionic::Ensemble_TI_Superionic(
  const char** params, int num_params, SuperionicStage input_stage)
  : stage(input_stage)
{
  atom = nullptr;
  box = nullptr;
  group = nullptr;
  thermo = nullptr;
  current_step = nullptr;
  total_steps = nullptr;

  temperature_coupling = 100;
  bool has_auto_spring = false;
  bool has_explicit_spring = false;
  int i = 2;
  while (i < num_params) {
    if (strcmp(params[i], "temp") == 0) {
      if (i + 1 >= num_params || !is_valid_real(params[i + 1], &temperature))
        PRINT_INPUT_ERROR("Wrong inputs for temp keyword.");
      has_temperature = true;
      i += 2;
    } else if (strcmp(params[i], "tperiod") == 0) {
      if (i + 1 >= num_params || !is_valid_real(params[i + 1], &temperature_coupling))
        PRINT_INPUT_ERROR("Wrong inputs for t_period keyword.");
      i += 2;
    } else if (strcmp(params[i], "tequil") == 0) {
      if (i + 1 >= num_params || !is_valid_int(params[i + 1], &t_equil))
        PRINT_INPUT_ERROR("Wrong inputs for t_equil keyword.");
      i += 2;
    } else if (strcmp(params[i], "tswitch") == 0) {
      if (i + 1 >= num_params || !is_valid_int(params[i + 1], &t_switch))
        PRINT_INPUT_ERROR("Wrong inputs for t_switch keyword.");
      i += 2;
    } else if (strcmp(params[i], "press") == 0) {
      if (i + 1 >= num_params || !is_valid_real(params[i + 1], &target_pressure))
        PRINT_INPUT_ERROR("Wrong inputs for press keyword.");
      target_pressure /= PRESSURE_UNIT_CONVERSION;
      i += 2;
    } else if (strcmp(params[i], "spring") == 0) {
      i++;
      if (i >= num_params || is_superionic_keyword(params[i]))
        PRINT_INPUT_ERROR("Missing inputs for spring keyword.");

      if (strcmp(params[i], "auto") == 0) {
        if (has_explicit_spring)
          PRINT_INPUT_ERROR("Cannot mix auto and explicit spring inputs.");
        has_auto_spring = true;
        auto_k = true;
        i++;
        while (i < num_params && !is_superionic_keyword(params[i])) {
          auto_spring_species.push_back(params[i]);
          i++;
        }
      } else {
        if (has_auto_spring)
          PRINT_INPUT_ERROR("Cannot mix auto and explicit spring inputs.");
        has_explicit_spring = true;
        auto_k = false;
        while (i < num_params && !is_superionic_keyword(params[i])) {
          double k = 0.0;
          if (i + 1 >= num_params || is_superionic_keyword(params[i + 1]) ||
              !is_valid_real(params[i + 1], &k))
            PRINT_INPUT_ERROR("Wrong inputs for spring keyword.");
          if (k <= 0.0)
            PRINT_INPUT_ERROR("Spring constant must be positive.");
          spring_map[params[i]] = k;
          i += 2;
        }
      }
    } else if (strcmp(params[i], "uf") == 0) {
      if (i + 4 >= num_params)
        PRINT_INPUT_ERROR("Wrong inputs for uf keyword.");
      SuperionicUFPair pair;
      pair.element_i = params[i + 1];
      pair.element_j = params[i + 2];
      if (!is_valid_real(params[i + 3], &pair.p) || !is_valid_real(params[i + 4], &pair.sigma))
        PRINT_INPUT_ERROR("Wrong inputs for uf keyword.");
      uf_pairs.push_back(pair);
      i += 5;
    } else {
      PRINT_INPUT_ERROR("Unknown keyword.");
    }
  }

  if (t_equil < 0 || t_switch < 0)
    PRINT_INPUT_ERROR("Please specify both t_equil and t_switch.");
  if (!has_temperature)
    PRINT_INPUT_ERROR("Please specify temp.");
  if (temperature <= 0)
    PRINT_INPUT_ERROR("Temperature should > 0.");
  beta = 1.0 / (temperature * K_B);
  if (temperature_coupling < 1)
    PRINT_INPUT_ERROR("Temperature coupling should >= 1.");
  if (t_switch <= 0)
    PRINT_INPUT_ERROR("t_switch should be > 0.");
  if (auto_k)
    PRINT_INPUT_ERROR("Automatic spring constants are not implemented yet.");
  if (spring_map.empty() && auto_spring_species.empty())
    PRINT_INPUT_ERROR("Please specify at least one spring species.");
  if (uf_pairs.empty())
    PRINT_INPUT_ERROR("Please specify at least one uf pair.");

  printf(
    "Thermostat: target temperature is %f k, t_period is %f timesteps.\n",
    temperature,
    temperature_coupling);
  type = 3;
  c1 = exp(-0.5 / temperature_coupling);
  c2 = sqrt((1 - c1 * c1) * K_B * temperature);
}

bool Ensemble_TI_Superionic::is_supported_self_p(double p) const
{
  return uf_reference::supports_p(p);
}

int Ensemble_TI_Superionic::find_type_for_symbol(const std::string& symbol) const
{
  for (int i = 0; i < atom->number_of_atoms; ++i) {
    if (atom->cpu_atom_symbol[i] == symbol)
      return atom->cpu_type[i];
  }
  return -1;
}

void Ensemble_TI_Superionic::validate_species()
{
  int local_num_types = static_cast<int>(atom->cpu_type_size.size());
  if (auto_k) {
    for (const auto& symbol : auto_spring_species) {
      if (find_type_for_symbol(symbol) < 0)
        PRINT_INPUT_ERROR("spring element does not exist in the structure.");
    }
  } else {
    for (const auto& entry : spring_map) {
      if (find_type_for_symbol(entry.first) < 0)
        PRINT_INPUT_ERROR("spring element does not exist in the structure.");
    }
  }

  bool has_self_pair = false;
  std::vector<int> seen_uf_pair(local_num_types * local_num_types, 0);
  for (const auto& pair : uf_pairs) {
    int type_i = find_type_for_symbol(pair.element_i);
    int type_j = find_type_for_symbol(pair.element_j);
    if (type_i < 0 || type_j < 0)
      PRINT_INPUT_ERROR("uf element does not exist in the structure.");
    if (pair.p <= 0.0 || pair.sigma <= 0.0)
      PRINT_INPUT_ERROR("UF p and sigma must be positive.");
    if (pair.element_i == pair.element_j) {
      has_self_pair = true;
      if (!is_supported_self_p(pair.p))
        PRINT_INPUT_ERROR("Self UF p must be 1, 25, 50, 75, or 100.");
    }
    int type_min = type_i < type_j ? type_i : type_j;
    int type_max = type_i < type_j ? type_j : type_i;
    int key = type_min * local_num_types + type_max;
    if (seen_uf_pair[key])
      PRINT_INPUT_ERROR("Duplicate UF pair.");
    seen_uf_pair[key] = 1;
  }
  if (!has_self_pair)
    PRINT_INPUT_ERROR("Please specify at least one self uf pair.");
}

void Ensemble_TI_Superionic::prepare_reference_state()
{
  validate_species();
  int N = atom->number_of_atoms;
  num_types = static_cast<int>(atom->cpu_type_size.size());
  cpu_k.assign(N, 0.0);
  cpu_spring_mask.assign(N, 0.0);
  cpu_uf_p.assign(num_types * num_types, 0.0);
  cpu_uf_sigma_sqrd.assign(num_types * num_types, 1.0);
  cpu_uf_kind.assign(num_types * num_types, 0);

  for (int i = 0; i < N; ++i) {
    std::string symbol = atom->cpu_atom_symbol[i];
    if (auto_k) {
      for (const auto& auto_symbol : auto_spring_species) {
        if (symbol == auto_symbol)
          cpu_spring_mask[i] = 1.0;
      }
    } else {
      auto spring = spring_map.find(symbol);
      if (spring != spring_map.end()) {
        cpu_spring_mask[i] = 1.0;
        cpu_k[i] = spring->second;
      }
    }
  }

  for (const auto& pair : uf_pairs) {
    int type_i = find_type_for_symbol(pair.element_i);
    int type_j = find_type_for_symbol(pair.element_j);
    int kind = pair.element_i == pair.element_j ? 1 : 2;
    int ij = type_i * num_types + type_j;
    int ji = type_j * num_types + type_i;
    cpu_uf_p[ij] = pair.p;
    cpu_uf_p[ji] = pair.p;
    cpu_uf_sigma_sqrd[ij] = pair.sigma * pair.sigma;
    cpu_uf_sigma_sqrd[ji] = pair.sigma * pair.sigma;
    cpu_uf_kind[ij] = kind;
    cpu_uf_kind[ji] = kind;
  }

  gpu_k.resize(N);
  gpu_k.copy_from_host(cpu_k.data());
  gpu_spring_mask.resize(N);
  gpu_spring_mask.copy_from_host(cpu_spring_mask.data());
  gpu_uf_p.resize(num_types * num_types);
  gpu_uf_p.copy_from_host(cpu_uf_p.data());
  gpu_uf_sigma_sqrd.resize(num_types * num_types);
  gpu_uf_sigma_sqrd.copy_from_host(cpu_uf_sigma_sqrd.data());
  gpu_uf_kind.resize(num_types * num_types);
  gpu_uf_kind.copy_from_host(cpu_uf_kind.data());
  gpu_einstein.resize(N, 0.0);
  gpu_uf_self.resize(N, 0.0);
  gpu_uf_cross.resize(N, 0.0);
  gpu_aux_fx.resize(N, 0.0);
  gpu_aux_fy.resize(N, 0.0);
  gpu_aux_fz.resize(N, 0.0);
  gpu_cross_fx.resize(N, 0.0);
  gpu_cross_fy.resize(N, 0.0);
  gpu_cross_fz.resize(N, 0.0);
  position_0.resize(3 * N);
  CHECK(gpuMemcpy(
    position_0.data(),
    atom->position_per_atom.data(),
    sizeof(double) * position_0.size(),
    gpuMemcpyDeviceToDevice));
}

void Ensemble_TI_Superionic::write_yaml_pair_list(
  FILE* file, const char* key, bool self_pairs) const
{
  bool has_pair = false;
  for (const auto& pair : uf_pairs) {
    bool is_self = pair.element_i == pair.element_j;
    if (is_self == self_pairs)
      has_pair = true;
  }
  if (!has_pair) {
    fprintf(file, "%s: []\n", key);
    return;
  }

  fprintf(file, "%s:\n", key);
  for (const auto& pair : uf_pairs) {
    bool is_self = pair.element_i == pair.element_j;
    if (is_self == self_pairs) {
      fprintf(
        file,
        "  - {element_i: \"%s\", element_j: \"%s\", p: %.17g, sigma: %.17g}\n",
        pair.element_i.c_str(),
        pair.element_j.c_str(),
        pair.p,
        pair.sigma);
    }
  }
}

double Ensemble_TI_Superionic::get_uf_fe_for_pair(const SuperionicUFPair& pair, int count)
{
  if (count <= 0)
    PRINT_INPUT_ERROR("Self UF species has no atoms.");

  double kT = K_B * temperature;
  double species_volume = box->get_volume() / count;
  double sigma_sqrd = pair.sigma * pair.sigma;
  double x_UF = pow(PI * sigma_sqrd, 1.5) / (2.0 * species_volume);
  int index = 0;
  if (x_UF < 0.1) {
    index = static_cast<int>(x_UF * 400);
  } else if (x_UF < 1) {
    index = 40 + static_cast<int>(x_UF * 40 - 4);
  } else if (x_UF < 4) {
    index = 76 + static_cast<int>(x_UF * 10 - 10);
  } else {
    index = 105;
  }

  const auto uf_data = uf_reference::get_data(pair.p);
  const std::vector<double>& sum_spline = uf_data.sum_spline;
  const std::vector<std::vector<double>>& spline = uf_data.spline;
  double coef[4] = {spline[index][0], spline[index][1], spline[index][2], spline[index][3]};

  double F_UF = uf_reference::fe(x_UF, coef, sum_spline, index) * kT * count;
  double mass = 0.0;
  int type = find_type_for_symbol(pair.element_i);
  for (int i = 0; i < atom->number_of_atoms; ++i) {
    if (atom->cpu_type[i] == type) {
      mass = atom->cpu_mass[i];
      break;
    }
  }
  if (mass <= 0.0)
    PRINT_INPUT_ERROR("Mass is not available for a self UF species.");
  double de_broglie = log(HBAR * sqrt(2 * PI / (mass * kT)));
  double F_IG = count * kT * (log(1.0 / species_volume) - 1.0) + 3.0 * kT * count * de_broglie;
  return (F_UF + F_IG) / atom->number_of_atoms;
}

void Ensemble_TI_Superionic::compute_reference_free_energy()
{
  if (
    cpu_spring_mask.size() != static_cast<size_t>(atom->number_of_atoms) ||
    cpu_k.size() != static_cast<size_t>(atom->number_of_atoms))
    PRINT_INPUT_ERROR("Reference state is not available for free-energy calculation.");

  double kT = K_B * temperature;
  F_Einstein = 0.0;
  for (int i = 0; i < atom->number_of_atoms; ++i) {
    if (cpu_spring_mask[i] > 0.5) {
      if (cpu_k[i] <= 0.0)
        PRINT_INPUT_ERROR("Spring constant is not available for a spring atom.");
      double omega = sqrt(cpu_k[i] / atom->cpu_mass[i]);
      F_Einstein += log(omega * HBAR / kT);
    }
  }
  F_Einstein = 3.0 * kT * F_Einstein / atom->number_of_atoms;

  F_UF_self = 0.0;
  for (const auto& pair : uf_pairs) {
    if (pair.element_i == pair.element_j) {
      int count = 0;
      int type = find_type_for_symbol(pair.element_i);
      for (int i = 0; i < atom->number_of_atoms; ++i) {
        if (atom->cpu_type[i] == type)
          count++;
      }
      F_UF_self += get_uf_fe_for_pair(pair, count);
    }
  }
  F_ref = F_Einstein + F_UF_self;
}

void Ensemble_TI_Superionic::init()
{
  printf("The number of steps should be set to %d!\n", 2 * (t_equil + t_switch));
  printf(
    "Superionic TI stage %d: t_switch is %d timestep, t_equil is %d timesteps.\n",
    stage_number(stage),
    t_switch,
    t_equil);

  output_file = my_fopen(csv_filename(stage), "w");
  if (stage == SuperionicStage::stage1) {
    fprintf(output_file, "lambda,dlambda,U_einstein,U_uf_self,U_uf_cross,dHdlambda\n");
  } else {
    fprintf(
      output_file,
      "lambda,dlambda,U_target,U_einstein,U_uf_self,U_uf_cross,U_aux,dHdlambda\n");
  }

  int N = atom->number_of_atoms;
  curand_states.resize(N);
  int grid_size = (N - 1) / 128 + 1;
  initialize_curand_states<<<grid_size, 128>>>(curand_states.data(), N, rand());
  GPU_CHECK_KERNEL

  thermo_cpu.resize(thermo->size());
  prepare_reference_state();
  initialized = true;
}

Ensemble_TI_Superionic::~Ensemble_TI_Superionic(void)
{
  int N = 0;
  if (atom != nullptr) {
    N = atom->number_of_atoms;
  }
  if (atom != nullptr && box != nullptr && N > 0) {
    V = box->get_volume() / N;
    compute_reference_free_energy();
  }

  FILE* yaml_file = my_fopen(yaml_filename(stage), "w");
  fprintf(yaml_file, "stage: %d\n", stage_number(stage));
  fprintf(yaml_file, "T: %.17g\n", temperature);
  fprintf(yaml_file, "V: %.17g\n", V);
  fprintf(yaml_file, "P: %.17g\n", target_pressure);
  fprintf(yaml_file, "N_total: %d\n", N);

  fprintf(yaml_file, "spring_species:\n");
  if (auto_k) {
    for (const auto& symbol : auto_spring_species)
      fprintf(yaml_file, "  - \"%s\"\n", symbol.c_str());
  } else {
    for (const auto& entry : spring_map) {
      fprintf(yaml_file, "  - \"%s\"\n", entry.first.c_str());
    }
  }
  write_yaml_pair_list(yaml_file, "uf_self_pairs", true);
  write_yaml_pair_list(yaml_file, "uf_cross_pairs", false);

  fprintf(yaml_file, "W_forward: %.17g\n", W_forward);
  fprintf(yaml_file, "W_backward: %.17g\n", W_backward);
  fprintf(yaml_file, "delta_F: %.17g\n", delta_F);
  fprintf(yaml_file, "F_Einstein: %.17g\n", F_Einstein);
  fprintf(yaml_file, "F_UF_self: %.17g\n", F_UF_self);
  fprintf(yaml_file, "F_ref: %.17g\n", F_ref);

  if (output_file != nullptr) {
    printf("Closing %s output file...\n", csv_filename(stage));
    fclose(output_file);
  }
  fclose(yaml_file);
}

void Ensemble_TI_Superionic::compute1(
  const double time_step,
  const std::vector<Group>& group,
  Box& box,
  Atom& atoms,
  GPU_Vector<double>& thermo)
{
  if (*current_step == 0 && !initialized)
    init();
  Ensemble_LAN::compute1(time_step, group, box, atoms, thermo);
}

void Ensemble_TI_Superionic::find_thermo()
{
  Ensemble::find_thermo(
    false,
    box->get_volume(),
    *group,
    atom->mass,
    atom->potential_per_atom,
    atom->velocity_per_atom,
    atom->virial_per_atom,
    *thermo);
  thermo->copy_to_host(thermo_cpu.data());
  pe = thermo_cpu[1];
}

double Ensemble_TI_Superionic::get_sum(GPU_Vector<double>& data)
{
  double value = 0.0;
  gpu_sum_array<<<1, 1024>>>(atom->number_of_atoms, data.data());
  GPU_CHECK_KERNEL
  data.copy_to_host(&value, 1);
  return value;
}

void Ensemble_TI_Superionic::find_reference_forces(Force& force)
{
  int N = atom->number_of_atoms;
  const int grid_size = (N - 1) / 128 + 1;

  gpu_zero_superionic_arrays<<<grid_size, 128>>>(
    N,
    gpu_einstein.data(),
    gpu_uf_self.data(),
    gpu_uf_cross.data(),
    gpu_aux_fx.data(),
    gpu_aux_fy.data(),
    gpu_aux_fz.data(),
    gpu_cross_fx.data(),
    gpu_cross_fy.data(),
    gpu_cross_fz.data());
  GPU_CHECK_KERNEL

  gpu_find_superionic_spring<<<grid_size, 128>>>(
    N,
    *box,
    gpu_k.data(),
    atom->position_per_atom.data(),
    atom->position_per_atom.data() + N,
    atom->position_per_atom.data() + 2 * N,
    position_0.data(),
    position_0.data() + N,
    position_0.data() + 2 * N,
    gpu_einstein.data(),
    gpu_aux_fx.data(),
    gpu_aux_fy.data(),
    gpu_aux_fz.data());
  GPU_CHECK_KERNEL

  if (is_nep_small_box(force.potentials[0].get(), *box))
    PRINT_INPUT_ERROR(
      "ti_superionic requires the main NEP potential to expose the active radial neighbor list; "
      "please use a larger periodic box.");

  const GPU_Vector<int>& NN = force.potentials[0]->get_NN_radial_ptr();
  const GPU_Vector<int>& NL = force.potentials[0]->get_NL_radial_ptr();
  if (NN.size() == 0 || NL.size() == 0)
    PRINT_INPUT_ERROR("The main potential must provide a radial neighbor list for ti_superionic.");

  gpu_find_superionic_uf<<<grid_size, 128>>>(
    N,
    num_types,
    *box,
    beta,
    atom->type.data(),
    atom->position_per_atom.data(),
    atom->position_per_atom.data() + N,
    atom->position_per_atom.data() + 2 * N,
    gpu_uf_p.data(),
    gpu_uf_sigma_sqrd.data(),
    gpu_uf_kind.data(),
    NN.data(),
    NL.data(),
    gpu_uf_self.data(),
    gpu_uf_cross.data(),
    gpu_aux_fx.data(),
    gpu_aux_fy.data(),
    gpu_aux_fz.data(),
    gpu_cross_fx.data(),
    gpu_cross_fy.data(),
    gpu_cross_fz.data());
  GPU_CHECK_KERNEL

  U_einstein = get_sum(gpu_einstein);
  U_uf_self = get_sum(gpu_uf_self);
  U_uf_cross = get_sum(gpu_uf_cross);
  U_aux = U_einstein + U_uf_self + U_uf_cross;
}

void Ensemble_TI_Superionic::accumulate_work()
{
  double increment = dHdlambda * dlambda / atom->number_of_atoms;
  if (dlambda > 0.0) {
    W_forward += increment;
  } else if (dlambda < 0.0) {
    W_backward += increment;
  }
  delta_F = 0.5 * (W_forward - W_backward);
}

void Ensemble_TI_Superionic::find_lambda()
{
  lambda = 0.0;
  dlambda = 0.0;
  lambda_active = false;
  V = box->get_volume() / atom->number_of_atoms;

  const int t = *current_step - t_equil;
  const double r_switch = 1.0 / t_switch;

  if ((t >= 0) && (t <= t_switch)) {
    lambda = switch_func(t * r_switch);
    dlambda = dswitch_func(t * r_switch);
    lambda_active = true;
  } else if ((t > t_switch) && (t < t_equil + t_switch)) {
    lambda = 1.0;
  } else if ((t >= t_equil + t_switch) && (t <= (t_equil + 2 * t_switch))) {
    lambda = switch_func(1.0 - (t - t_switch - t_equil) * r_switch);
    dlambda = -dswitch_func(1.0 - (t - t_switch - t_equil) * r_switch);
    lambda_active = true;
  }
}

void Ensemble_TI_Superionic::compute3(
  const double time_step,
  const std::vector<Group>& group,
  Box& box,
  Atom& atoms,
  GPU_Vector<double>& thermo,
  Force& force)
{
  int N = atom->number_of_atoms;
  const int grid_size = (N - 1) / 128 + 1;

  find_lambda();
  find_thermo();
  find_reference_forces(force);

  if (stage == SuperionicStage::stage1) {
    dHdlambda = U_uf_cross;
    gpu_apply_superionic_stage1<<<grid_size, 128>>>(
      N,
      lambda,
      gpu_aux_fx.data(),
      gpu_aux_fy.data(),
      gpu_aux_fz.data(),
      gpu_cross_fx.data(),
      gpu_cross_fy.data(),
      gpu_cross_fz.data(),
      atom->force_per_atom.data(),
      atom->force_per_atom.data() + N,
      atom->force_per_atom.data() + 2 * N);
    GPU_CHECK_KERNEL
  } else {
    dHdlambda = pe - U_aux;
    gpu_add_cross_to_aux<<<grid_size, 128>>>(
      N,
      gpu_cross_fx.data(),
      gpu_cross_fy.data(),
      gpu_cross_fz.data(),
      gpu_aux_fx.data(),
      gpu_aux_fy.data(),
      gpu_aux_fz.data());
    GPU_CHECK_KERNEL

    gpu_apply_superionic_stage2<<<grid_size, 128>>>(
      N,
      lambda,
      gpu_aux_fx.data(),
      gpu_aux_fy.data(),
      gpu_aux_fz.data(),
      atom->force_per_atom.data(),
      atom->force_per_atom.data() + N,
      atom->force_per_atom.data() + 2 * N);
    GPU_CHECK_KERNEL
  }

  if (lambda_active) {
    accumulate_work();
    if (output_file != nullptr) {
      if (stage == SuperionicStage::stage1) {
        fprintf(
          output_file,
          "%.17e,%.17e,%.17e,%.17e,%.17e,%.17e\n",
          lambda,
          dlambda,
          U_einstein / N,
          U_uf_self / N,
          U_uf_cross / N,
          dHdlambda / N);
      } else {
        fprintf(
          output_file,
          "%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e\n",
          lambda,
          dlambda,
          pe / N,
          U_einstein / N,
          U_uf_self / N,
          U_uf_cross / N,
          U_aux / N,
          dHdlambda / N);
      }
    }
  }

  Ensemble_LAN::compute2(time_step, group, box, atoms, thermo);
}

double Ensemble_TI_Superionic::switch_func(double t)
{
  double t2 = t * t;
  double t5 = t2 * t2 * t;
  return ((70.0 * t2 * t2 - 315.0 * t2 * t + 540.0 * t2 - 420.0 * t + 126.0) * t5);
}

double Ensemble_TI_Superionic::dswitch_func(double t)
{
  double t2 = t * t;
  double t4 = t2 * t2;
  return ((630 * t2 * t2 - 2520 * t2 * t + 3780 * t2 - 2520 * t + 630) * t4) /
         t_switch;
}
