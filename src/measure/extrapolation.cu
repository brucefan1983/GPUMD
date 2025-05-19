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

#include "extrapolation.cuh"
#include "utilities/gpu_macro.cuh"
#include <cstring>

#define PRINT_RUNTIME_ERROR(text)                                                                  \
  do {                                                                                             \
    fprintf(stderr, "Runtime Error:\n");                                                           \
    fprintf(stderr, "    File:       %s\n", __FILE__);                                             \
    fprintf(stderr, "    Line:       %d\n", __LINE__);                                             \
    fprintf(stderr, "    Error text: %s\n", text);                                                 \
    exit(1);                                                                                       \
  } while (0)

__global__ void gpu_calculate_max_gamma(
  double* gamma_full, double* gamma, int number_of_particles, int B_size_per_atom)
{
  double a;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < number_of_particles) {
    gamma[i] = 0;
    for (int j = 0; j < B_size_per_atom; j++) {
      a = std::abs(gamma_full[i * B_size_per_atom + j]);
      if (a > gamma[i]) {
        gamma[i] = a;
      }
    }
  }
}

Extrapolation::Extrapolation(const char** params, int num_params)
{
  property_name = "compute_extrapolation";
  int i = 1;
  while (i < num_params) {
    if (strcmp(params[i], "asi_file") == 0) {
      asi_file_name.assign(params[i + 1]);
      i += 2;
    } else if (strcmp(params[i], "gamma_low") == 0) {
      if (!is_valid_real(params[i + 1], &gamma_low)) {
        PRINT_INPUT_ERROR("Wrong input for gamma_low.\n");
      }
      i += 2;
    } else if (strcmp(params[i], "gamma_high") == 0) {
      if (!is_valid_real(params[i + 1], &gamma_high)) {
        PRINT_INPUT_ERROR("Wrong input for gamma_high.\n");
      }
      i += 2;
    } else if (strcmp(params[i], "check_interval") == 0) {
      if (!is_valid_int(params[i + 1], &check_interval)) {
        PRINT_INPUT_ERROR("Wrong input for check_interval.\n");
      }
      i += 2;
    } else if (strcmp(params[i], "dump_interval") == 0) {
      if (!is_valid_int(params[i + 1], &dump_interval)) {
        PRINT_INPUT_ERROR("Wrong input for dump_interval.\n");
      }
      i += 2;
    } else {
      PRINT_INPUT_ERROR("Wrong input parameter!");
    }
  }
}

void Extrapolation::preprocess(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  int N = atom.number_of_atoms;
  printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
  printf("Initializing extrapolation grade calculation...\n");
  B_size_per_atom = force.potentials[0]->B_projection_size;
  if (B_size_per_atom == 0)
    PRINT_INPUT_ERROR("This potential cannot be used to calculate the extrapolation grade!");
  else
    printf("The length of B vector for each atom: %d\n", B_size_per_atom);
  B.resize(B_size_per_atom * N);
  gamma_full.resize(B_size_per_atom * N);
  gamma.resize(N, Memory_Type::managed);
  force.potentials[0]->B_projection = B.data();
  force.potentials[0]->need_B_projection = true;
  this->atom = &atom;
  this->box = &box;
  f = my_fopen("extrapolation_dump.xyz", "a");

  blas_A.resize(N, Memory_Type::managed);
  blas_x.resize(N, Memory_Type::managed);
  blas_y.resize(N, Memory_Type::managed);
  load_asi();
  for (int i = 0; i < N; i++) {
    blas_x[i] = B.data() + i * B_size_per_atom;
    blas_y[i] = gamma_full.data() + i * B_size_per_atom;
  }

  gpublasCreate(&handle);
  printf("gamma_low:      %f\n", gamma_low);
  printf("gamma_high:     %f\n", gamma_high);
  printf("check_interval: %d\n", check_interval);
  printf("dump_interval:  %d\n", dump_interval);
  printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
}

void Extrapolation::postprocess(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature)
{
  printf("Closing extrapolation dump file...\n");
  fclose(f);
  gpublasDestroy(handle);
};

void Extrapolation::load_asi()
{
  printf("Loading the Active Set Inversion file (ASI): %s\n", asi_file_name.c_str());
  std::ifstream f(asi_file_name);
  std::string token;
  if (f.is_open()) {
    while (f >> token) {
      int type_of_atom = -1;
      std::string element = token;
      for (int m = 0; m < atom->number_of_atoms; ++m) {
        if (element == atom->cpu_atom_symbol[m]) {
          type_of_atom = atom->cpu_type[m];
          break;
        }
      }
      f >> token;
      int shape1 = std::stoi(token);
      f >> token;
      int shape2 = std::stoi(token);
      int B_size = shape1 * shape2;
      printf(
        "    Loading the ASI of %s (%d): shape %d x %d, ",
        element.c_str(),
        type_of_atom,
        shape1,
        shape2);
      asi_list.emplace_back(
        std::unique_ptr<GPU_Vector<double>>(new GPU_Vector<double>(B_size, Memory_Type::managed)));
      auto& asi = asi_list.back();
      for (int i = 0; i < B_size; ++i) {
        f >> (*asi)[i];
      }
      printf("[%f %f ... %f]\n", (*asi)[0], (*asi)[1], (*asi)[B_size - 1]);

      for (int j = 0; j < atom->number_of_atoms; j++) {
        if (atom->cpu_type[j] == type_of_atom) {
          blas_A[j] = asi->data();
        }
      }
    }
    printf("ASI successfully loaded!\n");
    f.close();
  } else {
    PRINT_INPUT_ERROR("Fail to open ASI file!");
  }
}

void Extrapolation::process(
  const int number_of_steps,
  int step,
  const int fixed_group,
  const int move_group,
  const double global_time,
  const double temperature,
  Integrate& integrate,
  Box& box,
  std::vector<Group>& group,
  GPU_Vector<double>& thermo,
  Atom& atom,
  Force& force)
{
  if (step % check_interval == 0) {
    calculate_gamma();
    max_gamma = 0;
    for (int i = 0; i < atom.number_of_atoms; i++) {
      if (gamma[i] > max_gamma)
        max_gamma = gamma[i];
    }
    if (max_gamma > gamma_high) {
      dump();
      printf("Current step: %d, gamma = %f\n", step, max_gamma);
      PRINT_RUNTIME_ERROR(
        "The extrapolation grade exceeds the upperlimit. Terminating the simulation.");
    }
    if (max_gamma >= gamma_low) {
      if (step == 0 || step - last_dump >= dump_interval) {
        last_dump = step;
        dump();
      }
    }
  }
}

void Extrapolation::calculate_gamma()
{
  int N = atom->number_of_atoms;

  double alpha = 1.0, beta = 0.0;

#if defined(__CUDACC__) && (CUDA_VERSION < 12000)
  for (int i = 0; i < N; i++) {
    gpublasDgemv(
      handle,
      GPUBLAS_OP_N,
      B_size_per_atom,
      B_size_per_atom,
      &alpha,
      blas_A[i],
      B_size_per_atom,
      blas_x[i],
      1,
      &beta,
      blas_y[i],
      1);
  }
#else
  gpublasDgemvBatched(
    handle,
    GPUBLAS_OP_N,
    B_size_per_atom,
    B_size_per_atom,
    &alpha,
    blas_A.data(),
    B_size_per_atom,
    blas_x.data(),
    1,
    &beta,
    blas_y.data(),
    1,
    N);
#endif

  gpu_calculate_max_gamma<<<(N - 1) / 128 + 1, 128>>>(
    gamma_full.data(), gamma.data(), N, B_size_per_atom);
  gpuDeviceSynchronize();
}

void Extrapolation::dump()
{
  const int num_atoms_total = atom->position_per_atom.size() / 3;
  atom->position_per_atom.copy_to_host(atom->cpu_position_per_atom.data());

  // line 1
  fprintf(f, "%d\n", num_atoms_total);

  // line 2
  fprintf(f, "max_gamma=%.8f", max_gamma);

  // PBC
  fprintf(
    f, " pbc=\"%c %c %c\"", box->pbc_x ? 'T' : 'F', box->pbc_y ? 'T' : 'F', box->pbc_z ? 'T' : 'F');

  // box
  fprintf(
    f,
    " Lattice=\"%.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f\"",
    box->cpu_h[0],
    box->cpu_h[3],
    box->cpu_h[6],
    box->cpu_h[1],
    box->cpu_h[4],
    box->cpu_h[7],
    box->cpu_h[2],
    box->cpu_h[5],
    box->cpu_h[8]);

  fprintf(f, " Properties=species:S:1:pos:R:3");
  fprintf(f, ":gamma:R:1\n");

  // other lines
  for (int n = 0; n < num_atoms_total; n++) {
    fprintf(f, "%s", atom->cpu_atom_symbol[n].c_str());
    for (int d = 0; d < 3; ++d) {
      fprintf(f, " %.8f", atom->cpu_position_per_atom[n + num_atoms_total * d]);
    }
    fprintf(f, " %8f\n", gamma[n]);
  }
}
