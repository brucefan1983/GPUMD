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

__global__ void gpu_calculate_gamma(
  float* gamma,
  float* B,
  int* atom_type,
  double** asi,
  int number_of_particles,
  int B_size_per_atom)
{
  double max_gamma = 0;
  double current_gamma;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < number_of_particles) {
    double* current_asi = asi[atom_type[i]];
    for (int j = 0; j < B_size_per_atom; j++) {
      current_gamma = 0;
      for (int k = 0; k < B_size_per_atom; k++) {
        current_gamma += B[i * B_size_per_atom + k] * current_asi[j * B_size_per_atom + k];
      }
      current_gamma = std::abs(current_gamma);
      if (current_gamma >= max_gamma) {
        max_gamma = current_gamma;
      }
    }
    gamma[i] = max_gamma;
  }
}

void Extrapolation::parse(const char** params, int num_params)
{
  int i = 1;
  while (i < num_params) {
    if (strcmp(params[i], "asi_file") == 0) {
      load_asi(params[i + 1]);
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
  printf("gamma_low:      %f\n", gamma_low);
  printf("gamma_high:     %f\n", gamma_high);
  printf("check_interval: %d\n", check_interval);
  printf("dump_interval:  %d\n", dump_interval);
  printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
}

void Extrapolation::allocate_memory(Force& force, Atom& atom, Box& box)
{
  printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
  printf("Initializing extrapolation grade calculation...\n");
  B_size_per_atom = force.potentials[0]->B_projection_size;
  if (B_size_per_atom == 0)
    PRINT_INPUT_ERROR("This potential cannot be used to calculate the extrapolation grade!");
  else
    printf("The length of B vector for each atom: %d\n", B_size_per_atom);
  B.resize(B_size_per_atom * atom.number_of_atoms);
  gamma.resize(atom.number_of_atoms);
  gamma_cpu.resize(atom.number_of_atoms);
  force.potentials[0]->B_projection = B.data();
  force.potentials[0]->need_B_projection = true;
  this->atom = &atom;
  this->box = &box;
  activated = true;
  f = my_fopen("extrapolation_dump.xyz", "w");
}

void Extrapolation::load_asi(std::string asi_file_name)
{
  printf("Loading the Active Set Inversion file (ASI): %s\n", asi_file_name.c_str());
  std::ifstream f(asi_file_name);
  std::string token;
  int type_of_atom = 0;
  if (f.is_open()) {
    while (f >> token) {
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
      std::vector<double> asi_temp(B_size);
      for (int i = 0; i < B_size; ++i) {
        f >> asi_temp[i];
      }
      printf("[%f %f ... %f]\n", asi_temp[0], asi_temp[1], asi_temp[B_size - 1]);

      GPU_Vector<double>* a = new GPU_Vector<double>(B_size);
      a->copy_from_host(asi_temp.data());
      asi_data.push_back(a);
      asi_cpu[type_of_atom] = a->data();
      asi_gpu.copy_from_host(asi_cpu.data());
    }
    printf("ASI successfully loaded!\n");
    f.close();
  } else {
    PRINT_INPUT_ERROR("Fail to open ASI file!");
  }
}

void Extrapolation::process(int step)
{
  if (activated) {
    if (step % check_interval == 0) {
      calculate_gamma();
      max_gamma = 0;
      for (double g : gamma_cpu) {
        if (g > max_gamma)
          max_gamma = g;
      }
      if (max_gamma > gamma_high) {
        dump();
        printf("Current step: %d, gamma = %f\n", step, max_gamma);
        PRINT_RUMTIME_ERROR(
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
}

void Extrapolation::calculate_gamma()
{
  int N = atom->number_of_atoms;
  gpu_calculate_gamma<<<(N - 1) / 128 + 1, 128>>>(
    gamma.data(), B.data(), atom->type.data(), asi_gpu.data(), N, B_size_per_atom);
  gamma.copy_to_host(gamma_cpu.data());
}

void Extrapolation::dump()
{
  const int num_atoms_total = atom->position_per_atom.size() / 3;
  atom->position_per_atom.copy_to_host(atom->cpu_position_per_atom.data());

  // line 1
  fprintf(f, "%d\n", num_atoms_total);

  // line 2
  output_line2();

  // other lines
  for (int n = 0; n < num_atoms_total; n++) {
    fprintf(f, "%s", atom->cpu_atom_symbol[n].c_str());
    for (int d = 0; d < 3; ++d) {
      fprintf(f, " %.8f", atom->cpu_position_per_atom[n + num_atoms_total * d]);
    }
    fprintf(f, " %8f\n", gamma_cpu[n]);
  }
}

void Extrapolation::output_line2()
{
  fprintf(f, "max_gamma=%.8f", max_gamma);

  // PBC
  fprintf(
    f, " pbc=\"%c %c %c\"", box->pbc_x ? 'T' : 'F', box->pbc_y ? 'T' : 'F', box->pbc_z ? 'T' : 'F');

  // box
  if (box->triclinic == 0) {
    fprintf(
      f,
      " Lattice=\"%.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f\"",
      box->cpu_h[0],
      0.0,
      0.0,
      0.0,
      box->cpu_h[1],
      0.0,
      0.0,
      0.0,
      box->cpu_h[2]);
  } else {
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
  }
  fprintf(f, " Properties=species:S:1:pos:R:3");
  fprintf(f, ":gamma:R:1\n");
}

Extrapolation::~Extrapolation()
{
  printf("Closing extrapolation dump file...\n");
  fclose(f);
}