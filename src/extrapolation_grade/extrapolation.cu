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

const std::string ELEMENTS[NUM_ELEMENTS] = {
  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",  "S",
  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
  "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
  "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
  "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
  "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu"};

__global__ void gpu_calculate_gamma(
  double* gamma,
  double* B,
  int* atom_type,
  std::map<int, double*> asi,
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
      i += 1;
    } else if (strcmp(params[i], "gamma_low") == 0) {
      if (!is_valid_real(params[i], &gamma_low)) {
        PRINT_INPUT_ERROR("Wrong input for gamma_low.\n");
      }
    } else if (strcmp(params[i], "gamma_high") == 0) {
      if (!is_valid_real(params[i], &gamma_low)) {
        PRINT_INPUT_ERROR("Wrong input for gamma_high.\n");
      }
    } else if (strcmp(params[i], "check_interval") == 0) {
      if (!is_valid_int(params[i], &check_interval)) {
        PRINT_INPUT_ERROR("Wrong input for check_interval.\n");
      }
    } else if (strcmp(params[i], "dump_interval") == 0) {
      if (!is_valid_int(params[i], &dump_interval)) {
        PRINT_INPUT_ERROR("Wrong input for dump_interval.\n");
      }
    }
  }
}

void Extrapolation::allocate_memory(Force& force, Atom& atom, Box& box)
{
  B_size_per_atom = force.potentials[0]->B_projection_size;
  if (B_size_per_atom == 0)
    PRINT_INPUT_ERROR("This potential cannot be used to calculate the extrapolation grade!");
  else
    printf("The length of B vector for each atom: %d.\n", B_size_per_atom);
  B.resize(B_size_per_atom * atom.number_of_atoms);
  gamma.resize(atom.number_of_atoms);
  gamma_cpu.resize(atom.number_of_atoms);
  force.potentials[0]->B_projection = B.data();
  force.potentials[0]->need_B_projection = true;
  this->atom = &atom;
  activated = true;
  f = my_fopen("extrapolation_dump.xyz", "w");
}

void Extrapolation::load_asi(std::string asi_file_name)
{
  printf("Loading the Active Set Inversion file (ASI): %s\n", asi_file_name);
  std::ifstream f(asi_file_name);
  std::string token;
  int atomic_number = 0;
  if (f.is_open()) {
    while (f >> token) {
      std::string element = token;
      for (int m = 0; m < NUM_ELEMENTS; ++m) {
        if (element == ELEMENTS[m]) {
          atomic_number = m + 1;
          break;
        }
      }
      f >> token;
      int shape1 = std::stoi(token);
      f >> token;
      int shape2 = std::stoi(token);
      int B_size = shape1 * shape2;
      printf("    Loading the ASI of %s (%d): shape %d x %d, ", element, atomic_number, shape2);
      std::vector<double> B(B_size);
      for (int i = 0; i < B_size; ++i) {
        f >> B[i];
      }
      printf("[%f %f ... %f]\n", B[0], B[1], B[B_size - 1]);

      GPU_Vector<double>* B_gpu = new GPU_Vector<double>(B_size);
      B_gpu->copy_from_host(B.data());
      asi_data.push_back(B_gpu);
      asi[atomic_number] = B_gpu->data();
    }
    printf("ASI successfully loaded!");
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
      if (max_gamma > gamma_low) {
        if (step - last_dump >= dump_interval) {
          last_dump = step;
          // dump
        }
      }
      if (max_gamma > gamma_high) {
        printf("Current step: %d.\n", step);
        PRINT_RUMTIME_ERROR(
          "The extrapolation grade exceeds the upperlimit. Terminating the simulation.");
      }
    }
  }
}

void Extrapolation::calculate_gamma()
{
  int N = atom->number_of_atoms;
  gpu_calculate_gamma<<<(N - 1) / 128 + 1, 128>>>(
    gamma.data(), B.data(), atom->type.data(), asi, N, B_size_per_atom);
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
    fprintf(f, " %8f", gamma_cpu[n]);
  }
  fclose(f);
}

void Extrapolation::output_line2()
{
  // time
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
}