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

void Extrapolation::parse(const char** params, int num_params)
{
  int i = 1;
  while (i < num_params) {
    if (strcmp(params[i], "asi_file") == 0) {
      // load asi file
      load_asi(params[i + 1]);
      i += 1;
    }
  }
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