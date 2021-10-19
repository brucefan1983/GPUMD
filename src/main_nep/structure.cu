/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, Mikko Ervasti, and Ari Harju
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

#include "parameters.cuh"
#include "structure.cuh"
#include <chrono>
#include <random>
#include <vector>

static void read_Nc(FILE* fid, std::vector<Structure>& structures)
{
  int Nc;
  int count = fscanf(fid, "%d", &Nc);
  PRINT_SCANF_ERROR(count, 1, "reading error for number of configurations in train.in.");
  printf("Number of configurations = %d.\n", Nc);

  structures.resize(Nc);
}

static void read_Na(FILE* fid, std::vector<Structure>& structures)
{
  for (int nc = 0; nc < structures.size(); ++nc) {
    int count = fscanf(fid, "%d%d", &structures[nc].num_atom, &structures[nc].has_virial);
    PRINT_SCANF_ERROR(count, 2, "reading error for number of atoms and virial flag in train.in.");
    if (structures[nc].num_atom < 1) {
      PRINT_INPUT_ERROR("Number of atoms for one configuration should >= 1.");
    }
    structures[nc].num_atom_original = structures[nc].num_atom;
  }
}

static void read_energy_virial(FILE* fid, int nc, std::vector<Structure>& structures)
{
  if (structures[nc].has_virial) {
    int count = fscanf(
      fid, "%f%f%f%f%f%f%f", &structures[nc].energy, &structures[nc].virial[0],
      &structures[nc].virial[1], &structures[nc].virial[2], &structures[nc].virial[3],
      &structures[nc].virial[4], &structures[nc].virial[5]);
    PRINT_SCANF_ERROR(count, 7, "reading error for energy and virial in train.in.");
    for (int k = 0; k < 6; ++k) {
      structures[nc].virial[k] /= structures[nc].num_atom;
    }
  } else {
    int count = fscanf(fid, "%f", &structures[nc].energy);
    PRINT_SCANF_ERROR(count, 1, "reading error for energy in train.in.");
  }
  structures[nc].energy /= structures[nc].num_atom;
}

static float get_area(const float* a, const float* b)
{
  float s1 = a[1] * b[2] - a[2] * b[1];
  float s2 = a[2] * b[0] - a[0] * b[2];
  float s3 = a[0] * b[1] - a[1] * b[0];
  return sqrt(s1 * s1 + s2 * s2 + s3 * s3);
}

static float get_det(const float* box)
{
  return box[0] * (box[4] * box[8] - box[5] * box[7]) +
         box[1] * (box[5] * box[6] - box[3] * box[8]) +
         box[2] * (box[3] * box[7] - box[4] * box[6]);
}

static void read_box(FILE* fid, int nc, Parameters& para, std::vector<Structure>& structures)
{
  float a[3], b[3], c[3];
  int count = fscanf(
    fid, "%f%f%f%f%f%f%f%f%f", &a[0], &a[1], &a[2], &b[0], &b[1], &b[2], &c[0], &c[1], &c[2]);
  PRINT_SCANF_ERROR(count, 9, "reading error for box in train.in.");

  structures[nc].box_original[0] = a[0];
  structures[nc].box_original[3] = a[1];
  structures[nc].box_original[6] = a[2];
  structures[nc].box_original[1] = b[0];
  structures[nc].box_original[4] = b[1];
  structures[nc].box_original[7] = b[2];
  structures[nc].box_original[2] = c[0];
  structures[nc].box_original[5] = c[1];
  structures[nc].box_original[8] = c[2];

  float det = get_det(structures[nc].box_original);
  float volume = abs(det);
  structures[nc].num_cell_a = int(ceil(2.0f * para.rc_radial / (volume / get_area(b, c))));
  structures[nc].num_cell_b = int(ceil(2.0f * para.rc_radial / (volume / get_area(c, a))));
  structures[nc].num_cell_c = int(ceil(2.0f * para.rc_radial / (volume / get_area(a, b))));

  structures[nc].box[0] = structures[nc].box_original[0] * structures[nc].num_cell_a;
  structures[nc].box[3] = structures[nc].box_original[3] * structures[nc].num_cell_a;
  structures[nc].box[6] = structures[nc].box_original[6] * structures[nc].num_cell_a;
  structures[nc].box[1] = structures[nc].box_original[1] * structures[nc].num_cell_b;
  structures[nc].box[4] = structures[nc].box_original[4] * structures[nc].num_cell_b;
  structures[nc].box[7] = structures[nc].box_original[7] * structures[nc].num_cell_b;
  structures[nc].box[2] = structures[nc].box_original[2] * structures[nc].num_cell_c;
  structures[nc].box[5] = structures[nc].box_original[5] * structures[nc].num_cell_c;
  structures[nc].box[8] = structures[nc].box_original[8] * structures[nc].num_cell_c;

  structures[nc].box[9] =
    structures[nc].box[4] * structures[nc].box[8] - structures[nc].box[5] * structures[nc].box[7];
  structures[nc].box[10] =
    structures[nc].box[2] * structures[nc].box[7] - structures[nc].box[1] * structures[nc].box[8];
  structures[nc].box[11] =
    structures[nc].box[1] * structures[nc].box[5] - structures[nc].box[2] * structures[nc].box[4];
  structures[nc].box[12] =
    structures[nc].box[5] * structures[nc].box[6] - structures[nc].box[3] * structures[nc].box[8];
  structures[nc].box[13] =
    structures[nc].box[0] * structures[nc].box[8] - structures[nc].box[2] * structures[nc].box[6];
  structures[nc].box[14] =
    structures[nc].box[2] * structures[nc].box[3] - structures[nc].box[0] * structures[nc].box[5];
  structures[nc].box[15] =
    structures[nc].box[3] * structures[nc].box[7] - structures[nc].box[4] * structures[nc].box[6];
  structures[nc].box[16] =
    structures[nc].box[1] * structures[nc].box[6] - structures[nc].box[0] * structures[nc].box[7];
  structures[nc].box[17] =
    structures[nc].box[0] * structures[nc].box[4] - structures[nc].box[1] * structures[nc].box[3];

  det *= structures[nc].num_cell_a * structures[nc].num_cell_b * structures[nc].num_cell_c;
  for (int n = 9; n < 18; n++) {
    structures[nc].box[n] /= det;
  }
}

static void read_force(FILE* fid, int nc, Parameters& para, std::vector<Structure>& structures)
{
  structures[nc].num_atom *=
    structures[nc].num_cell_a * structures[nc].num_cell_b * structures[nc].num_cell_c;

  structures[nc].atomic_number.resize(structures[nc].num_atom);
  structures[nc].x.resize(structures[nc].num_atom);
  structures[nc].y.resize(structures[nc].num_atom);
  structures[nc].z.resize(structures[nc].num_atom);
  structures[nc].fx.resize(structures[nc].num_atom);
  structures[nc].fy.resize(structures[nc].num_atom);
  structures[nc].fz.resize(structures[nc].num_atom);

  for (int na = 0; na < structures[nc].num_atom_original; ++na) {
    int count = fscanf(
      fid, "%d%f%f%f%f%f%f", &structures[nc].atomic_number[na], &structures[nc].x[na],
      &structures[nc].y[na], &structures[nc].z[na], &structures[nc].fx[na], &structures[nc].fy[na],
      &structures[nc].fz[na]);
    PRINT_SCANF_ERROR(count, 7, "reading error for force in train.in.");

    if (structures[nc].atomic_number[na] < 0) {
      PRINT_INPUT_ERROR("Atom type should >= 0.\n");
    }
  }

  for (int ia = 0; ia < structures[nc].num_cell_a; ++ia) {
    for (int ib = 0; ib < structures[nc].num_cell_b; ++ib) {
      for (int ic = 0; ic < structures[nc].num_cell_c; ++ic) {
        if (ia != 0 || ib != 0 || ic != 0) {
          for (int na = 0; na < structures[nc].num_atom_original; ++na) {
            int na_new =
              na + (ia + (ib + ic * structures[nc].num_cell_b) * structures[nc].num_cell_a) *
                     structures[nc].num_atom_original;
            float delta_x = structures[nc].box_original[0] * ia +
                            structures[nc].box_original[1] * ib +
                            structures[nc].box_original[2] * ic;
            float delta_y = structures[nc].box_original[3] * ia +
                            structures[nc].box_original[4] * ib +
                            structures[nc].box_original[5] * ic;
            float delta_z = structures[nc].box_original[6] * ia +
                            structures[nc].box_original[7] * ib +
                            structures[nc].box_original[8] * ic;
            structures[nc].atomic_number[na_new] = structures[nc].atomic_number[na];
            structures[nc].x[na_new] = structures[nc].x[na] + delta_x;
            structures[nc].y[na_new] = structures[nc].y[na] + delta_y;
            structures[nc].z[na_new] = structures[nc].z[na] + delta_z;
            structures[nc].fx[na_new] = structures[nc].fx[na];
            structures[nc].fy[na_new] = structures[nc].fy[na];
            structures[nc].fz[na_new] = structures[nc].fz[na];
          }
        }
      }
    }
  }
}

static void find_permuted_indices(std::vector<int>& permuted_indices)
{
  std::mt19937 rng;
#ifdef DEBUG
  rng = std::mt19937(54321);
#else
  rng = std::mt19937(std::chrono::system_clock::now().time_since_epoch().count());
#endif
  for (int i = 0; i < permuted_indices.size(); ++i) {
    permuted_indices[i] = i;
  }
  std::uniform_int_distribution<int> rand_int(0, INT_MAX);
  for (int i = 0; i < permuted_indices.size(); ++i) {
    int j = rand_int(rng) % (permuted_indices.size() - i) + i;
    int temp = permuted_indices[i];
    permuted_indices[i] = permuted_indices[j];
    permuted_indices[j] = temp;
  }
}

static void reorder(std::vector<Structure>& structures)
{
  std::vector<int> configuration_id(structures.size());
  find_permuted_indices(configuration_id);

  std::vector<Structure> structures_copy(structures.size());

  for (int nc = 0; nc < structures.size(); ++nc) {
    structures_copy[nc].num_atom = structures[nc].num_atom;
    structures_copy[nc].num_atom_original = structures[nc].num_atom_original;
    structures_copy[nc].has_virial = structures[nc].has_virial;
    structures_copy[nc].energy = structures[nc].energy;
    for (int k = 0; k < 6; ++k) {
      structures_copy[nc].virial[k] = structures[nc].virial[k];
    }
    for (int k = 0; k < 18; ++k) {
      structures_copy[nc].box[k] = structures[nc].box[k];
    }
    structures_copy[nc].atomic_number.resize(structures[nc].num_atom);
    structures_copy[nc].x.resize(structures[nc].num_atom);
    structures_copy[nc].y.resize(structures[nc].num_atom);
    structures_copy[nc].z.resize(structures[nc].num_atom);
    structures_copy[nc].fx.resize(structures[nc].num_atom);
    structures_copy[nc].fy.resize(structures[nc].num_atom);
    structures_copy[nc].fz.resize(structures[nc].num_atom);
    for (int na = 0; na < structures[nc].num_atom; ++na) {
      structures_copy[nc].atomic_number[na] = structures[nc].atomic_number[na];
      structures_copy[nc].x[na] = structures[nc].x[na];
      structures_copy[nc].y[na] = structures[nc].y[na];
      structures_copy[nc].z[na] = structures[nc].z[na];
      structures_copy[nc].fx[na] = structures[nc].fx[na];
      structures_copy[nc].fy[na] = structures[nc].fy[na];
      structures_copy[nc].fz[na] = structures[nc].fz[na];
    }
  }

  for (int nc = 0; nc < structures.size(); ++nc) {
    structures[nc].num_atom = structures_copy[configuration_id[nc]].num_atom;
    structures[nc].num_atom_original = structures_copy[configuration_id[nc]].num_atom_original;
    structures[nc].has_virial = structures_copy[configuration_id[nc]].has_virial;
    structures[nc].energy = structures_copy[configuration_id[nc]].energy;
    for (int k = 0; k < 6; ++k) {
      structures[nc].virial[k] = structures_copy[configuration_id[nc]].virial[k];
    }
    for (int k = 0; k < 18; ++k) {
      structures[nc].box[k] = structures_copy[configuration_id[nc]].box[k];
    }
    structures[nc].atomic_number.resize(structures[nc].num_atom);
    structures[nc].x.resize(structures[nc].num_atom);
    structures[nc].y.resize(structures[nc].num_atom);
    structures[nc].z.resize(structures[nc].num_atom);
    structures[nc].fx.resize(structures[nc].num_atom);
    structures[nc].fy.resize(structures[nc].num_atom);
    structures[nc].fz.resize(structures[nc].num_atom);
    for (int na = 0; na < structures[nc].num_atom; ++na) {
      structures[nc].atomic_number[na] = structures_copy[configuration_id[nc]].atomic_number[na];
      structures[nc].x[na] = structures_copy[configuration_id[nc]].x[na];
      structures[nc].y[na] = structures_copy[configuration_id[nc]].y[na];
      structures[nc].z[na] = structures_copy[configuration_id[nc]].z[na];
      structures[nc].fx[na] = structures_copy[configuration_id[nc]].fx[na];
      structures[nc].fy[na] = structures_copy[configuration_id[nc]].fy[na];
      structures[nc].fz[na] = structures_copy[configuration_id[nc]].fz[na];
    }
  }
}

void read_structures(
  bool is_train, char* input_dir, Parameters& para, std::vector<Structure>& structures)
{
  char file_train[200];
  strcpy(file_train, input_dir);
  if (is_train) {
    strcat(file_train, "/train.in");
  } else {
    strcat(file_train, "/test.in");
  }
  FILE* fid = my_fopen(file_train, "r");

  read_Nc(fid, structures);
  read_Na(fid, structures);
  for (int n = 0; n < structures.size(); ++n) {
    read_energy_virial(fid, n, structures);
    read_box(fid, n, para, structures);
    read_force(fid, n, para, structures);
  }

  fclose(fid);

  if (is_train) {
    reorder(structures);
  }
}
