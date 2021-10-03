/*
compile:
    g++ -O3 split_train.cpp
run:
    ./a.out test_size
*/

#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// some global variables:
int Nc, Nc_train, Nc_test;
struct Structure {
  int num_atom;
  int has_virial;
  float energy;
  float virial[6];
  float box[9];
  std::vector<int> atomic_number;
  std::vector<float> x;
  std::vector<float> y;
  std::vector<float> z;
  std::vector<float> fx;
  std::vector<float> fy;
  std::vector<float> fz;
};
std::vector<Structure> structures;
std::vector<float> energy_array;
std::vector<int> index_array;

// check the return value of scanf
#define PRINT_SCANF_ERROR(count, n, text)                                                          \
  do {                                                                                             \
    if (count != n) {                                                                              \
      fprintf(stderr, "Input Error:\n");                                                           \
      fprintf(stderr, "    File:       %s\n", __FILE__);                                           \
      fprintf(stderr, "    Line:       %d\n", __LINE__);                                           \
      fprintf(stderr, "    Error text: %s\n", text);                                               \
      exit(1);                                                                                     \
    }                                                                                              \
  } while (0)

// report an input error
#define PRINT_INPUT_ERROR(text)                                                                    \
  do {                                                                                             \
    fprintf(stderr, "Input Error:\n");                                                             \
    fprintf(stderr, "    File:       %s\n", __FILE__);                                             \
    fprintf(stderr, "    Line:       %d\n", __LINE__);                                             \
    fprintf(stderr, "    Error text: %s\n", text);                                                 \
    exit(1);                                                                                       \
  } while (0)

// open a file safely
FILE* my_fopen(const char* filename, const char* mode)
{
  FILE* fid = fopen(filename, mode);
  if (fid == NULL) {
    printf("Failed to open %s!\n", filename);
    printf("%s\n", strerror(errno));
    exit(EXIT_FAILURE);
  }
  return fid;
}

// read train.in
void read(FILE* fid)
{
  // number of configurations
  int count = fscanf(fid, "%d", &Nc);
  PRINT_SCANF_ERROR(count, 1, "reading error for number of configurations in train.in.");
  printf("Total number of structures = %d.\n", Nc);
  structures.resize(Nc);

  // number of atoms
  for (int nc = 0; nc < Nc; ++nc) {
    int count = fscanf(fid, "%d%d", &structures[nc].num_atom, &structures[nc].has_virial);
    PRINT_SCANF_ERROR(count, 2, "reading error for number of atoms and virial flag in train.in.");
    if (structures[nc].num_atom < 1) {
      PRINT_INPUT_ERROR("Number of atoms for one configuration should >= 1.");
    }
  }

  // per-configuration:
  for (int nc = 0; nc < Nc; ++nc) {
    // energy and virial
    if (structures[nc].has_virial) {
      int count = fscanf(
        fid, "%f%f%f%f%f%f%f", &structures[nc].energy, &structures[nc].virial[0],
        &structures[nc].virial[1], &structures[nc].virial[2], &structures[nc].virial[3],
        &structures[nc].virial[4], &structures[nc].virial[5]);
      PRINT_SCANF_ERROR(count, 7, "reading error for energy and virial in train.in.");
    } else {
      int count = fscanf(fid, "%f", &structures[nc].energy);
      PRINT_SCANF_ERROR(count, 1, "reading error for energy in train.in.");
    }

    // box
    int count = fscanf(
      fid, "%f%f%f%f%f%f%f%f%f", &structures[nc].box[0], &structures[nc].box[1],
      &structures[nc].box[2], &structures[nc].box[3], &structures[nc].box[4],
      &structures[nc].box[5], &structures[nc].box[6], &structures[nc].box[7],
      &structures[nc].box[8]);
    PRINT_SCANF_ERROR(count, 9, "reading error for box in train.in.");

    // type, position, force
    structures[nc].atomic_number.resize(structures[nc].num_atom);
    structures[nc].x.resize(structures[nc].num_atom);
    structures[nc].y.resize(structures[nc].num_atom);
    structures[nc].z.resize(structures[nc].num_atom);
    structures[nc].fx.resize(structures[nc].num_atom);
    structures[nc].fy.resize(structures[nc].num_atom);
    structures[nc].fz.resize(structures[nc].num_atom);
    for (int na = 0; na < structures[nc].num_atom; ++na) {
      int count = fscanf(
        fid, "%d%f%f%f%f%f%f", &structures[nc].atomic_number[na], &structures[nc].x[na],
        &structures[nc].y[na], &structures[nc].z[na], &structures[nc].fx[na],
        &structures[nc].fy[na], &structures[nc].fz[na]);
      PRINT_SCANF_ERROR(count, 7, "reading error for force in train.in.");
      if (structures[nc].atomic_number[na] < 0) {
        PRINT_INPUT_ERROR("Atomic number should >= 0.\n");
      }
    }
  }
}

void find_permuted_indices(std::vector<int>& permuted_indices)
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

void shuffle()
{
  std::vector<int> configuration_id(structures.size());
  find_permuted_indices(configuration_id);

  std::vector<Structure> structures_copy(structures.size());

  for (int nc = 0; nc < structures.size(); ++nc) {
    structures_copy[nc].num_atom = structures[nc].num_atom;
    structures_copy[nc].has_virial = structures[nc].has_virial;
    structures_copy[nc].energy = structures[nc].energy;
    for (int k = 0; k < 6; ++k) {
      structures_copy[nc].virial[k] = structures[nc].virial[k];
    }
    for (int k = 0; k < 9; ++k) {
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
    structures[nc].has_virial = structures_copy[configuration_id[nc]].has_virial;
    structures[nc].energy = structures_copy[configuration_id[nc]].energy;
    for (int k = 0; k < 6; ++k) {
      structures[nc].virial[k] = structures_copy[configuration_id[nc]].virial[k];
    }
    for (int k = 0; k < 9; ++k) {
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

// write sturctures to train.out or test.out
void write(FILE* fid, int nc1, int nc2)
{
  fprintf(fid, "%d\n", nc2 - nc1);

  for (int nc = nc1; nc < nc2; ++nc) {
    fprintf(fid, "%d %d\n", structures[nc].num_atom, structures[nc].has_virial);
  }

  for (int nc = nc1; nc < nc2; ++nc) {
    if (structures[nc].has_virial) {
      fprintf(
        fid, "%g %g %g %g %g %g %g\n", structures[nc].energy, structures[nc].virial[0],
        structures[nc].virial[1], structures[nc].virial[2], structures[nc].virial[3],
        structures[nc].virial[4], structures[nc].virial[5]);
    } else {
      fprintf(fid, "%g\n", structures[nc].energy);
    }
    fprintf(
      fid, "%g %g %g %g %g %g %g %g %g\n", structures[nc].box[0], structures[nc].box[1],
      structures[nc].box[2], structures[nc].box[3], structures[nc].box[4], structures[nc].box[5],
      structures[nc].box[6], structures[nc].box[7], structures[nc].box[8]);
    for (int n = 0; n < structures[nc].num_atom; ++n) {
      fprintf(
        fid, "%d %g %g %g %g %g %g\n", structures[nc].atomic_number[n], structures[nc].x[n],
        structures[nc].y[n], structures[nc].z[n], structures[nc].fx[n], structures[nc].fy[n],
        structures[nc].fz[n]);
    }
  }
}

int main(int argc, char* argv[])
{
  // read Nc_test from command line
  if (argc != 2) {
    printf("usage: %s test_size\n", argv[0]);
    exit(1);
  }
  Nc_test = atoi(argv[1]);

  // read train.in
  FILE* fid_in = my_fopen("train.in", "r");
  read(fid_in);
  fclose(fid_in);

  Nc_train = Nc - Nc_test;
  printf("Number of structures to be output to train.out = %d.\n", Nc_train);
  printf("Number of structures to be output to test.out = %d.\n", Nc_test);

  // shuffle the structures
  shuffle();

  // output train.out
  FILE* fid_train_out = my_fopen("train.out", "w");
  write(fid_train_out, 0, Nc_train);
  fclose(fid_train_out);

  // output test.out
  FILE* fid_test_out = my_fopen("test.out", "w");
  write(fid_test_out, Nc_train, Nc);
  fclose(fid_test_out);

  // Done
  printf("Done.\n");
  return EXIT_SUCCESS;
}
