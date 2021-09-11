/*
compile:
    g++ -O3 split_train.cpp
run:
    ./a.out num_batches
*/

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// some global variables:
int num_batches, Nc;
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
  if (Nc > 100000) {
    PRINT_INPUT_ERROR("Number of configurations should <= 100000");
  }
  printf("Total number of structures = %d.\n", Nc);
  structures.resize(Nc);

  // number of atoms
  for (int nc = 0; nc < Nc; ++nc) {
    int count = fscanf(fid, "%d%d", &structures[nc].num_atom, &structures[nc].has_virial);
    PRINT_SCANF_ERROR(count, 2, "reading error for number of atoms and virial flag in train.in.");
    if (structures[nc].num_atom < 1) {
      PRINT_INPUT_ERROR("Number of atoms for one configuration should >= 1.");
    }
    if (structures[nc].num_atom > 1024) {
      PRINT_INPUT_ERROR("Number of atoms for one configuration should <=1024.");
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
      if (structures[nc].atomic_number[na] == 52) {
        structures[nc].atomic_number[na] = 0;
      }
      if (structures[nc].atomic_number[na] == 82) {
        structures[nc].atomic_number[na] = 1;
      }
    }
  }
}

// a simple sorting function
static void insertion_sort(float array[], int index[], int n)
{
  for (int i = 1; i < n; i++) {
    float key = array[i];
    int j = i - 1;
    while (j >= 0 && array[j] > key) {
      array[j + 1] = array[j];
      index[j + 1] = index[j];
      --j;
    }
    array[j + 1] = key;
    index[j + 1] = i;
  }
}

// sort the sturctures according to the per-atom energy
void sort()
{
  energy_array.resize(Nc);
  index_array.resize(Nc);
  for (int nc = 0; nc < Nc; ++nc) {
    energy_array[nc] = structures[nc].energy / structures[nc].num_atom;
    index_array[nc] = nc;
  }
  insertion_sort(energy_array.data(), index_array.data(), Nc);
}

// write the sturctures for a batch
void write(FILE* fid, int batch)
{
  int Nc_batch = 0;
  for (int nc_old = batch; nc_old < structures.size(); nc_old += num_batches) {
    Nc_batch++;
  }
  printf("Selected number of structures = %d.\n", Nc_batch);

  fprintf(fid, "%d\n", Nc_batch);

  for (int nc_old = batch; nc_old < structures.size(); nc_old += num_batches) {
    int nc = index_array[nc_old];
    fprintf(fid, "%d %d\n", structures[nc].num_atom, structures[nc].has_virial);
  }

  for (int nc_old = batch; nc_old < structures.size(); nc_old += num_batches) {
    int nc = index_array[nc_old];
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
  // read in
  if (argc != 2) {
    printf("usage: %s batch_size\n", argv[0]);
    exit(1);
  }
  num_batches = atoi(argv[1]);

  FILE* fid_in = my_fopen("train.in", "r");
  read(fid_in);
  fclose(fid_in);

  // sort
  sort();

  // output
  char outputfile[100];
  FILE* fid_train_out = my_fopen("train.out", "w");
  write(fid_train_out, 0);
  fclose(fid_train_out);

  // Done
  printf("Done.\n");
  return EXIT_SUCCESS;
}
