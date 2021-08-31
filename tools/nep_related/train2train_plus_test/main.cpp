/*
compile:
    g++ -O3 main.cpp
run:
    ./a.out < input.txt
input files:
    input.txt (with the following format)
	    cutoff               8 4            # same as in nep.in
        n_max                12 6           # same as in nep.in
        l_max                4              # same as in nep.in
        distance_threshold   0.07           # must be within [0, 0.1]
    train.in (same format as used by nep)
output files:
    train.out (same format as train.in)
    test.out (same format as train.in)
*/

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

const int NUM_OF_ABC = 24; // 3 + 5 + 7 + 9 for L_max = 4
const float YLM[NUM_OF_ABC] = {
  0.238732414637843f, 0.119366207318922f, 0.119366207318922f, 0.099471839432435f,
  0.596831036594608f, 0.596831036594608f, 0.149207759148652f, 0.149207759148652f,
  0.139260575205408f, 0.104445431404056f, 0.104445431404056f, 1.044454314040563f,
  1.044454314040563f, 0.174075719006761f, 0.174075719006761f, 0.011190581936149f,
  0.223811638722978f, 0.223811638722978f, 0.111905819361489f, 0.111905819361489f,
  1.566681471060845f, 1.566681471060845f, 0.195835183882606f, 0.195835183882606f};

// some global variables:
int Nc, n_max_radial, n_max_angular, L_max, N_des;
float rc_radial, rc_angular, rcinv_radial, rcinv_angular, distance_threshold;
struct Structure {
  int num_cell_a;
  int num_cell_b;
  int num_cell_c;
  int num_atom_original;
  int num_atom;
  int has_virial;
  float energy;
  float virial[6];
  float box_original[9];
  float box[18];
  std::vector<int> atomic_number;
  std::vector<float> atomic_number_float;
  std::vector<float> x;
  std::vector<float> y;
  std::vector<float> z;
  std::vector<float> fx;
  std::vector<float> fy;
  std::vector<float> fz;
  std::vector<float> descriptor_min;
  std::vector<float> descriptor_max;
};
std::vector<Structure> structures;

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

// read in some parameters
void read_para()
{
  printf("Started reading parameters.\n");
  char name[20];
  int count = scanf("%s%f%f", name, &rc_radial, &rc_angular);
  PRINT_SCANF_ERROR(count, 3, "reading error for cutoff.");
  printf("    radial cutoff = %g A.\n", rc_radial);
  printf("    angular cutoff = %g A.\n", rc_angular);
  if (rc_angular > rc_radial) {
    PRINT_INPUT_ERROR("angular cutoff should <= radial cutoff.");
  }
  if (rc_angular < 1.0f) {
    PRINT_INPUT_ERROR("angular cutoff should >= 1 A.");
  }
  if (rc_radial > 10.0f) {
    PRINT_INPUT_ERROR("radial cutoff should <= 10 A.");
  }
  rcinv_radial = 1.0f / rc_radial;
  rcinv_angular = 1.0f / rc_angular;

  count = scanf("%s%d%d", name, &n_max_radial, &n_max_angular);
  PRINT_SCANF_ERROR(count, 3, "reading error for n_max.");
  printf("    n_max_radial = %d.\n", n_max_radial);
  printf("    n_max_angular = %d.\n", n_max_angular);
  if (n_max_radial < 0) {
    PRINT_INPUT_ERROR("n_max_radial should >= 0.");
  } else if (n_max_radial > 19) {
    PRINT_INPUT_ERROR("n_max_radial should <= 19.");
  }
  if (n_max_angular < 0) {
    PRINT_INPUT_ERROR("n_max_angular should >= 0.");
  } else if (n_max_angular > 19) {
    PRINT_INPUT_ERROR("n_max_angular should <= 19.");
  }

  count = scanf("%s%d", name, &L_max);
  PRINT_SCANF_ERROR(count, 2, "reading error for l_max.");
  printf("    l_max = %d.\n", L_max);
  if (L_max != 4) {
    PRINT_INPUT_ERROR("l_max should = 4.");
  }

  N_des = (n_max_radial + 1) + (n_max_angular + 1) * L_max;
  printf("    dim = %d.\n", N_des);

  count = scanf("%s%f", name, &distance_threshold);
  PRINT_SCANF_ERROR(count, 2, "reading error for distance_threshold.");
  printf("    distance_threshold = %g.\n", distance_threshold);
  if (distance_threshold > 0.1f || distance_threshold < 0.0f) {
    PRINT_INPUT_ERROR("distance_threshold should be within [0, 0.1].");
  }

  printf("Finished reading parameters.\n");
}

// read the first line of train.in
void read_Nc(FILE* fid)
{
  int count = fscanf(fid, "%d", &Nc);
  PRINT_SCANF_ERROR(count, 1, "reading error for number of configurations in train.in.");
  if (Nc > 100000) {
    PRINT_INPUT_ERROR("Number of configurations should <= 100000");
  }
  printf("    Number of configurations = %d.\n", Nc);
}

// read the next Nc lines of train.in
void read_Na(FILE* fid)
{
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
}

// read in energy and virial for one configuration
void read_energy_virial(FILE* fid, int nc)
{
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

void read_box(FILE* fid, int nc)
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
  structures[nc].num_cell_a = int(ceil(2.0f * rc_radial / (volume / get_area(b, c))));
  structures[nc].num_cell_b = int(ceil(2.0f * rc_radial / (volume / get_area(c, a))));
  structures[nc].num_cell_c = int(ceil(2.0f * rc_radial / (volume / get_area(a, b))));

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

void read_force(FILE* fid, int nc, float& atomic_number_max)
{
  structures[nc].num_atom_original = structures[nc].num_atom;
  structures[nc].num_atom *=
    structures[nc].num_cell_a * structures[nc].num_cell_b * structures[nc].num_cell_c;
  if (structures[nc].num_atom > 1024) {
    PRINT_INPUT_ERROR("Number of atoms for one configuration after replication should <=1024; "
                      "consider using smaller cutoff.");
  }

  structures[nc].atomic_number.resize(structures[nc].num_atom_original);
  structures[nc].atomic_number_float.resize(structures[nc].num_atom);
  structures[nc].x.resize(structures[nc].num_atom);
  structures[nc].y.resize(structures[nc].num_atom);
  structures[nc].z.resize(structures[nc].num_atom);
  structures[nc].fx.resize(structures[nc].num_atom_original);
  structures[nc].fy.resize(structures[nc].num_atom_original);
  structures[nc].fz.resize(structures[nc].num_atom_original);

  for (int na = 0; na < structures[nc].num_atom_original; ++na) {
    int count = fscanf(
      fid, "%d%f%f%f%f%f%f", &structures[nc].atomic_number[na], &structures[nc].x[na],
      &structures[nc].y[na], &structures[nc].z[na], &structures[nc].fx[na], &structures[nc].fy[na],
      &structures[nc].fz[na]);
    PRINT_SCANF_ERROR(count, 7, "reading error for force in train.in.");
    if (structures[nc].atomic_number[na] < 1) {
      PRINT_INPUT_ERROR("Atomic number should > 0.\n");
    }
    structures[nc].atomic_number_float[na] = structures[nc].atomic_number[na];
    if (structures[nc].atomic_number[na] > atomic_number_max) {
      atomic_number_max = structures[nc].atomic_number[na];
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
            structures[nc].atomic_number_float[na_new] = structures[nc].atomic_number_float[na];
            structures[nc].x[na_new] = structures[nc].x[na] + delta_x;
            structures[nc].y[na_new] = structures[nc].y[na] + delta_y;
            structures[nc].z[na_new] = structures[nc].z[na] + delta_z;
          }
        }
      }
    }
  }
}

void normalize_atomic_number(float atomic_number_max)
{
  for (int nc = 0; nc < Nc; ++nc) {
    for (int na = 0; na < structures[nc].num_atom; ++na) {
      structures[nc].atomic_number_float[na] =
        sqrt(structures[nc].atomic_number_float[na] / atomic_number_max);
    }
  }
}

void apply_mic(const float* box, float& x12, float& y12, float& z12)
{
  float sx12 = box[9] * x12 + box[10] * y12 + box[11] * z12;
  float sy12 = box[12] * x12 + box[13] * y12 + box[14] * z12;
  float sz12 = box[15] * x12 + box[16] * y12 + box[17] * z12;
  sx12 -= nearbyint(sx12);
  sy12 -= nearbyint(sy12);
  sz12 -= nearbyint(sz12);
  x12 = box[0] * sx12 + box[1] * sy12 + box[2] * sz12;
  y12 = box[3] * sx12 + box[4] * sy12 + box[5] * sz12;
  z12 = box[6] * sx12 + box[7] * sy12 + box[8] * sz12;
}

void find_fc(float rc, float rcinv, float d12, float& fc)
{
  if (d12 < rc) {
    float x = d12 * rcinv;
    fc = 0.5f * cos(3.1415927f * x) + 0.5f;
  } else {
    fc = 0.0f;
  }
}

void find_fn(const int n_max, const float rcinv, const float d12, const float fc12, float* fn)
{
  float x = 2.0f * (d12 * rcinv - 1.0f) * (d12 * rcinv - 1.0f) - 1.0f;
  fn[0] = 1.0f;
  fn[1] = x;
  for (int m = 2; m <= n_max; ++m) {
    fn[m] = 2.0f * x * fn[m - 1] - fn[m - 2];
  }
  for (int m = 0; m <= n_max; ++m) {
    fn[m] = (fn[m] + 1.0f) * 0.5f * fc12;
  }
}

void find_fn(const int n, const float rcinv, const float d12, const float fc12, float& fn)
{
  if (n == 0) {
    fn = fc12;
  } else if (n == 1) {
    float x = 2.0f * (d12 * rcinv - 1.0f) * (d12 * rcinv - 1.0f) - 1.0f;
    fn = (x + 1.0f) * 0.5f * fc12;
  } else {
    float x = 2.0f * (d12 * rcinv - 1.0f) * (d12 * rcinv - 1.0f) - 1.0f;
    float t0 = 1.0f;
    float t1 = x;
    float t2;
    for (int m = 2; m <= n; ++m) {
      t2 = 2.0f * x * t1 - t0;
      t0 = t1;
      t1 = t2;
    }
    fn = (t2 + 1.0f) * 0.5f * fc12;
  }
}

void find_descriptors_radial(int nc)
{
  structures[nc].descriptor_min.resize(N_des, +1.0e5f);
  structures[nc].descriptor_max.resize(N_des, -1.0e5f);
  for (int n1 = 0; n1 < structures[nc].num_atom; ++n1) {
    std::vector<float> q(N_des, 0.0f);
    for (int n2 = 0; n2 < structures[nc].num_atom; ++n2) {
      if (n1 == n2) {
        continue;
      }
      float x12 = structures[nc].x[n2] - structures[nc].x[n1];
      float y12 = structures[nc].y[n2] - structures[nc].y[n1];
      float z12 = structures[nc].z[n2] - structures[nc].z[n1];
      apply_mic(structures[nc].box, x12, y12, z12);
      float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      if (d12 > rc_radial) {
        continue;
      }
      float fc12;
      find_fc(rc_radial, rcinv_radial, d12, fc12);
      fc12 *= structures[nc].atomic_number_float[n1] * structures[nc].atomic_number_float[n2];
      float fn12[20];
      find_fn(n_max_radial, rcinv_radial, d12, fc12, fn12);
      for (int n = 0; n <= n_max_radial; ++n) {
        q[n] += fn12[n];
      }
    }
    for (int n = 0; n <= n_max_radial; ++n) {
      if (q[n] < structures[nc].descriptor_min[n]) {
        structures[nc].descriptor_min[n] = q[n];
      }
      if (q[n] > structures[nc].descriptor_max[n]) {
        structures[nc].descriptor_max[n] = q[n];
      }
    }
  }
}

void accumulate_s(const float d12, float x12, float y12, float z12, const float fn, float* s)
{
  float d12inv = 1.0f / d12;
  x12 *= d12inv;
  y12 *= d12inv;
  z12 *= d12inv;
  float x12sq = x12 * x12;
  float y12sq = y12 * y12;
  float z12sq = z12 * z12;
  float x12sq_minus_y12sq = x12sq - y12sq;
  s[0] += z12 * fn;                                                             // Y10
  s[1] += x12 * fn;                                                             // Y11_real
  s[2] += y12 * fn;                                                             // Y11_imag
  s[3] += (3.0f * z12sq - 1.0f) * fn;                                           // Y20
  s[4] += x12 * z12 * fn;                                                       // Y21_real
  s[5] += y12 * z12 * fn;                                                       // Y21_imag
  s[6] += x12sq_minus_y12sq * fn;                                               // Y22_real
  s[7] += 2.0f * x12 * y12 * fn;                                                // Y22_imag
  s[8] += (5.0f * z12sq - 3.0f) * z12 * fn;                                     // Y30
  s[9] += (5.0f * z12sq - 1.0f) * x12 * fn;                                     // Y31_real
  s[10] += (5.0f * z12sq - 1.0f) * y12 * fn;                                    // Y31_imag
  s[11] += x12sq_minus_y12sq * z12 * fn;                                        // Y32_real
  s[12] += 2.0f * x12 * y12 * z12 * fn;                                         // Y32_imag
  s[13] += (x12 * x12 - 3.0f * y12 * y12) * x12 * fn;                           // Y33_real
  s[14] += (3.0f * x12 * x12 - y12 * y12) * y12 * fn;                           // Y33_imag
  s[15] += ((35.0f * z12sq - 30.0f) * z12sq + 3.0f) * fn;                       // Y40
  s[16] += (7.0f * z12sq - 3.0f) * x12 * z12 * fn;                              // Y41_real
  s[17] += (7.0f * z12sq - 3.0f) * y12 * z12 * fn;                              // Y41_iamg
  s[18] += (7.0f * z12sq - 1.0f) * x12sq_minus_y12sq * fn;                      // Y42_real
  s[19] += (7.0f * z12sq - 1.0f) * x12 * y12 * 2.0f * fn;                       // Y42_imag
  s[20] += (x12sq - 3.0f * y12sq) * x12 * z12 * fn;                             // Y43_real
  s[21] += (3.0f * x12sq - y12sq) * y12 * z12 * fn;                             // Y43_imag
  s[22] += (x12sq_minus_y12sq * x12sq_minus_y12sq - 4.0f * x12sq * y12sq) * fn; // Y44_real
  s[23] += (4.0f * x12 * y12 * x12sq_minus_y12sq) * fn;                         // Y44_imag
}

void find_q(const int n_max_angular_plus_1, const int n, const float* s, float* q)
{
  q[n] = YLM[0] * s[0] * s[0] + 2.0f * (YLM[1] * s[1] * s[1] + YLM[2] * s[2] * s[2]);
  q[n_max_angular_plus_1 + n] =
    YLM[3] * s[3] * s[3] + 2.0f * (YLM[4] * s[4] * s[4] + YLM[5] * s[5] * s[5] +
                                   YLM[6] * s[6] * s[6] + YLM[7] * s[7] * s[7]);
  q[2 * n_max_angular_plus_1 + n] =
    YLM[8] * s[8] * s[8] +
    2.0f * (YLM[9] * s[9] * s[9] + YLM[10] * s[10] * s[10] + YLM[11] * s[11] * s[11] +
            YLM[12] * s[12] * s[12] + YLM[13] * s[13] * s[13] + YLM[14] * s[14] * s[14]);
  q[3 * n_max_angular_plus_1 + n] =
    YLM[15] * s[15] * s[15] +
    2.0f * (YLM[16] * s[16] * s[16] + YLM[17] * s[17] * s[17] + YLM[18] * s[18] * s[18] +
            YLM[19] * s[19] * s[19] + YLM[20] * s[20] * s[20] + YLM[21] * s[21] * s[21] +
            YLM[22] * s[22] * s[22] + YLM[23] * s[23] * s[23]);
}

void find_descriptors_angular(int nc)
{
  for (int n1 = 0; n1 < structures[nc].num_atom; ++n1) {
    std::vector<float> q(N_des, 0.0f);

    for (int n = 0; n <= n_max_angular; ++n) {
      float s[NUM_OF_ABC] = {0.0f};
      for (int n2 = 0; n2 < structures[nc].num_atom; ++n2) {
        if (n1 == n2) {
          continue;
        }
        float x12 = structures[nc].x[n2] - structures[nc].x[n1];
        float y12 = structures[nc].y[n2] - structures[nc].y[n1];
        float z12 = structures[nc].z[n2] - structures[nc].z[n1];
        apply_mic(structures[nc].box, x12, y12, z12);
        float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
        if (d12 > rc_angular) {
          continue;
        }
        float fc12;
        find_fc(rc_angular, rcinv_angular, d12, fc12);
        fc12 *= structures[nc].atomic_number_float[n1] * structures[nc].atomic_number_float[n2];
        float fn;
        find_fn(n, rcinv_angular, d12, fc12, fn);
        accumulate_s(d12, x12, y12, z12, fn, s);
      }
      find_q(n_max_angular + 1, n, s, q.data());
    }
    for (int l = 0; l < L_max; ++l) {
      for (int n = 0; n <= n_max_angular; ++n) {
        int ln = l * (n_max_angular + 1) + n;
        int index = (n_max_radial + 1) + ln;
        if (q[ln] < structures[nc].descriptor_min[index]) {
          structures[nc].descriptor_min[index] = q[ln];
        }
        if (q[ln] > structures[nc].descriptor_max[index]) {
          structures[nc].descriptor_max[index] = q[ln];
        }
      }
    }
  }
}

void normalize_descriptor()
{
  std::vector<float> descriptor_min(N_des, +1.0e5);
  std::vector<float> descriptor_max(N_des, -1.0e5);
  for (int nc = 0; nc < Nc; ++nc) {
    for (int d = 0; d < N_des; ++d) {
      if (descriptor_min[d] > structures[nc].descriptor_min[d]) {
        descriptor_min[d] = structures[nc].descriptor_min[d];
      }
      if (descriptor_max[d] < structures[nc].descriptor_max[d]) {
        descriptor_max[d] = structures[nc].descriptor_max[d];
      }
    }
  }
  for (int nc = 0; nc < Nc; ++nc) {
    for (int d = 0; d < N_des; ++d) {
      float scaler = 1.0f / (descriptor_max[d] - descriptor_min[d]);
      structures[nc].descriptor_min[d] =
        (structures[nc].descriptor_min[d] - descriptor_min[d]) * scaler;
      structures[nc].descriptor_max[d] =
        (structures[nc].descriptor_max[d] - descriptor_min[d]) * scaler;
    }
  }
}

void select_structures(
  std::vector<int>& structure_id_in_train, std::vector<int>& structure_id_in_test)
{
  std::vector<int> selected_structure;
  structure_id_in_train.push_back(0); // start from structure 0
  for (int nc = 1; nc < Nc; ++nc) {
    bool too_close = false;
    for (int s = 0; s < structure_id_in_train.size(); ++s) {
      float distance_square = 0.0f;
      for (int d = 0; d < N_des; ++d) {
        float diff_min =
          structures[structure_id_in_train[s]].descriptor_min[d] - structures[nc].descriptor_min[d];
        float diff_max =
          structures[structure_id_in_train[s]].descriptor_max[d] - structures[nc].descriptor_max[d];
        distance_square += diff_min * diff_min + diff_max * diff_max;
      }
      distance_square /= N_des;
      if (distance_square < distance_threshold * distance_threshold) {
        structure_id_in_test.push_back(nc); // add one structure to test set
        too_close = true;
        break;
      }
    }
    if (!too_close) {
      structure_id_in_train.push_back(nc); // add one structure to train set
    }
  }
}

void write_structure(FILE* fid, std::vector<int> structure_id)
{
  fprintf(fid, "%zd\n", structure_id.size());
  for (int s = 0; s < structure_id.size(); ++s) {
    int nc = structure_id[s];
    fprintf(fid, "%d %d\n", structures[nc].num_atom_original, structures[nc].has_virial);
  }
  for (int s = 0; s < structure_id.size(); ++s) {
    int nc = structure_id[s];
    if (structures[nc].has_virial) {
      fprintf(
        fid, "%g %g %g %g %g %g %g\n", structures[nc].energy, structures[nc].virial[0],
        structures[nc].virial[1], structures[nc].virial[2], structures[nc].virial[3],
        structures[nc].virial[4], structures[nc].virial[5]);
    } else {
      fprintf(fid, "%g\n", structures[nc].energy);
    }
    fprintf(
      fid, "%g %g %g %g %g %g %g %g %g\n", structures[nc].box_original[0],
      structures[nc].box_original[1], structures[nc].box_original[2],
      structures[nc].box_original[3], structures[nc].box_original[4],
      structures[nc].box_original[5], structures[nc].box_original[6],
      structures[nc].box_original[7], structures[nc].box_original[8]);
    for (int n = 0; n < structures[nc].num_atom_original; ++n) {
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
  read_para();
  printf("Started reading train.in.\n");
  FILE* fid_in = my_fopen("train.in", "r");
  read_Nc(fid_in);
  structures.resize(Nc);
  read_Na(fid_in);
  float atomic_number_max = 0.0f;
  for (int nc = 0; nc < Nc; ++nc) {
    read_energy_virial(fid_in, nc);
    read_box(fid_in, nc);
    read_force(fid_in, nc, atomic_number_max);
  }
  fclose(fid_in);
  printf("Finished reading train.in.\n");

  // normalize atomic number
  normalize_atomic_number(atomic_number_max);

  // calculate descriptor
  printf("Started calculating the descriptors.\n");
  for (int nc = 0; nc < Nc; ++nc) {
    find_descriptors_radial(nc);
    find_descriptors_angular(nc);
  }
  printf("Finished calculating the descriptors.\n");

  // normalize descriptor
  normalize_descriptor();

  // select structures
  printf("Started selecting the structures.\n");
  std::vector<int> structure_id_in_train;
  std::vector<int> structure_id_in_test;
  select_structures(structure_id_in_train, structure_id_in_test);
  printf("    there are %zd structures in train set.\n", structure_id_in_train.size());
  printf("    there are %zd structures in test set.\n", structure_id_in_test.size());
  printf("Finished selecting the structures.\n");

  // write out
  printf("Started writing train.out.\n");
  FILE* fid_train_out = my_fopen("train.out", "w");
  write_structure(fid_train_out, structure_id_in_train);
  fclose(fid_train_out);
  FILE* fid_test_out = my_fopen("test.out", "w");
  write_structure(fid_test_out, structure_id_in_test);
  fclose(fid_test_out);
  printf("Finished writing train.out.\n");

  // Done
  printf("Done.\n");
  return EXIT_SUCCESS;
}
