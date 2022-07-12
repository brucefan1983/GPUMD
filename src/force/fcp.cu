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

/*----------------------------------------------------------------------------80
The force constant potential (FCP)
------------------------------------------------------------------------------*/

#include "fcp.cuh"
#include "utilities/error.cuh"
#include <vector>

FCP::FCP(FILE* fid, char* input_dir, const int N, const Box& box)
{
  // get the highest order of the force constants
  int count = fscanf(fid, "%d%d", &order, &heat_order);
  PRINT_SCANF_ERROR(count, 2, "Reading error for force constant potential.");

  printf("Use the force constant potential.\n");
  printf("    up to order-%d.\n", order);
  printf("    and compute heat current up to order-%d.\n", heat_order);
  if (heat_order != 2 && heat_order != 3) {
    PRINT_INPUT_ERROR("heat current order should be 2 or 3.");
  }

  // get the path of the files related to the force constants
  count = fscanf(fid, "%s", file_path);
  PRINT_SCANF_ERROR(count, 1, "Reading error for force constant potential.");
  printf("    Use the force constant data in %s.\n", file_path);

  // allocate memeory
  fcp_data.u.resize(N * 3);
  fcp_data.r0.resize(N * 3, Memory_Type::managed);
  fcp_data.pfv.resize(N * 13);

  // read in the equilibrium positions and force constants
  read_r0(N);
  read_fc2(N, box);
  if (order >= 3)
    read_fc3(N, box);
  if (order >= 4)
    read_fc4(N);
  if (order >= 5)
    read_fc5(N);
  if (order >= 6)
    read_fc6(N);
}

FCP::~FCP(void)
{
  // nothing
}

void FCP::read_r0(const int N)
{
  char file[200];
  strcpy(file, file_path);
  strcat(file, "/r0.in");
  FILE* fid = my_fopen(file, "r");

  for (int n = 0; n < N; n++) {
    int count =
      fscanf(fid, "%f%f%f", &fcp_data.r0[n], &fcp_data.r0[n + N], &fcp_data.r0[n + N + N]);
    PRINT_SCANF_ERROR(count, 3, "Reading error for r0.in.");
  }
  fclose(fid);
  printf("    Data in r0.in have been read in.\n");
}

void FCP::read_fc2(const int N, const Box& box)
{
  // reading fcs_order2.in
  char file_fc[200];
  strcpy(file_fc, file_path);
  strcat(file_fc, "/fcs_order2.in");
  FILE* fid_fc = my_fopen(file_fc, "r");

  printf("    Reading data from %s\n", file_fc);

  int num_fcs = 0;
  int count = fscanf(fid_fc, "%d", &num_fcs);
  PRINT_SCANF_ERROR(count, 1, "Reading error for fcs_order2.in.");
  if (num_fcs <= 0) {
    PRINT_INPUT_ERROR("number of force constant matrix should > 0.");
  }

  fcp_data.phi2.resize(num_fcs * 9, Memory_Type::managed);
  for (int n = 0; n < num_fcs; ++n)
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) {
        int index = n * 9 + a * 3 + b;
        int aa, bb; // not used
        count = fscanf(fid_fc, "%d%d%f", &aa, &bb, &fcp_data.phi2[index]);
        PRINT_SCANF_ERROR(count, 3, "Reading error for fcs_order2.in.");
      }

  // reading clusters_order2.in
  char file_cluster[200];
  strcpy(file_cluster, file_path);
  strcat(file_cluster, "/clusters_order2.in");
  FILE* fid_cluster = my_fopen(file_cluster, "r");

  printf("    Reading data from %s\n", file_cluster);

  count = fscanf(fid_cluster, "%d", &number2);
  PRINT_SCANF_ERROR(count, 1, "Reading error for clusters_order2.in.");
  if (number2 <= 0) {
    PRINT_INPUT_ERROR("number of clusters should > 0.");
  }

  fcp_data.i2.resize(number2, Memory_Type::managed);
  fcp_data.j2.resize(number2, Memory_Type::managed);
  fcp_data.index2.resize(number2, Memory_Type::managed);
  fcp_data.xij2.resize(number2, Memory_Type::managed);
  fcp_data.yij2.resize(number2, Memory_Type::managed);
  fcp_data.zij2.resize(number2, Memory_Type::managed);

  for (int nc = 0; nc < number2; nc++) {
    int i, j, index;
    count = fscanf(fid_cluster, "%d%d%d", &i, &j, &index);
    PRINT_SCANF_ERROR(count, 3, "Reading error for clusters_order2.in.");

    if (i < 0 || i >= N) {
      PRINT_INPUT_ERROR("i < 0 or >= N.");
    }
    if (j < 0 || j >= N) {
      PRINT_INPUT_ERROR("j < 0 or >= N.");
    }
    if (index < 0 || index >= num_fcs) {
      PRINT_INPUT_ERROR("idx_fcs < 0 or >= num_fcs");
    }

    fcp_data.i2[nc] = i;
    fcp_data.j2[nc] = j;
    fcp_data.index2[nc] = index;

    double xij2 = fcp_data.r0[j] - fcp_data.r0[i];
    double yij2 = fcp_data.r0[j + N] - fcp_data.r0[i + N];
    double zij2 = fcp_data.r0[j + N * 2] - fcp_data.r0[i + N * 2];
    apply_mic(box, xij2, yij2, zij2);
    fcp_data.xij2[nc] = xij2 * 0.5;
    fcp_data.yij2[nc] = yij2 * 0.5;
    fcp_data.zij2[nc] = zij2 * 0.5;
  }

  fclose(fid_fc);
  fclose(fid_cluster);
}

void FCP::read_fc3(const int N, const Box& box)
{
  // reading fcs_order3.in
  char file_fc[200];
  strcpy(file_fc, file_path);
  strcat(file_fc, "/fcs_order3.in");
  FILE* fid_fc = my_fopen(file_fc, "r");

  printf("    Reading data from %s\n", file_fc);

  int num_fcs = 0;
  int count = fscanf(fid_fc, "%d", &num_fcs);
  PRINT_SCANF_ERROR(count, 1, "Reading error for fcs_order3.in.");
  if (num_fcs <= 0) {
    PRINT_INPUT_ERROR("number of force constant matrix should > 0.");
  }

  fcp_data.phi3.resize(num_fcs * 27, Memory_Type::managed);
  for (int n = 0; n < num_fcs; ++n)
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b)
        for (int c = 0; c < 3; ++c) {
          int index = n * 27 + a * 9 + b * 3 + c;
          int aa, bb, cc; // not used
          count = fscanf(fid_fc, "%d%d%d%f", &aa, &bb, &cc, &fcp_data.phi3[index]);
          PRINT_SCANF_ERROR(count, 4, "Reading error for fcs_order3.in.");
        }

  // reading clusters_order3.in
  char file_cluster[200];
  strcpy(file_cluster, file_path);
  strcat(file_cluster, "/clusters_order3.in");
  FILE* fid_cluster = my_fopen(file_cluster, "r");

  printf("    Reading data from %s\n", file_cluster);

  count = fscanf(fid_cluster, "%d", &number3);
  PRINT_SCANF_ERROR(count, 1, "Reading error for clusters_order3.in.");
  if (number3 <= 0) {
    PRINT_INPUT_ERROR("number of clusters should > 0.");
  }

  fcp_data.i3.resize(number3, Memory_Type::managed);
  fcp_data.j3.resize(number3, Memory_Type::managed);
  fcp_data.k3.resize(number3, Memory_Type::managed);
  fcp_data.index3.resize(number3, Memory_Type::managed);
  fcp_data.xij3.resize(number3, Memory_Type::managed);
  fcp_data.yij3.resize(number3, Memory_Type::managed);
  fcp_data.zij3.resize(number3, Memory_Type::managed);

  for (int nc = 0; nc < number3; nc++) {
    int i, j, k, index;
    count = fscanf(fid_cluster, "%d%d%d%d", &i, &j, &k, &index);
    PRINT_SCANF_ERROR(count, 4, "Reading error for clusters_order3.in.");

    if (i < 0 || i >= N) {
      PRINT_INPUT_ERROR("i < 0 or >= N.");
    }
    if (j < 0 || j >= N) {
      PRINT_INPUT_ERROR("j < 0 or >= N.");
    }
    if (k < 0 || k >= N) {
      PRINT_INPUT_ERROR("k < 0 or >= N.");
    }
    if (index < 0 || index >= num_fcs) {
      PRINT_INPUT_ERROR("idx_fcs < 0 or >= num_fcs");
    }

    fcp_data.i3[nc] = i;
    fcp_data.j3[nc] = j;
    fcp_data.k3[nc] = k;
    fcp_data.index3[nc] = index;

    double xij3 = fcp_data.r0[j] - fcp_data.r0[i];
    double yij3 = fcp_data.r0[j + N] - fcp_data.r0[i + N];
    double zij3 = fcp_data.r0[j + N * 2] - fcp_data.r0[i + N * 2];
    apply_mic(box, xij3, yij3, zij3);
    fcp_data.xij3[nc] = xij3 / 3.0;
    fcp_data.yij3[nc] = yij3 / 3.0;
    fcp_data.zij3[nc] = zij3 / 3.0;
  }

  fclose(fid_fc);
  fclose(fid_cluster);
}

void FCP::read_fc4(const int N)
{
  // reading fcs_order4.in
  char file_fc[200];
  strcpy(file_fc, file_path);
  strcat(file_fc, "/fcs_order4.in");
  FILE* fid_fc = my_fopen(file_fc, "r");

  printf("    Reading data from %s\n", file_fc);

  int num_fcs = 0;
  int count = fscanf(fid_fc, "%d", &num_fcs);
  PRINT_SCANF_ERROR(count, 1, "Reading error for fcs_order4.in.");
  if (num_fcs <= 0) {
    PRINT_INPUT_ERROR("number of force constant matrix should > 0.");
  }

  fcp_data.phi4.resize(num_fcs * 81, Memory_Type::managed);
  for (int n = 0; n < num_fcs; ++n)
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b)
        for (int c = 0; c < 3; ++c)
          for (int d = 0; d < 3; ++d) {
            int index = n * 81 + a * 27 + b * 9 + c * 3 + d;
            int aa, bb, cc, dd; // not used
            count = fscanf(fid_fc, "%d%d%d%d%f", &aa, &bb, &cc, &dd, &fcp_data.phi4[index]);
            PRINT_SCANF_ERROR(count, 5, "Reading error for fcs_order4.in.");
          }

  // reading clusters_order4.in
  char file_cluster[200];
  strcpy(file_cluster, file_path);
  strcat(file_cluster, "/clusters_order4.in");
  FILE* fid_cluster = my_fopen(file_cluster, "r");

  printf("    Reading data from %s\n", file_cluster);

  count = fscanf(fid_cluster, "%d", &number4);
  PRINT_SCANF_ERROR(count, 1, "Reading error for clusters_order4.in.");
  if (number4 <= 0) {
    PRINT_INPUT_ERROR("number of clusters should > 0.");
  }

  fcp_data.i4.resize(number4, Memory_Type::managed);
  fcp_data.j4.resize(number4, Memory_Type::managed);
  fcp_data.k4.resize(number4, Memory_Type::managed);
  fcp_data.l4.resize(number4, Memory_Type::managed);
  fcp_data.index4.resize(number4, Memory_Type::managed);
  fcp_data.weight4.resize(number4, 1.0f, Memory_Type::managed);

  for (int nc = 0; nc < number4; nc++) {
    int i, j, k, l, index;
    count = fscanf(fid_cluster, "%d%d%d%d%d", &i, &j, &k, &l, &index);
    PRINT_SCANF_ERROR(count, 5, "Reading error for clusters_order4.in.");

    if (i < 0 || i >= N) {
      PRINT_INPUT_ERROR("i < 0 or >= N.");
    }
    if (j < 0 || j >= N) {
      PRINT_INPUT_ERROR("j < 0 or >= N.");
    }
    if (k < 0 || k >= N) {
      PRINT_INPUT_ERROR("k < 0 or >= N.");
    }
    if (l < 0 || l >= N) {
      PRINT_INPUT_ERROR("l < 0 or >= N.");
    }
    if (i > j) {
      PRINT_INPUT_ERROR("i > j.");
    }
    if (j > k) {
      PRINT_INPUT_ERROR("j > k.");
    }
    if (k > l) {
      PRINT_INPUT_ERROR("k > l.");
    }
    if (index < 0 || index >= num_fcs) {
      PRINT_INPUT_ERROR("idx_fcs < 0 or >= num_fcs");
    }

    fcp_data.i4[nc] = i;
    fcp_data.j4[nc] = j;
    fcp_data.k4[nc] = k;
    fcp_data.l4[nc] = l;
    fcp_data.index4[nc] = index;

    // 2^3-1 = 7 cases:
    if (i == j && j != k && k != l) {
      fcp_data.weight4[nc] /= 2;
    } // 1123
    if (i != j && j == k && k != l) {
      fcp_data.weight4[nc] /= 2;
    } // 1223
    if (i != j && j != k && k == l) {
      fcp_data.weight4[nc] /= 2;
    } // 1233
    if (i == j && j != k && k == l) {
      fcp_data.weight4[nc] /= 4;
    } // 1122
    if (i == j && j == k && k != l) {
      fcp_data.weight4[nc] /= 6;
    } // 1112
    if (i != j && j == k && k == l) {
      fcp_data.weight4[nc] /= 6;
    } // 1222
    if (i == j && j == k && k == l) {
      fcp_data.weight4[nc] /= 24;
    } // 1111
  }

  fclose(fid_fc);
  fclose(fid_cluster);
}

void FCP::read_fc5(const int N)
{
  // reading fcs_order5.in
  char file_fc[200];
  strcpy(file_fc, file_path);
  strcat(file_fc, "/fcs_order5.in");
  FILE* fid_fc = my_fopen(file_fc, "r");

  printf("    Reading data from %s\n", file_fc);

  int num_fcs = 0;
  int count = fscanf(fid_fc, "%d", &num_fcs);
  PRINT_SCANF_ERROR(count, 1, "Reading error for fcs_order5.in.");
  if (num_fcs <= 0) {
    PRINT_INPUT_ERROR("number of force constant matrix should > 0.");
  }

  fcp_data.phi5.resize(num_fcs * 243, Memory_Type::managed);
  for (int n = 0; n < num_fcs; ++n)
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b)
        for (int c = 0; c < 3; ++c)
          for (int d = 0; d < 3; ++d)
            for (int e = 0; e < 3; ++e) {
              int index = n * 243 + a * 81 + b * 27 + c * 9 + d * 3 + e;
              int aa, bb, cc, dd, ee; // not used
              count =
                fscanf(fid_fc, "%d%d%d%d%d%f", &aa, &bb, &cc, &dd, &ee, &fcp_data.phi5[index]);
              PRINT_SCANF_ERROR(count, 6, "Reading error for fcs_order5.in.");
            }

  // reading clusters_order5.in
  char file_cluster[200];
  strcpy(file_cluster, file_path);
  strcat(file_cluster, "/clusters_order5.in");
  FILE* fid_cluster = my_fopen(file_cluster, "r");

  printf("    Reading data from %s\n", file_cluster);

  count = fscanf(fid_cluster, "%d", &number5);
  PRINT_SCANF_ERROR(count, 1, "Reading error for clusters_order5.in.");
  if (number5 <= 0) {
    PRINT_INPUT_ERROR("number of clusters should > 0.");
  }

  fcp_data.i5.resize(number5, Memory_Type::managed);
  fcp_data.j5.resize(number5, Memory_Type::managed);
  fcp_data.k5.resize(number5, Memory_Type::managed);
  fcp_data.l5.resize(number5, Memory_Type::managed);
  fcp_data.m5.resize(number5, Memory_Type::managed);
  fcp_data.index5.resize(number5, Memory_Type::managed);
  fcp_data.weight5.resize(number5, 1.0f, Memory_Type::managed);

  for (int nc = 0; nc < number5; nc++) {
    int i, j, k, l, m, index;
    count = fscanf(fid_cluster, "%d%d%d%d%d%d", &i, &j, &k, &l, &m, &index);
    PRINT_SCANF_ERROR(count, 6, "Reading error for clusters_order5.in.");

    if (i < 0 || i >= N) {
      PRINT_INPUT_ERROR("i < 0 or >= N.");
    }
    if (j < 0 || j >= N) {
      PRINT_INPUT_ERROR("j < 0 or >= N.");
    }
    if (k < 0 || k >= N) {
      PRINT_INPUT_ERROR("k < 0 or >= N.");
    }
    if (l < 0 || l >= N) {
      PRINT_INPUT_ERROR("l < 0 or >= N.");
    }
    if (m < 0 || m >= N) {
      PRINT_INPUT_ERROR("m < 0 or >= N.");
    }
    if (i > j) {
      PRINT_INPUT_ERROR("i > j.");
    }
    if (j > k) {
      PRINT_INPUT_ERROR("j > k.");
    }
    if (k > l) {
      PRINT_INPUT_ERROR("k > l.");
    }
    if (l > m) {
      PRINT_INPUT_ERROR("l > m.");
    }
    if (index < 0 || index >= num_fcs) {
      PRINT_INPUT_ERROR("idx_fcs < 0 or >= num_fcs");
    }

    fcp_data.i5[nc] = i;
    fcp_data.j5[nc] = j;
    fcp_data.k5[nc] = k;
    fcp_data.l5[nc] = l;
    fcp_data.m5[nc] = m;
    fcp_data.index5[nc] = index;

    // 2^4-1 = 15 cases:
    if (i == j && j != k && k != l && l != m) {
      fcp_data.weight5[nc] /= 2;
    } // 11234
    if (i != j && j == k && k != l && l != m) {
      fcp_data.weight5[nc] /= 2;
    } // 12234
    if (i != j && j != k && k == l && l != m) {
      fcp_data.weight5[nc] /= 2;
    } // 12334
    if (i != j && j != k && k != l && l == m) {
      fcp_data.weight5[nc] /= 2;
    } // 12344
    if (i == j && j == k && k != l && l != m) {
      fcp_data.weight5[nc] /= 6;
    } // 11123
    if (i != j && j == k && k == l && l != m) {
      fcp_data.weight5[nc] /= 6;
    } // 12223
    if (i != j && j != k && k == l && l == m) {
      fcp_data.weight5[nc] /= 6;
    } // 12333
    if (i == j && j == k && k == l && l != m) {
      fcp_data.weight5[nc] /= 24;
    } // 11112
    if (i != j && j == k && k == l && l == m) {
      fcp_data.weight5[nc] /= 24;
    } // 12222
    if (i == j && j == k && k == l && l == m) {
      fcp_data.weight5[nc] /= 120;
    } // 11111
    if (i == j && j != k && k == l && l != m) {
      fcp_data.weight5[nc] /= 4;
    } // 11223
    if (i == j && j != k && k != l && l == m) {
      fcp_data.weight5[nc] /= 4;
    } // 11233
    if (i != j && j == k && k != l && l == m) {
      fcp_data.weight5[nc] /= 4;
    } // 12233
    if (i == j && j != k && k == l && l == m) {
      fcp_data.weight5[nc] /= 12;
    } // 11222
    if (i == j && j == k && k != l && l == m) {
      fcp_data.weight5[nc] /= 12;
    } // 11122
  }

  fclose(fid_fc);
  fclose(fid_cluster);
}

void FCP::read_fc6(const int N)
{
  // reading fcs_order6.in
  char file_fc[200];
  strcpy(file_fc, file_path);
  strcat(file_fc, "/fcs_order6.in");
  FILE* fid_fc = my_fopen(file_fc, "r");

  printf("    Reading data from %s\n", file_fc);

  int num_fcs = 0;
  int count = fscanf(fid_fc, "%d", &num_fcs);
  PRINT_SCANF_ERROR(count, 1, "Reading error for fcs_order6.in.");
  if (num_fcs <= 0) {
    PRINT_INPUT_ERROR("number of force constant matrix should > 0.");
  }

  fcp_data.phi6.resize(num_fcs * 729, Memory_Type::managed);
  for (int n = 0; n < num_fcs; ++n)
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b)
        for (int c = 0; c < 3; ++c)
          for (int d = 0; d < 3; ++d)
            for (int e = 0; e < 3; ++e)
              for (int f = 0; f < 3; ++f) {
                int index = n * 729 + a * 243 + b * 81 + c * 27 + d * 9 + e * 3 + f;
                int aa, bb, cc, dd, ee, ff; // not used
                count = fscanf(
                  fid_fc, "%d%d%d%d%d%d%f", &aa, &bb, &cc, &dd, &ee, &ff, &fcp_data.phi6[index]);
                PRINT_SCANF_ERROR(count, 7, "Reading error for fcs_order6.in.");
              }

  // reading clusters_order6.in
  char file_cluster[200];
  strcpy(file_cluster, file_path);
  strcat(file_cluster, "/clusters_order6.in");
  FILE* fid_cluster = my_fopen(file_cluster, "r");

  printf("    Reading data from %s\n", file_cluster);

  count = fscanf(fid_cluster, "%d", &number6);
  PRINT_SCANF_ERROR(count, 1, "Reading error for clusters_order6.in.");
  if (number6 <= 0) {
    PRINT_INPUT_ERROR("number of clusters should > 0.");
  }

  fcp_data.i6.resize(number6, Memory_Type::managed);
  fcp_data.j6.resize(number6, Memory_Type::managed);
  fcp_data.k6.resize(number6, Memory_Type::managed);
  fcp_data.l6.resize(number6, Memory_Type::managed);
  fcp_data.m6.resize(number6, Memory_Type::managed);
  fcp_data.n6.resize(number6, Memory_Type::managed);
  fcp_data.index6.resize(number6, Memory_Type::managed);
  fcp_data.weight6.resize(number6, 1.0f, Memory_Type::managed);

  for (int nc = 0; nc < number6; nc++) {
    int i, j, k, l, m, n, index;
    count = fscanf(fid_cluster, "%d%d%d%d%d%d%d", &i, &j, &k, &l, &m, &n, &index);
    PRINT_SCANF_ERROR(count, 7, "Reading error for clusters_order6.in.");

    if (i < 0 || i >= N) {
      PRINT_INPUT_ERROR("i < 0 or >= N.");
    }
    if (j < 0 || j >= N) {
      PRINT_INPUT_ERROR("j < 0 or >= N.");
    }
    if (k < 0 || k >= N) {
      PRINT_INPUT_ERROR("k < 0 or >= N.");
    }
    if (l < 0 || l >= N) {
      PRINT_INPUT_ERROR("l < 0 or >= N.");
    }
    if (m < 0 || m >= N) {
      PRINT_INPUT_ERROR("m < 0 or >= N.");
    }
    if (n < 0 || n >= N) {
      PRINT_INPUT_ERROR("n < 0 or >= N.");
    }
    if (i > j) {
      PRINT_INPUT_ERROR("i > j.");
    }
    if (j > k) {
      PRINT_INPUT_ERROR("j > k.");
    }
    if (k > l) {
      PRINT_INPUT_ERROR("k > l.");
    }
    if (l > m) {
      PRINT_INPUT_ERROR("l > m.");
    }
    if (m > n) {
      PRINT_INPUT_ERROR("m > n.");
    }
    if (index < 0 || index >= num_fcs) {
      PRINT_INPUT_ERROR("idx_fcs < 0 or >= num_fcs");
    }

    fcp_data.i6[nc] = i;
    fcp_data.j6[nc] = j;
    fcp_data.k6[nc] = k;
    fcp_data.l6[nc] = l;
    fcp_data.m6[nc] = m;
    fcp_data.n6[nc] = n;
    fcp_data.index6[nc] = index;

    // 2^5-1 = 31 cases:
    if (i == j && j != k && k != l && l != m && m != n) {
      fcp_data.weight6[nc] /= 2;
    } // 112345
    if (i != j && j == k && k != l && l != m && m != n) {
      fcp_data.weight6[nc] /= 2;
    } // 122345
    if (i != j && j != k && k == l && l != m && m != n) {
      fcp_data.weight6[nc] /= 2;
    } // 123345
    if (i != j && j != k && k != l && l == m && m != n) {
      fcp_data.weight6[nc] /= 2;
    } // 123445
    if (i != j && j != k && k != l && l != m && m == n) {
      fcp_data.weight6[nc] /= 2;
    } // 123455
    if (i == j && j == k && k != l && l != m && m != n) {
      fcp_data.weight6[nc] /= 6;
    } // 111234
    if (i != j && j == k && k == l && l != m && m != n) {
      fcp_data.weight6[nc] /= 6;
    } // 122234
    if (i != j && j != k && k == l && l == m && m != n) {
      fcp_data.weight6[nc] /= 6;
    } // 123334
    if (i != j && j != k && k != l && l == m && m == n) {
      fcp_data.weight6[nc] /= 6;
    } // 123444
    if (i == j && j == k && k == l && l != m && m != n) {
      fcp_data.weight6[nc] /= 24;
    } // 111123
    if (i != j && j == k && k == l && l == m && m != n) {
      fcp_data.weight6[nc] /= 24;
    } // 122223
    if (i != j && j != k && k == l && l == m && m == n) {
      fcp_data.weight6[nc] /= 24;
    } // 122223
    if (i == j && j == k && k == l && l == m && m != n) {
      fcp_data.weight6[nc] /= 120;
    } // 111112
    if (i != j && j == k && k == l && l == m && m == n) {
      fcp_data.weight6[nc] /= 120;
    } // 122222
    if (i == j && j == k && k == l && l == m && m == n) {
      fcp_data.weight6[nc] /= 720;
    } // 111111
    if (i == j && j != k && k == l && l != m && m != n) {
      fcp_data.weight6[nc] /= 4;
    } // 112234
    if (i == j && j != k && k != l && l == m && m != n) {
      fcp_data.weight6[nc] /= 4;
    } // 112334
    if (i == j && j != k && k != l && l != m && m == n) {
      fcp_data.weight6[nc] /= 4;
    } // 112344
    if (i != j && j == k && k != l && l == m && m != n) {
      fcp_data.weight6[nc] /= 4;
    } // 122334
    if (i != j && j == k && k != l && l != m && m == n) {
      fcp_data.weight6[nc] /= 4;
    } // 122344
    if (i != j && j != k && k == l && l != m && m == n) {
      fcp_data.weight6[nc] /= 4;
    } // 123344
    if (i == j && j != k && k == l && l == m && m != n) {
      fcp_data.weight6[nc] /= 12;
    } // 112223
    if (i == j && j != k && k != l && l == m && m == n) {
      fcp_data.weight6[nc] /= 12;
    } // 112333
    if (i != j && j == k && k != l && l == m && m == n) {
      fcp_data.weight6[nc] /= 12;
    } // 122333
    if (i == j && j == k && k != l && l == m && m != n) {
      fcp_data.weight6[nc] /= 12;
    } // 111223
    if (i == j && j == k && k != l && l != m && m == n) {
      fcp_data.weight6[nc] /= 12;
    } // 111233
    if (i != j && j == k && k == l && l != m && m == n) {
      fcp_data.weight6[nc] /= 12;
    } // 122233
    if (i == j && j != k && k == l && l == m && m == n) {
      fcp_data.weight6[nc] /= 48;
    } // 112222
    if (i == j && j == k && k == l && l != m && m == n) {
      fcp_data.weight6[nc] /= 48;
    } // 111122
    if (i == j && j == k && k != l && l == m && m == n) {
      fcp_data.weight6[nc] /= 36;
    } // 111222
    if (i == j && j != k && k == l && l != m && m == n) {
      fcp_data.weight6[nc] /= 8;
    } // 112233
  }

  fclose(fid_fc);
  fclose(fid_cluster);
}

// potential, force, and virial from the second-order force constants
static __global__ void gpu_find_force_fcp2(
  const int N,
  const int number_of_clusters,
  const int* g_i,
  const int* g_j,
  const int* g_index,
  const float* g_phi,
  const float* __restrict__ g_u,
  const float* g_xij2,
  const float* g_yij2,
  const float* g_zij2,
  float* g_pfv)
{
  const int nc = blockIdx.x * blockDim.x + threadIdx.x;
  if (nc >= number_of_clusters)
    return;

  const int i = g_i[nc];
  const int j = g_j[nc];
  const int index = g_index[nc];
  const float xij2 = g_xij2[nc];
  const float yij2 = g_yij2[nc];
  const float zij2 = g_zij2[nc];

  // for virial tensor
  const int x[3] = {4, 7, 8};
  const int y[3] = {10, 5, 9};
  const int z[3] = {11, 12, 6};

  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) {
      const int ab = a * 3 + b;
      const int ia = i + a * N;
      const int jb = j + b * N;
      const float phi = g_phi[index * 9 + ab];
      const float uia = g_u[ia];
      const float ujb = g_u[jb];

      // potential
      atomicAdd(&g_pfv[i], 0.5f * phi * uia * ujb);
      // force
      atomicAdd(&g_pfv[ia + N], -phi * ujb);
      // virial tensor
      atomicAdd(&g_pfv[i + N * x[a]], xij2 * phi * ujb);
      atomicAdd(&g_pfv[i + N * y[a]], yij2 * phi * ujb);
      atomicAdd(&g_pfv[i + N * z[a]], zij2 * phi * ujb);
    }
}

// potential and force from the third-order force constants
static __global__ void gpu_find_force_fcp3(
  const int heat_order,
  const int N,
  const int number_of_clusters,
  const int* g_i,
  const int* g_j,
  const int* g_k,
  const int* g_index,
  const float* g_phi,
  const float* __restrict__ g_u,
  const float* g_xij3,
  const float* g_yij3,
  const float* g_zij3,
  float* g_pfv)
{
  const int nc = blockIdx.x * blockDim.x + threadIdx.x;
  if (nc >= number_of_clusters)
    return;

  const int i = g_i[nc];
  const int j = g_j[nc];
  const int k = g_k[nc];
  const int index = g_index[nc];
  const float xij3 = g_xij3[nc];
  const float yij3 = g_yij3[nc];
  const float zij3 = g_zij3[nc];

  const float one_over_factorial3 = 1.0f / 6.0f;

  // for virial tensor
  const int x[3] = {4, 7, 8};
  const int y[3] = {10, 5, 9};
  const int z[3] = {11, 12, 6};

  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
      for (int c = 0; c < 3; ++c) {
        const int abc = a * 9 + b * 3 + c;
        const int ia = i + a * N;
        const int jb = j + b * N;
        const int kc = k + c * N;
        const float phi = g_phi[index * 27 + abc];
        const float uia = g_u[ia];
        const float ujb = g_u[jb];
        const float ukc = g_u[kc];

        const float phi_ujb_ukc = phi * ujb * ukc;
        // potential
        atomicAdd(&g_pfv[i], one_over_factorial3 * phi_ujb_ukc * uia);
        // force
        atomicAdd(&g_pfv[ia + N], -0.5f * phi_ujb_ukc);
        // virial tensor
        if (heat_order == 3) {
          atomicAdd(&g_pfv[i + N * x[a]], xij3 * phi_ujb_ukc);
          atomicAdd(&g_pfv[i + N * y[a]], yij3 * phi_ujb_ukc);
          atomicAdd(&g_pfv[i + N * z[a]], zij3 * phi_ujb_ukc);
        }
      }
}

// potential and force from the fourth-order force constants
static __global__ void gpu_find_force_fcp4(
  const int N,
  const int number_of_clusters,
  const int* g_i,
  const int* g_j,
  const int* g_k,
  const int* g_l,
  const int* g_index,
  const float* g_weight,
  const float* g_phi,
  const float* __restrict__ g_u,
  float* g_pf)
{
  const int nc = blockIdx.x * blockDim.x + threadIdx.x;
  if (nc >= number_of_clusters)
    return;

  const int i = g_i[nc];
  const int j = g_j[nc];
  const int k = g_k[nc];
  const int l = g_l[nc];
  const int index = g_index[nc];
  const float weight = g_weight[nc];

  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
      for (int c = 0; c < 3; ++c)
        for (int d = 0; d < 3; ++d) {
          const int abcd = a * 27 + b * 9 + c * 3 + d;
          const int ia = i + a * N;
          const int jb = j + b * N;
          const int kc = k + c * N;
          const int ld = l + d * N;
          const float phi = weight * g_phi[index * 81 + abcd];
          const float uia = g_u[ia];
          const float ujb = g_u[jb];
          const float ukc = g_u[kc];
          const float uld = g_u[ld];
          atomicAdd(&g_pf[i], phi * uia * ujb * ukc * uld);
          atomicAdd(&g_pf[ia + N], -phi * ujb * ukc * uld);
          atomicAdd(&g_pf[jb + N], -phi * uia * ukc * uld);
          atomicAdd(&g_pf[kc + N], -phi * uia * ujb * uld);
          atomicAdd(&g_pf[ld + N], -phi * uia * ujb * ukc);
        }
}

// potential and force from the fifth-order force constants
static __global__ void gpu_find_force_fcp5(
  const int N,
  const int number_of_clusters,
  const int* g_i,
  const int* g_j,
  const int* g_k,
  const int* g_l,
  const int* g_m,
  const int* g_index,
  const float* g_weight,
  const float* g_phi,
  const float* __restrict__ g_u,
  float* g_pf)
{
  const int nc = blockIdx.x * blockDim.x + threadIdx.x;
  if (nc >= number_of_clusters)
    return;

  const int i = g_i[nc];
  const int j = g_j[nc];
  const int k = g_k[nc];
  const int l = g_l[nc];
  const int m = g_m[nc];
  const int index = g_index[nc];
  const float weight = g_weight[nc];

  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
      for (int c = 0; c < 3; ++c)
        for (int d = 0; d < 3; ++d)
          for (int e = 0; e < 3; ++e) {
            const int abcde = a * 81 + b * 27 + c * 9 + d * 3 + e;
            const int ia = i + a * N;
            const int jb = j + b * N;
            const int kc = k + c * N;
            const int ld = l + d * N;
            const int me = m + e * N;
            const float phi = weight * g_phi[index * 243 + abcde];
            const float uia = g_u[ia];
            const float ujb = g_u[jb];
            const float ukc = g_u[kc];
            const float uld = g_u[ld];
            const float ume = g_u[me];
            atomicAdd(&g_pf[i], phi * uia * ujb * ukc * uld * ume);
            atomicAdd(&g_pf[ia + N], -phi * ujb * ukc * uld * ume);
            atomicAdd(&g_pf[jb + N], -phi * uia * ukc * uld * ume);
            atomicAdd(&g_pf[kc + N], -phi * uia * ujb * uld * ume);
            atomicAdd(&g_pf[ld + N], -phi * uia * ujb * ukc * ume);
            atomicAdd(&g_pf[me + N], -phi * uia * ujb * ukc * uld);
          }
}

// potential and force from the sixth-order force constants
static __global__ void gpu_find_force_fcp6(
  const int N,
  const int number_of_clusters,
  const int* g_i,
  const int* g_j,
  const int* g_k,
  const int* g_l,
  const int* g_m,
  const int* g_n,
  const int* g_index,
  const float* g_weight,
  const float* g_phi,
  const float* __restrict__ g_u,
  float* g_pf)
{
  const int nc = blockIdx.x * blockDim.x + threadIdx.x;
  if (nc >= number_of_clusters)
    return;

  const int i = g_i[nc];
  const int j = g_j[nc];
  const int k = g_k[nc];
  const int l = g_l[nc];
  const int m = g_m[nc];
  const int n = g_n[nc];
  const int index = g_index[nc];
  const float weight = g_weight[nc];

  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
      for (int c = 0; c < 3; ++c)
        for (int d = 0; d < 3; ++d)
          for (int e = 0; e < 3; ++e)
            for (int f = 0; f < 3; ++f) {
              const int abcdef = a * 243 + b * 81 + c * 27 + d * 9 + e * 3 + f;
              const int ia = i + a * N;
              const int jb = j + b * N;
              const int kc = k + c * N;
              const int ld = l + d * N;
              const int me = m + e * N;
              const int nf = n + f * N;
              const float phi = weight * g_phi[index * 729 + abcdef];
              const float uia = g_u[ia];
              const float ujb = g_u[jb];
              const float ukc = g_u[kc];
              const float uld = g_u[ld];
              const float ume = g_u[me];
              const float unf = g_u[nf];
              atomicAdd(&g_pf[i], phi * uia * ujb * ukc * uld * ume * unf);
              atomicAdd(&g_pf[ia + N], -phi * ujb * ukc * uld * ume * unf);
              atomicAdd(&g_pf[jb + N], -phi * uia * ukc * uld * ume * unf);
              atomicAdd(&g_pf[kc + N], -phi * uia * ujb * uld * ume * unf);
              atomicAdd(&g_pf[ld + N], -phi * uia * ujb * ukc * ume * unf);
              atomicAdd(&g_pf[me + N], -phi * uia * ujb * ukc * uld * unf);
              atomicAdd(&g_pf[nf + N], -phi * uia * ujb * ukc * uld * ume);
            }
}

// get the displacement (u=r-r0)
static __global__ void
gpu_get_u(const int N, const double* x, const double* y, const double* z, const float* r0, float* u)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    u[n] = x[n] - r0[n];
    u[n + N] = y[n] - r0[n + N];
    u[n + N * 2] = z[n] - r0[n + N + N];
  }
}

// save potential (p), force (f), and virial (v)
static __global__ void gpu_save_pfv(
  const int N, const float* pfv, double* p, double* fx, double* fy, double* fz, double* v)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    p[n] = pfv[n];              // potential energy
    fx[n] = pfv[n + N];         // fx
    fy[n] = pfv[n + N * 2];     // fy
    fz[n] = pfv[n + N * 3];     // fz
    for (int m = 0; m < 9; ++m) // virial tensor
    {
      v[n + N * m] = pfv[n + N * (m + 4)];
    }
  }
}

// Wrapper of the above kernels
void FCP::compute(
  const int group_method,
  std::vector<Group>& group,
  const int type_begin,
  const int type_end,
  const int type_shift,
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const int number_of_atoms = type.size();
  const int block_size = 1024;

  gpu_get_u<<<(number_of_atoms - 1) / block_size + 1, block_size>>>(
    number_of_atoms, position_per_atom.data(), position_per_atom.data() + number_of_atoms,
    position_per_atom.data() + number_of_atoms * 2, fcp_data.r0.data(), fcp_data.u.data());
  CUDA_CHECK_KERNEL

  fcp_data.pfv.fill(0.0f);

  gpu_find_force_fcp2<<<(number2 - 1) / block_size + 1, block_size>>>(
    number_of_atoms, number2, fcp_data.i2.data(), fcp_data.j2.data(), fcp_data.index2.data(),
    fcp_data.phi2.data(), fcp_data.u.data(), fcp_data.xij2.data(), fcp_data.yij2.data(),
    fcp_data.zij2.data(), fcp_data.pfv.data());

  if (order >= 3)
    gpu_find_force_fcp3<<<(number3 - 1) / block_size + 1, block_size>>>(
      heat_order, number_of_atoms, number3, fcp_data.i3.data(), fcp_data.j3.data(),
      fcp_data.k3.data(), fcp_data.index3.data(), fcp_data.phi3.data(), fcp_data.u.data(),
      fcp_data.xij3.data(), fcp_data.yij3.data(), fcp_data.zij3.data(), fcp_data.pfv.data());

  if (order >= 4)
    gpu_find_force_fcp4<<<(number4 - 1) / block_size + 1, block_size>>>(
      number_of_atoms, number4, fcp_data.i4.data(), fcp_data.j4.data(), fcp_data.k4.data(),
      fcp_data.l4.data(), fcp_data.index4.data(), fcp_data.weight4.data(), fcp_data.phi4.data(),
      fcp_data.u.data(), fcp_data.pfv.data());

  if (order >= 5)
    gpu_find_force_fcp5<<<(number5 - 1) / block_size + 1, block_size>>>(
      number_of_atoms, number5, fcp_data.i5.data(), fcp_data.j5.data(), fcp_data.k5.data(),
      fcp_data.l5.data(), fcp_data.m5.data(), fcp_data.index5.data(), fcp_data.weight5.data(),
      fcp_data.phi5.data(), fcp_data.u.data(), fcp_data.pfv.data());

  if (order >= 6)
    gpu_find_force_fcp6<<<(number6 - 1) / block_size + 1, block_size>>>(
      number_of_atoms, number6, fcp_data.i6.data(), fcp_data.j6.data(), fcp_data.k6.data(),
      fcp_data.l6.data(), fcp_data.m6.data(), fcp_data.n6.data(), fcp_data.index6.data(),
      fcp_data.weight6.data(), fcp_data.phi6.data(), fcp_data.u.data(), fcp_data.pfv.data());

  gpu_save_pfv<<<(number_of_atoms - 1) / block_size + 1, block_size>>>(
    number_of_atoms, fcp_data.pfv.data(), potential_per_atom.data(), force_per_atom.data(),
    force_per_atom.data() + number_of_atoms, force_per_atom.data() + 2 * number_of_atoms,
    virial_per_atom.data());

  CUDA_CHECK_KERNEL
}
