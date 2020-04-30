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
#include "atom.cuh"
#include "mic.cuh"
#include "error.cuh"
#include <vector>


FCP::FCP(FILE* fid, char *input_dir, Atom *atom)
{
    // get the highest order of the force constants
    int count = fscanf(fid, "%d", &order);
    PRINT_SCANF_ERROR(count, 1, "Reading error for force constant potential.");

    printf("Use the force constant potential.\n");
    printf("    up to order-%d.\n", order);

    // get the path of the files related to the force constants
    count = fscanf(fid, "%s", file_path);
    PRINT_SCANF_ERROR(count, 1, "Reading error for force constant potential.");
    printf("    Use the force constant data in %s.\n", file_path);

    // allocate memeory
    fcp_data.u.resize(atom->N * 3);
    fcp_data.r0.resize(atom->N * 3, Memory_Type::managed);
    fcp_data.pfv.resize(atom->N * 13);

    // read in the equilibrium positions and force constants
    read_r0(atom);
    read_fc2(atom);
    if (order >= 3) read_fc3(atom);
    if (order >= 4) read_fc4(atom);
    if (order >= 5) read_fc5(atom);
    if (order >= 6) read_fc6(atom);
}


FCP::~FCP(void)
{
  // nothing
}


void FCP::read_r0(Atom *atom)
{
    char file[200];
    strcpy(file, file_path);
    strcat(file, "/r0.in");
    FILE *fid = my_fopen(file, "r");

    int N = atom->N;
    for (int n = 0; n < N; n++)
    {
        int count = fscanf
        (
            fid, "%f%f%f", &fcp_data.r0[n], 
            &fcp_data.r0[n + N], &fcp_data.r0[n + N + N]
        );
        PRINT_SCANF_ERROR(count, 3, "Reading error for r0.in.");
    }
    fclose(fid);
    printf("    Data in r0.in have been read in.\n");
}


void FCP::read_fc2(Atom *atom)
{
    // reading fcs_order2.in
    char file_fc[200];
    strcpy(file_fc, file_path);
    strcat(file_fc, "/fcs_order2.in");
    FILE *fid_fc = my_fopen(file_fc, "r");
    
    printf("    Reading data from %s\n", file_fc);

    int num_fcs = 0;
    int count = fscanf(fid_fc, "%d", &num_fcs);
    PRINT_SCANF_ERROR(count, 1, "Reading error for fcs_order2.in.");
    if (num_fcs <= 0)
    {
        PRINT_INPUT_ERROR("number of force constant matrix should > 0.");
    }

    std::vector<float> fc(num_fcs * 9);
    for (int n = 0; n < num_fcs; ++n)
    {
        for (int a = 0; a < 3; ++a)
        {
            for (int b = 0; b < 3; ++b)
            {
                int index = n*9 + a*3 + b;
                int aa, bb; // not used
                count = fscanf(fid_fc, "%d%d%f", &aa, &bb, &fc[index]);
                PRINT_SCANF_ERROR(count, 3, "Reading error for fcs_order2.in.");
            }
        }
    }
        
    // reading clusters_order2.in
    char file_cluster[200];
    strcpy(file_cluster, file_path);
    strcat(file_cluster, "/clusters_order2.in");
    FILE *fid_cluster = my_fopen(file_cluster, "r");
    
    printf("    Reading data from %s\n", file_cluster);
    
    int num_clusters = 0;
    count = fscanf(fid_cluster, "%d", &num_clusters);
    PRINT_SCANF_ERROR(count, 1, "Reading error for clusters_order2.in.");
    if (num_clusters <= 0)
    {
        PRINT_INPUT_ERROR("number of clusters should > 0.");
    }
    number2 = num_clusters * 9;

    fcp_data.ia2.resize(number2 * 2, Memory_Type::managed);
    fcp_data.jb2.resize(number2 * 2, Memory_Type::managed);
    fcp_data.phi2.resize(number2 * 2, Memory_Type::managed);
    fcp_data.xij2.resize(number2 * 2, Memory_Type::managed);
    fcp_data.yij2.resize(number2 * 2, Memory_Type::managed);
    fcp_data.zij2.resize(number2 * 2, Memory_Type::managed);
	
    int idx_clusters_new = 0;
    for (int idx_clusters = 0; idx_clusters < num_clusters; idx_clusters++)
    {
        int i, j, idx_fcs;
        count = fscanf(fid_cluster, "%d%d%d", &i, &j, &idx_fcs);
        PRINT_SCANF_ERROR(count, 3, "Reading error for clusters_order2.in.");

        if (i < 0 || i >= atom->N) { PRINT_INPUT_ERROR("i < 0 or >= N."); } 
        if (j < 0 || j >= atom->N) { PRINT_INPUT_ERROR("j < 0 or >= N."); } 
        if (i > j) { PRINT_INPUT_ERROR("i > j."); } 
        if (idx_fcs < 0 || idx_fcs >= num_fcs) 
        {
            PRINT_INPUT_ERROR("idx_fcs < 0 or >= num_fcs");
        } 

        for (int a = 0; a < 3; ++a)
        {
            for (int b = 0; b < 3; ++b)
            {
                int ab = a*3 + b;
                int index = idx_clusters_new*9 + ab;
                fcp_data.ia2[index] = a * atom->N + i;
                fcp_data.jb2[index] = b * atom->N + j;
                fcp_data.phi2[index] = fc[idx_fcs*9 + ab];

                double xij2 = fcp_data.r0[j] - fcp_data.r0[i];
                double yij2 = fcp_data.r0[j + atom->N] 
                            - fcp_data.r0[i + atom->N];
                double zij2 = fcp_data.r0[j + atom->N*2] 
                            - fcp_data.r0[i + atom->N*2];
                apply_mic
                (
                    atom->box.triclinic, atom->box.pbc_x, atom->box.pbc_y, 
                    atom->box.pbc_z, atom->box.cpu_h, xij2, yij2, zij2
                );
                fcp_data.xij2[index] = xij2 * 0.5;
                fcp_data.yij2[index] = yij2 * 0.5;
                fcp_data.zij2[index] = zij2 * 0.5;
            }
        }
        ++idx_clusters_new;
		
        if (i != j)
        {
            for (int ab = 0; ab < 9; ++ab)
            {
                int index = idx_clusters_new*9 + ab;
                int index_old = index - 9;
                fcp_data.ia2[index] = fcp_data.jb2[index_old];
                fcp_data.jb2[index] = fcp_data.ia2[index_old];
                fcp_data.phi2[index] = fcp_data.phi2[index_old];
                fcp_data.xij2[index] = -fcp_data.xij2[index_old];
                fcp_data.yij2[index] = -fcp_data.yij2[index_old];
                fcp_data.zij2[index] = -fcp_data.zij2[index_old];
            }
            ++idx_clusters_new;
        }
    }
    number2 = idx_clusters_new * 9;

    fclose(fid_fc);
    fclose(fid_cluster);
}


void FCP::read_fc3(Atom *atom)
{
    // reading fcs_order3.in
    char file_fc[200];
    strcpy(file_fc, file_path);
    strcat(file_fc, "/fcs_order3.in");
    FILE *fid_fc = my_fopen(file_fc, "r");
    
    printf("    Reading data from %s\n", file_fc);

    int num_fcs = 0;
    int count = fscanf(fid_fc, "%d", &num_fcs);
    PRINT_SCANF_ERROR(count, 1, "Reading error for fcs_order3.in.");
    if (num_fcs <= 0)
    {
        PRINT_INPUT_ERROR("number of force constant matrix should > 0.");
    }

    std::vector<float> fc(num_fcs * 27);
    for (int n = 0; n < num_fcs; ++n)
    for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
    for (int c = 0; c < 3; ++c)
    {
        int index = n*27 + a*9 + b*3 + c;
        int aa, bb, cc; // not used
        count = fscanf(fid_fc, "%d%d%d%f", &aa, &bb, &cc, &fc[index]);
        PRINT_SCANF_ERROR(count, 4, "Reading error for fcs_order3.in.");
    }
        
    // reading clusters_order3.in
    char file_cluster[200];
    strcpy(file_cluster, file_path);
    strcat(file_cluster, "/clusters_order3.in");
    FILE *fid_cluster = my_fopen(file_cluster, "r");
    
    printf("    Reading data from %s\n", file_cluster);
    
    int num_clusters = 0;
    count = fscanf(fid_cluster, "%d", &num_clusters);
    PRINT_SCANF_ERROR(count, 1, "Reading error for clusters_order3.in.");
    if (num_clusters <= 0)
    {
        PRINT_INPUT_ERROR("number of clusters should > 0.");
    }
    number3 = num_clusters * 27;

    fcp_data.ia3.resize(number3, Memory_Type::managed);
    fcp_data.jb3.resize(number3, Memory_Type::managed);
    fcp_data.kc3.resize(number3, Memory_Type::managed);
    fcp_data.phi3.resize(number3, Memory_Type::managed);

    for (int idx_clusters = 0; idx_clusters < num_clusters; idx_clusters++)
    {
        int i, j, k, idx_fcs;
        count = fscanf(fid_cluster, "%d%d%d%d", &i, &j, &k, &idx_fcs);
        PRINT_SCANF_ERROR(count, 4, "Reading error for clusters_order3.in.");
        
        if (i < 0 || i >= atom->N) { PRINT_INPUT_ERROR("i < 0 or >= N."); } 
        if (j < 0 || j >= atom->N) { PRINT_INPUT_ERROR("j < 0 or >= N."); } 
        if (k < 0 || k >= atom->N) { PRINT_INPUT_ERROR("k < 0 or >= N."); } 
        if (i > j) { PRINT_INPUT_ERROR("i > j."); } 
        if (j > k) { PRINT_INPUT_ERROR("j > k."); } 
        if (idx_fcs < 0 || idx_fcs >= num_fcs) 
        {
            PRINT_INPUT_ERROR("idx_fcs < 0 or >= num_fcs");
        } 

        for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
        for (int c = 0; c < 3; ++c)
        {
            int abc = a*9 + b*3 + c;
            int index = idx_clusters*27 + abc;
            fcp_data.ia3[index] = a * atom->N + i;
            fcp_data.jb3[index] = b * atom->N + j;
            fcp_data.kc3[index] = c * atom->N + k;
            fcp_data.phi3[index] = fc[idx_fcs*27 + abc];

            // 2^2-1 = 3 cases:
            if (i == j && j != k) { fcp_data.phi3[index] /= 2; } // 112
            if (i != j && j == k) { fcp_data.phi3[index] /= 2; } // 122
            if (i == j && j == k) { fcp_data.phi3[index] /= 6; } // 111
        }
    } 

    fclose(fid_fc);
    fclose(fid_cluster);
}


void FCP::read_fc4(Atom *atom)
{
    // reading fcs_order4.in
    char file_fc[200];
    strcpy(file_fc, file_path);
    strcat(file_fc, "/fcs_order4.in");
    FILE *fid_fc = my_fopen(file_fc, "r");
    
    printf("    Reading data from %s\n", file_fc);

    int num_fcs = 0;
    int count = fscanf(fid_fc, "%d", &num_fcs);
    PRINT_SCANF_ERROR(count, 1, "Reading error for fcs_order4.in.");
    if (num_fcs <= 0)
    {
        PRINT_INPUT_ERROR("number of force constant matrix should > 0.");
    }

    std::vector<float> fc(num_fcs * 81);
    for (int n = 0; n < num_fcs; ++n)
    for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
    for (int c = 0; c < 3; ++c)
    for (int d = 0; d < 3; ++d)
    {
        int index = n*81 + a*27 + b*9 + c*3 + d;
        int aa, bb, cc, dd; // not used
        count = fscanf(fid_fc, "%d%d%d%d%f", &aa, &bb, &cc, &dd, &fc[index]);
        PRINT_SCANF_ERROR(count, 5, "Reading error for fcs_order4.in.");
    }
        
    // reading clusters_order4.in
    char file_cluster[200];
    strcpy(file_cluster, file_path);
    strcat(file_cluster, "/clusters_order4.in");
    FILE *fid_cluster = my_fopen(file_cluster, "r");
    
    printf("    Reading data from %s\n", file_cluster);
    
    int num_clusters = 0;
    count = fscanf(fid_cluster, "%d", &num_clusters);
    PRINT_SCANF_ERROR(count, 1, "Reading error for clusters_order4.in.");
    if (num_clusters <= 0)
    {
        PRINT_INPUT_ERROR("number of clusters should > 0.");
    }
    number4 = num_clusters * 81;

    fcp_data.ia4.resize(number4, Memory_Type::managed);
    fcp_data.jb4.resize(number4, Memory_Type::managed);
    fcp_data.kc4.resize(number4, Memory_Type::managed);
    fcp_data.ld4.resize(number4, Memory_Type::managed);
    fcp_data.phi4.resize(number4, Memory_Type::managed);

    for (int idx_clusters = 0; idx_clusters < num_clusters; idx_clusters++)
    {
        int i, j, k, l, idx_fcs;
        count = fscanf(fid_cluster, "%d%d%d%d%d", &i, &j, &k, &l, &idx_fcs);
        PRINT_SCANF_ERROR(count, 5, "Reading error for clusters_order4.in.");
        
        if (i < 0 || i >= atom->N) { PRINT_INPUT_ERROR("i < 0 or >= N."); } 
        if (j < 0 || j >= atom->N) { PRINT_INPUT_ERROR("j < 0 or >= N."); } 
        if (k < 0 || k >= atom->N) { PRINT_INPUT_ERROR("k < 0 or >= N."); } 
        if (l < 0 || l >= atom->N) { PRINT_INPUT_ERROR("l < 0 or >= N."); } 
        if (i > j) { PRINT_INPUT_ERROR("i > j."); } 
        if (j > k) { PRINT_INPUT_ERROR("j > k."); } 
        if (k > l) { PRINT_INPUT_ERROR("k > l."); } 
        if (idx_fcs < 0 || idx_fcs >= num_fcs) 
        {
            PRINT_INPUT_ERROR("idx_fcs < 0 or >= num_fcs");
        } 

        for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
        for (int c = 0; c < 3; ++c)
        for (int d = 0; d < 3; ++d)
        {
            int abcd = a*27 + b*9 + c*3 + d;
            int index = idx_clusters*81 + abcd;
            fcp_data.ia4[index] = a * atom->N + i;
            fcp_data.jb4[index] = b * atom->N + j;
            fcp_data.kc4[index] = c * atom->N + k;
            fcp_data.ld4[index] = d * atom->N + l;
            fcp_data.phi4[index] = fc[idx_fcs*81 + abcd];

            // 2^3-1 = 7 cases:
            if (i == j && j != k && k != l) 
            { fcp_data.phi4[index] /= 2; }  // 1123
            if (i != j && j == k && k != l) 
            { fcp_data.phi4[index] /= 2; }  // 1223
            if (i != j && j != k && k == l) 
            { fcp_data.phi4[index] /= 2; }  // 1233
            if (i == j && j != k && k == l) 
            { fcp_data.phi4[index] /= 4; }  // 1122
            if (i == j && j == k && k != l) 
            { fcp_data.phi4[index] /= 6; }  // 1112
            if (i != j && j == k && k == l) 
            { fcp_data.phi4[index] /= 6; }  // 1222
            if (i == j && j == k && k == l) 
            { fcp_data.phi4[index] /= 24; } // 1111
        }
    }

    fclose(fid_fc);
    fclose(fid_cluster);
}


void FCP::read_fc5(Atom *atom)
{
    // reading fcs_order5.in
    char file_fc[200];
    strcpy(file_fc, file_path);
    strcat(file_fc, "/fcs_order5.in");
    FILE *fid_fc = my_fopen(file_fc, "r");
    
    printf("    Reading data from %s\n", file_fc);

    int num_fcs = 0;
    int count = fscanf(fid_fc, "%d", &num_fcs);
    PRINT_SCANF_ERROR(count, 1, "Reading error for fcs_order5.in.");
    if (num_fcs <= 0)
    {
        PRINT_INPUT_ERROR("number of force constant matrix should > 0.");
    }

    std::vector<float> fc(num_fcs * 243);
    for (int n = 0; n < num_fcs; ++n)
    for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
    for (int c = 0; c < 3; ++c)
    for (int d = 0; d < 3; ++d)
    for (int e = 0; e < 3; ++e)
    {
        int index = n*243 + a*81 + b*27 + c*9 + d*3 + e;
        int aa, bb, cc, dd, ee; // not used
        count = fscanf
        (fid_fc, "%d%d%d%d%d%f", &aa, &bb, &cc, &dd, &ee, &fc[index]);
        PRINT_SCANF_ERROR(count, 6, "Reading error for fcs_order5.in.");
    }
        
    // reading clusters_order5.in
    char file_cluster[200];
    strcpy(file_cluster, file_path);
    strcat(file_cluster, "/clusters_order5.in");
    FILE *fid_cluster = my_fopen(file_cluster, "r");
    
    printf("    Reading data from %s\n", file_cluster);
    
    int num_clusters = 0;
    count = fscanf(fid_cluster, "%d", &num_clusters);
    PRINT_SCANF_ERROR(count, 1, "Reading error for clusters_order5.in.");
    if (num_clusters <= 0)
    {
        PRINT_INPUT_ERROR("number of clusters should > 0.");
    }
    number5 = num_clusters * 243;

    fcp_data.ia5.resize(number5, Memory_Type::managed);
    fcp_data.jb5.resize(number5, Memory_Type::managed);
    fcp_data.kc5.resize(number5, Memory_Type::managed);
    fcp_data.ld5.resize(number5, Memory_Type::managed);
    fcp_data.me5.resize(number5, Memory_Type::managed);
    fcp_data.phi5.resize(number5, Memory_Type::managed);

    for (int idx_clusters = 0; idx_clusters < num_clusters; idx_clusters++)
    {
        int i, j, k, l, m, idx_fcs;
        count = fscanf
        (fid_cluster, "%d%d%d%d%d%d", &i, &j, &k, &l, &m, &idx_fcs);
        PRINT_SCANF_ERROR(count, 6, "Reading error for clusters_order5.in.");
        
        if (i < 0 || i >= atom->N) { PRINT_INPUT_ERROR("i < 0 or >= N."); } 
        if (j < 0 || j >= atom->N) { PRINT_INPUT_ERROR("j < 0 or >= N."); } 
        if (k < 0 || k >= atom->N) { PRINT_INPUT_ERROR("k < 0 or >= N."); } 
        if (l < 0 || l >= atom->N) { PRINT_INPUT_ERROR("l < 0 or >= N."); } 
        if (m < 0 || m >= atom->N) { PRINT_INPUT_ERROR("m < 0 or >= N."); }
        if (i > j) { PRINT_INPUT_ERROR("i > j."); } 
        if (j > k) { PRINT_INPUT_ERROR("j > k."); } 
        if (k > l) { PRINT_INPUT_ERROR("k > l."); } 
        if (l > m) { PRINT_INPUT_ERROR("l > m."); }
        if (idx_fcs < 0 || idx_fcs >= num_fcs) 
        {
            PRINT_INPUT_ERROR("idx_fcs < 0 or >= num_fcs");
        } 

        for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
        for (int c = 0; c < 3; ++c)
        for (int d = 0; d < 3; ++d)
        for (int e = 0; e < 3; ++e)
        {
            int abcde = a*81 + b*27 + c*9 + d*3 + e;
            int index = idx_clusters*243 + abcde;
            fcp_data.ia5[index] = a * atom->N + i;
            fcp_data.jb5[index] = b * atom->N + j;
            fcp_data.kc5[index] = c * atom->N + k;
            fcp_data.ld5[index] = d * atom->N + l;
            fcp_data.me5[index] = e * atom->N + m;
            fcp_data.phi5[index] = fc[idx_fcs*243 + abcde];

            // 2^4-1 = 15 cases:
            if (i == j && j != k && k != l && l != m) 
            { fcp_data.phi5[index] /= 2; }   // 11234
            if (i != j && j == k && k != l && l != m) 
            { fcp_data.phi5[index] /= 2; }   // 12234
            if (i != j && j != k && k == l && l != m) 
            { fcp_data.phi5[index] /= 2; }   // 12334
            if (i != j && j != k && k != l && l == m) 
            { fcp_data.phi5[index] /= 2; }   // 12344
            if (i == j && j == k && k != l && l != m) 
            { fcp_data.phi5[index] /= 6; }   // 11123
            if (i != j && j == k && k == l && l != m) 
            { fcp_data.phi5[index] /= 6; }   // 12223
            if (i != j && j != k && k == l && l == m) 
            { fcp_data.phi5[index] /= 6; }   // 12333
            if (i == j && j == k && k == l && l != m) 
            { fcp_data.phi5[index] /= 24; }  // 11112
            if (i != j && j == k && k == l && l == m) 
            { fcp_data.phi5[index] /= 24; }  // 12222
            if (i == j && j == k && k == l && l == m) 
            { fcp_data.phi5[index] /= 120; } // 11111
            if (i == j && j != k && k == l && l != m) 
            { fcp_data.phi5[index] /= 4; }   // 11223
            if (i == j && j != k && k != l && l == m) 
            { fcp_data.phi5[index] /= 4; }   // 11233
            if (i != j && j == k && k != l && l == m) 
            { fcp_data.phi5[index] /= 4; }   // 12233
            if (i == j && j != k && k == l && l == m) 
            { fcp_data.phi5[index] /= 12; }  // 11222
            if (i == j && j == k && k != l && l == m) 
            { fcp_data.phi5[index] /= 12; }  // 11122
        }
    } 

    fclose(fid_fc);
    fclose(fid_cluster);
}


void FCP::read_fc6(Atom *atom)
{
    // reading fcs_order6.in
    char file_fc[200];
    strcpy(file_fc, file_path);
    strcat(file_fc, "/fcs_order6.in");
    FILE *fid_fc = my_fopen(file_fc, "r");
    
    printf("    Reading data from %s\n", file_fc);

    int num_fcs = 0;
    int count = fscanf(fid_fc, "%d", &num_fcs);
    PRINT_SCANF_ERROR(count, 1, "Reading error for fcs_order6.in.");
    if (num_fcs <= 0)
    {
        PRINT_INPUT_ERROR("number of force constant matrix should > 0.");
    }

    std::vector<float> fc(num_fcs * 729);
    for (int n = 0; n < num_fcs; ++n)
    for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
    for (int c = 0; c < 3; ++c)
    for (int d = 0; d < 3; ++d)
    for (int e = 0; e < 3; ++e)
    for (int f = 0; f < 3; ++f)
    {
        int index = n*729 + a*243 + b*81 + c*27 + d*9 + e*3 + f;
        int aa, bb, cc, dd, ee, ff; // not used
        count = fscanf
        (fid_fc, "%d%d%d%d%d%d%f", &aa, &bb, &cc, &dd, &ee, &ff, &fc[index]);
        PRINT_SCANF_ERROR(count, 7, "Reading error for fcs_order6.in.");
    }
        
    // reading clusters_order6.in
    char file_cluster[200];
    strcpy(file_cluster, file_path);
    strcat(file_cluster, "/clusters_order6.in");
    FILE *fid_cluster = my_fopen(file_cluster, "r");
    
    printf("    Reading data from %s\n", file_cluster);
    
    int num_clusters = 0;
    count = fscanf(fid_cluster, "%d", &num_clusters);
    PRINT_SCANF_ERROR(count, 1, "Reading error for clusters_order6.in.");
    if (num_clusters <= 0)
    {
        PRINT_INPUT_ERROR("number of clusters should > 0.");
    }
    number6 = num_clusters * 729;

    fcp_data.ia6.resize(number6, Memory_Type::managed);
    fcp_data.jb6.resize(number6, Memory_Type::managed);
    fcp_data.kc6.resize(number6, Memory_Type::managed);
    fcp_data.ld6.resize(number6, Memory_Type::managed);
    fcp_data.me6.resize(number6, Memory_Type::managed);
    fcp_data.nf6.resize(number6, Memory_Type::managed);
    fcp_data.phi6.resize(number6, Memory_Type::managed);

    for (int idx_clusters = 0; idx_clusters < num_clusters; idx_clusters++)
    {
        int i, j, k, l, m, n, idx_fcs;
        count = fscanf
        (fid_cluster, "%d%d%d%d%d%d%d", &i, &j, &k, &l, &m, &n, &idx_fcs);
        PRINT_SCANF_ERROR(count, 7, "Reading error for clusters_order6.in.");
        
        if (i < 0 || i >= atom->N) { PRINT_INPUT_ERROR("i < 0 or >= N."); } 
        if (j < 0 || j >= atom->N) { PRINT_INPUT_ERROR("j < 0 or >= N."); } 
        if (k < 0 || k >= atom->N) { PRINT_INPUT_ERROR("k < 0 or >= N."); } 
        if (l < 0 || l >= atom->N) { PRINT_INPUT_ERROR("l < 0 or >= N."); } 
        if (m < 0 || m >= atom->N) { PRINT_INPUT_ERROR("m < 0 or >= N."); }
        if (n < 0 || n >= atom->N) { PRINT_INPUT_ERROR("n < 0 or >= N."); }
        if (i > j) { PRINT_INPUT_ERROR("i > j."); } 
        if (j > k) { PRINT_INPUT_ERROR("j > k."); } 
        if (k > l) { PRINT_INPUT_ERROR("k > l."); } 
        if (l > m) { PRINT_INPUT_ERROR("l > m."); }
        if (m > n) { PRINT_INPUT_ERROR("m > n."); }
        if (idx_fcs < 0 || idx_fcs >= num_fcs) 
        {
            PRINT_INPUT_ERROR("idx_fcs < 0 or >= num_fcs");
        } 

        for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
        for (int c = 0; c < 3; ++c)
        for (int d = 0; d < 3; ++d)
        for (int e = 0; e < 3; ++e)
        for (int f = 0; f < 3; ++f)
        {
            int abcdef = a*243 + b*81 + c*27 + d*9 + e*3 + f;
            int index = idx_clusters*729 + abcdef;
            fcp_data.ia6[index] = a * atom->N + i;
            fcp_data.jb6[index] = b * atom->N + j;
            fcp_data.kc6[index] = c * atom->N + k;
            fcp_data.ld6[index] = d * atom->N + l;
            fcp_data.me6[index] = e * atom->N + m;
            fcp_data.nf6[index] = f * atom->N + n;
            fcp_data.phi6[index] = fc[idx_fcs*729 + abcdef];

            // 2^5-1 = 31 cases:
            if (i == j && j != k && k != l && l != m && m != n) 
            { fcp_data.phi6[index] /= 2; }   // 112345
            if (i != j && j == k && k != l && l != m && m != n) 
            { fcp_data.phi6[index] /= 2; }   // 122345
            if (i != j && j != k && k == l && l != m && m != n) 
            { fcp_data.phi6[index] /= 2; }   // 123345
            if (i != j && j != k && k != l && l == m && m != n) 
            { fcp_data.phi6[index] /= 2; }   // 123445
            if (i != j && j != k && k != l && l != m && m == n) 
            { fcp_data.phi6[index] /= 2; }   // 123455
            if (i == j && j == k && k != l && l != m && m != n) 
            { fcp_data.phi6[index] /= 6; }   // 111234
            if (i != j && j == k && k == l && l != m && m != n) 
            { fcp_data.phi6[index] /= 6; }   // 122234
            if (i != j && j != k && k == l && l == m && m != n) 
            { fcp_data.phi6[index] /= 6; }   // 123334
            if (i != j && j != k && k != l && l == m && m == n) 
            { fcp_data.phi6[index] /= 6; }   // 123444
            if (i == j && j == k && k == l && l != m && m != n) 
            { fcp_data.phi6[index] /= 24; }  // 111123
            if (i != j && j == k && k == l && l == m && m != n) 
            { fcp_data.phi6[index] /= 24; }  // 122223
            if (i != j && j != k && k == l && l == m && m == n) 
            { fcp_data.phi6[index] /= 24; }  // 122223
            if (i == j && j == k && k == l && l == m && m != n) 
            { fcp_data.phi6[index] /= 120; } // 111112
            if (i != j && j == k && k == l && l == m && m == n) 
            { fcp_data.phi6[index] /= 120; } // 122222
            if (i == j && j == k && k == l && l == m && m == n) 
            { fcp_data.phi6[index] /= 720; } // 111111
            if (i == j && j != k && k == l && l != m && m != n) 
            { fcp_data.phi6[index] /= 4; }   // 112234
            if (i == j && j != k && k != l && l == m && m != n) 
            { fcp_data.phi6[index] /= 4; }   // 112334
            if (i == j && j != k && k != l && l != m && m == n) 
            { fcp_data.phi6[index] /= 4; }   // 112344
            if (i != j && j == k && k != l && l == m && m != n) 
            { fcp_data.phi6[index] /= 4; }   // 122334
            if (i != j && j == k && k != l && l != m && m == n) 
            { fcp_data.phi6[index] /= 4; }   // 122344
            if (i != j && j != k && k == l && l != m && m == n) 
            { fcp_data.phi6[index] /= 4; }   // 123344
            if (i == j && j != k && k == l && l == m && m != n) 
            { fcp_data.phi6[index] /= 12; }  // 112223
            if (i == j && j != k && k != l && l == m && m == n) 
            { fcp_data.phi6[index] /= 12; }  // 112333
            if (i != j && j == k && k != l && l == m && m == n) 
            { fcp_data.phi6[index] /= 12; }  // 122333
            if (i == j && j == k && k != l && l == m && m != n) 
            { fcp_data.phi6[index] /= 12; }  // 111223
            if (i == j && j == k && k != l && l != m && m == n) 
            { fcp_data.phi6[index] /= 12; }  // 111233
            if (i != j && j == k && k == l && l != m && m == n) 
            { fcp_data.phi6[index] /= 12; }  // 122233
            if (i == j && j != k && k == l && l == m && m == n) 
            { fcp_data.phi6[index] /= 48; }  // 112222
            if (i == j && j == k && k == l && l != m && m == n) 
            { fcp_data.phi6[index] /= 48; }  // 111122
            if (i == j && j == k && k != l && l == m && m == n) 
            { fcp_data.phi6[index] /= 36; }  // 111222
            if (i == j && j != k && k == l && l != m && m == n) 
            { fcp_data.phi6[index] /= 8; }   // 112233

        }
    } 

    fclose(fid_fc);
    fclose(fid_cluster);
}


// potential, force, and virial from the second-order force constants
static __global__ void gpu_find_force_fcp2
(
    int N, int number2, int *g_ia2, int *g_jb2, float *g_phi2, float* g_u, 
    float *g_xij2, float *g_yij2, float *g_zij2, float *g_pfv
)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < number2)
    {
        int ia = g_ia2[n]; 
        int jb = g_jb2[n];
        float phi = g_phi2[n];
        float xij2 = g_xij2[n];
        float yij2 = g_yij2[n];
        float zij2 = g_zij2[n];
        float uia = LDG(g_u, ia);
        float ujb = LDG(g_u, jb);
        
        int atom_id = ia % N;
        atomicAdd(&g_pfv[atom_id], 0.5f * phi * uia * ujb); // potential
        atomicAdd(&g_pfv[ia + N], - phi * ujb); // force
        
        // virial tensor
        int a = ia / N;
        int x[3] = {4, 7, 8};
        int y[3] = {10, 5, 9};
        int z[3] = {11, 12, 6};
        atomicAdd(&g_pfv[atom_id + N * x[a]], xij2 * phi * ujb);
        atomicAdd(&g_pfv[atom_id + N * y[a]], yij2 * phi * ujb);
        atomicAdd(&g_pfv[atom_id + N * z[a]], zij2 * phi * ujb);
    }
}


// potential and force from the third-order force constants
static __global__ void gpu_find_force_fcp3
(
    int N, int number3, int *g_ia3, int *g_jb3, int *g_kc3, float *g_phi3,
    const float* __restrict__ g_u, float *g_pf
)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < number3)
    {
        int ia = g_ia3[n];
        int jb = g_jb3[n]; 
        int kc = g_kc3[n]; 
        float phi = g_phi3[n];
        float uia = LDG(g_u, ia); 
        float ujb = LDG(g_u, jb);
        float ukc = LDG(g_u, kc);
        atomicAdd(&g_pf[ia % N], phi * uia * ujb * ukc);
        atomicAdd(&g_pf[ia + N], - phi * ujb * ukc);
        atomicAdd(&g_pf[jb + N], - phi * uia * ukc);
        atomicAdd(&g_pf[kc + N], - phi * uia * ujb);
    }
}


// potential and force from the fourth-order force constants
static __global__ void gpu_find_force_fcp4
(
    int N, int number4, int *g_ia4, int *g_jb4, int *g_kc4, int *g_ld4,
    float *g_phi4, const float* __restrict__ g_u, float *g_pf
)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < number4)
    {
        int ia = g_ia4[n];
        int jb = g_jb4[n]; 
        int kc = g_kc4[n];
        int ld = g_ld4[n];
        float phi = g_phi4[n];
        float uia = LDG(g_u, ia); 
        float ujb = LDG(g_u, jb);
        float ukc = LDG(g_u, kc);
        float uld = LDG(g_u, ld);
        atomicAdd(&g_pf[ia % N], phi * uia * ujb * ukc * uld);
        atomicAdd(&g_pf[ia + N], - phi * ujb * ukc * uld);
        atomicAdd(&g_pf[jb + N], - phi * uia * ukc * uld);
        atomicAdd(&g_pf[kc + N], - phi * uia * ujb * uld);
        atomicAdd(&g_pf[ld + N], - phi * uia * ujb * ukc);
    }
}


// potential and force from the fifth-order force constants
static __global__ void gpu_find_force_fcp5
(
    int N, int number5, int *g_ia5, int *g_jb5, int *g_kc5, int *g_ld5,
    int *g_me5, float *g_phi5, const float* __restrict__ g_u, float *g_pf
)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < number5)
    {
        int ia = g_ia5[n];
        int jb = g_jb5[n]; 
        int kc = g_kc5[n];
        int ld = g_ld5[n];
        int me = g_me5[n];
        float phi = g_phi5[n];
        float uia = LDG(g_u, ia); 
        float ujb = LDG(g_u, jb);
        float ukc = LDG(g_u, kc);
        float uld = LDG(g_u, ld);
        float ume = LDG(g_u, me);
        atomicAdd(&g_pf[ia % N], phi * uia * ujb * ukc * uld * ume);
        atomicAdd(&g_pf[ia + N], - phi * ujb * ukc * uld * ume);
        atomicAdd(&g_pf[jb + N], - phi * uia * ukc * uld * ume);
        atomicAdd(&g_pf[kc + N], - phi * uia * ujb * uld * ume);
        atomicAdd(&g_pf[ld + N], - phi * uia * ujb * ukc * ume);
        atomicAdd(&g_pf[me + N], - phi * uia * ujb * ukc * uld);
    }
}


// potential and force from the sixth-order force constants
static __global__ void gpu_find_force_fcp6
(
    int N, int number6, int *g_ia6, int *g_jb6, int *g_kc6, int *g_ld6,
    int *g_me6, int *g_nf6, float *g_phi6, const float* __restrict__ g_u, 
    float *g_pf
)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < number6)
    {
        int ia = g_ia6[n];
        int jb = g_jb6[n]; 
        int kc = g_kc6[n];
        int ld = g_ld6[n];
        int me = g_me6[n];
        int nf = g_nf6[n];
        float phi = g_phi6[n];
        float uia = LDG(g_u, ia); 
        float ujb = LDG(g_u, jb);
        float ukc = LDG(g_u, kc);
        float uld = LDG(g_u, ld);
        float ume = LDG(g_u, me);
        float unf = LDG(g_u, nf);
        atomicAdd(&g_pf[ia % N], phi * uia * ujb * ukc * uld * ume * unf);
        atomicAdd(&g_pf[ia + N], - phi * ujb * ukc * uld * ume * unf);
        atomicAdd(&g_pf[jb + N], - phi * uia * ukc * uld * ume * unf);
        atomicAdd(&g_pf[kc + N], - phi * uia * ujb * uld * ume * unf);
        atomicAdd(&g_pf[ld + N], - phi * uia * ujb * ukc * ume * unf);
        atomicAdd(&g_pf[me + N], - phi * uia * ujb * ukc * uld * unf);
        atomicAdd(&g_pf[nf + N], - phi * uia * ujb * ukc * uld * ume);
    }
}


// initialize the local potential (p), force (f), and virial (v)
static __global__ void gpu_initialize_pfv(int N, float *pfv)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N)
    {
        pfv[n] = 0.0f;
    }
}


// get the displacement (u=r-r0)
static __global__ void gpu_get_u
(
    int N, double *x, double *y, double *z, 
    float *r0, float *u
)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N)
    {
        u[n]         = x[n] - r0[n];
        u[n + N]     = y[n] - r0[n + N];
        u[n + N * 2] = z[n] - r0[n + N + N];
    }
}


// save potential (p), force (f), and virial (v)
static __global__ void gpu_save_pfv
(int N, float *pfv, double *p, double *fx, double *fy, double *fz, double *v)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N)
    {
        p[n]  = pfv[n];                // potential energy
        fx[n] = pfv[n + N];            // fx
        fy[n] = pfv[n + N * 2];        // fy
        fz[n] = pfv[n + N * 3];        // fz
        for (int m = 0; m < 9; ++m)    // virial tensor
        {
            v[n + N * m] = pfv[n + N * (m + 4)]; 
        }
    }
}


// Wrapper of the above kernels
void FCP::compute(Atom *atom, int potential_number)
{
    const int block_size = 1024;

    gpu_get_u<<<(atom->N - 1) / block_size + 1, block_size>>>
    (
        atom->N,
        atom->x,
        atom->y,
        atom->z,
        fcp_data.r0.data(),
        fcp_data.u.data()
    );

    gpu_initialize_pfv<<<(atom->N * 13 - 1) / block_size + 1, block_size>>>
    (
        atom->N * 13,
        fcp_data.pfv.data()
    );

    gpu_find_force_fcp2<<<(number2 - 1) / block_size + 1, block_size>>>
    (
        atom->N,
        number2,
        fcp_data.ia2.data(),
        fcp_data.jb2.data(),
        fcp_data.phi2.data(),
        fcp_data.u.data(),
        fcp_data.xij2.data(),
        fcp_data.yij2.data(),
        fcp_data.zij2.data(),
        fcp_data.pfv.data()
    );

    if (order >= 3)
    gpu_find_force_fcp3<<<(number3 - 1) / block_size + 1, block_size>>>
    (
        atom->N,
        number3,
        fcp_data.ia3.data(),
        fcp_data.jb3.data(),
        fcp_data.kc3.data(),
        fcp_data.phi3.data(),
        fcp_data.u.data(),
        fcp_data.pfv.data()
    );

    if (order >= 4)
    gpu_find_force_fcp4<<<(number4 - 1) / block_size + 1, block_size>>>
    (
        atom->N,
        number4,
        fcp_data.ia4.data(),
        fcp_data.jb4.data(),
        fcp_data.kc4.data(),
        fcp_data.ld4.data(),
        fcp_data.phi4.data(),
        fcp_data.u.data(),
        fcp_data.pfv.data()
    );

    if (order >= 5)
    gpu_find_force_fcp5<<<(number5 - 1) / block_size + 1, block_size>>>
    (
        atom->N,
        number5,
        fcp_data.ia5.data(),
        fcp_data.jb5.data(),
        fcp_data.kc5.data(),
        fcp_data.ld5.data(),
        fcp_data.me5.data(),
        fcp_data.phi5.data(),
        fcp_data.u.data(),
        fcp_data.pfv.data()
    );

    if (order >= 6)
    gpu_find_force_fcp6<<<(number6 - 1) / block_size + 1, block_size>>>
    (
        atom->N,
        number6,
        fcp_data.ia6.data(),
        fcp_data.jb6.data(),
        fcp_data.kc6.data(),
        fcp_data.ld6.data(),
        fcp_data.me6.data(),
        fcp_data.nf6.data(),
        fcp_data.phi6.data(),
        fcp_data.u.data(),
        fcp_data.pfv.data()
    ); 

    gpu_save_pfv<<<(atom->N - 1) / block_size + 1, block_size>>>
    (
        atom->N,
        fcp_data.pfv.data(),
        atom->potential_per_atom,
        atom->fx,
        atom->fy,
        atom->fz,
        atom->virial_per_atom
    );

    CUDA_CHECK_KERNEL
}


