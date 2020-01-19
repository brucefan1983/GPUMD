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
#include "measure.cuh"
#include "atom.cuh"
#include "mic.cuh"
#include "error.cuh"


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
    CHECK(cudaMalloc(&fcp_data.uv, sizeof(float) * atom->N * 6));
    CHECK(cudaMallocManaged(&fcp_data.r0, sizeof(float) * atom->N * 3));
    CHECK(cudaMalloc(&fcp_data.pfj, sizeof(float) * atom->N * 7));

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
    CHECK(cudaFree(fcp_data.uv));
    CHECK(cudaFree(fcp_data.r0));
    CHECK(cudaFree(fcp_data.pfj));
    CHECK(cudaFree(fcp_data.ia2));
    CHECK(cudaFree(fcp_data.jb2));
    CHECK(cudaFree(fcp_data.phi2));
    CHECK(cudaFree(fcp_data.xij2));
    CHECK(cudaFree(fcp_data.yij2));
    CHECK(cudaFree(fcp_data.zij2));
    if (order >= 3)
    {
        CHECK(cudaFree(fcp_data.ia3));
        CHECK(cudaFree(fcp_data.jb3));
        CHECK(cudaFree(fcp_data.kc3));
        CHECK(cudaFree(fcp_data.phi3));
    }
    if (order >= 4)
    {
        CHECK(cudaFree(fcp_data.ia4));
        CHECK(cudaFree(fcp_data.jb4));
        CHECK(cudaFree(fcp_data.kc4));
        CHECK(cudaFree(fcp_data.ld4));
        CHECK(cudaFree(fcp_data.phi4));
    }
    if (order >= 5)
    {
        CHECK(cudaFree(fcp_data.ia5));
        CHECK(cudaFree(fcp_data.jb5));
        CHECK(cudaFree(fcp_data.kc5));
        CHECK(cudaFree(fcp_data.ld5));
        CHECK(cudaFree(fcp_data.me5));
        CHECK(cudaFree(fcp_data.phi5));
    }
    if (order >= 6)
    {
        CHECK(cudaFree(fcp_data.ia6));
        CHECK(cudaFree(fcp_data.jb6));
        CHECK(cudaFree(fcp_data.kc6));
        CHECK(cudaFree(fcp_data.ld6));
        CHECK(cudaFree(fcp_data.me6));
        CHECK(cudaFree(fcp_data.nf6));
        CHECK(cudaFree(fcp_data.phi6));
    }
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
	// reading fcs_order2
    char file_fc[200];
    strcpy(file_fc, file_path);
    strcat(file_fc, "/fcs_order2");
    FILE *fid_fc = my_fopen(file_fc, "r");
    
    printf("    Reading data from %s\n", file_fc);

    int num_fcs = 0;
    int count = fscanf(fid_fc, "%d", &num_fcs);
    PRINT_SCANF_ERROR(count, 1, "Reading error for fcs_order2.");
    if (num_fcs <= 0)
    {
    	PRINT_INPUT_ERROR("number of force constant matrix should > 0.");
	}

    float *fc;
	MY_MALLOC(fc, float, num_fcs * 9);
    for (int n = 0; n < num_fcs; ++n)
    {
        for (int a = 0; a < 3; ++a)
        {
        	for (int b = 0; b < 3; ++b)
        	{
        		int index = n*9 + a*3 + b;
        		int aa, bb; // not used
        		count = fscanf(fid_fc, "%d%d%f", &aa, &bb, &fc[index]);
        		PRINT_SCANF_ERROR(count, 3, "Reading error for fcs_order2.");
			}
		}
    }
		
    // reading clusters_order2
    char file_cluster[200];
    strcpy(file_cluster, file_path);
    strcat(file_cluster, "/clusters_order2");
    FILE *fid_cluster = my_fopen(file_cluster, "r");
    
    printf("    Reading data from %s\n", file_cluster);
    
    int num_clusters = 0;
    count = fscanf(fid_cluster, "%d", &num_clusters);
    PRINT_SCANF_ERROR(count, 1, "Reading error for clusters_order2.");
    number2 = num_clusters * 9;

    CHECK(cudaMallocManaged(&fcp_data.ia2, sizeof(int) * number2));
    CHECK(cudaMallocManaged(&fcp_data.jb2, sizeof(int) * number2));
    CHECK(cudaMallocManaged(&fcp_data.phi2, sizeof(float) * number2));
    CHECK(cudaMallocManaged(&fcp_data.xij2, sizeof(float) * number2));
    CHECK(cudaMallocManaged(&fcp_data.yij2, sizeof(float) * number2));
    CHECK(cudaMallocManaged(&fcp_data.zij2, sizeof(float) * number2));

    for (int idx_clusters = 0; idx_clusters < num_clusters; idx_clusters++)
	{
        int i, j, idx_fcs;
        count = fscanf(fid_cluster, "%d%d%d", &i, &j, &idx_fcs);
        PRINT_SCANF_ERROR(count, 3, "Reading error for clusters_order2.");

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
        		int index = idx_clusters*9 + ab;
                fcp_data.ia2[index] = a * atom->N + i;
                fcp_data.jb2[index] = b * atom->N + j;
                fcp_data.phi2[index] = fc[idx_fcs*9 + ab];

                // 2^1-1 = 1 case:
                if (i == j) { fcp_data.phi2[index] /= 2; } // 11
        
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
    }

    fclose(fid_fc);
    fclose(fid_cluster);
    MY_FREE(fc);
}


void FCP::read_fc3(Atom *atom)
{
	// reading fcs_order3
    char file_fc[200];
    strcpy(file_fc, file_path);
    strcat(file_fc, "/fcs_order3");
    FILE *fid_fc = my_fopen(file_fc, "r");
    
    printf("    Reading data from %s\n", file_fc);

    int num_fcs = 0;
    int count = fscanf(fid_fc, "%d", &num_fcs);
    PRINT_SCANF_ERROR(count, 1, "Reading error for fcs_order3.");
    if (num_fcs <= 0)
    {
    	PRINT_INPUT_ERROR("number of force constant matrix should > 0.");
	}

    float *fc;
	MY_MALLOC(fc, float, num_fcs * 27);
    for (int n = 0; n < num_fcs; ++n)
    for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
    for (int c = 0; c < 3; ++c)
    {
        int index = n*27 + a*9 + b*3 + c;
        int aa, bb, cc; // not used
        count = fscanf(fid_fc, "%d%d%d%f", &aa, &bb, &cc, &fc[index]);
        PRINT_SCANF_ERROR(count, 4, "Reading error for fcs_order3.");
    }
		
    // reading clusters_order3
    char file_cluster[200];
    strcpy(file_cluster, file_path);
    strcat(file_cluster, "/clusters_order3");
    FILE *fid_cluster = my_fopen(file_cluster, "r");
    
    printf("    Reading data from %s\n", file_cluster);
    
    int num_clusters = 0;
    count = fscanf(fid_cluster, "%d", &num_clusters);
    PRINT_SCANF_ERROR(count, 1, "Reading error for clusters_order3.");
    number3 = num_clusters * 27;

    CHECK(cudaMallocManaged(&fcp_data.ia3, sizeof(int) * number3));
    CHECK(cudaMallocManaged(&fcp_data.jb3, sizeof(int) * number3));
    CHECK(cudaMallocManaged(&fcp_data.kc3, sizeof(int) * number3));
    CHECK(cudaMallocManaged(&fcp_data.phi3, sizeof(float) * number3));

    for (int idx_clusters = 0; idx_clusters < num_clusters; idx_clusters++)
	{
        int i, j, k, idx_fcs;
        count = fscanf(fid_cluster, "%d%d%d%d", &i, &j, &k, &idx_fcs);
        PRINT_SCANF_ERROR(count, 4, "Reading error for clusters_order3.");
        
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
    MY_FREE(fc);
}


void FCP::read_fc4(Atom *atom)
{
	// reading fcs_order4
    char file_fc[200];
    strcpy(file_fc, file_path);
    strcat(file_fc, "/fcs_order4");
    FILE *fid_fc = my_fopen(file_fc, "r");
    
    printf("    Reading data from %s\n", file_fc);

    int num_fcs = 0;
    int count = fscanf(fid_fc, "%d", &num_fcs);
    PRINT_SCANF_ERROR(count, 1, "Reading error for fcs_order4.");
    if (num_fcs <= 0)
    {
    	PRINT_INPUT_ERROR("number of force constant matrix should > 0.");
	}

    float *fc;
	MY_MALLOC(fc, float, num_fcs * 81);
    for (int n = 0; n < num_fcs; ++n)
    for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
    for (int c = 0; c < 3; ++c)
    for (int d = 0; d < 3; ++d)
    {
        int index = n*81 + a*27 + b*9 + c*3 + d;
        int aa, bb, cc, dd; // not used
        count = fscanf(fid_fc, "%d%d%d%d%f", &aa, &bb, &cc, &dd, &fc[index]);
        PRINT_SCANF_ERROR(count, 5, "Reading error for fcs_order4.");
    }
		
    // reading clusters_order4
    char file_cluster[200];
    strcpy(file_cluster, file_path);
    strcat(file_cluster, "/clusters_order4");
    FILE *fid_cluster = my_fopen(file_cluster, "r");
    
    printf("    Reading data from %s\n", file_cluster);
    
    int num_clusters = 0;
    count = fscanf(fid_cluster, "%d", &num_clusters);
    PRINT_SCANF_ERROR(count, 1, "Reading error for clusters_order4.");
    number4 = num_clusters * 81;

    CHECK(cudaMallocManaged(&fcp_data.ia4, sizeof(int) * number4));
    CHECK(cudaMallocManaged(&fcp_data.jb4, sizeof(int) * number4));
    CHECK(cudaMallocManaged(&fcp_data.kc4, sizeof(int) * number4));
    CHECK(cudaMallocManaged(&fcp_data.ld4, sizeof(int) * number4));
    CHECK(cudaMallocManaged(&fcp_data.phi4, sizeof(float) * number4));

    for (int idx_clusters = 0; idx_clusters < num_clusters; idx_clusters++)
	{
        int i, j, k, l, idx_fcs;
        count = fscanf(fid_cluster, "%d%d%d%d%d", &i, &j, &k, &l, &idx_fcs);
        PRINT_SCANF_ERROR(count, 5, "Reading error for clusters_order4.");
        
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
    MY_FREE(fc);
}


void FCP::read_fc5(Atom *atom)
{
	// reading fcs_order5
    char file_fc[200];
    strcpy(file_fc, file_path);
    strcat(file_fc, "/fcs_order5");
    FILE *fid_fc = my_fopen(file_fc, "r");
    
    printf("    Reading data from %s\n", file_fc);

    int num_fcs = 0;
    int count = fscanf(fid_fc, "%d", &num_fcs);
    PRINT_SCANF_ERROR(count, 1, "Reading error for fcs_order5.");
    if (num_fcs <= 0)
    {
    	PRINT_INPUT_ERROR("number of force constant matrix should > 0.");
	}

    float *fc;
	MY_MALLOC(fc, float, num_fcs * 243);
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
        PRINT_SCANF_ERROR(count, 6, "Reading error for fcs_order5.");
    }
		
    // reading clusters_order5
    char file_cluster[200];
    strcpy(file_cluster, file_path);
    strcat(file_cluster, "/clusters_order5");
    FILE *fid_cluster = my_fopen(file_cluster, "r");
    
    printf("    Reading data from %s\n", file_cluster);
    
    int num_clusters = 0;
    count = fscanf(fid_cluster, "%d", &num_clusters);
    PRINT_SCANF_ERROR(count, 1, "Reading error for clusters_order5.");
    number5 = num_clusters * 243;

    CHECK(cudaMallocManaged(&fcp_data.ia5, sizeof(int) * number5));
    CHECK(cudaMallocManaged(&fcp_data.jb5, sizeof(int) * number5));
    CHECK(cudaMallocManaged(&fcp_data.kc5, sizeof(int) * number5));
    CHECK(cudaMallocManaged(&fcp_data.ld5, sizeof(int) * number5));
    CHECK(cudaMallocManaged(&fcp_data.me5, sizeof(int) * number5));
    CHECK(cudaMallocManaged(&fcp_data.phi5, sizeof(float) * number5));

    for (int idx_clusters = 0; idx_clusters < num_clusters; idx_clusters++)
	{
        int i, j, k, l, m, idx_fcs;
        count = fscanf
		(fid_cluster, "%d%d%d%d%d%d", &i, &j, &k, &l, &m, &idx_fcs);
        PRINT_SCANF_ERROR(count, 6, "Reading error for clusters_order5.");
        
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
    MY_FREE(fc);
}


void FCP::read_fc6(Atom *atom)
{
	// reading fcs_order6
    char file_fc[200];
    strcpy(file_fc, file_path);
    strcat(file_fc, "/fcs_order6");
    FILE *fid_fc = my_fopen(file_fc, "r");
    
    printf("    Reading data from %s\n", file_fc);

    int num_fcs = 0;
    int count = fscanf(fid_fc, "%d", &num_fcs);
    PRINT_SCANF_ERROR(count, 1, "Reading error for fcs_order6.");
    if (num_fcs <= 0)
    {
    	PRINT_INPUT_ERROR("number of force constant matrix should > 0.");
	}

    float *fc;
	MY_MALLOC(fc, float, num_fcs * 729);
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
        PRINT_SCANF_ERROR(count, 7, "Reading error for fcs_order6.");
    }
		
    // reading clusters_order6
    char file_cluster[200];
    strcpy(file_cluster, file_path);
    strcat(file_cluster, "/clusters_order6");
    FILE *fid_cluster = my_fopen(file_cluster, "r");
    
    printf("    Reading data from %s\n", file_cluster);
    
    int num_clusters = 0;
    count = fscanf(fid_cluster, "%d", &num_clusters);
    PRINT_SCANF_ERROR(count, 1, "Reading error for clusters_order6.");
    number6 = num_clusters * 729;

    CHECK(cudaMallocManaged(&fcp_data.ia6, sizeof(int) * number6));
    CHECK(cudaMallocManaged(&fcp_data.jb6, sizeof(int) * number6));
    CHECK(cudaMallocManaged(&fcp_data.kc6, sizeof(int) * number6));
    CHECK(cudaMallocManaged(&fcp_data.ld6, sizeof(int) * number6));
    CHECK(cudaMallocManaged(&fcp_data.me6, sizeof(int) * number6));
    CHECK(cudaMallocManaged(&fcp_data.nf6, sizeof(int) * number6));
    CHECK(cudaMallocManaged(&fcp_data.phi6, sizeof(float) * number6));

    for (int idx_clusters = 0; idx_clusters < num_clusters; idx_clusters++)
	{
        int i, j, k, l, m, n, idx_fcs;
        count = fscanf
		(fid_cluster, "%d%d%d%d%d%d%d", &i, &j, &k, &l, &m, &n, &idx_fcs);
        PRINT_SCANF_ERROR(count, 7, "Reading error for clusters_order6.");
        
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
    MY_FREE(fc);
}


// potential, force, and heat current from the second-order force constants
static __global__ void gpu_find_force_fcp2
(
    int hnemd_compute, real fe_x, real fe_y, real fe_z,
    int N, int number2, int *g_ia2, int *g_jb2, float *g_phi2,
    const float* __restrict__ g_uv, 
    float *g_xij2, float *g_yij2, float *g_zij2, float *g_pfj
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

        float uia = LDG(g_uv, ia); 
        float ujb = LDG(g_uv, jb);
        float via = LDG(g_uv, ia + N * 3);
        float vjb = LDG(g_uv, jb + N * 3);

        int atom_id = ia % N;
        atomicAdd(&g_pfj[atom_id], phi * uia * ujb);
        float fia = - phi * ujb;
        float fjb = - phi * uia;
        if (hnemd_compute)
        {
            float fe_times_rij2 = (fe_x * xij2 + fe_y * yij2 + fe_z * zij2);
            fia = (1.0 - fe_times_rij2) * fia;
            fjb = (1.0 + fe_times_rij2) * fjb;
        }
        atomicAdd(&g_pfj[ia + N], fia);
        atomicAdd(&g_pfj[jb + N], fjb);

        float uvij = via * ujb - uia * vjb;
        atomicAdd(&g_pfj[atom_id + N * 4], phi * xij2 * uvij);
        atomicAdd(&g_pfj[atom_id + N * 5], phi * yij2 * uvij);
        atomicAdd(&g_pfj[atom_id + N * 6], phi * zij2 * uvij);
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


// initialize the local potential (p), force (f), and heat current (j)
static __global__ void gpu_initialize_pfj(int N, float *pfj)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N)
    {
        pfj[n] = 0.0;
        pfj[n + N] = 0.0;
        pfj[n + N * 2] = 0.0;
        pfj[n + N * 3] = 0.0;
        pfj[n + N * 4] = 0.0;
        pfj[n + N * 5] = 0.0;
        pfj[n + N * 6] = 0.0;
    }
}


// get the displacement (u=r-r0) and velocity (v)
static __global__ void gpu_get_uv
(
    int N, double *x, double *y, double *z, 
    double *vx, double *vy, double *vz, float *r0, float *uv
)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N)
    {
        uv[n]         = x[n] - r0[n];
        uv[n + N]     = y[n] - r0[n + N];
        uv[n + N * 2] = z[n] - r0[n + N + N];
        uv[n + N * 3] = vx[n];
        uv[n + N * 4] = vy[n];
        uv[n + N * 5] = vz[n];
    }
}


// save potential (p), force (f), and heat current (j)
static __global__ void gpu_save_pfj
(
    int N, float *pfj, double *p, double *fx, double *fy, double *fz, double *j
)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N)
    {
        p[n]  = pfj[n];                // potential energy
        fx[n] = pfj[n + N];            // fx
        fy[n] = pfj[n + N * 2];        // fy
        fz[n] = pfj[n + N * 3];        // fz
        j[n]  = pfj[n + N * 4];        // jx_in (jx_out = 0)
        j[n + N * 2] = pfj[n + N * 5]; // jy_in (jy_out = 0)
        j[n + N * 4] = pfj[n + N * 6]; // jz
    }
}


// Wrapper of the above kernels
void FCP::compute(Atom *atom, Measure *measure, int potential_number)
{
    const int block_size = 1024;

    gpu_get_uv<<<(atom->N - 1) / block_size + 1, block_size>>>
    (
        atom->N, atom->x, atom->y, atom->z, atom->vx, atom->vy, atom->vz, 
        fcp_data.r0, fcp_data.uv
    );

    gpu_initialize_pfj<<<(atom->N - 1) / block_size + 1, block_size>>>
    (atom->N, fcp_data.pfj);

    gpu_find_force_fcp2<<<(number2 - 1) / block_size + 1, block_size>>>
    (
        measure->hnemd.compute,
        measure->hnemd.fe_x, measure->hnemd.fe_y, measure->hnemd.fe_z,
        atom->N, number2, fcp_data.ia2, fcp_data.jb2, fcp_data.phi2,
        fcp_data.uv, fcp_data.xij2, fcp_data.yij2, fcp_data.zij2, fcp_data.pfj
    );

    if (order >= 3)
    gpu_find_force_fcp3<<<(number3 - 1) / block_size + 1, block_size>>>
    (
        atom->N, number3, fcp_data.ia3, fcp_data.jb3, fcp_data.kc3,
        fcp_data.phi3, fcp_data.uv, fcp_data.pfj
    );

    if (order >= 4)
    gpu_find_force_fcp4<<<(number4 - 1) / block_size + 1, block_size>>>
    (
        atom->N, number4, fcp_data.ia4, fcp_data.jb4, fcp_data.kc4,
        fcp_data.ld4, fcp_data.phi4, fcp_data.uv, fcp_data.pfj
    );

    if (order >= 5)
    gpu_find_force_fcp5<<<(number5 - 1) / block_size + 1, block_size>>>
    (
        atom->N, number5, fcp_data.ia5, fcp_data.jb5, fcp_data.kc5,
        fcp_data.ld5, fcp_data.me5, fcp_data.phi5, fcp_data.uv, fcp_data.pfj
    );

    if (order >= 6)
    gpu_find_force_fcp6<<<(number6 - 1) / block_size + 1, block_size>>>
    (
        atom->N, number6, fcp_data.ia6, fcp_data.jb6, fcp_data.kc6,
        fcp_data.ld6, fcp_data.me6, fcp_data.nf6, fcp_data.phi6, 
        fcp_data.uv, fcp_data.pfj
    ); 

    gpu_save_pfj<<<(atom->N - 1) / block_size + 1, block_size>>>
    (
        atom->N, fcp_data.pfj, atom->potential_per_atom,
        atom->fx, atom->fy, atom->fz, atom->heat_per_atom
    );

    CUDA_CHECK_KERNEL
}


