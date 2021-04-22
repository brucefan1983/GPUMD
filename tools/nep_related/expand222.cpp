/*
    Purpose:
        make the cells in train.in 2*2*2 times larger
    Compile:
        g++ -O3 expand222.cpp
    Run:
        ./a.out
    Author:
        Zheyong Fan <brucenju(at)gmail.com>
*/

#include <string.h>
#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>

#define PRINT_ERROR(count, n)                            \
    {                                                    \
        if (count != n)                                  \
        {                                                \
            fprintf(stderr, "Reading error:\n");         \
            fprintf(stderr, "    File: %s\n", __FILE__); \
            fprintf(stderr, "    Line: %d\n", __LINE__); \
            exit(1);                                     \
        }                                                \
    }

FILE *my_fopen(const char *filename, const char *mode)
{
    FILE *fid = fopen(filename, mode);
    if (fid == NULL)
    {
        printf("Failed to open %s!\n", filename);
        printf("%s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    return fid;
}

int main()
{
    // open files
    FILE *fid_train_in = my_fopen("train.in", "r");
    FILE *fid_train_out = my_fopen("train.out", "w");

    // first line: number of configurations
    int Nc, Nc_force;
    int count = fscanf(fid_train_in, "%d%d", &Nc, &Nc_force);
    PRINT_ERROR(count, 2);
    fprintf(fid_train_out, "%d %d\n", Nc, Nc_force);

    // the next Nc lines
    std::vector<int> Na(Nc);
    for (int n = 0; n < Nc; ++n)
    {
        int count = fscanf(fid_train_in, "%d", &Na[n]);
        PRINT_ERROR(count, 1);
        fprintf(fid_train_out, "%d\n", Na[n] * 8);
    }

    // force configurations
    for (int nf = 0; nf < Nc_force; ++nf)
    {
        // box
        float box[9];
        for (int k = 0; k < 9; ++k)
        {
            int count = fscanf(fid_train_in, "%f", &box[k]);
            PRINT_ERROR(count, 1);
            fprintf(fid_train_out, "%g ", box[k] * 2);
        }
        fprintf(fid_train_out, "\n");

        // type, position, and force
        int type;
        float r[3], f[3];
        for (int na = 0; na < Na[nf]; ++na)
        {
            count = fscanf(
                fid_train_in, "%d%f%f%f%f%f%f", &type, &r[0], &r[1], &r[2],
                &f[0], &f[1], &f[2]);
            PRINT_ERROR(count, 7);

            for (int na = 0; na < 2; ++na)
                for (int nb = 0; nb < 2; ++nb)
                    for (int nc = 0; nc < 2; ++nc)
                    {
                        float dx = box[0] * na + box[3] * nb + box[6] * nc;
                        float dy = box[1] * na + box[4] * nb + box[7] * nc;
                        float dz = box[2] * na + box[5] * nb + box[8] * nc;

                        fprintf(
                            fid_train_out, "%d %g %g %g %g %g %g\n", type,
                            r[0] + dx, r[1] + dy, r[2] + dz, f[0], f[1], f[2]);
                    }
        }
    }

    // energy/virial configurations
    for (int ne = Nc_force; ne < Nc; ++ne)
    {
        // energy and virial
        float energy_virial[7];
        for (int k = 0; k < 7; ++k)
        {
            int count = fscanf(fid_train_in, "%f", &energy_virial[k]);
            PRINT_ERROR(count, 1);
            fprintf(fid_train_out, "%g ", energy_virial[k] * 8);
        }
        fprintf(fid_train_out, "\n");

        // box
        float box[9];
        for (int k = 0; k < 9; ++k)
        {
            int count = fscanf(fid_train_in, "%f", &box[k]);
            PRINT_ERROR(count, 1);
            fprintf(fid_train_out, "%g ", box[k] * 2);
        }
        fprintf(fid_train_out, "\n");

        // type and positon
        int type;
        float r[3];
        for (int na = 0; na < Na[ne]; ++na)
        {
            count = fscanf(
                fid_train_in, "%d%f%f%f", &type, &r[0], &r[1], &r[2]);
            PRINT_ERROR(count, 4);

            for (int na = 0; na < 2; ++na)
                for (int nb = 0; nb < 2; ++nb)
                    for (int nc = 0; nc < 2; ++nc)
                    {
                        float dx = box[0] * na + box[3] * nb + box[6] * nc;
                        float dy = box[1] * na + box[4] * nb + box[7] * nc;
                        float dz = box[2] * na + box[5] * nb + box[8] * nc;

                        fprintf(
                            fid_train_out, "%d %g %g %g\n", type,
                            r[0] + dx, r[1] + dy, r[2] + dz);
                    }
        }
    }

    // close files
    fclose(fid_train_in);
    fclose(fid_train_out);
    return 0;
}
