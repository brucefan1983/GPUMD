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




#include "common.cuh"
#include "gpumd.cuh"
#include "force.cuh"
#include "validate.cuh"
#include "integrate.cuh"
#include "ensemble.cuh" 
#include "measure.cuh"
#include "parse.cuh" 
#include "velocity.cuh"
#include "neighbor.cuh"
#include "memory.cuh"

#define DIM 3
#define NUM_OF_HEAT_COMPONENTS 5




static FILE *my_fopen(const char *filename, const char *mode)
{
    FILE *fid = fopen(filename, mode);
    if (fid == NULL) 
    {
        printf ("Failed to open %s!\n", filename);
        printf ("%s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    return fid;
}




static void print_error (const char *str)
{
    printf("ERROR: %s", str);
    exit(EXIT_FAILURE);
}




GPUMD::GPUMD(char *input_dir)
{ 
    // Data structures:
    Parameters  para;
    CPU_Data    cpu_data;
    Atom    atom;
    Force       force;
    Integrate   integrate;
    Measure     measure(input_dir);

    initialize(input_dir, &para, &cpu_data, &atom);
    run(input_dir, &para, &cpu_data, &atom, &force, &integrate, &measure);
    finalize(&cpu_data, &atom);
}




GPUMD::~GPUMD(void)
{
    // nothing
} 




static void initialize_position
(char *input_dir, Parameters *para, CPU_Data *cpu_data, Atom* atom)
{  
    printf("---------------------------------------------------------------\n");
    printf("INFO:  read in initial positions and related parameters.\n");

    int count = 0;
    char file_xyz[FILE_NAME_LENGTH];
    strcpy(file_xyz, input_dir);
    strcat(file_xyz, "/xyz.in");
    FILE *fid_xyz = my_fopen(file_xyz, "r"); 

    // the first line of the xyz.in file
    double rc;
    count = fscanf(fid_xyz, "%d%d%lf", &para->N, &para->neighbor.MN, &rc);
    if (count != 3) print_error("reading error for line 1 of xyz.in.\n");
    para->neighbor.rc = rc;
    if (para->N < 1)
        print_error("number of atoms should >= 1\n");
    else
        printf("INPUT: number of atoms is %d.\n", para->N);
    
    if (para->neighbor.MN < 0)
        print_error("maximum number of neighbors should >= 0\n");
    else
        printf("INPUT: maximum number of neighbors is %d.\n",para->neighbor.MN);

    if (para->neighbor.rc < 0)
        print_error("initial cutoff for neighbor list should >= 0\n");
    else
        printf
        (
            "INPUT: initial cutoff for neighbor list is %g A.\n", 
            para->neighbor.rc
        );    

    // now we have enough information to allocate memroy for the major data
    MY_MALLOC(cpu_data->type,       int, para->N);
    MY_MALLOC(cpu_data->type_local, int, para->N);
    MY_MALLOC(cpu_data->label,      int, para->N);
    MY_MALLOC(atom->cpu_mass, real, para->N);
    MY_MALLOC(atom->cpu_x,    real, para->N);
    MY_MALLOC(atom->cpu_y,    real, para->N);
    MY_MALLOC(atom->cpu_z,    real, para->N);
    MY_MALLOC(atom->cpu_box_length, real, 3);
    MY_MALLOC(atom->cpu_box_matrix, real, 9);
    MY_MALLOC(atom->cpu_box_matrix_inv, real, 9);

#ifdef TRICLINIC

    // second line: boundary conditions
    count = fscanf
    (fid_xyz, "%d%d%d", &(para->pbc_x), &(para->pbc_y), &(para->pbc_z));
    if (count != 3) print_error("reading error for line 2 of xyz.in.\n");

    // third line: triclinic box parameters
    double box[9];   
    count = fscanf
    (
        fid_xyz, "%lf%lf%lf%lf%lf%lf%lf%lf%lf", &box[0], &box[1], &box[2], 
        &box[3], &box[4], &box[5], &box[6], &box[7], &box[8]
    ); 
    if (count != 9) print_error("reading error for line 3 of xyz.in.\n");
    for (int n = 0; n < 9; ++n) atom->cpu_box_matrix[n] = box[n];

    real volume = atom->cpu_box_matrix[0]
                * atom->cpu_box_matrix[4]
                * atom->cpu_box_matrix[8] 
                + atom->cpu_box_matrix[1]
                * atom->cpu_box_matrix[5]
                * atom->cpu_box_matrix[6] 
                + atom->cpu_box_matrix[2]
                * atom->cpu_box_matrix[3]
                * atom->cpu_box_matrix[7]
                - atom->cpu_box_matrix[2]
                * atom->cpu_box_matrix[4]
                * atom->cpu_box_matrix[6] 
                - atom->cpu_box_matrix[1]
                * atom->cpu_box_matrix[3]
                * atom->cpu_box_matrix[8] 
                - atom->cpu_box_matrix[0]
                * atom->cpu_box_matrix[5]
                * atom->cpu_box_matrix[7];

    atom->cpu_box_matrix_inv[0] = atom->cpu_box_matrix[4]
                                * atom->cpu_box_matrix[8] 
                                - atom->cpu_box_matrix[5]
                                * atom->cpu_box_matrix[7];
    atom->cpu_box_matrix_inv[1] = atom->cpu_box_matrix[2]
                                * atom->cpu_box_matrix[7] 
                                - atom->cpu_box_matrix[1]
                                * atom->cpu_box_matrix[8];
    atom->cpu_box_matrix_inv[2] = atom->cpu_box_matrix[1]
                                * atom->cpu_box_matrix[5] 
                                - atom->cpu_box_matrix[2]
                                * atom->cpu_box_matrix[4];
    atom->cpu_box_matrix_inv[3] = atom->cpu_box_matrix[5]
                                * atom->cpu_box_matrix[6] 
                                - atom->cpu_box_matrix[3]
                                * atom->cpu_box_matrix[8];
    atom->cpu_box_matrix_inv[4] = atom->cpu_box_matrix[0]
                                * atom->cpu_box_matrix[8] 
                                - atom->cpu_box_matrix[2]
                                * atom->cpu_box_matrix[6];
    atom->cpu_box_matrix_inv[5] = atom->cpu_box_matrix[2]
                                * atom->cpu_box_matrix[3] 
                                - atom->cpu_box_matrix[0]
                                * atom->cpu_box_matrix[5];
    atom->cpu_box_matrix_inv[6] = atom->cpu_box_matrix[3]
                                * atom->cpu_box_matrix[7] 
                                - atom->cpu_box_matrix[4]
                                * atom->cpu_box_matrix[6];
    atom->cpu_box_matrix_inv[7] = atom->cpu_box_matrix[1]
                                * atom->cpu_box_matrix[6] 
                                - atom->cpu_box_matrix[0]
                                * atom->cpu_box_matrix[7];
    atom->cpu_box_matrix_inv[8] = atom->cpu_box_matrix[0]
                                * atom->cpu_box_matrix[4] 
                                - atom->cpu_box_matrix[1]
                                * atom->cpu_box_matrix[3];

    for (int n = 0; n < 9; n++) atom->cpu_box_matrix_inv[n] /= volume;

#else // #ifdef TRICLINIC

    // the second line of the xyz.in file (boundary conditions and box size)
    double lx, ly, lz;
    count = fscanf
    (
        fid_xyz, "%d%d%d%lf%lf%lf", 
        &(para->pbc_x), &(para->pbc_y), &(para->pbc_z), &lx, &ly, &lz
    );
    if (count != 6) print_error("reading error for line 2 of xyz.in.\n");
    atom->cpu_box_length[0] = lx;
    atom->cpu_box_length[1] = ly;
    atom->cpu_box_length[2] = lz;

#endif // #ifdef TRICLINIC

    if (para->pbc_x == 1)
        printf("INPUT: use periodic boundary conditions along x.\n");
    else if (para->pbc_x == 0)
        printf("INPUT: use     free boundary conditions along x.\n");
    else
        print_error("invalid boundary conditions along x.\n");

    if (para->pbc_y == 1)
        printf("INPUT: use periodic boundary conditions along y.\n");
    else if (para->pbc_y == 0)
        printf("INPUT: use     free boundary conditions along y.\n");
    else
        print_error("invalid boundary conditions along y.\n");

    if (para->pbc_z == 1)
        printf("INPUT: use periodic boundary conditions along z.\n");
    else if (para->pbc_z == 0)
        printf("INPUT: use     free boundary conditions along z.\n");
    else
        print_error("invalid boundary conditions along z.\n");

    // the remaining lines in the xyz.in file (type, label, mass, and positions)
    int max_label = -1; // used to determine the number of groups
    int max_type = -1; // used to determine the number of types
    for (int n = 0; n < para->N; n++) 
    {
        double mass, x, y, z;
        count = fscanf
        (
            fid_xyz, "%d%d%lf%lf%lf%lf", 
            &(cpu_data->type[n]), &(cpu_data->label[n]), &mass, &x, &y, &z
        );
        if (count != 6) print_error("reading error for xyz.in.\n");
        atom->cpu_mass[n] = mass;
        atom->cpu_x[n] = x;
        atom->cpu_y[n] = y;
        atom->cpu_z[n] = z;

        if (cpu_data->label[n] > max_label)
            max_label = cpu_data->label[n];

        if (cpu_data->type[n] > max_type)
            max_type = cpu_data->type[n];

        // copy
        cpu_data->type_local[n] = cpu_data->type[n];
    }

    fclose(fid_xyz);

    // number of groups determined
    para->number_of_groups = max_label + 1;
    if (para->number_of_groups == 1)
        printf("INPUT: there is only one group of atoms.\n");
    else
        printf("INPUT: there are %d groups of atoms.\n",para->number_of_groups);

    // determine the number of atoms in each group
    MY_MALLOC(cpu_data->group_size, int, para->number_of_groups);
    MY_MALLOC(cpu_data->group_size_sum, int, para->number_of_groups);
    for (int m = 0; m < para->number_of_groups; m++)
    {
        cpu_data->group_size[m] = 0;
        cpu_data->group_size_sum[m] = 0;
    }
    for (int n = 0; n < para->N; n++) 
        cpu_data->group_size[cpu_data->label[n]]++;
    for (int m = 0; m < para->number_of_groups; m++)
        printf("       %d atoms in group %d.\n", cpu_data->group_size[m], m);   
    
    // calculate the number of atoms before a group
    for (int m = 1; m < para->number_of_groups; m++)
        for (int n = 0; n < m; n++)
            cpu_data->group_size_sum[m] += cpu_data->group_size[n];

    // determine the atom indices from the first to the last group
    MY_MALLOC(cpu_data->group_contents, int, para->N);
    int *offset;
    MY_MALLOC(offset, int, para->number_of_groups);
    for (int m = 0; m < para->number_of_groups; m++) offset[m] = 0;
    for (int n = 0; n < para->N; n++) 
        for (int m = 0; m < para->number_of_groups; m++)
            if (cpu_data->label[n] == m)
            {
                cpu_data->group_contents[cpu_data->group_size_sum[m]+offset[m]] 
                    = n;
                offset[m]++;
            }
    MY_FREE(offset);

    // number of types determined
    para->number_of_types = max_type + 1;
    if (para->number_of_types == 1)
        printf("INPUT: there is only one atom type.\n");
    else
        printf("INPUT: there are %d atom types.\n", para->number_of_types);

    // determine the number of atoms in each type
    MY_MALLOC(cpu_data->type_size, int, para->number_of_types);
    for (int m = 0; m < para->number_of_types; m++)
        cpu_data->type_size[m] = 0;
    for (int n = 0; n < para->N; n++) 
        cpu_data->type_size[cpu_data->type[n]]++;
    for (int m = 0; m < para->number_of_types; m++)
        printf("       %d atoms of type %d.\n", cpu_data->type_size[m], m); 

    printf("INFO:  positions and related parameters initialized.\n");
    printf("---------------------------------------------------------------\n");
    printf("\n");
}




static void allocate_memory_gpu(Parameters *para, Atom *atom)
{
    // memory amount
    int m1 = sizeof(int) * para->N;
    int m2 = m1 * para->neighbor.MN;
    int m3 = sizeof(int) * para->number_of_groups;
    int m4 = sizeof(real) * para->N;
    int m5 = m4 * NUM_OF_HEAT_COMPONENTS;

    // for indexing
    CHECK(cudaMalloc((void**)&atom->NN, m1)); 
    CHECK(cudaMalloc((void**)&atom->NL, m2)); 
#ifndef FIXED_NL
    CHECK(cudaMalloc((void**)&atom->NN_local, m1)); 
    CHECK(cudaMalloc((void**)&atom->NL_local, m2));
#endif
    CHECK(cudaMalloc((void**)&atom->type, m1));  
    CHECK(cudaMalloc((void**)&atom->type_local, m1));
    CHECK(cudaMalloc((void**)&atom->label, m1)); 
    CHECK(cudaMalloc((void**)&atom->group_size, m3)); 
    CHECK(cudaMalloc((void**)&atom->group_size_sum, m3));
    CHECK(cudaMalloc((void**)&atom->group_contents, m1));

    // for atoms
    CHECK(cudaMalloc((void**)&atom->mass, m4));
    CHECK(cudaMalloc((void**)&atom->x0,   m4));
    CHECK(cudaMalloc((void**)&atom->y0,   m4));
    CHECK(cudaMalloc((void**)&atom->z0,   m4));
    CHECK(cudaMalloc((void**)&atom->x,    m4));
    CHECK(cudaMalloc((void**)&atom->y,    m4));
    CHECK(cudaMalloc((void**)&atom->z,    m4));
    CHECK(cudaMalloc((void**)&atom->vx,   m4));
    CHECK(cudaMalloc((void**)&atom->vy,   m4));
    CHECK(cudaMalloc((void**)&atom->vz,   m4));
    CHECK(cudaMalloc((void**)&atom->fx,   m4));
    CHECK(cudaMalloc((void**)&atom->fy,   m4));
    CHECK(cudaMalloc((void**)&atom->fz,   m4));

    CHECK(cudaMalloc((void**)&atom->heat_per_atom, m5));

    // per-atom stress and potential energy, which are always needed
    CHECK(cudaMalloc((void**)&atom->virial_per_atom_x,  m4));
    CHECK(cudaMalloc((void**)&atom->virial_per_atom_y,  m4));
    CHECK(cudaMalloc((void**)&atom->virial_per_atom_z,  m4));
    CHECK(cudaMalloc((void**)&atom->potential_per_atom, m4));

    // box lengths
    CHECK(cudaMalloc((void**)&atom->box_matrix,     sizeof(real) * 9));
    CHECK(cudaMalloc((void**)&atom->box_matrix_inv, sizeof(real) * 9));
    CHECK(cudaMalloc((void**)&atom->box_length, sizeof(real) * DIM));

    // 6 thermodynamic quantities
    CHECK(cudaMalloc((void**)&atom->thermo, sizeof(real) * 6));

}




static void copy_from_cpu_to_gpu
(Parameters *para, CPU_Data *cpu_data, Atom *atom)
{
    int m1 = sizeof(int) * para->N;
    int m2 = sizeof(int) * para->number_of_groups;
    int m3 = sizeof(real) * para->N;
    int m4 = sizeof(real) * DIM;

    cudaMemcpy(atom->type, cpu_data->type, m1, cudaMemcpyHostToDevice); 
    cudaMemcpy
    (atom->type_local, cpu_data->type, m1, cudaMemcpyHostToDevice);
    cudaMemcpy(atom->label, cpu_data->label, m1, cudaMemcpyHostToDevice); 

    cudaMemcpy
    (atom->group_size, cpu_data->group_size, m2, cudaMemcpyHostToDevice);
    cudaMemcpy
    (
        atom->group_size_sum, cpu_data->group_size_sum, m2, 
        cudaMemcpyHostToDevice
    );
    cudaMemcpy
    (
        atom->group_contents, cpu_data->group_contents, m1, 
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(atom->mass, atom->cpu_mass, m3, cudaMemcpyHostToDevice);
    cudaMemcpy(atom->x, atom->cpu_x, m3, cudaMemcpyHostToDevice); 
    cudaMemcpy(atom->y, atom->cpu_y, m3, cudaMemcpyHostToDevice); 
    cudaMemcpy(atom->z, atom->cpu_z, m3, cudaMemcpyHostToDevice);

    cudaMemcpy
    (
        atom->box_matrix, atom->cpu_box_matrix, 
        9 * sizeof(real), cudaMemcpyHostToDevice
    );
    cudaMemcpy
    (
        atom->box_matrix_inv, atom->cpu_box_matrix_inv, 
        9 * sizeof(real), cudaMemcpyHostToDevice
    );
    cudaMemcpy
    (atom->box_length, atom->cpu_box_length, m4, cudaMemcpyHostToDevice);
}




void GPUMD::initialize
(char *input_dir, Parameters *para, CPU_Data *cpu_data, Atom *atom)
{ 
    initialize_position(input_dir, para, cpu_data, atom);
    allocate_memory_gpu(para, atom);
    copy_from_cpu_to_gpu(para, cpu_data, atom);

    // build the initial neighbor list
    int is_first = 1;
    find_neighbor(para, cpu_data, atom, is_first);
}




void GPUMD::finalize(CPU_Data *cpu_data, Atom *atom)
{
    // Free the memory allocated on the GPU
    CHECK(cudaFree(atom->NN)); 
    CHECK(cudaFree(atom->NL)); 
    CHECK(cudaFree(atom->NN_local)); 
    CHECK(cudaFree(atom->NL_local));
    CHECK(cudaFree(atom->type));  
    CHECK(cudaFree(atom->type_local));
    CHECK(cudaFree(atom->label)); 
    CHECK(cudaFree(atom->group_size)); 
    CHECK(cudaFree(atom->group_size_sum));
    CHECK(cudaFree(atom->group_contents));
    CHECK(cudaFree(atom->mass));
    CHECK(cudaFree(atom->x0));  
    CHECK(cudaFree(atom->y0));  
    CHECK(cudaFree(atom->z0));
    CHECK(cudaFree(atom->x));  
    CHECK(cudaFree(atom->y));  
    CHECK(cudaFree(atom->z));
    CHECK(cudaFree(atom->vx)); 
    CHECK(cudaFree(atom->vy)); 
    CHECK(cudaFree(atom->vz));
    CHECK(cudaFree(atom->fx)); 
    CHECK(cudaFree(atom->fy)); 
    CHECK(cudaFree(atom->fz));
    CHECK(cudaFree(atom->virial_per_atom_x));
    CHECK(cudaFree(atom->virial_per_atom_y));
    CHECK(cudaFree(atom->virial_per_atom_z));
    CHECK(cudaFree(atom->potential_per_atom));
    CHECK(cudaFree(atom->heat_per_atom));    
    //#ifdef TRICLINIC
    CHECK(cudaFree(atom->box_matrix));
    CHECK(cudaFree(atom->box_matrix_inv));
    //#else
    CHECK(cudaFree(atom->box_length));
    //#endif
    CHECK(cudaFree(atom->thermo));

    // Free the major memory allocated on the CPU
    MY_FREE(cpu_data->type);
    MY_FREE(cpu_data->type_local);
    MY_FREE(cpu_data->label);
    MY_FREE(cpu_data->group_size);
    MY_FREE(cpu_data->group_size_sum);
    MY_FREE(cpu_data->group_contents);
    MY_FREE(cpu_data->type_size);
    MY_FREE(atom->cpu_mass);
    MY_FREE(atom->cpu_x);
    MY_FREE(atom->cpu_y);
    MY_FREE(atom->cpu_z);
    MY_FREE(atom->cpu_box_length);
    MY_FREE(atom->cpu_box_matrix);
    MY_FREE(atom->cpu_box_matrix_inv);
}




/*----------------------------------------------------------------------------80
    run a number of steps for a given set of inputs
------------------------------------------------------------------------------*/
static void process_run 
(
    char **param, 
    unsigned int num_param, 
    char *input_dir,  
    Parameters *para, 
    CPU_Data *cpu_data,
    Atom *atom,
    Force *force,
    Integrate *integrate,
    Measure *measure
)
{
    integrate->initialize(para, cpu_data); 
    measure->initialize(para, cpu_data, atom);

    // record the starting time for this run
    clock_t time_begin = clock();

    // Now, start to run!
    for (int step = 0; step < para->number_of_steps; ++step)
    {  
        // update the neighbor list
        if (para->neighbor.update)
        {
            find_neighbor(para, cpu_data, atom, 0);
        }

        // set the current temperature;
        if (integrate->ensemble->type >= 1 && integrate->ensemble->type <= 20)
        {
            integrate->ensemble->temperature = para->temperature1 
                + (para->temperature2 - para->temperature1)
                * real(step) / para->number_of_steps;   
        }

        // integrate by one time-step:
        integrate->compute(para, cpu_data, atom, force, measure);

        // measure
        measure->compute(input_dir, para, cpu_data, atom, integrate, step);

        if (para->number_of_steps >= 10)
        {
            if ((step + 1) % (para->number_of_steps / 10) == 0)
            {
                printf("INFO:  %d steps completed.\n", step + 1);
            }
        }
    }
    
    // only for myself
    if (0)
    {
        validate_force(force, para, cpu_data, atom, measure);
    }

    printf("INFO:  This run is completed.\n\n");

    // report the time used for this run and its speed:
    clock_t time_finish = clock();
    real time_used = (time_finish - time_begin) / (real) CLOCKS_PER_SEC;
    printf("INFO:  Time used for this run = %g s.\n", time_used);
    real run_speed = para->N * (para->number_of_steps / time_used);
    printf("INFO:  Speed of this run = %g atom*step/second.\n\n", run_speed);

    measure->finalize(input_dir, para, cpu_data, atom, integrate);
    integrate->finalize();
}




/*----------------------------------------------------------------------------80
    set some default values after each run
------------------------------------------------------------------------------*/
static void initialize_run(Parameters *para, Measure* measure)
{
    para->neighbor.update = 0;
    measure->heat.sample     = 0;
    measure->shc.compute     = 0;
    measure->vac.compute     = 0;
    measure->hac.compute     = 0;
    measure->hnemd.compute   = 0;
    para->fixed_group     = -1; // no group has an index of -1
}




/*----------------------------------------------------------------------------80
	Read the input file to memory in the beginning, because
	we do not want to keep the FILE handle open all the time
------------------------------------------------------------------------------*/
static char *get_file_contents (char *filename)
{

    char *contents;
    int contents_size;
    FILE *in = my_fopen(filename, "r");

    // Find file size
    fseek(in, 0, SEEK_END);
    contents_size = ftell(in);
    rewind(in);

    MY_MALLOC(contents, char, contents_size + 1);
    int size_read_in = fread(contents, sizeof(char), contents_size, in);
    if (size_read_in != contents_size)
    {
        print_error ("File size mismatch.");
    }

    fclose(in);
    contents[contents_size] = '\0'; // Assures proper null termination

    return contents;
}




/*----------------------------------------------------------------------------80
	Parse a single row
------------------------------------------------------------------------------*/
static char *row_find_param (char *s, char *param[], int *num_param)
{
    *num_param = 0;
    int start_new_word = 1, comment_found = 0;
    if (s == NULL) return NULL;

    while(*s)
    {
        if(*s == '\n')
        {
            *s = '\0';
            return s + sizeof(char);
        }
        else if (comment_found)
        {
            // Do nothing
        }
        else if (*s == '#')
        {
            *s = '\0';
            comment_found = 1;
        }
        else if(isspace(*s))
        {
            *s = '\0';
            start_new_word = 1;
        }
        else if (start_new_word)
        {
            param[*num_param] = s;
            ++(*num_param);
            start_new_word = 0;			
        }
        ++s;
    }
    return NULL;
}




/*----------------------------------------------------------------------------80
    Read and process the inputs from the "run.in" file.
------------------------------------------------------------------------------*/
void GPUMD::run
(
    char *input_dir,  
    Parameters *para,
    CPU_Data *cpu_data, 
    Atom *atom,
    Force *force,
    Integrate *integrate,
    Measure *measure 
)
{
    char file_run[FILE_NAME_LENGTH];
    strcpy(file_run, input_dir);
    strcat(file_run, "/run.in");
    char *input = get_file_contents(file_run);
    char *input_ptr = input; // Keep the pointer in order to free later

    // Iterate the rows
    const int max_num_param = 10; // never use more than 9 parameters
    int num_param;
    char *param[max_num_param];

    initialize_run(para, measure); // set some default values before the first run

    while (input_ptr)
    {
        // get one line from the input file
        input_ptr = row_find_param(input_ptr, param, &num_param);
        if (num_param == 0) { continue; } 

        // set default values
        int is_potential = 0;
        int is_velocity = 0;
        int is_run = 0;

        // parse a line of the input file 
        parse
        (
            param, num_param, para, force, integrate, measure,
            &is_potential, &is_velocity, &is_run
        );

        // check for some special keywords
        if (is_potential) 
        {  
            force->initialize(input_dir, para, cpu_data, atom);
            force->compute(para, atom, measure);
            #ifdef FORCE
            // output the initial forces (for lattice dynamics calculations)
            int m = sizeof(real) * para->N;
            real *cpu_fx = cpu_data->fx;
            real *cpu_fy = cpu_data->fy;
            real *cpu_fz = cpu_data->fz;
            CHECK(cudaMemcpy(cpu_fx, atom->fx, m, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(cpu_fy, atom->fy, m, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(cpu_fz, atom->fz, m, cudaMemcpyDeviceToHost));
            char file_force[FILE_NAME_LENGTH];
            strcpy(file_force, input_dir);
            strcat(file_force, "/f.out");
            FILE *fid_force = my_fopen(file_force, "w");
            for (int n = 0; n < para->N; n++)
            {
                fprintf
                (
                    fid_force, "%20.10e%20.10e%20.10e\n", 
                    cpu_fx[n], cpu_fy[n], cpu_fz[n]
                );
            }
            fflush(fid_force);
            fclose(fid_force);
            #endif
        }
        if (is_velocity)  
        {
            process_velocity(para, atom);
        }
        if (is_run)
        { 
            process_run
            (
                param, num_param, input_dir, para, cpu_data, atom, 
                force, integrate, measure
            );
            
            initialize_run(para, measure); // change back to the default
        }
    }

    MY_FREE(input); // Free the input file contents
}




