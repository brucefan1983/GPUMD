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
Dump atom positions using AMBER conventions for NetCDF files. Additional
readers, such as VMD, OVITO, and ASE, can read/visualize the outputs.

Contributing author: Alexander Gabourie (Stanford University)

Code was written for NetCDF version 4.6.3 and the style was influenced by
LAMMPS' implementation by Lars Pastewka (University of Freiburg). The netCDF
documentation used can be found here:
https://www.unidata.ucar.edu/software/netcdf/docs/index.html
and the AMBER conventions followed are here:
http://ambermd.org/netcdf/nctraj.xhtml
------------------------------------------------------------------------------*/

#ifdef USE_NETCDF

#include "dump_netcdf.cuh"
#include "model/box.cuh"
#include "netcdf.h"
#include "parse_utilities.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include <unistd.h>

const int FILE_NAME_LENGTH = 200;
#define GPUMD_VERSION "2.5"

/* Handle errors by printing an error message and exiting with a
 * non-zero status. */
#define ERRCODE 2
#define ERR(e)                                                                                     \
  {                                                                                                \
    printf("Error: %s\n", nc_strerror(e));                                                         \
    exit(ERRCODE);                                                                                 \
  }
#define NC_CHECK(s)                                                                                \
  {                                                                                                \
    if (s != NC_NOERR)                                                                             \
      ERR(s);                                                                                      \
  }

const char SPATIAL_STR[] = "spatial";
const char FRAME_STR[] = "frame";
const char ATOM_STR[] = "atom";
const char CELL_SPATIAL_STR[] = "cell_spatial";
const char CELL_ANGULAR_STR[] = "cell_angular";
const char LABEL_STR[] = "label";
const char TIME_STR[] = "time";
const char COORDINATES_STR[] = "coordinates";
const char TYPE_STR[] = "type";
const char CELL_LENGTHS_STR[] = "cell_lengths";
const char CELL_ANGLES_STR[] = "cell_angles";
const char UNITS_STR[] = "units";
bool DUMP_NETCDF::append = false;

void DUMP_NETCDF::parse(char** param, int num_param)
{
  dump_ = true;
  printf("Dump positions in netCDF format.\n");

  if (num_param < 2) {
    PRINT_INPUT_ERROR("dump_netcdf should have at least 1 parameter.");
  }
  if (num_param > 4) {
    PRINT_INPUT_ERROR("dump_netcdf has too many parameters.");
  }

  if (!is_valid_int(param[1], &interval)) {
    PRINT_INPUT_ERROR("position dump interval should be an integer.");
  }
  if (dump_interval_ <= 0) {
    PRINT_INPUT_ERROR("position dump interval should > 0.");
  }
  printf("    every %d steps.\n", interval);

  for (int k = 2; k < num_param; k++) {
    if (strcmp(param[k], "precision") == 0) {
      parse_precision(param, num_param, k, precision);
    } else {
      PRINT_INPUT_ERROR("Unrecognized argument in dump_netcdf.\n");
    }
  }

  if (precision == 1) {
    printf("Note: Single precision netCDF output does not follow AMBER conventions.\n"
           "      However, it will still work for many readers.\n");
  }
}

void DUMP_NETCDF::preprocess(char* input_dir, const int N)
{
  if (!dump_)
    return;

  strcpy(file_position, input_dir);
  strcat(file_position, "/movie.nc");

  // find appropriate file name
  // Append if same simulation, new file otherwise
  bool done = false;
  char filename[FILE_NAME_LENGTH];
  int filenum = 1;
  while (!done) {

    if (access(file_position, F_OK) != -1) {
      strcpy(file_position, input_dir);
      filenum++;
      sprintf(filename, "/movie_%d.nc", filenum);
      strcat(file_position, filename);
    } else {
      done = true;
    }
  }

  if (append) {
    if (filenum == 2) {
      strcpy(file_position, input_dir);
      strcat(file_position, "/movie.nc");
    } else {
      strcpy(file_position, input_dir);
      sprintf(filename, "/movie_%d.nc", filenum - 1);
      strcat(file_position, filename);
    }
    return; // creation of file & other info not needed
  }

  // create file (automatically placed in 'define' mode)
  NC_CHECK(nc_create(file_position, NC_64BIT_OFFSET, &ncid));

  // Global attributes
  NC_CHECK(nc_put_att_text(ncid, NC_GLOBAL, "program", 5, "GPUMD"));
  NC_CHECK(
    nc_put_att_text(ncid, NC_GLOBAL, "programVersion", strlen(GPUMD_VERSION), GPUMD_VERSION));
  NC_CHECK(nc_put_att_text(ncid, NC_GLOBAL, "Conventions", 5, "AMBER"));
  NC_CHECK(nc_put_att_text(ncid, NC_GLOBAL, "ConventionVersion", 3, "1.0"));

  // dimensions
  NC_CHECK(nc_def_dim(ncid, FRAME_STR, NC_UNLIMITED, &frame_dim));
  NC_CHECK(nc_def_dim(ncid, SPATIAL_STR, 3, &spatial_dim));
  NC_CHECK(nc_def_dim(ncid, ATOM_STR, N, &atom_dim));
  NC_CHECK(nc_def_dim(ncid, CELL_SPATIAL_STR, 3, &cell_spatial_dim));
  NC_CHECK(nc_def_dim(ncid, CELL_ANGULAR_STR, 3, &cell_angular_dim));
  NC_CHECK(nc_def_dim(ncid, LABEL_STR, 10, &label_dim));

  // Label variables
  int dimids[3];
  dimids[0] = spatial_dim;
  NC_CHECK(nc_def_var(ncid, SPATIAL_STR, NC_CHAR, 1, dimids, &spatial_var));
  dimids[0] = cell_spatial_dim;
  NC_CHECK(nc_def_var(ncid, CELL_SPATIAL_STR, NC_CHAR, 1, dimids, &cell_spatial_var));
  dimids[0] = cell_angular_dim;
  dimids[1] = label_dim;
  NC_CHECK(nc_def_var(ncid, CELL_ANGULAR_STR, NC_CHAR, 2, dimids, &cell_angular_var));

  // Data variables
  dimids[0] = frame_dim;
  NC_CHECK(nc_def_var(ncid, TIME_STR, NC_DOUBLE, 1, dimids, &time_var));
  dimids[1] = cell_spatial_dim;
  NC_CHECK(nc_def_var(ncid, CELL_LENGTHS_STR, NC_DOUBLE, 2, dimids, &cell_lengths_var));
  dimids[1] = cell_angular_dim;
  NC_CHECK(nc_def_var(ncid, CELL_ANGLES_STR, NC_DOUBLE, 2, dimids, &cell_angles_var));

  // More extensive data variables (type, coordinates)
  dimids[0] = frame_dim;
  dimids[1] = atom_dim;
  dimids[2] = spatial_dim;

  if (precision == 1) // single precision
  {
    NC_CHECK(nc_def_var(ncid, COORDINATES_STR, NC_FLOAT, 3, dimids, &coordinates_var));
  } else {
    NC_CHECK(nc_def_var(ncid, COORDINATES_STR, NC_DOUBLE, 3, dimids, &coordinates_var));
  }
  NC_CHECK(nc_def_var(ncid, TYPE_STR, NC_INT, 2, dimids, &type_var));

  // Units
  NC_CHECK(nc_put_att_text(ncid, time_var, UNITS_STR, 10, "picosecond"));
  NC_CHECK(nc_put_att_text(ncid, cell_lengths_var, UNITS_STR, 8, "angstrom"));
  NC_CHECK(nc_put_att_text(ncid, coordinates_var, UNITS_STR, 8, "angstrom"));
  NC_CHECK(nc_put_att_text(ncid, cell_angles_var, UNITS_STR, 6, "degree"));

  // Definitions are complete -> leave define mode
  NC_CHECK(nc_enddef(ncid));

  // Write the Label Variables
  NC_CHECK(nc_put_var_text(ncid, spatial_var, "xyz"));
  NC_CHECK(nc_put_var_text(ncid, cell_spatial_var, "abc"));
  // 2D data, startp defines index for data to start
  //  countp determines the size of data in each dimension
  size_t startp[2] = {0, 0};
  size_t countp[2] = {1, 5};
  NC_CHECK(nc_put_vara_text(ncid, cell_angular_var, startp, countp, "alpha"));
  startp[0] = 1;
  countp[1] = 4;
  NC_CHECK(nc_put_vara_text(ncid, cell_angular_var, startp, countp, "beta"));
  startp[0] = 2;
  countp[1] = 5;
  NC_CHECK(nc_put_vara_text(ncid, cell_angular_var, startp, countp, "gamma"));

  // File not used until first dump. Close for now.
  NC_CHECK(nc_close(ncid));

  // Append to this file for the rest of this simulation
  append = true;
}

void DUMP_NETCDF::open_file(int frame_in_run)
{
  if (access(file_position, F_OK) != -1) {
    NC_CHECK(nc_open(file_position, NC_WRITE, &ncid));
  }

  /*
   * If another dump in simulation, must reload IDs from object.
   * Don't reload IDs more than once.
   */
  if (append && frame_in_run == 1) {
    // get all dimension ids
    NC_CHECK(nc_inq_dimid(ncid, FRAME_STR, &frame_dim));
    NC_CHECK(nc_inq_dimid(ncid, SPATIAL_STR, &spatial_dim));
    NC_CHECK(nc_inq_dimid(ncid, ATOM_STR, &atom_dim));
    NC_CHECK(nc_inq_dimid(ncid, CELL_SPATIAL_STR, &cell_spatial_dim));
    NC_CHECK(nc_inq_dimid(ncid, CELL_ANGULAR_STR, &cell_angular_dim));
    NC_CHECK(nc_inq_dimid(ncid, LABEL_STR, &label_dim));

    // Label Variables
    NC_CHECK(nc_inq_varid(ncid, SPATIAL_STR, &spatial_var));
    NC_CHECK(nc_inq_varid(ncid, CELL_SPATIAL_STR, &cell_spatial_var));
    NC_CHECK(nc_inq_varid(ncid, CELL_ANGULAR_STR, &cell_angular_var));

    // Data Variables
    NC_CHECK(nc_inq_varid(ncid, TIME_STR, &time_var));
    NC_CHECK(nc_inq_varid(ncid, CELL_LENGTHS_STR, &cell_lengths_var));
    NC_CHECK(nc_inq_varid(ncid, CELL_ANGLES_STR, &cell_angles_var));

    NC_CHECK(nc_inq_varid(ncid, COORDINATES_STR, &coordinates_var));
    NC_CHECK(nc_inq_varid(ncid, TYPE_STR, &type_var));

    // check data type -> must use precision of last run
    nc_type typep;
    int prev_precision = 0;
    NC_CHECK(nc_inq_vartype(ncid, coordinates_var, &typep));
    if (typep == NC_FLOAT)
      prev_precision = 1;
    else if (typep == NC_DOUBLE)
      prev_precision = 2;

    if (prev_precision != precision) {
      printf("Warning: GPUMD currently does not support changing netCDF\n"
             "         file precision after an initial definition.\n");
      precision = prev_precision;
    }
  }

  // get frame number
  NC_CHECK(nc_inq_dimlen(ncid, frame_dim, &lenp))
}

void DUMP_NETCDF::write(
  const double global_time,
  const Box& box,
  const std::vector<int>& cpu_type,
  GPU_Vector<double>& position_per_atom,
  std::vector<double>& cpu_position_per_atom)
{
  const int N = cpu_type.size();

  //// Write Frame Header ////
  // Get cell lengths and angles
  double cell_lengths[3];
  double cell_angles[3];
  if (box.triclinic) {
    const double* t = box.cpu_h;
    double cosgamma, cosbeta, cosalpha;
    cell_lengths[0] = sqrt(t[0] * t[0] + t[3] * t[3] + t[6] * t[6]); // a-side
    cell_lengths[1] = sqrt(t[1] * t[1] + t[4] * t[4] + t[7] * t[7]); // b-side
    cell_lengths[2] = sqrt(t[2] * t[2] + t[5] * t[5] + t[8] * t[8]); // c-side

    cosgamma = (t[0] * t[1] + t[3] * t[4] + t[6] * t[7]) / (cell_lengths[0] * cell_lengths[1]);
    cosbeta = (t[0] * t[2] + t[3] * t[5] + t[6] * t[8]) / (cell_lengths[0] * cell_lengths[2]);
    cosalpha = (t[1] * t[2] + t[4] * t[5] + t[7] * t[8]) / (cell_lengths[1] * cell_lengths[2]);

    cell_angles[0] = acos(cosalpha) * 180.0 / PI;
    cell_angles[1] = acos(cosbeta) * 180.0 / PI;
    cell_angles[2] = acos(cosgamma) * 180.0 / PI;

  } else {
    cell_lengths[0] = box.cpu_h[0];
    cell_lengths[1] = box.cpu_h[1];
    cell_lengths[2] = box.cpu_h[2];

    cell_angles[0] = 90;
    cell_angles[1] = 90;
    cell_angles[2] = 90;
  }

  // Set lengths to 0 if PBC is off
  if (!box.pbc_x)
    cell_lengths[0] = 0;
  if (!box.pbc_y)
    cell_lengths[1] = 0;
  if (!box.pbc_z)
    cell_lengths[2] = 0;

  size_t countp[3] = {1, 3, 0}; // 3rd dimension unused until per-atom
  size_t startp[3] = {lenp, 0, 0};
  double time = global_time / 1000.0 * TIME_UNIT_CONVERSION; // convert fs to ps
  NC_CHECK(nc_put_var1_double(ncid, time_var, startp, &time));
  NC_CHECK(nc_put_vara_double(ncid, cell_lengths_var, startp, countp, cell_lengths));
  NC_CHECK(nc_put_vara_double(ncid, cell_angles_var, startp, countp, cell_angles));

  //// Write Per-Atom Data ////
  position_per_atom.copy_to_host(cpu_position_per_atom.data());

  if (precision == 1) // single precision
  {
    // must convert data
    float cpu_x[N];
    float cpu_y[N];
    float cpu_z[N];
    for (int i = 0; i < N; i++) {
      cpu_x[i] = (float)cpu_position_per_atom[i];
      cpu_y[i] = (float)cpu_position_per_atom[i + N];
      cpu_z[i] = (float)cpu_position_per_atom[i + N * 2];
    }

    countp[0] = 1;
    countp[1] = N;
    countp[2] = 1;
    NC_CHECK(nc_put_vara_int(ncid, type_var, startp, countp, cpu_type.data()));
    NC_CHECK(nc_put_vara_float(ncid, coordinates_var, startp, countp, cpu_x));
    startp[2] = 1;
    NC_CHECK(nc_put_vara_float(ncid, coordinates_var, startp, countp, cpu_y));
    startp[2] = 2;
    NC_CHECK(nc_put_vara_float(ncid, coordinates_var, startp, countp, cpu_z));
  } else {
    countp[0] = 1;
    countp[1] = N;
    countp[2] = 1;
    NC_CHECK(nc_put_vara_int(ncid, type_var, startp, countp, cpu_type.data()));
    NC_CHECK(
      nc_put_vara_double(ncid, coordinates_var, startp, countp, cpu_position_per_atom.data()));
    startp[2] = 1;
    NC_CHECK(
      nc_put_vara_double(ncid, coordinates_var, startp, countp, cpu_position_per_atom.data() + N));
    startp[2] = 2;
    NC_CHECK(nc_put_vara_double(
      ncid, coordinates_var, startp, countp, cpu_position_per_atom.data() + N * 2));
  }
}

void DUMP_NETCDF::postprocess()
{
  if (dump_) {
    dump_ = false;
    precision = 2;
  }
}

void DUMP_NETCDF::process(
  const int step,
  const double global_time,
  const Box& box,
  const std::vector<int>& cpu_type,
  GPU_Vector<double>& position_per_atom,
  std::vector<double>& cpu_position_per_atom)
{
  if (!dump_)
    return;
  if ((step + 1) % interval != 0)
    return;

  int frame_in_run = (step + 1) / interval;
  open_file(frame_in_run);
  write(global_time, box, cpu_type, position_per_atom, cpu_position_per_atom);
  NC_CHECK(nc_close(ncid));
}

#endif
