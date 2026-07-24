/*
    Copyright 2017 Zheyong Fan and GPUMD development team
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
Write atom types, positions, and optional velocities to NetCDF trajectory
files. The layout is based on the AMBER 1.0 trajectory conventions, with
GPUMD extensions for atom types, group metadata, selectable precision, and
optional NetCDF4 deflate compression.

Contributing authors: Alexander Gabourie (Stanford University)
                      Liang Ting (The Chinese University of Hong Kong)

The implementation was influenced by LAMMPS' NetCDF output developed by
Lars Pastewka (University of Freiburg). Documentation can be found at:
https://docs.unidata.ucar.edu/netcdf-c/current/
https://ambermd.org/netcdf/nctraj.pdf
------------------------------------------------------------------------------*/

#ifdef USE_NETCDF

#include "dump_netcdf.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include "netcdf.h"
#include "parse_utilities.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/read_file.cuh"
#include <algorithm>
#include <cmath>
#include <cstring>

#define GPUMD_VERSION "5.6"

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
const char VELOCITIES_STR[] = "velocities"; // maybe not use
const char TYPE_STR[] = "type";
const char CELL_LENGTHS_STR[] = "cell_lengths";
const char CELL_ANGLES_STR[] = "cell_angles";
const char UNITS_STR[] = "units";
std::vector<std::string> DUMP_NETCDF::initialized_files_;

DUMP_NETCDF::DUMP_NETCDF(
  const char** param, int num_param, const std::vector<Group>& groups)
{
  parse(param, num_param, groups);
  property_name = "dump_netcdf";
}

void DUMP_NETCDF::parse(
  const char** param, int num_param, const std::vector<Group>& groups)
{
  dump_ = true;
  printf("Dump positions and optional velocities in NetCDF format.\n");

  if (num_param < 6) {
    PRINT_INPUT_ERROR("dump_netcdf should have at least 5 parameters.\n");
  }
  if (num_param > 11) {
    PRINT_INPUT_ERROR("dump_netcdf has too many parameters.");
  }

  if (!is_valid_int(param[1], &grouping_method_)) {
    PRINT_INPUT_ERROR("grouping method of dump_netcdf should be integer.");
  }
  if (grouping_method_ < 0) {
    printf("    for the whole system.\n");
  } else {
    if (grouping_method_ >= int(groups.size())) {
      PRINT_INPUT_ERROR("grouping method exceeds the bound.");
    }
    printf("    for grouping method %d.\n", grouping_method_);
  }

  if (!is_valid_int(param[2], &group_id_)) {
    PRINT_INPUT_ERROR("group id of dump_netcdf should be integer.");
  }
  if (grouping_method_ >= 0) {
    if (group_id_ < 0) {
      PRINT_INPUT_ERROR("group id is negative.");
    }
    if (group_id_ >= groups[grouping_method_].number) {
      PRINT_INPUT_ERROR("group id exceeds the bound.");
    }
    if (groups[grouping_method_].cpu_size[group_id_] <= 0) {
      PRINT_INPUT_ERROR("dump_netcdf cannot output an empty group.");
    }
    printf("    for group id %d.\n", group_id_);
  }

  if (!is_valid_int(param[3], &interval_)) {
    PRINT_INPUT_ERROR("dump interval should be an integer.");
  }
  if (interval_ <= 0) {
    PRINT_INPUT_ERROR("dump interval should > 0.");
  }
  printf("    every %d steps.\n", interval_);

  if (!is_valid_int(param[4], &has_velocity_)) {
    PRINT_INPUT_ERROR("has_velocity should be an integer.");
  }
  if (has_velocity_ != 0 && has_velocity_ != 1) {
    PRINT_INPUT_ERROR("has_velocity should be 0 or 1.");
  }
  if (has_velocity_ == 0) {
    printf("    without velocity data.\n");
  } else {
    printf("    with velocity data.\n");
  }

  filename_ = param[5];
  if (filename_.empty()) {
    PRINT_INPUT_ERROR("dump_netcdf filename should not be empty.");
  }
  printf("    into file %s.\n", filename_.c_str());

  bool precision_seen = false;
  bool compression_seen = false;
  for (int k = 6; k < num_param; k++) {
    if (strcmp(param[k], "precision") == 0) {
      if (precision_seen) {
        PRINT_INPUT_ERROR("Option 'precision' is specified more than once in dump_netcdf.\n");
      }
      parse_precision(param, num_param, k, precision_);
      precision_seen = true;
    } else if (strcmp(param[k], "compression") == 0) {
      if (compression_seen) {
        PRINT_INPUT_ERROR("Option 'compression' is specified more than once in dump_netcdf.\n");
      }
      if (k + 1 >= num_param) {
        PRINT_INPUT_ERROR("Not enough arguments for option 'compression'.\n");
      }
      if (strcmp(param[k + 1], "none") == 0) {
        compression_level_ = -1;
        printf("    without compression.\n");
        ++k;
      } else if (strcmp(param[k + 1], "deflate") == 0) {
        if (k + 2 >= num_param) {
          PRINT_INPUT_ERROR("A deflate level is required for dump_netcdf compression.\n");
        }
        if (!is_valid_int(param[k + 2], &compression_level_)) {
          PRINT_INPUT_ERROR("The dump_netcdf deflate level should be an integer.\n");
        }
        if (compression_level_ < 0 || compression_level_ > 9) {
          PRINT_INPUT_ERROR("The dump_netcdf deflate level should be between 0 and 9.\n");
        }
        printf("    with lossless deflate compression at level %d.\n", compression_level_);
        k += 2;
      } else {
        PRINT_INPUT_ERROR("Compression should be 'none' or 'deflate <0-9>'.\n");
      }
      compression_seen = true;
    } else {
      PRINT_INPUT_ERROR("Unrecognized argument in dump_netcdf.\n");
    }
  }

  if (precision_ == 1) {
    printf("    using single precision for NetCDF output.\n");
  } else {
    printf("    using double precision for NetCDF output.\n");
  }
}

void DUMP_NETCDF::preprocess(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  if (!dump_)
    return;

  if (grouping_method_ < 0) {
    number_of_atoms_to_dump_ = atom.number_of_atoms;
  } else {
    const Group& selected_group = group[grouping_method_];
    number_of_atoms_to_dump_ = selected_group.cpu_size[group_id_];
    cpu_type_to_dump_.resize(number_of_atoms_to_dump_);
    group_position_.resize(number_of_atoms_to_dump_ * 3);
    cpu_group_position_.resize(number_of_atoms_to_dump_ * 3);
    if (has_velocity_) {
      group_velocity_.resize(number_of_atoms_to_dump_ * 3);
      cpu_group_velocity_.resize(number_of_atoms_to_dump_ * 3);
    }
  }

  const size_t frame_values = size_t(number_of_atoms_to_dump_) * 3;
  if (precision_ == 1) {
    cpu_position_float_.resize(frame_values);
    if (has_velocity_) {
      cpu_velocity_float_.resize(frame_values);
    }
  } else {
    cpu_position_double_.resize(frame_values);
    if (has_velocity_) {
      cpu_velocity_double_.resize(frame_values);
    }
  }

  const bool initialized =
    std::find(initialized_files_.begin(), initialized_files_.end(), filename_) !=
    initialized_files_.end();
  if (initialized) {
    NC_CHECK(nc_open(filename_.c_str(), NC_WRITE, &ncid));
    load_file_definition();
    validate_file_definition();
    NC_CHECK(nc_inq_dimlen(ncid, frame_dim, &lenp));
  } else {
    create_file();
    initialized_files_.push_back(filename_);
  }
}

void DUMP_NETCDF::create_file()
{
  const int creation_mode = compression_level_ >= 0 ? NC_NETCDF4 : NC_64BIT_OFFSET;
  const int create_status = nc_create(filename_.c_str(), creation_mode, &ncid);
  if (create_status != NC_NOERR) {
    if (
      compression_level_ >= 0 &&
      (create_status == NC_ENOTBUILT || create_status == NC_ENOTNC4)) {
      fprintf(
        stderr,
        "Error: dump_netcdf deflate compression requires a NetCDF-C build with "
        "NetCDF4/HDF5 support.\n");
    }
    ERR(create_status);
  }

  // Global attributes
  NC_CHECK(nc_put_att_text(ncid, NC_GLOBAL, "program", 5, "GPUMD"));
  NC_CHECK(
    nc_put_att_text(ncid, NC_GLOBAL, "programVersion", strlen(GPUMD_VERSION), GPUMD_VERSION));
  NC_CHECK(nc_put_att_text(ncid, NC_GLOBAL, "Conventions", 5, "AMBER"));
  NC_CHECK(nc_put_att_text(ncid, NC_GLOBAL, "ConventionVersion", 3, "1.0"));
  NC_CHECK(nc_put_att_int(
    ncid, NC_GLOBAL, "gpumd_grouping_method", NC_INT, 1, &grouping_method_));
  NC_CHECK(nc_put_att_int(ncid, NC_GLOBAL, "gpumd_group_id", NC_INT, 1, &group_id_));
  NC_CHECK(nc_put_att_int(
    ncid, NC_GLOBAL, "gpumd_has_velocity", NC_INT, 1, &has_velocity_));
  NC_CHECK(nc_put_att_int(
    ncid, NC_GLOBAL, "gpumd_compression_level", NC_INT, 1, &compression_level_));

  // dimensions
  NC_CHECK(nc_def_dim(
    ncid, FRAME_STR, NC_UNLIMITED, &frame_dim)); // unlimited number of steps (can append)
  NC_CHECK(nc_def_dim(ncid, SPATIAL_STR, 3, &spatial_dim));         // number of spatial dimensions
  NC_CHECK(nc_def_dim(ncid, ATOM_STR, number_of_atoms_to_dump_, &atom_dim));
  NC_CHECK(nc_def_dim(ncid, CELL_SPATIAL_STR, 3, &cell_spatial_dim)); // unitcell lengths
  NC_CHECK(nc_def_dim(ncid, CELL_ANGULAR_STR, 3, &cell_angular_dim)); // unitcell angles
  NC_CHECK(nc_def_dim(ncid, LABEL_STR, 10, &label_dim));              // needed for cell_angular

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

  // More extensive data variables (type, coordinates, velocities)
  dimids[0] = frame_dim;
  dimids[1] = atom_dim;
  dimids[2] = spatial_dim;

  if (precision_ == 1) // single precision
  {
    NC_CHECK(nc_def_var(ncid, COORDINATES_STR, NC_FLOAT, 3, dimids, &coordinates_var));
    if (has_velocity_) {
      NC_CHECK(nc_def_var(ncid, VELOCITIES_STR, NC_FLOAT, 3, dimids, &velocities_var));
    }
  } else {
    NC_CHECK(nc_def_var(ncid, COORDINATES_STR, NC_DOUBLE, 3, dimids, &coordinates_var));
    if (has_velocity_) {
      NC_CHECK(nc_def_var(ncid, VELOCITIES_STR, NC_DOUBLE, 3, dimids, &velocities_var));
    }
  }
  NC_CHECK(nc_def_var(ncid, TYPE_STR, NC_INT, 2, dimids, &type_var));

  if (compression_level_ >= 0) {
    const size_t bytes_per_value = precision_ == 1 ? sizeof(float) : sizeof(double);
    const size_t target_chunk_bytes = 1024 * 1024;
    const size_t max_chunk_atoms =
      std::max<size_t>(1, target_chunk_bytes / (bytes_per_value * 3));
    size_t chunks[3] = {
      1, std::min<size_t>(number_of_atoms_to_dump_, max_chunk_atoms), 3};
    NC_CHECK(nc_def_var_chunking(ncid, coordinates_var, NC_CHUNKED, chunks));
    NC_CHECK(nc_def_var_deflate(ncid, coordinates_var, 1, 1, compression_level_));
    if (has_velocity_) {
      NC_CHECK(nc_def_var_chunking(ncid, velocities_var, NC_CHUNKED, chunks));
      NC_CHECK(nc_def_var_deflate(ncid, velocities_var, 1, 1, compression_level_));
    }
    size_t type_chunks[2] = {
      1,
      std::min<size_t>(
        number_of_atoms_to_dump_, target_chunk_bytes / sizeof(int))};
    NC_CHECK(nc_def_var_chunking(ncid, type_var, NC_CHUNKED, type_chunks));
    NC_CHECK(nc_def_var_deflate(ncid, type_var, 1, 1, compression_level_));
  }

  // Units
  NC_CHECK(nc_put_att_text(ncid, time_var, UNITS_STR, 10, "picosecond"));
  NC_CHECK(nc_put_att_text(ncid, cell_lengths_var, UNITS_STR, 8, "angstrom"));
  NC_CHECK(nc_put_att_text(ncid, coordinates_var, UNITS_STR, 8, "angstrom"));
  NC_CHECK(nc_put_att_text(ncid, cell_angles_var, UNITS_STR, 6, "degree"));

  if (has_velocity_) {
    NC_CHECK(nc_put_att_text(
      ncid, velocities_var, UNITS_STR, 19, "angstrom/picosecond")); // AMBER conventions
  }
  // Definitions are complete -> leave define mode
  NC_CHECK(nc_enddef(ncid));

  // Write the Label Variables
  NC_CHECK(nc_put_var_text(ncid, spatial_var, "xyz"));
  NC_CHECK(nc_put_var_text(ncid, cell_spatial_var, "abc"));
  // 2D data, startp defines index for data to start
  // countp determines the size of data in each dimension
  size_t startp[2] = {0, 0};
  size_t countp[2] = {1, 5};
  NC_CHECK(nc_put_vara_text(ncid, cell_angular_var, startp, countp, "alpha"));
  startp[0] = 1;
  countp[1] = 4;
  NC_CHECK(nc_put_vara_text(ncid, cell_angular_var, startp, countp, "beta"));
  startp[0] = 2;
  countp[1] = 5;
  NC_CHECK(nc_put_vara_text(ncid, cell_angular_var, startp, countp, "gamma"));
  lenp = 0;
}

void DUMP_NETCDF::load_file_definition()
{
  NC_CHECK(nc_inq_dimid(ncid, FRAME_STR, &frame_dim));
  NC_CHECK(nc_inq_dimid(ncid, SPATIAL_STR, &spatial_dim));
  NC_CHECK(nc_inq_dimid(ncid, ATOM_STR, &atom_dim));
  NC_CHECK(nc_inq_dimid(ncid, CELL_SPATIAL_STR, &cell_spatial_dim));
  NC_CHECK(nc_inq_dimid(ncid, CELL_ANGULAR_STR, &cell_angular_dim));
  NC_CHECK(nc_inq_dimid(ncid, LABEL_STR, &label_dim));
  NC_CHECK(nc_inq_varid(ncid, SPATIAL_STR, &spatial_var));
  NC_CHECK(nc_inq_varid(ncid, CELL_SPATIAL_STR, &cell_spatial_var));
  NC_CHECK(nc_inq_varid(ncid, CELL_ANGULAR_STR, &cell_angular_var));
  NC_CHECK(nc_inq_varid(ncid, TIME_STR, &time_var));
  NC_CHECK(nc_inq_varid(ncid, CELL_LENGTHS_STR, &cell_lengths_var));
  NC_CHECK(nc_inq_varid(ncid, CELL_ANGLES_STR, &cell_angles_var));
  NC_CHECK(nc_inq_varid(ncid, COORDINATES_STR, &coordinates_var));
  NC_CHECK(nc_inq_varid(ncid, TYPE_STR, &type_var));
}

void DUMP_NETCDF::validate_file_definition()
{
  size_t previous_number_of_atoms = 0;
  NC_CHECK(nc_inq_dimlen(ncid, atom_dim, &previous_number_of_atoms));
  if (previous_number_of_atoms != size_t(number_of_atoms_to_dump_)) {
    PRINT_INPUT_ERROR("Cannot append dump_netcdf data with a different number of atoms.\n");
  }

  nc_type coordinate_type;
  NC_CHECK(nc_inq_vartype(ncid, coordinates_var, &coordinate_type));
  const nc_type expected_type = precision_ == 1 ? NC_FLOAT : NC_DOUBLE;
  if (coordinate_type != expected_type) {
    PRINT_INPUT_ERROR("Cannot change dump_netcdf precision between run commands.\n");
  }

  int previous_grouping_method;
  int previous_group_id;
  int previous_has_velocity;
  int previous_compression_level;
  NC_CHECK(nc_get_att_int(
    ncid, NC_GLOBAL, "gpumd_grouping_method", &previous_grouping_method));
  NC_CHECK(nc_get_att_int(ncid, NC_GLOBAL, "gpumd_group_id", &previous_group_id));
  NC_CHECK(nc_get_att_int(
    ncid, NC_GLOBAL, "gpumd_has_velocity", &previous_has_velocity));
  NC_CHECK(nc_get_att_int(
    ncid, NC_GLOBAL, "gpumd_compression_level", &previous_compression_level));
  if (previous_grouping_method != grouping_method_ || previous_group_id != group_id_) {
    PRINT_INPUT_ERROR("Cannot change the dump_netcdf group between run commands.\n");
  }
  if (previous_has_velocity != has_velocity_) {
    PRINT_INPUT_ERROR("Cannot change dump_netcdf velocity output between run commands.\n");
  }
  if (has_velocity_) {
    NC_CHECK(nc_inq_varid(ncid, VELOCITIES_STR, &velocities_var));
  }
  if (previous_compression_level != compression_level_) {
    PRINT_INPUT_ERROR("Cannot change dump_netcdf compression between run commands.\n");
  }
}

static bool build_netcdf_transform(
  const Box& box, double cell_lengths[3], double cell_angles[3], double transform[9])
{
  // AMBER NetCDF stores only cell lengths and angles. Readers reconstruct a
  // restricted cell with a along +x and b in the xy plane. Use the same
  // general-to-restricted transformation for every GPUMD cell.
  const double* h = box.cpu_h;
  const double a[3] = {h[0], h[3], h[6]};
  const double b[3] = {h[1], h[4], h[7]};
  const double c[3] = {h[2], h[5], h[8]};
  const auto dot = [](const double x[3], const double y[3]) {
    return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
  };
  const auto clamp_cosine = [](double value) {
    return std::max(-1.0, std::min(1.0, value));
  };

  cell_lengths[0] = sqrt(dot(a, a));
  cell_lengths[1] = sqrt(dot(b, b));
  cell_lengths[2] = sqrt(dot(c, c));

  // The rows of transform are the axes of the restricted NetCDF cell written
  // in GPUMD Cartesian coordinates.
  for (int d = 0; d < 3; ++d) {
    transform[d] = a[d] / cell_lengths[0];
  }
  const double bx = transform[0] * b[0] + transform[1] * b[1] + transform[2] * b[2];
  double by = 0.0;
  for (int d = 0; d < 3; ++d) {
    transform[3 + d] = b[d] - bx * transform[d];
    by += transform[3 + d] * transform[3 + d];
  }
  by = sqrt(by);
  for (int d = 0; d < 3; ++d) {
    transform[3 + d] /= by;
  }

  transform[6] = transform[1] * transform[5] - transform[2] * transform[4];
  transform[7] = transform[2] * transform[3] - transform[0] * transform[5];
  transform[8] = transform[0] * transform[4] - transform[1] * transform[3];
  if (transform[6] * c[0] + transform[7] * c[1] + transform[8] * c[2] < 0.0) {
    for (int d = 0; d < 3; ++d) {
      transform[6 + d] = -transform[6 + d];
    }
  }

  const double cosalpha =
    clamp_cosine(dot(b, c) / (cell_lengths[1] * cell_lengths[2]));
  const double cosbeta =
    clamp_cosine(dot(a, c) / (cell_lengths[0] * cell_lengths[2]));
  const double cosgamma =
    clamp_cosine(dot(a, b) / (cell_lengths[0] * cell_lengths[1]));
  cell_angles[0] = acos(cosalpha) * 180.0 / PI;
  cell_angles[1] = acos(cosbeta) * 180.0 / PI;
  cell_angles[2] = acos(cosgamma) * 180.0 / PI;

  bool transform_vectors = false;
  for (int i = 0; i < 9; ++i) {
    const double identity_value = i % 4 == 0 ? 1.0 : 0.0;
    transform_vectors |= fabs(transform[i] - identity_value) > 1.0e-12;
  }
  return transform_vectors;
}

template <typename T>
static void pack_netcdf_frame(
  const int number_of_atoms,
  const bool has_velocity,
  const bool transform_vectors,
  const double transform[9],
  const double velocity_scale,
  const std::vector<double>& position,
  const std::vector<double>& velocity,
  std::vector<T>& packed_position,
  std::vector<T>& packed_velocity)
{
  for (int i = 0; i < number_of_atoms; ++i) {
    for (int output_dim = 0; output_dim < 3; ++output_dim) {
      double position_value = position[i + number_of_atoms * output_dim];
      double velocity_value =
        has_velocity ? velocity[i + number_of_atoms * output_dim] : 0.0;
      if (transform_vectors) {
        position_value = 0.0;
        velocity_value = 0.0;
        for (int input_dim = 0; input_dim < 3; ++input_dim) {
          const double coefficient = transform[output_dim * 3 + input_dim];
          position_value +=
            coefficient * position[i + number_of_atoms * input_dim];
          if (has_velocity) {
            velocity_value +=
              coefficient * velocity[i + number_of_atoms * input_dim];
          }
        }
      }
      packed_position[i * 3 + output_dim] = static_cast<T>(position_value);
      if (has_velocity) {
        packed_velocity[i * 3 + output_dim] =
          static_cast<T>(velocity_value * velocity_scale);
      }
    }
  }
}

void DUMP_NETCDF::write(
  const double global_time,
  const Box& box,
  const std::vector<int>& cpu_type,
  const std::vector<double>& cpu_position_per_atom,
  const std::vector<double>& cpu_velocity_per_atom)
{
  const int number_of_atoms = number_of_atoms_to_dump_;

  double cell_lengths[3];
  double cell_angles[3];
  double cell_transform[9];
  const bool transform_vectors =
    build_netcdf_transform(box, cell_lengths, cell_angles, cell_transform);

  // Set lengths to 0 if PBC is off
  if (!box.pbc_x)
    cell_lengths[0] = 0;
  if (!box.pbc_y)
    cell_lengths[1] = 0;
  if (!box.pbc_z)
    cell_lengths[2] = 0;

  size_t frame_start[1] = {lenp};
  size_t cell_start[2] = {lenp, 0};
  size_t cell_count[2] = {1, 3};
  double time = global_time / 1000.0 * TIME_UNIT_CONVERSION; // convert fs to ps
  NC_CHECK(nc_put_var1_double(ncid, time_var, frame_start, &time));
  NC_CHECK(nc_put_vara_double(ncid, cell_lengths_var, cell_start, cell_count, cell_lengths));
  NC_CHECK(nc_put_vara_double(ncid, cell_angles_var, cell_start, cell_count, cell_angles));

  const double natural_to_A_per_ps =
    1.0 / TIME_UNIT_CONVERSION * 1000.0; // * 1000 from A/fs to A/ps
  size_t atom_start[2] = {lenp, 0};
  size_t atom_count[2] = {1, size_t(number_of_atoms)};
  size_t vector_start[3] = {lenp, 0, 0};
  size_t vector_count[3] = {1, size_t(number_of_atoms), 3};
  NC_CHECK(nc_put_vara_int(ncid, type_var, atom_start, atom_count, cpu_type.data()));

  if (precision_ == 1) // single precision
  {
    pack_netcdf_frame(
      number_of_atoms,
      has_velocity_,
      transform_vectors,
      cell_transform,
      natural_to_A_per_ps,
      cpu_position_per_atom,
      cpu_velocity_per_atom,
      cpu_position_float_,
      cpu_velocity_float_);
    NC_CHECK(nc_put_vara_float(
      ncid, coordinates_var, vector_start, vector_count, cpu_position_float_.data()));
    if (has_velocity_) {
      NC_CHECK(nc_put_vara_float(
        ncid, velocities_var, vector_start, vector_count, cpu_velocity_float_.data()));
    }
  } else {
    pack_netcdf_frame(
      number_of_atoms,
      has_velocity_,
      transform_vectors,
      cell_transform,
      natural_to_A_per_ps,
      cpu_position_per_atom,
      cpu_velocity_per_atom,
      cpu_position_double_,
      cpu_velocity_double_);
    NC_CHECK(nc_put_vara_double(
      ncid, coordinates_var, vector_start, vector_count, cpu_position_double_.data()));
    if (has_velocity_) {
      NC_CHECK(nc_put_vara_double(
        ncid, velocities_var, vector_start, vector_count, cpu_velocity_double_.data()));
    }
  }
  ++lenp;
}

void DUMP_NETCDF::postprocess(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature)
{
  if (dump_) {
    NC_CHECK(nc_close(ncid));
    ncid = -1;
    dump_ = false;
  }
}

static __global__ void gather_netcdf_group(
  const int number_of_atoms_in_group,
  const int number_of_atoms,
  const int group_offset,
  const int* group_contents,
  const double* position,
  const double* velocity,
  double* group_position,
  double* group_velocity)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < number_of_atoms_in_group) {
    const int atom_index = group_contents[group_offset + n];
    for (int d = 0; d < 3; ++d) {
      group_position[n + number_of_atoms_in_group * d] =
        position[atom_index + number_of_atoms * d];
      if (group_velocity != nullptr) {
        group_velocity[n + number_of_atoms_in_group * d] =
          velocity[atom_index + number_of_atoms * d];
      }
    }
  }
}

void DUMP_NETCDF::process(
  const int number_of_steps,
  int step,
  const int fixed_group,
  const int move_group,
  const double global_time,
  const double temperature,
  Integrate& integrate,
  Box& box,
  std::vector<Group>& group,
  GPU_Vector<double>& thermo,
  Atom& atom,
  Force& force)
{
  if (!dump_)
    return;
  if ((step + 1) % interval_ != 0)
    return;

  const std::vector<double>* cpu_position = &atom.cpu_position_per_atom;
  const std::vector<double>* cpu_velocity = &atom.cpu_velocity_per_atom;
  const std::vector<int>* cpu_type = &atom.cpu_type;
  if (grouping_method_ < 0) {
    atom.position_per_atom.copy_to_host(atom.cpu_position_per_atom.data());
    if (has_velocity_) {
      atom.velocity_per_atom.copy_to_host(atom.cpu_velocity_per_atom.data());
    }
  } else {
    const Group& selected_group = group[grouping_method_];
    const int group_offset = selected_group.cpu_size_sum[group_id_];
    gather_netcdf_group<<<(number_of_atoms_to_dump_ - 1) / 128 + 1, 128>>>(
      number_of_atoms_to_dump_,
      atom.number_of_atoms,
      group_offset,
      selected_group.contents.data(),
      atom.position_per_atom.data(),
      has_velocity_ ? atom.velocity_per_atom.data() : nullptr,
      group_position_.data(),
      has_velocity_ ? group_velocity_.data() : nullptr);
    GPU_CHECK_KERNEL
    group_position_.copy_to_host(cpu_group_position_.data());
    if (has_velocity_) {
      group_velocity_.copy_to_host(cpu_group_velocity_.data());
    }
    for (int n = 0; n < number_of_atoms_to_dump_; ++n) {
      const int atom_index = selected_group.cpu_contents[group_offset + n];
      cpu_type_to_dump_[n] = atom.cpu_type[atom_index];
    }
    cpu_position = &cpu_group_position_;
    cpu_velocity = &cpu_group_velocity_;
    cpu_type = &cpu_type_to_dump_;
  }

  write(global_time, box, *cpu_type, *cpu_position, *cpu_velocity);
}

#endif
