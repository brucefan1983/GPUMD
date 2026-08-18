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
Write atom types, positions, and the optional per-atom quantities shared with
dump_xyz to NetCDF trajectory files. The layout is based on the AMBER 1.0
trajectory conventions, with GPUMD extensions for atom types, the quantities
AMBER does not define, group metadata, selectable precision, and optional
NetCDF4 deflate compression.

Contributing authors: Alexander Gabourie (Stanford University)
                      Liang Ting (The Chinese University of Hong Kong)

The implementation was influenced by LAMMPS' NetCDF output developed by
Lars Pastewka (University of Freiburg). Documentation can be found at:
https://docs.unidata.ucar.edu/netcdf-c/current/
https://ambermd.org/netcdf/nctraj.pdf
------------------------------------------------------------------------------*/

#ifdef USE_NETCDF

#include "dump_netcdf.cuh"
#include "force/force.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include "netcdf.h"
#include "netcdf_meta.h"
#include "parse_utilities.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/read_file.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>

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
const char GROUPING_METHOD_STR[] = "grouping_method";
const char FORCES_STR[] = "forces";
const char UNWRAPPED_COORDINATES_STR[] = "unwrapped_coordinates";
const char POTENTIAL_STR[] = "potential_energy";
const char CHARGE_STR[] = "charge";
const char BEC_STR[] = "bec";
const char VIRIAL_STR[] = "virial";
const char MASS_STR[] = "mass";
const char GROUP_LABELS_STR[] = "group_labels";

// The AMBER convention specifies forces in kcal/mol/A, so they are converted on the way out,
// as the velocities already are. 1 eV = 96.48533212331 kJ/mol and 1 kcal = 4.184 kJ.
const double EV_TO_KCAL_PER_MOL = 23.06054783061903;

// GPUMD keeps the per-atom virial in its own component order; this maps it to the row-major
// order the file uses, as Dump_XYZ does. The BEC is already row-major.
const int VIRIAL_COMPONENT[9] = {0, 3, 4, 6, 1, 5, 7, 8, 2};
const int ROW_MAJOR_COMPONENT[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};

std::vector<std::string> DUMP_NETCDF::active_files_;

// The three packing helpers below take one per-atom array laid out as src[atom + N * component],
// select the atoms of the output through `indices`, apply the rotation into the restricted
// NetCDF cell where the quantity calls for one, and write the result out atom-major in the
// element type the file uses.

template <typename T, typename S>
static void
pack_scalar(const std::vector<int>& indices, const std::vector<S>& src, std::vector<T>& dst)
{
  for (size_t n = 0; n < indices.size(); ++n) {
    dst[n] = static_cast<T>(src[indices[n]]);
  }
}

template <typename T, typename S>
static void pack_vector(
  const std::vector<int>& indices,
  const int number_of_atoms,
  const bool rotate,
  const double transform[9],
  const double scale,
  const std::vector<S>& src,
  std::vector<T>& dst)
{
  for (size_t n = 0; n < indices.size(); ++n) {
    const int m = indices[n];
    for (int i = 0; i < 3; ++i) {
      double value = 0.0;
      if (rotate) {
        for (int j = 0; j < 3; ++j) {
          value += transform[i * 3 + j] * src[m + number_of_atoms * j];
        }
      } else {
        value = src[m + number_of_atoms * i];
      }
      dst[n * 3 + i] = static_cast<T>(value * scale);
    }
  }
}

template <typename T, typename S>
static void pack_tensor(
  const std::vector<int>& indices,
  const int number_of_atoms,
  const bool rotate,
  const double transform[9],
  const int component[9],
  const std::vector<S>& src,
  std::vector<T>& dst)
{
  for (size_t n = 0; n < indices.size(); ++n) {
    const int m = indices[n];
    double tensor[9];
    for (int c = 0; c < 9; ++c) {
      tensor[c] = src[m + number_of_atoms * component[c]];
    }
    if (rotate) {
      // A rank-2 tensor follows the cell as R T R^T, which holds for any orthogonal R and so
      // also when R is the reflection produced for a left-handed cell.
      double temp[9];
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          double sum = 0.0;
          for (int k = 0; k < 3; ++k) {
            sum += transform[i * 3 + k] * tensor[k * 3 + j];
          }
          temp[i * 3 + j] = sum;
        }
      }
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          double sum = 0.0;
          for (int k = 0; k < 3; ++k) {
            sum += temp[i * 3 + k] * transform[j * 3 + k];
          }
          tensor[i * 3 + j] = sum;
        }
      }
    }
    for (int c = 0; c < 9; ++c) {
      dst[n * 9 + c] = static_cast<T>(tensor[c]);
    }
  }
}

// The file holds floats or doubles depending on the precision option, so every pack goes through
// one of these, which pick the buffer and instantiate the helper for its element type.

template <typename S>
static void pack_scalar_by_precision(
  const int precision,
  const std::vector<int>& indices,
  const std::vector<S>& src,
  std::vector<float>& pack_float,
  std::vector<double>& pack_double)
{
  if (precision == 1) {
    pack_scalar(indices, src, pack_float);
  } else {
    pack_scalar(indices, src, pack_double);
  }
}

template <typename S>
static void pack_vector_by_precision(
  const int precision,
  const std::vector<int>& indices,
  const int number_of_atoms,
  const bool rotate,
  const double transform[9],
  const double scale,
  const std::vector<S>& src,
  std::vector<float>& pack_float,
  std::vector<double>& pack_double)
{
  if (precision == 1) {
    pack_vector(indices, number_of_atoms, rotate, transform, scale, src, pack_float);
  } else {
    pack_vector(indices, number_of_atoms, rotate, transform, scale, src, pack_double);
  }
}

template <typename S>
static void pack_tensor_by_precision(
  const int precision,
  const std::vector<int>& indices,
  const int number_of_atoms,
  const bool rotate,
  const double transform[9],
  const int component[9],
  const std::vector<S>& src,
  std::vector<float>& pack_float,
  std::vector<double>& pack_double)
{
  if (precision == 1) {
    pack_tensor(indices, number_of_atoms, rotate, transform, component, src, pack_float);
  } else {
    pack_tensor(indices, number_of_atoms, rotate, transform, component, src, pack_double);
  }
}

DUMP_NETCDF::DUMP_NETCDF(
  const char** param, int num_param, const std::vector<Group>& groups, Atom& atom)
{
  is_nep_charge_ = check_is_nep_charge();

  parse(param, num_param, groups);

  // The integrator only maintains the unwrapped positions while this array is non-empty, so
  // asking for them has to allocate it. Unlike dump_xyz this is done only when the quantity was
  // requested, so that a run that does not ask for it does not pay for the unwrapping.
  if (quantities_.has_unwrapped_position_) {
    if (atom.unwrapped_position.size() < atom.number_of_atoms * 3) {
      atom.unwrapped_position.resize(atom.number_of_atoms * 3);
      atom.unwrapped_position.copy_from_device(atom.position_per_atom.data());
    }
    if (atom.position_temp.size() < atom.number_of_atoms * 3) {
      atom.position_temp.resize(atom.number_of_atoms * 3);
    }
  }

  property_name = "dump_netcdf";
}

void DUMP_NETCDF::parse(const char** param, int num_param, const std::vector<Group>& groups)
{
  dump_ = true;
  printf("Dump per-atom data in NetCDF format.\n");

  if (num_param < 3) {
    PRINT_INPUT_ERROR("dump_netcdf should have at least 2 parameters.\n");
  }

  // The old syntax started with <grouping_method> <group_id> <interval>, so param[2] and param[3]
  // were both integers. In the current syntax param[2] is the file name and param[3] is an option
  // or a quantity keyword, neither of which is an integer.
  int scratch;
  if (num_param >= 4 && is_valid_int(param[2], &scratch) && is_valid_int(param[3], &scratch)) {
    PRINT_INPUT_ERROR(
      "dump_netcdf no longer takes <grouping_method> <group_id> as its first two parameters, and "
      "<has_velocity> is now the optional quantity 'velocity'. Use dump_netcdf <interval> "
      "<filename> [group <grouping_method> <group_id>] [velocity] instead.");
  }

  if (!is_valid_int(param[1], &interval_)) {
    PRINT_INPUT_ERROR("dump interval should be an integer.");
  }
  if (interval_ <= 0) {
    PRINT_INPUT_ERROR("dump interval should > 0.");
  }
  printf("    every %d steps.\n", interval_);

  filename_ = param[2];
  printf("    into file %s.\n", filename_.c_str());

  number_of_grouping_methods_ = groups.size();

  bool group_seen = false;
  bool precision_seen = false;
  bool compression_seen = false;
  for (int k = 3; k < num_param; k++) {
    if (strcmp(param[k], "group") == 0) {
      if (group_seen) {
        PRINT_INPUT_ERROR("Option 'group' is specified more than once in dump_netcdf.\n");
      }
      parse_group(param, num_param, false, groups, k, grouping_method_, group_id_);
      // parse_group checks the bounds but not the size, and an empty group is not caught further
      // down either. NC_UNLIMITED is 0, so defining the atom dimension with a length of zero
      // makes it unlimited rather than failing: with compression the run then succeeds and
      // writes a file whose coordinates have shape (frames, 0, 3), and without it the run dies
      // mid-way with "NC_UNLIMITED size already in use", since the classic format allows only
      // one unlimited dimension.
      if (groups[grouping_method_].cpu_size[group_id_] <= 0) {
        PRINT_INPUT_ERROR("dump_netcdf cannot output an empty group.");
      }
      group_seen = true;
    } else if (strcmp(param[k], "precision") == 0) {
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
    } else if (!parse_dump_quantity(param[k], quantities_, is_nep_charge_, groups, "dump_netcdf")) {
      PRINT_INPUT_ERROR("Unrecognized argument in dump_netcdf.\n");
    }
  }

  // A canonical listing of what was requested, independent of the order the arguments came in.
  // It goes into the file as the gpumd_quantities attribute and is what appending compares.
  const std::pair<bool, const char*> quantity_names[] = {
    {quantities_.has_velocity_, "velocity"},
    {quantities_.has_force_, "force"},
    {quantities_.has_potential_, "potential"},
    {quantities_.has_unwrapped_position_, "unwrapped_position"},
    {quantities_.has_mass_, "mass"},
    {quantities_.has_charge_, "charge"},
    {quantities_.has_bec_, "bec"},
    {quantities_.has_virial_, "virial"},
    {quantities_.has_group_, "group_labels"}};
  for (const auto& quantity : quantity_names) {
    if (quantity.first) {
      if (!quantity_list_.empty()) {
        quantity_list_ += " ";
      }
      quantity_list_ += quantity.second;
    }
  }

  if (!group_seen) {
    printf("    for the whole system.\n");
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

  if (std::find(active_files_.begin(), active_files_.end(), filename_) != active_files_.end()) {
    PRINT_INPUT_ERROR("dump_netcdf filenames must be unique within one run.");
  }
  active_files_.push_back(filename_);

  // The atoms of the output, in the order they are written. Group membership does not change
  // during a run, so this is built once and then serves both the grouped and the whole-system
  // case, which lets every quantity share one packing path.
  if (grouping_method_ < 0) {
    number_of_atoms_to_dump_ = atom.number_of_atoms;
    dump_indices_.resize(number_of_atoms_to_dump_);
    for (int n = 0; n < number_of_atoms_to_dump_; ++n) {
      dump_indices_[n] = n;
    }
  } else {
    const Group& selected_group = group[grouping_method_];
    number_of_atoms_to_dump_ = selected_group.cpu_size[group_id_];
    const int group_offset = selected_group.cpu_size_sum[group_id_];
    dump_indices_.resize(number_of_atoms_to_dump_);
    for (int n = 0; n < number_of_atoms_to_dump_; ++n) {
      dump_indices_[n] = selected_group.cpu_contents[group_offset + n];
    }
  }
  cpu_type_to_dump_.resize(number_of_atoms_to_dump_);

  // one variable of one frame at a time, so nine components per atom is the widest case
  const size_t pack_values = size_t(number_of_atoms_to_dump_) * 9;
  if (precision_ == 1) {
    pack_float_.resize(pack_values);
  } else {
    pack_double_.resize(pack_values);
  }

  // host copies of the arrays Atom does not already keep on the host, sized for the whole
  // system since that is what is copied back from the device
  if (quantities_.has_force_) {
    cpu_force_per_atom_.resize(atom.number_of_atoms * 3);
  }
  if (quantities_.has_potential_) {
    cpu_potential_per_atom_.resize(atom.number_of_atoms);
  }
  if (quantities_.has_unwrapped_position_) {
    cpu_unwrapped_position_.resize(atom.number_of_atoms * 3);
  }
  if (quantities_.has_virial_) {
    cpu_virial_per_atom_.resize(atom.number_of_atoms * 9);
  }
  if (quantities_.has_bec_) {
    cpu_bec_.resize(atom.number_of_atoms * 9);
  }

  // An existing file is extended rather than replaced, as dump_xyz does. Whether the file is
  // there is decided by looking for it rather than by inferring absence from a NetCDF error
  // code, which is not the same value in every netcdf-c build. This also covers a file written
  // earlier in this execution: the previous command closed it, so it is simply on disk.
  std::ifstream existing_file(filename_.c_str());
  if (existing_file.good()) {
    existing_file.close();
    const int open_status = nc_open(filename_.c_str(), NC_WRITE, &ncid);
    if (open_status != NC_NOERR) {
      fprintf(
        stderr,
        "Error: dump_netcdf cannot open %s to append to it. Remove or rename it to start a new "
        "trajectory.\n",
        filename_.c_str());
      ERR(open_status);
    }
    load_file_definition();
    validate_file_definition();
    NC_CHECK(nc_inq_dimlen(ncid, frame_dim, &lenp));
  } else {
    create_file(group, atom);
  }
}

// Defines one per-atom variable of the trajectory. `rank` is 2 for a scalar, 3 for a vector and
// 4 for a rank-2 tensor, and `values_per_atom` the matching 1, 3 or 9, used to size the chunks.
void DUMP_NETCDF::define_per_frame_variable(
  const char* name, const int rank, const int values_per_atom, const char* units, int& var)
{
  const int dimids[4] = {frame_dim, atom_dim, spatial_dim, spatial_dim};
  const nc_type type = precision_ == 1 ? NC_FLOAT : NC_DOUBLE;
  NC_CHECK(nc_def_var(ncid, name, type, rank, dimids, &var));
  if (units != nullptr) {
    NC_CHECK(nc_put_att_text(ncid, var, UNITS_STR, strlen(units), units));
  }
  if (compression_level_ >= 0) {
#if defined(NC_HAS_HDF5) && NC_HAS_HDF5
    const size_t bytes_per_value = precision_ == 1 ? sizeof(float) : sizeof(double);
    const size_t target_chunk_bytes = 1024 * 1024;
    const size_t max_chunk_atoms =
      std::max<size_t>(1, target_chunk_bytes / (bytes_per_value * values_per_atom));
    size_t chunks[4] = {1, std::min<size_t>(number_of_atoms_to_dump_, max_chunk_atoms), 3, 3};
    NC_CHECK(nc_def_var_chunking(ncid, var, NC_CHUNKED, chunks));
    NC_CHECK(nc_def_var_deflate(ncid, var, 1, 1, compression_level_));
#endif // NC_HAS_HDF5
#if !(defined(NC_HAS_HDF5) && NC_HAS_HDF5)
    (void)values_per_atom;
#endif
  }
}

void DUMP_NETCDF::create_file(const std::vector<Group>& groups, const Atom& atom)
{
  const int creation_mode = compression_level_ >= 0 ? NC_NETCDF4 : NC_64BIT_OFFSET;
  const int create_status = nc_create(filename_.c_str(), creation_mode, &ncid);
  if (create_status != NC_NOERR) {
    if (compression_level_ >= 0 && (create_status == NC_ENOTBUILT || create_status == NC_ENOTNC4)) {
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
  NC_CHECK(nc_put_att_int(ncid, NC_GLOBAL, "gpumd_grouping_method", NC_INT, 1, &grouping_method_));
  NC_CHECK(nc_put_att_int(ncid, NC_GLOBAL, "gpumd_group_id", NC_INT, 1, &group_id_));
  NC_CHECK(nc_put_att_text(
    ncid, NC_GLOBAL, "gpumd_quantities", quantity_list_.size(), quantity_list_.c_str()));
  NC_CHECK(
    nc_put_att_int(ncid, NC_GLOBAL, "gpumd_compression_level", NC_INT, 1, &compression_level_));

  // dimensions
  NC_CHECK(nc_def_dim(
    ncid, FRAME_STR, NC_UNLIMITED, &frame_dim)); // unlimited number of steps (can append)
  NC_CHECK(nc_def_dim(ncid, SPATIAL_STR, 3, &spatial_dim)); // number of spatial dimensions
  NC_CHECK(nc_def_dim(ncid, ATOM_STR, number_of_atoms_to_dump_, &atom_dim));
  NC_CHECK(nc_def_dim(ncid, CELL_SPATIAL_STR, 3, &cell_spatial_dim)); // unitcell lengths
  NC_CHECK(nc_def_dim(ncid, CELL_ANGULAR_STR, 3, &cell_angular_dim)); // unitcell angles
  NC_CHECK(nc_def_dim(ncid, LABEL_STR, 10, &label_dim));              // needed for cell_angular
  if (quantities_.has_group_) {
    NC_CHECK(
      nc_def_dim(ncid, GROUPING_METHOD_STR, number_of_grouping_methods_, &grouping_method_dim));
  }

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

  // Units of the cell and time variables
  NC_CHECK(nc_put_att_text(ncid, time_var, UNITS_STR, 10, "picosecond"));
  NC_CHECK(nc_put_att_text(ncid, cell_lengths_var, UNITS_STR, 8, "angstrom"));
  NC_CHECK(nc_put_att_text(ncid, cell_angles_var, UNITS_STR, 6, "degree"));

  // Per-atom data variables. The type is written per frame, as it always has been; the
  // quantities that cannot change during a run are defined further down without a frame
  // dimension, so that they are stored once rather than repeated every frame.
  dimids[0] = frame_dim;
  dimids[1] = atom_dim;
  NC_CHECK(nc_def_var(ncid, TYPE_STR, NC_INT, 2, dimids, &type_var));
  if (compression_level_ >= 0) {
#if defined(NC_HAS_HDF5) && NC_HAS_HDF5
    size_t type_chunks[2] = {
      1, std::min<size_t>(number_of_atoms_to_dump_, (1024 * 1024) / sizeof(int))};
    NC_CHECK(nc_def_var_chunking(ncid, type_var, NC_CHUNKED, type_chunks));
    NC_CHECK(nc_def_var_deflate(ncid, type_var, 1, 1, compression_level_));
#endif // NC_HAS_HDF5
  }

  define_per_frame_variable(COORDINATES_STR, 3, 3, "angstrom", coordinates_var);
  if (quantities_.has_velocity_) {
    // AMBER conventions
    define_per_frame_variable(VELOCITIES_STR, 3, 3, "angstrom/picosecond", velocities_var);
  }
  if (quantities_.has_force_) {
    define_per_frame_variable(FORCES_STR, 3, 3, "kilocalorie/mole/angstrom", forces_var);
  }
  if (quantities_.has_unwrapped_position_) {
    define_per_frame_variable(
      UNWRAPPED_COORDINATES_STR, 3, 3, "angstrom", unwrapped_coordinates_var);
  }
  if (quantities_.has_potential_) {
    define_per_frame_variable(POTENTIAL_STR, 2, 1, "eV", potential_var);
  }
  if (quantities_.has_charge_) {
    define_per_frame_variable(CHARGE_STR, 2, 1, "e", charge_var);
  }
  if (quantities_.has_bec_) {
    define_per_frame_variable(BEC_STR, 4, 9, "e", bec_var);
  }
  if (quantities_.has_virial_) {
    define_per_frame_variable(VIRIAL_STR, 4, 9, "eV", virial_var);
  }

  // Quantities that are constant over the run. They are written once and left uncompressed,
  // being one copy rather than one per frame.
  const nc_type per_atom_type = precision_ == 1 ? NC_FLOAT : NC_DOUBLE;
  if (quantities_.has_mass_) {
    dimids[0] = atom_dim;
    NC_CHECK(nc_def_var(ncid, MASS_STR, per_atom_type, 1, dimids, &mass_var));
    NC_CHECK(nc_put_att_text(ncid, mass_var, UNITS_STR, 3, "amu"));
  }
  if (quantities_.has_group_) {
    dimids[0] = atom_dim;
    dimids[1] = grouping_method_dim;
    NC_CHECK(nc_def_var(ncid, GROUP_LABELS_STR, NC_INT, 2, dimids, &group_labels_var));
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

  // The constant quantities, written once now rather than once per frame
  if (quantities_.has_mass_) {
    pack_scalar_by_precision(precision_, dump_indices_, atom.cpu_mass, pack_float_, pack_double_);
    const size_t mass_start[1] = {0};
    const size_t mass_count[1] = {size_t(number_of_atoms_to_dump_)};
    put_packed(mass_var, mass_start, mass_count);
  }
  if (quantities_.has_group_) {
    cpu_group_labels_.resize(size_t(number_of_atoms_to_dump_) * number_of_grouping_methods_);
    for (int n = 0; n < number_of_atoms_to_dump_; ++n) {
      for (int m = 0; m < number_of_grouping_methods_; ++m) {
        cpu_group_labels_[n * number_of_grouping_methods_ + m] =
          groups[m].cpu_label[dump_indices_[n]];
      }
    }
    const size_t label_start[2] = {0, 0};
    const size_t label_count[2] = {
      size_t(number_of_atoms_to_dump_), size_t(number_of_grouping_methods_)};
    NC_CHECK(
      nc_put_vara_int(ncid, group_labels_var, label_start, label_count, cpu_group_labels_.data()));
  }

  lenp = 0;
}

void DUMP_NETCDF::put_packed(const int var, const size_t* start, const size_t* count)
{
  if (precision_ == 1) {
    NC_CHECK(nc_put_vara_float(ncid, var, start, count, pack_float_.data()));
  } else {
    NC_CHECK(nc_put_vara_double(ncid, var, start, count, pack_double_.data()));
  }
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

// The layout of a NetCDF file is fixed when it is created, so appending is only meaningful when
// the run produces exactly what the file already holds. The file is named because removing or
// renaming it is what the user has to do next.
void DUMP_NETCDF::append_mismatch(const char* what)
{
  char message[512];
  snprintf(
    message,
    sizeof(message),
    "Cannot append to %s, which was written with a different %s. Remove or rename it to start a "
    "new trajectory.\n",
    filename_.c_str(),
    what);
  PRINT_INPUT_ERROR(message);
}

void DUMP_NETCDF::validate_file_definition()
{
  size_t previous_number_of_atoms = 0;
  NC_CHECK(nc_inq_dimlen(ncid, atom_dim, &previous_number_of_atoms));
  if (previous_number_of_atoms != size_t(number_of_atoms_to_dump_)) {
    append_mismatch("number of atoms");
  }

  nc_type coordinate_type;
  NC_CHECK(nc_inq_vartype(ncid, coordinates_var, &coordinate_type));
  const nc_type expected_type = precision_ == 1 ? NC_FLOAT : NC_DOUBLE;
  if (coordinate_type != expected_type) {
    append_mismatch("precision");
  }

  int previous_grouping_method;
  int previous_group_id;
  int previous_compression_level;
  NC_CHECK(nc_get_att_int(ncid, NC_GLOBAL, "gpumd_grouping_method", &previous_grouping_method));
  NC_CHECK(nc_get_att_int(ncid, NC_GLOBAL, "gpumd_group_id", &previous_group_id));
  NC_CHECK(nc_get_att_int(ncid, NC_GLOBAL, "gpumd_compression_level", &previous_compression_level));
  if (previous_grouping_method != grouping_method_ || previous_group_id != group_id_) {
    append_mismatch("group");
  }
  if (previous_compression_level != compression_level_) {
    append_mismatch("compression");
  }

  // One attribute covers every quantity, so the whole set is compared at once rather than one
  // flag at a time. Every file dump_netcdf writes records it, as an empty string when no quantity
  // was requested, so one without the attribute cannot be compared against at all. Saying so here
  // rather than where the file is opened leaves the checks above to speak first, so a file that
  // already differs in atom count or precision still gets that more specific message.
  int attribute_id;
  if (nc_inq_attid(ncid, NC_GLOBAL, "gpumd_quantities", &attribute_id) != NC_NOERR) {
    char message[512];
    snprintf(
      message,
      sizeof(message),
      "Cannot append to %s, which does not record which quantities it holds. Remove or rename it "
      "to start a new trajectory.\n",
      filename_.c_str());
    PRINT_INPUT_ERROR(message);
  }
  size_t previous_length = 0;
  NC_CHECK(nc_inq_attlen(ncid, NC_GLOBAL, "gpumd_quantities", &previous_length));
  std::string previous_quantities(previous_length, '\0');
  NC_CHECK(nc_get_att_text(ncid, NC_GLOBAL, "gpumd_quantities", &previous_quantities[0]));
  if (previous_quantities != quantity_list_) {
    append_mismatch("set of quantities");
  }

  // The quantities match, so every variable they call for is in the file.
  if (quantities_.has_velocity_) {
    NC_CHECK(nc_inq_varid(ncid, VELOCITIES_STR, &velocities_var));
  }
  if (quantities_.has_force_) {
    NC_CHECK(nc_inq_varid(ncid, FORCES_STR, &forces_var));
  }
  if (quantities_.has_unwrapped_position_) {
    NC_CHECK(nc_inq_varid(ncid, UNWRAPPED_COORDINATES_STR, &unwrapped_coordinates_var));
  }
  if (quantities_.has_potential_) {
    NC_CHECK(nc_inq_varid(ncid, POTENTIAL_STR, &potential_var));
  }
  if (quantities_.has_charge_) {
    NC_CHECK(nc_inq_varid(ncid, CHARGE_STR, &charge_var));
  }
  if (quantities_.has_bec_) {
    NC_CHECK(nc_inq_varid(ncid, BEC_STR, &bec_var));
  }
  if (quantities_.has_virial_) {
    NC_CHECK(nc_inq_varid(ncid, VIRIAL_STR, &virial_var));
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
  const auto clamp_cosine = [](double value) { return std::max(-1.0, std::min(1.0, value)); };

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

  const double cosalpha = clamp_cosine(dot(b, c) / (cell_lengths[1] * cell_lengths[2]));
  const double cosbeta = clamp_cosine(dot(a, c) / (cell_lengths[0] * cell_lengths[2]));
  const double cosgamma = clamp_cosine(dot(a, b) / (cell_lengths[0] * cell_lengths[1]));
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

void DUMP_NETCDF::write(const double global_time, const Box& box, const Atom& atom)
{
  const int number_of_atoms = atom.number_of_atoms;
  const int number_to_dump = number_of_atoms_to_dump_;

  double cell_lengths[3];
  double cell_angles[3];
  double cell_transform[9];
  const bool rotate = build_netcdf_transform(box, cell_lengths, cell_angles, cell_transform);

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

  size_t atom_start[2] = {lenp, 0};
  size_t atom_count[2] = {1, size_t(number_to_dump)};
  size_t vector_start[3] = {lenp, 0, 0};
  size_t vector_count[3] = {1, size_t(number_to_dump), 3};
  size_t tensor_start[4] = {lenp, 0, 0, 0};
  size_t tensor_count[4] = {1, size_t(number_to_dump), 3, 3};

  for (int n = 0; n < number_to_dump; ++n) {
    cpu_type_to_dump_[n] = atom.cpu_type[dump_indices_[n]];
  }
  NC_CHECK(nc_put_vara_int(ncid, type_var, atom_start, atom_count, cpu_type_to_dump_.data()));

  pack_vector_by_precision(
    precision_,
    dump_indices_,
    number_of_atoms,
    rotate,
    cell_transform,
    1.0,
    atom.cpu_position_per_atom,
    pack_float_,
    pack_double_);
  put_packed(coordinates_var, vector_start, vector_count);

  if (quantities_.has_velocity_) {
    const double natural_to_A_per_ps =
      1.0 / TIME_UNIT_CONVERSION * 1000.0; // * 1000 from A/fs to A/ps
    pack_vector_by_precision(
      precision_,
      dump_indices_,
      number_of_atoms,
      rotate,
      cell_transform,
      natural_to_A_per_ps,
      atom.cpu_velocity_per_atom,
      pack_float_,
      pack_double_);
    put_packed(velocities_var, vector_start, vector_count);
  }
  if (quantities_.has_force_) {
    pack_vector_by_precision(
      precision_,
      dump_indices_,
      number_of_atoms,
      rotate,
      cell_transform,
      EV_TO_KCAL_PER_MOL,
      cpu_force_per_atom_,
      pack_float_,
      pack_double_);
    put_packed(forces_var, vector_start, vector_count);
  }
  if (quantities_.has_unwrapped_position_) {
    pack_vector_by_precision(
      precision_,
      dump_indices_,
      number_of_atoms,
      rotate,
      cell_transform,
      1.0,
      cpu_unwrapped_position_,
      pack_float_,
      pack_double_);
    put_packed(unwrapped_coordinates_var, vector_start, vector_count);
  }
  if (quantities_.has_potential_) {
    pack_scalar_by_precision(
      precision_, dump_indices_, cpu_potential_per_atom_, pack_float_, pack_double_);
    put_packed(potential_var, atom_start, atom_count);
  }
  if (quantities_.has_charge_) {
    pack_scalar_by_precision(precision_, dump_indices_, atom.cpu_charge, pack_float_, pack_double_);
    put_packed(charge_var, atom_start, atom_count);
  }
  if (quantities_.has_bec_) {
    pack_tensor_by_precision(
      precision_,
      dump_indices_,
      number_of_atoms,
      rotate,
      cell_transform,
      ROW_MAJOR_COMPONENT,
      cpu_bec_,
      pack_float_,
      pack_double_);
    put_packed(bec_var, tensor_start, tensor_count);
  }
  if (quantities_.has_virial_) {
    pack_tensor_by_precision(
      precision_,
      dump_indices_,
      number_of_atoms,
      rotate,
      cell_transform,
      VIRIAL_COMPONENT,
      cpu_virial_per_atom_,
      pack_float_,
      pack_double_);
    put_packed(virial_var, tensor_start, tensor_count);
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
    const auto active_file = std::find(active_files_.begin(), active_files_.end(), filename_);
    if (active_file != active_files_.end()) {
      active_files_.erase(active_file);
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

  // Copy back what was asked for, in full. Selecting the group happens on the host while
  // packing, through dump_indices_, which is one path for the grouped and the whole-system case.
  atom.position_per_atom.copy_to_host(atom.cpu_position_per_atom.data());
  if (quantities_.has_velocity_) {
    atom.velocity_per_atom.copy_to_host(atom.cpu_velocity_per_atom.data());
  }
  if (quantities_.has_force_) {
    atom.force_per_atom.copy_to_host(cpu_force_per_atom_.data());
  }
  if (quantities_.has_potential_) {
    atom.potential_per_atom.copy_to_host(cpu_potential_per_atom_.data());
  }
  if (quantities_.has_unwrapped_position_) {
    atom.unwrapped_position.copy_to_host(cpu_unwrapped_position_.data());
  }
  if (quantities_.has_virial_) {
    atom.virial_per_atom.copy_to_host(cpu_virial_per_atom_.data());
  }
  if (quantities_.has_charge_) {
    if (is_nep_charge_) {
      GPU_Vector<float>& nep_charge = force.potentials[0]->get_charge_reference();
      nep_charge.copy_to_host(atom.cpu_charge.data());
    } else {
      atom.charge.copy_to_host(atom.cpu_charge.data());
    }
  }
  if (quantities_.has_bec_) {
    GPU_Vector<float>& gpu_bec = force.potentials[0]->get_bec_reference();
    gpu_bec.copy_to_host(cpu_bec_.data());
  }

  write(global_time, box, atom);
}

#endif
