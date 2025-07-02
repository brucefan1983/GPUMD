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

/*-----------------------------------------------------------------------------------------------100
Calculate:
    Angular-dependent Radial distribution function (AngularRDF)
--------------------------------------------------------------------------------------------------*/

#include "angular_rdf.cuh"
#include "integrate/integrate.cuh"
#include "force/neighbor.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include "parse_utilities.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <cmath>
#include <cstring>

namespace
{
static __global__ void gpu_find_rdf_ON1(
  const int N,                            // total number of atoms
  const double density,                   // system density
  const Box box,                          // simulation box
  const int* __restrict__ cell_counts,    // number of atoms in each cell
  const int* __restrict__ cell_count_sum, // cumulative sum of atoms in cells
  const int* __restrict__ cell_contents,  // atom indices in each cell
  const int nx,
  const int ny,
  const int nz,
  const double rc_inv,                // cutoff radius inverse
  const double* __restrict__ x,       // atom x coordinates
  const double* __restrict__ y,       // atom y coordinates
  const double* __restrict__ z,       // atom z coordinates
  const double* __restrict__ radial_, // radial distance array
  const double* __restrict__ theta_,  // angular distance array
  double* rdf_,                       // RDF results array
  const int rdf_bins_,                // number of bins
  const int rdf_theta_bins_,          // number of bins
  const double r_step_,               // radial step size
  const double theta_step_)           // theta step size
{
  // get current atom index
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  double rdf_PI = 3.14159265358979323846;

  if (n1 < N) {
    // get current atom coordinates
    const double x1 = x[n1];
    const double y1 = y[n1];
    const double z1 = z[n1];

    // calculate current atom cell ID
    int cell_id;
    int cell_id_x, cell_id_y, cell_id_z;
    find_cell_id(box, x1, y1, z1, rc_inv, nx, ny, nz, cell_id_x, cell_id_y, cell_id_z, cell_id);

    // set search range according to periodic boundary conditions
    const int z_lim = box.pbc_z ? 2 : 0;
    const int y_lim = box.pbc_y ? 2 : 0;
    const int x_lim = box.pbc_x ? 2 : 0;

    // loop over neighboring cells
    for (int k = -z_lim; k <= z_lim; ++k) {
      for (int j = -y_lim; j <= y_lim; ++j) {
        for (int i = -x_lim; i <= x_lim; ++i) {
          // calculate neighboring cell ID and handle periodic boundary conditions
          int neighbor_cell = cell_id + k * nx * ny + j * nx + i;
          // handle periodic boundary conditions
          if (cell_id_x + i < 0)
            neighbor_cell += nx;
          if (cell_id_x + i >= nx)
            neighbor_cell -= nx;
          if (cell_id_y + j < 0)
            neighbor_cell += ny * nx;
          if (cell_id_y + j >= ny)
            neighbor_cell -= ny * nx;
          if (cell_id_z + k < 0)
            neighbor_cell += nz * ny * nx;
          if (cell_id_z + k >= nz)
            neighbor_cell -= nz * ny * nx;

          // loop over atoms in neighboring cell
          const int num_atoms_neighbor_cell = cell_counts[neighbor_cell];
          const int num_atoms_previous_cells = cell_count_sum[neighbor_cell];

          for (int m = 0; m < num_atoms_neighbor_cell; ++m) {
            const int n2 = cell_contents[num_atoms_previous_cells + m];
            if (n2 >= 0 && n2 < N && n1 != n2) {
              // calculate distance between atom pairs
              double x12 = x[n2] - x1;
              double y12 = y[n2] - y1;
              double z12 = z[n2] - z1;
              apply_mic(box, x12, y12, z12); // minimum image convention
              const double d2 = x12 * x12 + y12 * y12 + z12 * z12;
              double theta = atan2(y12, x12);

              // update angular RDF histogram
              for (int w = 0; w < rdf_bins_; w++) {
                double r_low = radial_[w] - r_step_ / 2;
                double r_up = radial_[w] + r_step_ / 2;
                if (d2 > r_low * r_low && d2 <= r_up * r_up) {
                  for (int t = 0; t < rdf_theta_bins_; t++) {
                    double theta_low = theta_[t] - theta_step_ / 2;
                    double theta_up = theta_[t] + theta_step_ / 2;
                    if (theta > theta_low && theta <= theta_up) {
                      //  RDF normalization factor calculation
                      double shell_volume =
                        4.0 / 3.0 * rdf_PI * (r_up * r_up * r_up - r_low * r_low * r_low);
                      double theta_area = (theta_up - theta_low) / (2 * rdf_PI);
                      double bin_volume = theta_area * shell_volume;
                      rdf_[n1 * rdf_bins_ * rdf_theta_bins_ + w * rdf_theta_bins_ + t] +=
                        1 / (N * density * bin_volume);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

static __global__ void gpu_find_rdf_ON1(
  const int N,
  const double density1,
  const double density2,
  const double num_atom1_,
  const double num_atom2_,
  const double atom_id1_,
  const double atom_id2_,
  const Box box,
  const int* __restrict__ cell_counts,
  const int* __restrict__ cell_count_sum,
  const int* __restrict__ cell_contents,
  const int nx,
  const int ny,
  const int nz,
  const double rc_inv,
  const double* __restrict__ x,
  const double* __restrict__ y,
  const double* __restrict__ z,
  const int* __restrict__ type,
  const double* __restrict__ radial_,
  const double* __restrict__ theta_,
  double* rdf_,
  const int rdf_bins_,
  const int rdf_theta_bins_,
  const double r_step_,
  const double theta_step_)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  double rdf_PI = 3.14159265358979323846;
  if (n1 < N && type[n1] == atom_id1_) {
    const double x1 = x[n1];
    const double y1 = y[n1];
    const double z1 = z[n1];
    int cell_id;
    int cell_id_x;
    int cell_id_y;
    int cell_id_z;
    find_cell_id(box, x1, y1, z1, rc_inv, nx, ny, nz, cell_id_x, cell_id_y, cell_id_z, cell_id);

    const int z_lim = box.pbc_z ? 2 : 0;
    const int y_lim = box.pbc_y ? 2 : 0;
    const int x_lim = box.pbc_x ? 2 : 0;

    // get radial descriptors
    for (int k = -z_lim; k <= z_lim; ++k) {
      for (int j = -y_lim; j <= y_lim; ++j) {
        for (int i = -x_lim; i <= x_lim; ++i) {
          int neighbor_cell = cell_id + k * nx * ny + j * nx + i;
          if (cell_id_x + i < 0)
            neighbor_cell += nx;
          if (cell_id_x + i >= nx)
            neighbor_cell -= nx;
          if (cell_id_y + j < 0)
            neighbor_cell += ny * nx;
          if (cell_id_y + j >= ny)
            neighbor_cell -= ny * nx;
          if (cell_id_z + k < 0)
            neighbor_cell += nz * ny * nx;
          if (cell_id_z + k >= nz)
            neighbor_cell -= nz * ny * nx;

          const int num_atoms_neighbor_cell = cell_counts[neighbor_cell];
          const int num_atoms_previous_cells = cell_count_sum[neighbor_cell];

          for (int m = 0; m < num_atoms_neighbor_cell; ++m) {
            const int n2 = cell_contents[num_atoms_previous_cells + m];
            if (n2 >= 0 && n2 < N && n1 != n2 && type[n2] == atom_id2_) {
              double x12 = x[n2] - x1;
              double y12 = y[n2] - y1;
              double z12 = z[n2] - z1;
              apply_mic(box, x12, y12, z12);
              const double d2 = x12 * x12 + y12 * y12 + z12 * z12;
              double theta = atan2(y12, x12);
              for (int w = 0; w < rdf_bins_; w++) {
                double r_low = radial_[w] - r_step_ / 2;
                double r_up = radial_[w] + r_step_ / 2;
                if (d2 > r_low * r_low && d2 <= r_up * r_up) {
                  for (int t = 0; t < rdf_theta_bins_; t++) {
                    double theta_low = theta_[t] - theta_step_ / 2;
                    double theta_up = theta_[t] + theta_step_ / 2;
                    if (theta > theta_low && theta <= theta_up) {
                      double shell_volume =
                        4.0 / 3.0 * rdf_PI * (r_up * r_up * r_up - r_low * r_low * r_low);
                      double theta_area = (theta_up - theta_low) / (2 * rdf_PI);
                      double bin_volume = theta_area * shell_volume;
                      rdf_[n1 * rdf_bins_ * rdf_theta_bins_ + w * rdf_theta_bins_ + t] +=
                        1 / (num_atom1_ * density2 * bin_volume);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  } else if (n1 < N && type[n1] == atom_id2_) {
    const double x1 = x[n1];
    const double y1 = y[n1];
    const double z1 = z[n1];
    int cell_id;
    int cell_id_x;
    int cell_id_y;
    int cell_id_z;
    find_cell_id(box, x1, y1, z1, rc_inv, nx, ny, nz, cell_id_x, cell_id_y, cell_id_z, cell_id);

    const int z_lim = box.pbc_z ? 2 : 0;
    const int y_lim = box.pbc_y ? 2 : 0;
    const int x_lim = box.pbc_x ? 2 : 0;

    // get radial descriptors
    for (int k = -z_lim; k <= z_lim; ++k) {
      for (int j = -y_lim; j <= y_lim; ++j) {
        for (int i = -x_lim; i <= x_lim; ++i) {
          int neighbor_cell = cell_id + k * nx * ny + j * nx + i;
          if (cell_id_x + i < 0)
            neighbor_cell += nx;
          if (cell_id_x + i >= nx)
            neighbor_cell -= nx;
          if (cell_id_y + j < 0)
            neighbor_cell += ny * nx;
          if (cell_id_y + j >= ny)
            neighbor_cell -= ny * nx;
          if (cell_id_z + k < 0)
            neighbor_cell += nz * ny * nx;
          if (cell_id_z + k >= nz)
            neighbor_cell -= nz * ny * nx;

          const int num_atoms_neighbor_cell = cell_counts[neighbor_cell];
          const int num_atoms_previous_cells = cell_count_sum[neighbor_cell];

          for (int m = 0; m < num_atoms_neighbor_cell; ++m) {
            const int n2 = cell_contents[num_atoms_previous_cells + m];
            if (n2 >= 0 && n2 < N && n1 != n2 && type[n2] == atom_id1_) {
              double x12 = x1 - x[n2];
              double y12 = y1 - y[n2];
              double z12 = z1 - z[n2];
              apply_mic(box, x12, y12, z12);
              const double d2 = x12 * x12 + y12 * y12 + z12 * z12;
              double theta = atan2(y12, x12);
              for (int w = 0; w < rdf_bins_; w++) {
                double r_low = radial_[w] - r_step_ / 2;
                double r_up = radial_[w] + r_step_ / 2;
                if (d2 > r_low * r_low && d2 <= r_up * r_up) {
                  for (int t = 0; t < rdf_theta_bins_; t++) {
                    double theta_low = theta_[t] - theta_step_ / 2;
                    double theta_up = theta_[t] + theta_step_ / 2;
                    if (theta > theta_low && theta <= theta_up) {
                      double shell_volume =
                        4.0 / 3.0 * rdf_PI * (r_up * r_up * r_up - r_low * r_low * r_low);
                      double theta_area = (theta_up - theta_low) / (2 * rdf_PI);
                      double bin_volume = theta_area * shell_volume;
                      rdf_[n1 * rdf_bins_ * rdf_theta_bins_ + w * rdf_theta_bins_ + t] +=
                        1 / (num_atom2_ * density1 * bin_volume);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
} // namespace

// Function to calculate angular radial distribution function (RDF)
// Parameters:
// bead - current bead index
// rdf_atom_count - total number of atom types for RDF calculation
// rdf_atom_ - current atom type index
// atom_id1_ - array of first atom type IDs
// atom_id2_ - array of second atom type IDs
// atom_id1_typesize - number of atoms of first type for each pair
// atom_id2_typesize - number of atoms of second type for each pair
// density1 - number density of first atom type
// density2 - number density of second atom type
// rc - cutoff radius
// box - simulation box
// type - atom type array
// position_per_atom - atom positions array
// cell_count - number of atoms in each cell
// cell_count_sum - cumulative sum of atoms in cells
// cell_contents - atom indices in each cell
// num_bins_0,1,2 - number of cells in x,y,z directions
// rc_inv_cell_list - inverse of cell list cutoff
// radial_ - radial distance array
// rdf_g_ - RDF array
// rdf_r_bins_ - number of RDF histogram bins
// r_step_ - RDF bin width
void AngularRDF::find_angular_rdf(
  const int bead,
  const int rdf_atom_count,
  const int rdf_atom_, // Current atom pair index
  int* atom_id1_,
  int* atom_id2_,
  std::vector<int>& atom_id1_typesize,
  std::vector<int>& atom_id2_typesize,
  std::vector<double>& density1,
  std::vector<double>& density2,
  double rc,
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& cell_count,
  GPU_Vector<int>& cell_count_sum,
  GPU_Vector<int>& cell_contents,
  int num_bins_0,
  int num_bins_1,
  int num_bins_2,
  const double rc_inv_cell_list,
  GPU_Vector<double>& radial_, // Size: num_atoms_ * rdf_r_bins_ * rdf_theta_bins_
  GPU_Vector<double>& theta_,
  GPU_Vector<double>& rdf_g_,
  const int rdf_r_bins_,
  const int rdf_theta_bins_,
  const double r_step_,
  const double theta_step_)
{
  const int N = position_per_atom.size() / 3;
  const int block_size = 256;
  const int grid_size = (N - 1) / block_size + 1;
  const double* x = position_per_atom.data();
  const double* y = position_per_atom.data() + N;
  const double* z = position_per_atom.data() + N * 2;

  double* rdf_g_ind = rdf_g_.data() + rdf_atom_ * N * rdf_r_bins_ * rdf_theta_bins_;

  if (rdf_atom_ == 0) {
    gpu_find_rdf_ON1<<<grid_size, block_size>>>(
      N,
      density1[rdf_atom_],
      box,
      cell_count.data(),
      cell_count_sum.data(),
      cell_contents.data(),
      num_bins_0,
      num_bins_1,
      num_bins_2,
      rc_inv_cell_list,
      x,
      y,
      z,
      radial_.data(),
      theta_.data(),
      rdf_g_ind,
      rdf_r_bins_,
      rdf_theta_bins_,
      r_step_,
      theta_step_);
    GPU_CHECK_KERNEL
    // GPU_CHECK_KERNEL is a macro for checking if the CUDA kernel execution is successful.

  } else {
    gpu_find_rdf_ON1<<<grid_size, block_size>>>(
      N,
      density1[rdf_atom_],
      density2[rdf_atom_],
      atom_id1_typesize[rdf_atom_ - 1],
      atom_id2_typesize[rdf_atom_ - 1],
      atom_id1_[rdf_atom_ - 1],
      atom_id2_[rdf_atom_ - 1],
      box,
      cell_count.data(),
      cell_count_sum.data(),
      cell_contents.data(),
      num_bins_0,
      num_bins_1,
      num_bins_2,
      rc_inv_cell_list,
      x,
      y,
      z,
      type.data(),
      radial_.data(),
      theta_.data(),
      rdf_g_ind,
      rdf_r_bins_,
      rdf_theta_bins_,
      r_step_,
      theta_step_);
    GPU_CHECK_KERNEL
  }
}

AngularRDF::AngularRDF(
  const char** param,
  const int num_param,
  Box& box,
  const int number_of_types,
  const int number_of_steps)
{
  parse(param, num_param, box, number_of_types, number_of_steps);
  property_name = "compute_angular_rdf";
}

void AngularRDF::preprocess(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  // if not compute RDF, return directly
  if (!compute_)
    return;

  // if PIMD, return directly, currently not support PIMD
  if (integrate.type >= 31) {
    return;
  }

  // calculate radial step size
  r_step_ = r_cut_ / rdf_r_bins_;

  // calculate theta step size
  theta_step_ = 2 * PI / rdf_theta_bins_; // total angle is 360 degrees, theta step size is 360
                                          // degrees divided by theta bin number

  // initialize radial distance array
  std::vector<double> radial_cpu(rdf_r_bins_);
  for (int i = 0; i < rdf_r_bins_; i++) {
    radial_cpu[i] = i * r_step_ + r_step_ / 2; // center of each bin
  }
  radial_.resize(rdf_r_bins_);
  radial_.copy_from_host(radial_cpu.data()); // copy data to GPU

  // initialize theta distance array
  std::vector<double> theta_cpu(rdf_theta_bins_);
  for (int i = 0; i < rdf_theta_bins_; i++) {
    theta_cpu[i] = -PI + i * theta_step_ +
                   theta_step_ / 2; // center of each bin, atan2 returns value range is -pi to pi
  }
  theta_.resize(rdf_theta_bins_);
  theta_.copy_from_host(theta_cpu.data());

  rdf_N_ = atom.number_of_atoms;
  num_atoms_ = atom.number_of_atoms * rdf_atom_count;

  density1.resize(rdf_atom_count);
  density2.resize(rdf_atom_count);

  atom_id1_typesize.resize(
    rdf_atom_count -
    1); // we always consider the calculation of all atom pairs, here only store Partial AngularRDF
  atom_id2_typesize.resize(rdf_atom_count - 1);

  // initialize atom type size array
  for (int a = 0; a < rdf_atom_count - 1; a++) {
    atom_id1_typesize[a] = atom.cpu_type_size[atom_id1_[a]];
    atom_id2_typesize[a] = atom.cpu_type_size[atom_id2_[a]];
  }

  rdf_g_.resize(num_atoms_ * rdf_r_bins_ * rdf_theta_bins_, 0);
  rdf_.resize(num_atoms_ * rdf_r_bins_ * rdf_theta_bins_, 0);
  cell_count.resize(atom.number_of_atoms);
  cell_count_sum.resize(atom.number_of_atoms);
  cell_contents.resize(atom.number_of_atoms);
}

void AngularRDF::process(
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
  // if PIMD, return directly, currently not support PIMD
  if (integrate.type >= 31) {
    return;
  }

  // if not compute RDF, return directly
  if (!compute_)
    return;

  // if step is not a multiple of num_interval_, return directly
  if ((step + 1) % num_interval_ != 0) {
    return;
  }

  // repeat times
  num_repeat_++;

  // calculate number density
  density1[0] = rdf_N_ / box.get_volume();
  density2[0] = rdf_N_ / box.get_volume();
  for (int a = 0; a < rdf_atom_count - 1; a++) {
    density1[a + 1] = atom_id1_typesize[a] / box.get_volume();
    density2[a + 1] = atom_id2_typesize[a] / box.get_volume();
  }

  int classical = 0;
  const double rc_cell_list = 0.5 * r_cut_;
  const double rc_inv_cell_list = 2.0 / r_cut_;
  int num_bins[3];
  box.get_num_bins(rc_cell_list, num_bins);
  find_cell_list(
    rc_cell_list, num_bins, box, atom.position_per_atom, cell_count, cell_count_sum, cell_contents);

  for (int a = 0; a < rdf_atom_count; a++) {
    find_angular_rdf(
      classical,
      rdf_atom_count,
      a,
      atom_id1_,
      atom_id2_,
      atom_id1_typesize,
      atom_id2_typesize,
      density1,
      density2,
      r_cut_,
      box,
      atom.type,
      atom.position_per_atom,
      cell_count,
      cell_count_sum,
      cell_contents,
      num_bins[0],
      num_bins[1],
      num_bins[2],
      rc_inv_cell_list,
      radial_,
      theta_,
      rdf_g_,
      rdf_r_bins_,
      rdf_theta_bins_,
      r_step_,
      theta_step_);
  }
}

void AngularRDF::postprocess(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature)
{
  if (!compute_)
    return;
  if (integrate.type >= 31)
    return;

  CHECK(gpuMemcpy(
    rdf_.data(),
    rdf_g_.data(),
    sizeof(double) * num_atoms_ * rdf_r_bins_ * rdf_theta_bins_,
    gpuMemcpyDeviceToHost));
  CHECK(gpuDeviceSynchronize()); // needed for pre-Pascal GPU

  std::vector<double> rdf_average(rdf_atom_count * rdf_r_bins_ * rdf_theta_bins_, 0.0);
  for (int a = 0; a < rdf_atom_count; a++) {
    for (int m = 0; m < rdf_N_; m++) {
      for (int x = 0; x < rdf_r_bins_; x++) {
        for (int t = 0; t < rdf_theta_bins_; t++) {
          rdf_average[a * rdf_r_bins_ * rdf_theta_bins_ + x * rdf_theta_bins_ + t] +=
            rdf_
              [a * rdf_N_ * rdf_r_bins_ * rdf_theta_bins_ + m * rdf_r_bins_ * rdf_theta_bins_ +
               x * rdf_theta_bins_ + t] /
            num_repeat_;
        }
      }
    }
  }
  FILE* fid = fopen("angular_rdf.out", "a");
  fprintf(fid, "#radius theta");
  // print the header
  for (int a = 0; a < rdf_atom_count; a++) {
    if (a == 0) {
      fprintf(fid, " total");
    } else {
      fprintf(fid, " type_%d_%d", atom_id1_[a - 1], atom_id2_[a - 1]);
    }
  }
  fprintf(fid, "\n");
  // print the data
  for (int nc = 0; nc < rdf_r_bins_; nc++) {
    for (int tc = 0; tc < rdf_theta_bins_; tc++) {
      fprintf(
        fid, "%.5f %.5f", nc * r_step_ + r_step_ / 2, -PI + tc * theta_step_ + theta_step_ / 2);
      for (int a = 0; a < rdf_atom_count; a++) {
        if (a == 0) {
          fprintf(fid, " %.5f", rdf_average[nc * rdf_theta_bins_ + tc]);
        } else {
          fprintf(
            fid,
            " %.5f",
            (atom_id1_[a - 1] == atom_id2_[a - 1])
              ? rdf_average[a * rdf_r_bins_ * rdf_theta_bins_ + nc * rdf_theta_bins_ + tc]
              : rdf_average[a * rdf_r_bins_ * rdf_theta_bins_ + nc * rdf_theta_bins_ + tc] / 2);
        }
      }
      fprintf(fid, "\n");
    }
  }
  fflush(fid);
  fclose(fid);

  compute_ = false;
  for (int s = 0; s < 6; s++) {
    atom_id1_[s] = -1;
    atom_id2_[s] = -1;
  }
  rdf_atom_count = 1;
  num_repeat_ = 0;
}

void AngularRDF::parse(
  const char** param,
  const int num_param,
  Box& box,
  const int number_of_types,
  const int number_of_steps)
{
  printf("Compute Angular RDF.\n");
  compute_ = true;

  if (num_param < 5) {
    PRINT_INPUT_ERROR("compute_angular_rdf should have at least 4 parameters.\n");
  }
  if (num_param > 23) {
    PRINT_INPUT_ERROR("compute_angular_rdf has too many parameters.\n");
  }

  // radial cutoff
  if (!is_valid_real(param[1], &r_cut_)) {
    PRINT_INPUT_ERROR("radial cutoff should be a number.\n");
  }
  if (r_cut_ <= 0) {
    PRINT_INPUT_ERROR("radial cutoff should be positive.\n");
  }
  double thickness_half[3] = {
    box.get_volume() / box.get_area(0) / 2.5,
    box.get_volume() / box.get_area(1) / 2.5,
    box.get_volume() / box.get_area(2) / 2.5};
  if (r_cut_ > thickness_half[0] || r_cut_ > thickness_half[1] || r_cut_ > thickness_half[2]) {
    std::string message =
      "The box has a thickness < 2.5 RDF radial cutoffs in a periodic direction.\n"
      "                Please increase the periodic direction(s).\n";
    PRINT_INPUT_ERROR(message.c_str());
  }
  printf("    radial cutoff %g.\n", r_cut_);

  // number of bins
  if (!is_valid_int(param[2], &rdf_r_bins_)) {
    PRINT_INPUT_ERROR("number of bins should be an integer.\n");
  }
  if (rdf_r_bins_ <= 20) {
    PRINT_INPUT_ERROR("A larger nbins is recommended.\n");
  }

  if (rdf_r_bins_ > 500) {
    PRINT_INPUT_ERROR("A smaller nbins is recommended.\n");
  }

  printf("    radial cutoff will be divided into %d bins.\n", rdf_r_bins_);

  // 角度方向的bin数量
  if (!is_valid_int(param[3], &rdf_theta_bins_)) {
    PRINT_INPUT_ERROR("number of theta bins should be an integer.\n");
  }
  if (rdf_theta_bins_ <= 20) {
    PRINT_INPUT_ERROR("A larger ntheta is recommended.\n");
  }
  printf("    theta cutoff will be divided into %d bins.\n", rdf_theta_bins_);

  // sample interval
  if (!is_valid_int(param[4], &num_interval_)) {
    PRINT_INPUT_ERROR("interval step per sample should be an integer.\n");
  }
  if (num_interval_ <= 0) {
    PRINT_INPUT_ERROR("interval step per sample should be positive.\n");
  }
  printf("    Angular RDF sample interval is %d step.\n", num_interval_);

  // Process optional arguments
  for (int k = 5; k < num_param; k += 3) {
    if (strcmp(param[k], "atom") == 0) {
      int k_a = (k - 5) / 3;
      rdf_atom_count++;
      if (!is_valid_int(param[k + 1], &atom_id1_[k_a])) {
        PRINT_INPUT_ERROR("atom type index1 should be an integer.\n");
      }
      if (atom_id1_[k_a] < 0) {
        PRINT_INPUT_ERROR("atom type index1 should be non-negative.\n");
      }
      if (atom_id1_[k_a] > number_of_types) {
        PRINT_INPUT_ERROR("atom type index1 should be less than number of atomic types.\n");
      }
      if (!is_valid_int(param[k + 2], &atom_id2_[k_a])) {
        PRINT_INPUT_ERROR("atom type index2 should be an integer.\n");
      }
      if (atom_id2_[k_a] < 0) {
        PRINT_INPUT_ERROR("atom type index2 should be non-negative.\n");
      }
      if (atom_id2_[k_a] > number_of_types) {
        PRINT_INPUT_ERROR("atom type index1 should be less than number of atomic types.\n");
      }
    } else {
      PRINT_INPUT_ERROR("Unrecognized argument in compute_angular_rdf.\n");
    }
  }
}
