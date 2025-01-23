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
#include "force/neighbor.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include "parse_utilities.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <cstring>

namespace
{
// GPU核函数：计算单一原子类型的RDF
static __global__ void gpu_find_rdf_ON1(
  //__global__,  CPU调用，GPU执行; __device__, 只能在GPU上调用和执行;
  //__host__, 只能在CPU上调用和执行.
  const int N,                            // 总原子数
  const double density,                   // 系统密度
  const Box box,                          // 模拟盒子
  const int* __restrict__ cell_counts,    // 每个晶胞的原子数
  const int* __restrict__ cell_count_sum, // 晶胞原子数累积和
  const int* __restrict__ cell_contents,  // 晶胞中的原子索引
  const int nx,
  const int ny,
  const int nz,                       // 晶胞网格维度
  const double rc_inv,                // 截断半径倒数
  const double* __restrict__ x,       // 原子x坐标
  const double* __restrict__ y,       // 原子y坐标
  const double* __restrict__ z,       // 原子z坐标
  const double* __restrict__ radial_, // 径向距离数组
  double* rdf_,                       // RDF结果数组
  const int rdf_bins_,                // bin数量
  const double r_step_)               // 步长
{
  // 获取当前线程处理的原子索引
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  double rdf_PI = 3.14159265358979323846;

  if (n1 < N) {
    // 获取当前原子的坐标
    const double x1 = x[n1];
    const double y1 = y[n1];
    const double z1 = z[n1];

    // 计算当前原子所在的晶胞ID
    int cell_id;
    int cell_id_x, cell_id_y, cell_id_z;
    find_cell_id(box, x1, y1, z1, rc_inv, nx, ny, nz, cell_id_x, cell_id_y, cell_id_z, cell_id);

    // 根据周期性边界条件设置搜索范围
    const int z_lim = box.pbc_z ? 2 : 0;
    const int y_lim = box.pbc_y ? 2 : 0;
    const int x_lim = box.pbc_x ? 2 : 0;

    // 遍历邻近晶胞
    for (int k = -z_lim; k <= z_lim; ++k) {
      for (int j = -y_lim; j <= y_lim; ++j) {
        for (int i = -x_lim; i <= x_lim; ++i) {
          // 计算邻近晶胞ID并处理周期性边界
          int neighbor_cell = cell_id + k * nx * ny + j * nx + i;
          // ... (周期性边界处理代码)
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

          // 遍历邻近晶胞中的原子
          const int num_atoms_neighbor_cell = cell_counts[neighbor_cell];
          const int num_atoms_previous_cells = cell_count_sum[neighbor_cell];

          for (int m = 0; m < num_atoms_neighbor_cell; ++m) {
            const int n2 = cell_contents[num_atoms_previous_cells + m];
            if (n2 >= 0 && n2 < N && n1 != n2) {
              // 计算原子对之间的距离
              double x12 = x[n2] - x1;
              double y12 = y[n2] - y1;
              double z12 = z[n2] - z1;
              apply_mic(box, x12, y12, z12); // 最小镜像约定
              const double d2 = x12 * x12 + y12 * y12 + z12 * z12;

              // 更新RDF直方图
              for (int w = 0; w < rdf_bins_; w++) {
                double r_low = (radial_[w] - r_step_ / 2) * (radial_[w] - r_step_ / 2);
                double r_up = (radial_[w] + r_step_ / 2) * (radial_[w] + r_step_ / 2);
                double r_mid_sqaure = radial_[w] * radial_[w];
                if (d2 > r_low && d2 <= r_up) {
                  // RDF归一化因子计算
                  rdf_[n1 * rdf_bins_ + w] +=
                    1 / (N * density * r_mid_sqaure * 4 * rdf_PI * r_step_);
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
  double* rdf_,
  const int rdf_bins_,
  const double r_step_)
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
              for (int w = 0; w < rdf_bins_; w++) {
                double r_low = (radial_[w] - r_step_ / 2) * (radial_[w] - r_step_ / 2);
                double r_up = (radial_[w] + r_step_ / 2) * (radial_[w] + r_step_ / 2);
                double r_mid_sqaure = radial_[w] * radial_[w];
                if (d2 > r_low && d2 <= r_up) {
                  rdf_[n1 * rdf_bins_ + w] +=
                    1 / (num_atom1_ * density2 * r_mid_sqaure * 4 * rdf_PI * r_step_);
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
              double x12 = x[n2] - x1;
              double y12 = y[n2] - y1;
              double z12 = z[n2] - z1;
              apply_mic(box, x12, y12, z12);
              const double d2 = x12 * x12 + y12 * y12 + z12 * z12;
              for (int w = 0; w < rdf_bins_; w++) {
                double r_low = (radial_[w] - r_step_ / 2) * (radial_[w] - r_step_ / 2);
                double r_up = (radial_[w] + r_step_ / 2) * (radial_[w] + r_step_ / 2);
                double r_mid_sqaure = radial_[w] * radial_[w];
                if (d2 > r_low && d2 <= r_up) {
                  rdf_[n1 * rdf_bins_ + w] +=
                    1 / (num_atom2_ * density1 * r_mid_sqaure * 4 * rdf_PI * r_step_);
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
  const int rdf_atom_,
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
  GPU_Vector<double>& radial_,
  GPU_Vector<double>& rdf_g_,
  const int rdf_r_bins_,
  const double r_step_)
{
  const int N = position_per_atom.size() / 3;
  const int block_size = 256;
  const int grid_size = (N - 1) / block_size + 1;
  const double* x = position_per_atom.data();
  const double* y = position_per_atom.data() + N;
  const double* z = position_per_atom.data() + N * 2;

  double* rdf_g_ind =
    rdf_g_.data() + bead * rdf_atom_count * N * rdf_r_bins_ + rdf_atom_ * N * rdf_r_bins_;

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
      rdf_g_ind,
      rdf_r_bins_,
      r_step_);
    GPU_CHECK_KERNEL
    // GPU_CHECK_KERNEL 是一个用于检查CUDA核函数执行是否成功的宏定义。

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
      rdf_g_ind,
      rdf_r_bins_,
      r_step_);
    GPU_CHECK_KERNEL
  }
}

// 预处理函数:初始化角度相关RDF计算所需的数据结构
void AngularRDF::preprocess(
  const bool is_pimd,              // 是否为路径积分分子动力学
  const int number_of_beads,       // PIMD珠子数量
  const int num_atoms,             // 系统总原子数
  std::vector<int>& cpu_type_size) // 每种原子类型的数量
{
  // 如果不需要计算RDF则直接返回
  if (!compute_)
    return;

  // 如果为PIMD则直接返回, 当前不支持PIMD
  if (is_pimd) {
    return;
  }

  // 计算径向步长
  r_step_ = r_cut_ / rdf_r_bins_;

  // 计算角度步长
  theta_step_ = M_PI / rdf_theta_bins_; // 由于对称性，总角度为180度，角度步长为180度除以角度bin数量

  // 初始化径向距离数组
  std::vector<double> radial_cpu(rdf_r_bins_);
  for (int i = 0; i < rdf_r_bins_; i++) {
    radial_cpu[i] = i * r_step_ + r_step_ / 2; // 每个bin的中心位置
  }
  radial_.resize(rdf_r_bins_);
  radial_.copy_from_host(radial_cpu.data()); // 将数据复制到GPU

  // 初始化角度距离数组
  std::vector<double> theta_cpu(rdf_theta_bins_);
  for (int i = 0; i < rdf_theta_bins_; i++) {
    theta_cpu[i] = i * theta_step_ + theta_step_ / 2; // 每个bin的中心位置
  }
  theta_.resize(rdf_theta_bins_);
  theta_.copy_from_host(theta_cpu.data()); // 将数据复制到GPU

  // 设置原子数相关参数
  rdf_N_ = num_atoms;
  num_atoms_ =
    num_atoms * rdf_atom_count; // rdf_atom_count为原子对数量, num_atoms_为考虑所有原子对后的原子数

  // 分配密度数组空间
  density1.resize(rdf_atom_count);
  density2.resize(rdf_atom_count);

  // 分配原子类型大小数组空间
  atom_id1_typesize.resize(
    rdf_atom_count - 1); // 我们总是考虑计算所有原子间的Angular RDF，这里只存储Partial AngularRDF
  atom_id2_typesize.resize(rdf_atom_count - 1);

  // 初始化原子类型数目数组
  for (int a = 0; a < rdf_atom_count - 1; a++) {
    atom_id1_typesize[a] = cpu_type_size[atom_id1_[a]];
    atom_id2_typesize[a] = cpu_type_size[atom_id2_[a]];
  }

  rdf_g_.resize(
    num_atoms_ * rdf_r_bins_,
    0); // 注意这里实际是一个二维数组，rdf_g_的维度为num_atoms_ * rdf_r_bins_
  rdf_.resize(num_atoms_ * rdf_r_bins_, 0);
  cell_count.resize(num_atoms);
  cell_count_sum.resize(num_atoms);
  cell_contents.resize(num_atoms);
}

void AngularRDF::process(
  const bool is_pimd, const int number_of_steps, const int step, Box& box, Atom& atom)
{
  // 如果为PIMD则直接返回, 当前不支持PIMD
  if (is_pimd) {
    return;
  }

  // 如果不需要计算RDF则直接返回
  if (!compute_)
    return;

  // 如果步数不是采样间隔的倍数则直接返回
  if ((step + 1) % num_interval_ != 0) {
    return;
  }

  // 重复次数
  num_repeat_++;

  // 计算数密度
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
      rdf_g_,
      rdf_r_bins_,
      r_step_);
  }
}

void AngularRDF::postprocess(const bool is_pimd, const int number_of_beads)
{
  if (!compute_)
    return;

  if (is_pimd) {

    CHECK(gpuMemcpy(
      rdf_.data(),
      rdf_g_.data(),
      sizeof(double) * number_of_beads * num_atoms_ * rdf_r_bins_,
      gpuMemcpyDeviceToHost));
    CHECK(gpuDeviceSynchronize()); // needed for pre-Pascal GPU

    std::vector<double> rdf_average(number_of_beads * rdf_atom_count * rdf_r_bins_, 0.0);
    for (int k = 0; k < number_of_beads; k++) {
      for (int a = 0; a < rdf_atom_count; a++) {
        for (int m = 0; m < rdf_N_; m++) {
          for (int x = 0; x < rdf_r_bins_; x++) {
            rdf_average[k * rdf_atom_count * rdf_r_bins_ + a * rdf_r_bins_ + x] +=
              rdf_[k * num_atoms_ * rdf_r_bins_ + a * rdf_N_ * rdf_r_bins_ + m * rdf_r_bins_ + x] /
              num_repeat_;
          }
        }
      }
    }

    std::vector<double> rdf_centroid(rdf_atom_count * rdf_r_bins_, 0.0);
    for (int k = 0; k < number_of_beads; k++) {
      for (int a = 0; a < rdf_atom_count; a++) {
        for (int x = 0; x < rdf_r_bins_; x++) {
          rdf_centroid[a * rdf_r_bins_ + x] +=
            rdf_average[k * rdf_atom_count * rdf_r_bins_ + a * rdf_r_bins_ + x] / number_of_beads;
        }
      }
    }

    FILE* fid = fopen("rdf.out", "a");
    fprintf(fid, "#radius");
    for (int a = 0; a < rdf_atom_count; a++) {
      if (a == 0) {
        fprintf(fid, " total");
      } else {
        fprintf(fid, " type_%d_%d", atom_id1_[a - 1], atom_id2_[a - 1]);
      }
    }
    fprintf(fid, "\n");
    for (int nc = 0; nc < rdf_r_bins_; nc++) {
      fprintf(fid, "%.5f", nc * r_step_ + r_step_ / 2);
      for (int a = 0; a < rdf_atom_count; a++) {
        if (a == 0) {
          fprintf(fid, " %.5f", rdf_centroid[nc]);
        } else {
          fprintf(
            fid,
            " %.5f",
            (atom_id1_[a - 1] == atom_id2_[a - 1]) ? rdf_centroid[a * rdf_r_bins_ + nc]
                                                   : rdf_centroid[a * rdf_r_bins_ + nc] / 2);
        }
      }
      fprintf(fid, "\n");
    }
    fflush(fid);
    fclose(fid);

  } else {

    CHECK(gpuMemcpy(
      rdf_.data(),
      rdf_g_.data(),
      sizeof(double) * num_atoms_ * rdf_r_bins_,
      gpuMemcpyDeviceToHost));
    CHECK(gpuDeviceSynchronize()); // needed for pre-Pascal GPU

    std::vector<double> rdf_average(rdf_atom_count * rdf_r_bins_, 0.0);
    for (int a = 0; a < rdf_atom_count; a++) {
      for (int m = 0; m < rdf_N_; m++) {
        for (int x = 0; x < rdf_r_bins_; x++) {
          rdf_average[a * rdf_r_bins_ + x] +=
            rdf_[a * rdf_N_ * rdf_r_bins_ + m * rdf_r_bins_ + x] / num_repeat_;
        }
      }
    }

    FILE* fid = fopen("rdf.out", "a");
    fprintf(fid, "#radius");
    for (int a = 0; a < rdf_atom_count; a++) {
      if (a == 0) {
        fprintf(fid, " total");
      } else {
        fprintf(fid, " type_%d_%d", atom_id1_[a - 1], atom_id2_[a - 1]);
      }
    }
    fprintf(fid, "\n");
    for (int nc = 0; nc < rdf_r_bins_; nc++) {
      fprintf(fid, "%.5f", nc * r_step_ + r_step_ / 2);
      for (int a = 0; a < rdf_atom_count; a++) {
        if (a == 0) {
          fprintf(fid, " %.5f", rdf_average[nc]);
        } else {
          fprintf(
            fid,
            " %.5f",
            (atom_id1_[a - 1] == atom_id2_[a - 1]) ? rdf_average[a * rdf_r_bins_ + nc]
                                                   : rdf_average[a * rdf_r_bins_ + nc] / 2);
        }
      }
      fprintf(fid, "\n");
    }
    fflush(fid);
    fclose(fid);
  }

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
