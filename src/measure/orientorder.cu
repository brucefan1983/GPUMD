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
    Steinhardt bond-orientational order parameters
Author:
    Yongchao Wu 2025-08-27
Email:
    934313174@qq.com
Reference:
    1. Mickel, W., Kapfer, S. C., Schröder-Turk, G. E., & Mecke, K. (2013).
    Shortcomings of the bond orientational order parameters for the analysis of disordered
    particulate matter. The Journal of chemical physics, 138(4).
    2. Steinhardt, P. J., Nelson, D. R., & Ronchetti, M. (1983).
    Bond-orientational order in liquids and glasses. Physical Review B, 28(2), 784.
--------------------------------------------------------------------------------------------------*/

#include "force/neighbor.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include "orientorder.cuh"
#include "parse_utilities.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <algorithm>
#include <cstring>
#include <numeric>

namespace
{

static constexpr double h_factorial[168] = {
  1,
  1,
  2,
  6,
  24,
  120,
  720,
  5040,
  40320,
  362880,
  3628800,
  39916800,
  479001600,
  6227020800,
  87178291200,
  1307674368000,
  20922789888000,
  355687428096000,
  6.402373705728e15,
  1.21645100408832e17,
  2.43290200817664e18,
  5.10909421717094e19,
  1.12400072777761e21,
  2.5852016738885e22,
  6.20448401733239e23,
  1.5511210043331e25,
  4.03291461126606e26,
  1.08888694504184e28,
  3.04888344611714e29,
  8.8417619937397e30,
  2.65252859812191e32,
  8.22283865417792e33,
  2.63130836933694e35,
  8.68331761881189e36,
  2.95232799039604e38,
  1.03331479663861e40,
  3.71993326789901e41,
  1.37637530912263e43,
  5.23022617466601e44,
  2.03978820811974e46,
  8.15915283247898e47,
  3.34525266131638e49,
  1.40500611775288e51,
  6.04152630633738e52,
  2.65827157478845e54,
  1.1962222086548e56,
  5.50262215981209e57,
  2.58623241511168e59,
  1.24139155925361e61,
  6.08281864034268e62,
  3.04140932017134e64,
  1.55111875328738e66,
  8.06581751709439e67,
  4.27488328406003e69,
  2.30843697339241e71,
  1.26964033536583e73,
  7.10998587804863e74,
  4.05269195048772e76,
  2.35056133128288e78,
  1.3868311854569e80,
  8.32098711274139e81,
  5.07580213877225e83,
  3.14699732603879e85,
  1.98260831540444e87,
  1.26886932185884e89,
  8.24765059208247e90,
  5.44344939077443e92,
  3.64711109181887e94,
  2.48003554243683e96,
  1.71122452428141e98,
  1.19785716699699e100,
  8.50478588567862e101,
  6.12344583768861e103,
  4.47011546151268e105,
  3.30788544151939e107,
  2.48091408113954e109,
  1.88549470166605e111,
  1.45183092028286e113,
  1.13242811782063e115,
  8.94618213078297e116,
  7.15694570462638e118,
  5.79712602074737e120,
  4.75364333701284e122,
  3.94552396972066e124,
  3.31424013456535e126,
  2.81710411438055e128,
  2.42270953836727e130,
  2.10775729837953e132,
  1.85482642257398e134,
  1.65079551609085e136,
  1.48571596448176e138,
  1.3520015276784e140,
  1.24384140546413e142,
  1.15677250708164e144,
  1.08736615665674e146,
  1.03299784882391e148,
  9.91677934870949e149,
  9.61927596824821e151,
  9.42689044888324e153,
  9.33262154439441e155,
  9.33262154439441e157,
  9.42594775983835e159,
  9.61446671503512e161,
  9.90290071648618e163,
  1.02990167451456e166,
  1.08139675824029e168,
  1.14628056373471e170,
  1.22652020319614e172,
  1.32464181945183e174,
  1.44385958320249e176,
  1.58824554152274e178,
  1.76295255109024e180,
  1.97450685722107e182,
  2.23119274865981e184,
  2.54355973347219e186,
  2.92509369349301e188,
  3.3931086844519e190,
  3.96993716080872e192,
  4.68452584975429e194,
  5.5745857612076e196,
  6.68950291344912e198,
  8.09429852527344e200,
  9.8750442008336e202,
  1.21463043670253e205,
  1.50614174151114e207,
  1.88267717688893e209,
  2.37217324288005e211,
  3.01266001845766e213,
  3.8562048236258e215,
  4.97450422247729e217,
  6.46685548922047e219,
  8.47158069087882e221,
  1.118248651196e224,
  1.48727070609069e226,
  1.99294274616152e228,
  2.69047270731805e230,
  3.65904288195255e232,
  5.01288874827499e234,
  6.91778647261949e236,
  9.61572319694109e238,
  1.34620124757175e241,
  1.89814375907617e243,
  2.69536413788816e245,
  3.85437071718007e247,
  5.5502938327393e249,
  8.04792605747199e251,
  1.17499720439091e254,
  1.72724589045464e256,
  2.55632391787286e258,
  3.80892263763057e260,
  5.71338395644585e262,
  8.62720977423323e264,
  1.31133588568345e267,
  2.00634390509568e269,
  3.08976961384735e271,
  4.78914290146339e273,
  7.47106292628289e275,
  1.17295687942641e278,
  1.85327186949373e280,
  2.94670227249504e282,
  4.71472363599206e284,
  7.59070505394721e286,
  1.22969421873945e289,
  2.0044015765453e291,
  3.28721858553429e293,
  5.42391066613159e295,
  9.00369170577843e297,
  1.503616514865e300,
};

double _factorial(int n)
{
  if (n > 167) {
    PRINT_INPUT_ERROR("One of input degrees is too large.");
  }

  return h_factorial[n];
}

void _init_clebsch_gordan(std::vector<double>& cglist, const std::vector<int>& llist)
{
  int idxcg_count{0};
  for (int il = 0; il < llist.size(); il++) {
    const int l = llist[il];
    for (int m1 = 0; m1 < 2 * l + 1; m1++) {
      int aa2 = m1 - l;
      for (int m2 = max(0, l - m1); m2 < min(2 * l + 1, 3 * l - m1 + 1); m2++) {
        int bb2 = m2 - l;
        int m = aa2 + bb2 + l;
        double sums{0.0};
        for (int z = max(0, max(-aa2, bb2)); z < min(l, min(l - aa2, l + bb2)) + 1; z++) {
          int ifac = 1;
          if (z % 2) {
            ifac = -1;
          }
          sums += ifac / (_factorial(z) * _factorial(l - z) * _factorial(l - aa2 - z) *
                          _factorial(l + bb2 - z) * _factorial(aa2 + z) * _factorial(-bb2 + z));
        }
        int cc2 = m - l;
        double sfaccg = sqrt(
          _factorial(l + aa2) * _factorial(l - aa2) * _factorial(l + bb2) * _factorial(l - bb2) *
          _factorial(l + cc2) * _factorial(l - cc2) * (2 * l + 1));
        double sfac1 = _factorial(3 * l + 1);
        double sfac2 = _factorial(l);
        double dcg = sqrt(sfac2 * sfac2 * sfac2 / sfac1);
        cglist[idxcg_count] = sums * dcg * sfaccg;
        idxcg_count++;
      }
    }
  }
}

int _get_idx(const std::vector<int>& llist)
{
  int idxcg_count{0};
  for (int il = 0; il < llist.size(); il++) {
    const int l = llist[il];
    for (int m1 = 0; m1 < 2 * l + 1; m1++) {
      for (int m2 = max(0, l - m1); m2 < min(2 * l + 1, 3 * l - m1 + 1); m2++) {
        idxcg_count++;
      }
    }
  }
  return idxcg_count;
}

__device__ double _associated_legendre(int l, int m, double x)
{
  double res{0.0};
  if (l >= m) {
    double p{1.0};
    double pm1{0.0};
    double pm2{0.0};
    if (m != 0) {
      double sqx{sqrt(1.0 - x * x)};
      for (int i = 1; i < m + 1; i++) {
        p *= (2 * i - 1) * sqx;
      }
    }
    for (int i = m + 1; i < l + 1; i++) {
      pm2 = pm1;
      pm1 = p;
      p = ((2 * i - 1) * x * pm1 - (i + m - 1) * pm2) / (i - m);
    }
    res = p;
  }
  return res;
}

__device__ double _polar_prefactor(int l, int m, double costheta)
{
  const double My_PI = 3.14159265358979323846;
  int mabs = abs(m);
  double prefactor{1.0};
  for (int i = l - mabs + 1; i < l + mabs + 1; i++) {
    prefactor *= i;
  }
  prefactor = sqrt((2 * l + 1) / (4 * My_PI * prefactor)) * _associated_legendre(l, mabs, costheta);

  if ((m < 0) & (m % 2)) {
    prefactor = -prefactor;
  }
  return prefactor;
}

static __global__ void compute_ql_step1(
  const int N,
  const Box box,
  const int nnn,
  const int* __restrict__ NN,
  const int* __restrict__ NL,
  const double* __restrict__ x,
  const double* __restrict__ y,
  const double* __restrict__ z,
  const int* llist,
  const int ndegrees,
  const int lmax,
  double* qlm_r,
  double* qlm_i)
{
  const double MY_EPSILON{1e-15};
  const int nz{lmax * 2 + 1};
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N) {
    const double x1 = x[i];
    const double y1 = y[i];
    const double z1 = z[i];

    const int i_neigh = nnn > 0 ? nnn : NN[i];

    for (int jj = 0; jj < i_neigh; jj++) {
      const int j = NL[i + jj * N];
      double xij = x[j] - x1;
      double yij = y[j] - y1;
      double zij = z[j] - z1;
      apply_mic(box, xij, yij, zij);
      const double rmag = sqrt(xij * xij + yij * yij + zij * zij);
      if (rmag > MY_EPSILON) {
        double costheta = zij / rmag;
        double expphi_r = xij;
        double expphi_i = yij;
        double rxymag = sqrt(expphi_r * expphi_r + expphi_i * expphi_i);
        if (rxymag < MY_EPSILON) {
          expphi_r = 1.0;
          expphi_i = 0.0;
        } else {
          double rxymaginv = 1.0 / rxymag;
          expphi_r *= rxymaginv;
          expphi_i *= rxymaginv;
        }
        for (int il = 0; il < ndegrees; il++) {
          const int l = llist[il];
          qlm_r[i * (ndegrees * nz) + il * nz + l] += _polar_prefactor(l, 0, costheta);
          double expphim_r = expphi_r;
          double expphim_i = expphi_i;
          for (int m = 1; m < l + 1; m++) {
            double prefactor = _polar_prefactor(l, m, costheta);
            double c_r = prefactor * expphim_r;
            double c_i = prefactor * expphim_i;
            int index1 = i * (ndegrees * nz) + il * nz + m + l;
            int index2 = i * (ndegrees * nz) + il * nz - m + l;
            qlm_r[index1] += c_r;
            qlm_i[index1] += c_i;
            if (m & 1) {
              qlm_r[index2] -= c_r;
              qlm_i[index2] += c_i;
            } else {
              qlm_r[index2] += c_r;
              qlm_i[index2] -= c_i;
            }
            double tmp_r = expphim_r * expphi_r - expphim_i * expphi_i;
            double tmp_i = expphim_r * expphi_i + expphim_i * expphi_r;
            expphim_r = tmp_r;
            expphim_i = tmp_i;
          }
        }
      }
    }

    const double facn = 1.0 / i_neigh;
    for (int il = 0; il < ndegrees; il++) {
      const int l = llist[il];
      for (int m = 0; m < 2 * l + 1; m++) {
        int index = i * (ndegrees * nz) + il * nz + m;
        qlm_r[index] *= facn;
        qlm_i[index] *= facn;
      }
    }
  }
}

static __global__ void compute_ql_step2(
  const int N,
  const int* NN,
  const int* llist,
  const int ndegrees,
  const int lmax,
  const int ncol,
  double* qnarray,
  double* qlm_r,
  double* qlm_i,
  const double* cglist,
  const bool wl,
  const bool wlhat,
  const int nnn)
{

  const double My_PI = 3.14159265358979323846;
  const double MY_EPSILON{1e-15};
  const int nz{lmax * 2 + 1};
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N) {
    if (NN[i] < nnn) {
      return;
    }
    for (int il = 0; il < ndegrees; il++) {
      const int l = llist[il];
      double qnormfac = sqrt(4 * My_PI / (2 * l + 1));
      double qm_sum = 0.0;
      for (int m = 0; m < 2 * l + 1; m++) {
        int index = i * (ndegrees * nz) + il * nz + m;
        qm_sum += qlm_r[index] * qlm_r[index] + qlm_i[index] * qlm_i[index];
      }
      qnarray[i * ncol + il] = qnormfac * sqrt(qm_sum);
    }

    if (wl | wlhat) {
      int idxcg_count{0};
      for (int il = 0; il < ndegrees; il++) {
        const int l = llist[il];
        double wlsum{0.0};
        for (int m1 = 0; m1 < 2 * l + 1; m1++) {
          for (int m2 = max(0, l - m1); m2 < min(2 * l + 1, 3 * l - m1 + 1); m2++) {
            int m = m1 + m2 - l;
            int index1 = i * (ndegrees * nz) + il * nz + m1;
            int index2 = i * (ndegrees * nz) + il * nz + m2;
            int index3 = i * (ndegrees * nz) + il * nz + m;
            double qm1qm2_r = qlm_r[index1] * qlm_r[index2] - qlm_i[index1] * qlm_i[index2];
            double qm1qm2_i = qlm_r[index1] * qlm_i[index2] + qlm_i[index1] * qlm_r[index2];
            wlsum += (qm1qm2_r * qlm_r[index3] + qm1qm2_i * qlm_i[index3]) * cglist[idxcg_count];
            idxcg_count++;
          }
        }
        if (wl) {
          qnarray[i * ncol + il + ndegrees] = wlsum / sqrt(2 * l + 1.0);
        }
        if (wlhat) {
          if (qnarray[i * ncol + il] > MY_EPSILON) {
            double qnormfac = sqrt(4 * My_PI / (2 * l + 1));
            double qnfac = qnormfac / qnarray[i * ncol + il];
            qnarray[i * ncol + il + wl * ndegrees + ndegrees] =
              wlsum / sqrt(2 * l + 1.0) * (qnfac * qnfac * qnfac);
          }
        }
      }
    }
  }
}

static __global__ void compute_ql_average(
  const int N,
  const int* __restrict__ NN,
  const int* __restrict__ NL,
  const int* llist,
  const int ndegrees,
  const int lmax,
  double* qlm_r,
  double* qlm_i,
  const double* aqlm_r,
  const double* aqlm_i,
  const int nnn)
{

  const int nz{lmax * 2 + 1};
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N) {

    const int i_neigh = nnn > 0 ? nnn : NN[i];
    const double i_neigh_inv = 1.0 / (i_neigh + 1);

    for (int jj = 0; jj < i_neigh; jj++) {
      const int j = NL[i + jj * N];
      for (int il = 0; il < ndegrees; il++) {
        const int l = llist[il];
        for (int m = 0; m < 2 * l + 1; m++) {
          const int index1 = i * (ndegrees * nz) + il * nz + m;
          const int index2 = j * (ndegrees * nz) + il * nz + m;
          qlm_r[index1] += aqlm_r[index2];
          qlm_i[index1] += aqlm_i[index2];
        }
      }
    }

    for (int il = 0; il < ndegrees; il++) {
      const int l = llist[il];
      for (int m = 0; m < 2 * l + 1; m++) {
        const int index = i * (ndegrees * nz) + il * nz + m;
        qlm_r[index] *= i_neigh_inv;
        qlm_i[index] *= i_neigh_inv;
      }
    }
  }
}

static __global__ void sort_neighbors(
  const int N,
  const Box box,
  const int* __restrict__ NN,
  int* __restrict__ NL,
  const double* __restrict__ x,
  const double* __restrict__ y,
  const double* __restrict__ z,
  double* NLD,
  const int nnn)
{

  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N) {

    const int i_neigh = NN[i];
    if (i_neigh >= nnn) {
      const double x1 = x[i];
      const double y1 = y[i];
      const double z1 = z[i];
      for (int jj = 0; jj < i_neigh; jj++) {
        const int j = NL[i + jj * N];
        double xij = x[j] - x1;
        double yij = y[j] - y1;
        double zij = z[j] - z1;
        apply_mic(box, xij, yij, zij);
        const double rsq = xij * xij + yij * yij + zij * zij;
        NLD[i + jj * N] = rsq;
      }

      for (int k = 0; k < nnn; k++) {
        int minIndex = k;
        for (int m = k + 1; m < i_neigh; m++) {
          if (NLD[i + m * N] < NLD[i + minIndex * N]) {
            minIndex = m;
          }
        }
        if (minIndex != k) {
          int tmp = NL[i + k * N];
          double tmp1 = NLD[i + k * N];
          NL[i + k * N] = NL[i + minIndex * N];
          NLD[i + k * N] = NLD[i + minIndex * N];
          NL[i + minIndex * N] = tmp;
          NLD[i + minIndex * N] = tmp1;
        }
      }
    }
  }
}

} // namespace

OrientOrder::OrientOrder(const char** param, const int num_param)
{
  parse(param, num_param);
  property_name = "compute_orientorder";
}

void OrientOrder::preprocess(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  if (!compute_) {
    return;
  }

  num_atoms_ = atom.number_of_atoms;
  lmax_ = *std::max_element(llist.begin(), llist.end());
  llist_gpu.resize(ndegrees_, 0);

  CHECK(
    gpuMemcpy(llist_gpu.data(), llist.data(), sizeof(int) * llist.size(), gpuMemcpyHostToDevice));
  CHECK(gpuDeviceSynchronize()); // needed for pre-Pascal GPU

  qlm_r_gpu.resize(num_atoms_ * ndegrees_ * (2 * lmax_ + 1), 0.0);
  qlm_i_gpu.resize(num_atoms_ * ndegrees_ * (2 * lmax_ + 1), 0.0);

  if (average_) {
    aqlm_r_gpu.resize(num_atoms_ * ndegrees_ * (2 * lmax_ + 1));
    aqlm_i_gpu.resize(num_atoms_ * ndegrees_ * (2 * lmax_ + 1));
  }

  ncol_ = ndegrees_;
  if (wl_) {
    ncol_ += ndegrees_;
  }
  if (wlhat_) {
    ncol_ += ndegrees_;
  }

  if (wl_ | wlhat_) {
    int idxcg_count = _get_idx(llist);
    cglist.resize(idxcg_count, 0.0);
    cglist_gpu.resize(idxcg_count, 0.0);
    _init_clebsch_gordan(cglist, llist);
    CHECK(gpuMemcpy(
      cglist_gpu.data(), cglist.data(), sizeof(double) * cglist.size(), gpuMemcpyHostToDevice));
    CHECK(gpuDeviceSynchronize()); // needed for pre-Pascal GPU
  }

  qnarray_gpu.resize(num_atoms_ * ncol_, 0.0);
  qnarray.resize(num_atoms_ * ncol_, 0.0);

  cell_count.resize(num_atoms_);
  cell_count_sum.resize(num_atoms_);
  cell_contents.resize(num_atoms_);
  NN.resize(num_atoms_);
  NL.resize(num_atoms_ * 200); // 200 is the maximum number of neighbors
  if (mode_ == "nnn") {
    NLD.resize(num_atoms_ * 200);
  }

  fid = fopen("orientorder.out", "a");
};

void OrientOrder::process(
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

  if (!compute_) {
    return;
  }

  if ((step + 1) % num_interval_ != 0) {
    return;
  }

  const int BLOCK_SIZE = 64;
  int grid_size = (num_atoms_ - 1) / BLOCK_SIZE + 1;
  find_neighbor(
    0,
    num_atoms_,
    rc_,
    box,
    atom.type,
    atom.position_per_atom,
    cell_count,
    cell_count_sum,
    cell_contents,
    NN,
    NL);

  if (mode_ == "nnn") {
    sort_neighbors<<<grid_size, BLOCK_SIZE>>>(
      num_atoms_,
      box,
      NN.data(),
      NL.data(),
      atom.position_per_atom.data(),
      atom.position_per_atom.data() + num_atoms_,
      atom.position_per_atom.data() + 2 * num_atoms_,
      NLD.data(),
      nnn_);
    GPU_CHECK_KERNEL
  }

  compute_ql_step1<<<grid_size, BLOCK_SIZE>>>(
    num_atoms_,
    box,
    nnn_,
    NN.data(),
    NL.data(),
    atom.position_per_atom.data(),
    atom.position_per_atom.data() + num_atoms_,
    atom.position_per_atom.data() + 2 * num_atoms_,
    llist_gpu.data(),
    ndegrees_,
    lmax_,
    qlm_r_gpu.data(),
    qlm_i_gpu.data());
  GPU_CHECK_KERNEL

  if (average_) {
    aqlm_r_gpu.copy_from_device(qlm_r_gpu.data());
    CHECK(gpuDeviceSynchronize());
    aqlm_i_gpu.copy_from_device(qlm_i_gpu.data());
    CHECK(gpuDeviceSynchronize());
    compute_ql_average<<<grid_size, BLOCK_SIZE>>>(
      num_atoms_,
      NN.data(),
      NL.data(),
      llist_gpu.data(),
      ndegrees_,
      lmax_,
      qlm_r_gpu.data(),
      qlm_i_gpu.data(),
      aqlm_r_gpu.data(),
      aqlm_i_gpu.data(),
      nnn_);
    GPU_CHECK_KERNEL
  }

  compute_ql_step2<<<grid_size, BLOCK_SIZE>>>(
    num_atoms_,
    NN.data(),
    llist_gpu.data(),
    ndegrees_,
    lmax_,
    ncol_,
    qnarray_gpu.data(),
    qlm_r_gpu.data(),
    qlm_i_gpu.data(),
    cglist_gpu.data(),
    wl_,
    wlhat_,
    nnn_);
  GPU_CHECK_KERNEL

  CHECK(gpuMemcpy(
    qnarray.data(), qnarray_gpu.data(), sizeof(double) * qnarray.size(), gpuMemcpyDeviceToHost));
  CHECK(gpuDeviceSynchronize()); // needed for pre-Pascal GPU

  qlm_r_gpu.fill(0.0);
  qlm_i_gpu.fill(0.0);
  qnarray_gpu.fill(0.0);
  if (mode_ == "nnn") {
    NLD.fill(0.0);
  }

  fprintf(fid, "step = %d\n", step + 1);
  fprintf(fid, "ql%d", llist[0]);
  for (int il = 1; il < ndegrees_; il++) {
    fprintf(fid, " ql%d", llist[il]);
  }
  if (wl_) {
    for (int il = 0; il < ndegrees_; il++) {
      fprintf(fid, " wl%d", llist[il]);
    }
  }
  if (wlhat_) {
    for (int il = 0; il < ndegrees_; il++) {
      fprintf(fid, " wlhat%d", llist[il]);
    }
  }
  fprintf(fid, "\n");

  for (int i = 0; i < num_atoms_; i++) {
    for (int j = 0; j < ncol_ - 1; j++) {
      fprintf(fid, "%f ", qnarray[i * ncol_ + j]);
    }
    fprintf(fid, "%f", qnarray[i * ncol_ + ncol_ - 1]);
    fprintf(fid, "\n");
  }

  fflush(fid);
};

void OrientOrder::postprocess(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature)
{
  if (!compute_)
    return;

  fclose(fid);
  compute_ = false;
};

// compute_orientorder <interval> <mode_type> <mode_parameters> <ndegrees> <degree1> <degree2>
// <average> <wl> <wlhat>

void OrientOrder::parse(const char** param, const int num_param)
{
  printf("Compute Steinhardt bond-orientational order parameters.\n");
  compute_ = true;

  if (num_param < 6) {
    PRINT_INPUT_ERROR("compute_orientorder should have at least 5 parameters.\n");
  }

  if (!is_valid_int(param[1], &num_interval_)) {
    PRINT_INPUT_ERROR("interval step per sample should be an integer.\n");
  }
  if (num_interval_ <= 0) {
    PRINT_INPUT_ERROR("interval step per sample should be positive.\n");
  }

  mode_ = param[2];

  if (mode_ == "cutoff") {
    if (!is_valid_real(param[3], &rc_)) {
      PRINT_INPUT_ERROR("cutoff should be an positive float.\n");
    }
    if (rc_ <= 0.) {
      PRINT_INPUT_ERROR("cutoff should be an positive float.\n");
    }
  } else if (mode_ == "nnn") {
    if (!is_valid_int(param[3], &nnn_)) {
      PRINT_INPUT_ERROR("nnn should be an positive integer.\n");
    }
    if (nnn_ <= 0) {
      PRINT_INPUT_ERROR("nnn should be an positive integer.\n");
    }
  } else {
    PRINT_INPUT_ERROR("mode_type should be cutoff or nnn.\n");
  }

  if (!is_valid_int(param[4], &ndegrees_)) {
    PRINT_INPUT_ERROR("ndegrees should be an positive integer.\n");
  }

  if (ndegrees_ <= 0) {
    PRINT_INPUT_ERROR("ndegrees should be an positive integer.\n");
  }

  if (num_param < 5 + ndegrees_) {
    std::string message = "Must include " + std::to_string(ndegrees_) + " degrees.\n";
    PRINT_INPUT_ERROR(message.c_str());
  }

  llist.resize(ndegrees_);

  for (int i = 1; i < ndegrees_ + 1; ++i) {
    int degree = 0;
    if (!is_valid_int(param[4 + i], &degree)) {
      std::string message = "Degree " + std::to_string(i) + " should be an positive integer.\n";
      PRINT_INPUT_ERROR(message.c_str());
    }
    if (degree < 0) {
      std::string message = "Degree " + std::to_string(i) + " should be an positive integer.\n";
      PRINT_INPUT_ERROR(message.c_str());
    }
    llist[i - 1] = degree;
  }

  if ((num_param > 5 + ndegrees_) & (num_param < 5 + ndegrees_ + 5)) {
    if (num_param > 5 + ndegrees_) {
      int ave = 0;
      if (!is_valid_int(param[4 + ndegrees_ + 1], &ave)) {
        PRINT_INPUT_ERROR("average should be 1 or 0.\n");
      }
      if (ave != 0) {
        average_ = true;
      }
    }
    if (num_param > 5 + ndegrees_ + 1) {
      int wl = 0;
      if (!is_valid_int(param[4 + ndegrees_ + 2], &wl)) {
        PRINT_INPUT_ERROR("wl should be 1 or 0.\n");
      }
      if (wl != 0) {
        wl_ = true;
      }
    }

    if (num_param > 5 + ndegrees_ + 2) {
      int wlhat = 0;
      if (!is_valid_int(param[4 + ndegrees_ + 3], &wlhat)) {
        PRINT_INPUT_ERROR("wlhat should be 1 or 0.\n");
      }
      if (wlhat != 0) {
        wlhat_ = true;
      }
    }
  } else {
    std::string message =
      "Number of paramaters exceeds " + std::to_string(5 + ndegrees_ + 4) + ".\n";
    PRINT_INPUT_ERROR(message.c_str());
  }

  printf("    every %d steps, \n", num_interval_);
  if (mode_ == "cutoff") {
    printf("    with %f angstrom cutoff,\n", rc_);
  } else if (mode_ == "nnn") {
    printf("    with %d nearest neighbor atoms,\n", nnn_);
  }
  if (ndegrees_ > 1) {
    printf("    with %d degrees:", ndegrees_);
  } else {
    printf("    with %d degree:", ndegrees_);
  }

  for (int i = 0; i < ndegrees_; ++i) {
    printf(" %d", llist[i]);
  }
  printf(",\n");
  printf(
    "    average is %s, wl is %s and wlhat is %s.\n",
    average_ ? "true" : "false",
    wl_ ? "true" : "false",
    wlhat_ ? "true" : "false");
}