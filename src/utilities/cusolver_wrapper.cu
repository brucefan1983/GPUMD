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
Some wrappers for the cuSOLVER library
------------------------------------------------------------------------------*/

#include "cusolver_wrapper.cuh"
#include "error.cuh"
#include "gpu_macro.cuh"
#include "gpu_vector.cuh"
#ifdef USE_HIP
  #include <hipsolver/hipsolver.h>
#else
  #include <cusolverDn.h>
#endif
#include <vector>
#include <cstring>

void eig_hermitian_QR(size_t N, double* AR, double* AI, double* W_cpu)
{
  // get A
  size_t N2 = N * N;
  GPU_Vector<gpuDoubleComplex> A(N2);
  std::vector<gpuDoubleComplex> A_cpu(N2);

  for (size_t n = 0; n < N2; ++n) {
    A_cpu[n].x = AR[n];
    A_cpu[n].y = AI[n];
  }
  A.copy_from_host(A_cpu.data());

  // define W
  GPU_Vector<double> W(N);

  // get handle
  gpusolverDnHandle_t handle = NULL;
  gpusolverDnCreate(&handle);
  gpusolverEigMode_t jobz = GPUSOLVER_EIG_MODE_NOVECTOR;
  gpusolverFillMode_t uplo = GPUSOLVER_FILL_MODE_LOWER;

  // get work
  int lwork = 0;
  gpusolverDnZheevd_bufferSize(handle, jobz, uplo, N, A.data(), N, W.data(), &lwork);
  GPU_Vector<gpuDoubleComplex> work(lwork);

  // get W
  GPU_Vector<int> info(1);
  gpusolverDnZheevd(handle, jobz, uplo, N, A.data(), N, W.data(), work.data(), lwork, info.data());
  W.copy_to_host(W_cpu);

  // free
  gpusolverDnDestroy(handle);
}

void eig_hermitian_Jacobi(size_t N, double* AR, double* AI, double* W_cpu)
{
  // get A
  size_t N2 = N * N;
  GPU_Vector<gpuDoubleComplex> A(N2);
  std::vector<gpuDoubleComplex> A_cpu(N2);
  for (size_t n = 0; n < N2; ++n) {
    A_cpu[n].x = AR[n];
    A_cpu[n].y = AI[n];
  }
  A.copy_from_host(A_cpu.data());

  // define W
  GPU_Vector<double> W(N);

  // get handle
  gpusolverDnHandle_t handle = NULL;
  gpusolverDnCreate(&handle);
  gpusolverEigMode_t jobz = GPUSOLVER_EIG_MODE_NOVECTOR;
  gpusolverFillMode_t uplo = GPUSOLVER_FILL_MODE_LOWER;

  // some parameters for the Jacobi method
  gpusolverSyevjInfo_t para = NULL;
  gpusolverDnCreateSyevjInfo(&para);

  // get work
  int lwork = 0;
  gpusolverDnZheevj_bufferSize(handle, jobz, uplo, N, A.data(), N, W.data(), &lwork, para);
  GPU_Vector<gpuDoubleComplex> work(lwork);

  // get W
  GPU_Vector<int> info(1);
  gpusolverDnZheevj(
    handle, jobz, uplo, N, A.data(), N, W.data(), work.data(), lwork, info.data(), para);
  W.copy_to_host(W_cpu);

  // free
  gpusolverDnDestroy(handle);
  gpusolverDnDestroySyevjInfo(para);
}

void eigenvectors_symmetric_Jacobi(size_t N, double* A_cpu, double* W_cpu, double* eigenvectors_cpu)
{
  // get A
  size_t N2 = N * N;
  GPU_Vector<double> A(N2);
  A.copy_from_host(A_cpu);

  // define W
  GPU_Vector<double> W(N);

  // get handle
  gpusolverDnHandle_t handle = NULL;
  gpusolverDnCreate(&handle);
  gpusolverEigMode_t jobz = GPUSOLVER_EIG_MODE_VECTOR;
  gpusolverFillMode_t uplo = GPUSOLVER_FILL_MODE_LOWER;

  // some parameters for the Jacobi method
  gpusolverSyevjInfo_t para = NULL;
  gpusolverDnCreateSyevjInfo(&para);

  // get work
  int lwork = 0;
  gpusolverDnDsyevj_bufferSize(handle, jobz, uplo, N, A.data(), N, W.data(), &lwork, para);
  GPU_Vector<double> work(lwork);

  // get W
  GPU_Vector<int> info(1);
  gpusolverDnDsyevj(
    handle, jobz, uplo, N, A.data(), N, W.data(), work.data(), lwork, info.data(), para);
  W.copy_to_host(W_cpu);
  A.copy_to_host(eigenvectors_cpu);

  // free
  gpusolverDnDestroy(handle);
  gpusolverDnDestroySyevjInfo(para);
}

void eig_hermitian_Jacobi_batch(size_t N, size_t batch_size, double* AR, double* AI, double* W_cpu)
{
  // get A
  size_t M = N * N * batch_size;
  GPU_Vector<gpuDoubleComplex> A(M);
  std::vector<gpuDoubleComplex> A_cpu(M);
  for (size_t n = 0; n < M; ++n) {
    A_cpu[n].x = AR[n];
    A_cpu[n].y = AI[n];
  }
  A.copy_from_host(A_cpu.data());

  // define W
  GPU_Vector<double> W(N * batch_size);

  // get handle
  gpusolverDnHandle_t handle = NULL;
  gpusolverDnCreate(&handle);
  gpusolverEigMode_t jobz = GPUSOLVER_EIG_MODE_NOVECTOR;
  gpusolverFillMode_t uplo = GPUSOLVER_FILL_MODE_LOWER;

  // some parameters for the Jacobi method
  gpusolverSyevjInfo_t para = NULL;
  gpusolverDnCreateSyevjInfo(&para);

  // get work
  int lwork = 0;
  gpusolverDnZheevjBatched_bufferSize(
    handle, jobz, uplo, N, A.data(), N, W.data(), &lwork, para, batch_size);
  GPU_Vector<gpuDoubleComplex> work(lwork);

  // get W
  GPU_Vector<int> info(batch_size);
  gpusolverDnZheevjBatched(
    handle,
    jobz,
    uplo,
    N,
    A.data(),
    N,
    W.data(),
    work.data(),
    lwork,
    info.data(),
    para,
    batch_size);
  W.copy_to_host(W_cpu);

  // free
  gpusolverDnDestroy(handle);
  gpusolverDnDestroySyevjInfo(para);
}
