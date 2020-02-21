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
Some wrappers for the cuSOLVER library
------------------------------------------------------------------------------*/


#include "cusolver_wrapper.cuh"
#include "error.cuh"
#include <cusolverDn.h>


void eig_hermitian_QR(size_t N, double* AR, double* AI, double* W_cpu)
{
    // get A
    size_t N2 = N * N;
    cuDoubleComplex *A, *A_cpu; 
    MY_MALLOC(A_cpu, cuDoubleComplex, N2);
    CHECK(cudaMalloc((void**)&A, sizeof(cuDoubleComplex) * N2));
    for (size_t n = 0; n < N2; ++n) { A_cpu[n].x = AR[n]; A_cpu[n].y = AI[n]; }
    CHECK(cudaMemcpy(A, A_cpu, sizeof(cuDoubleComplex) * N2, 
        cudaMemcpyHostToDevice));

    // define W
    double* W; CHECK(cudaMalloc((void**)&W, sizeof(double) * N));

    // get handle
    cusolverDnHandle_t handle = NULL;
    cusolverDnCreate(&handle);
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_NOVECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    // get work
    int lwork = 0;
    cusolverDnZheevd_bufferSize(handle, jobz, uplo, N, A, N, W, &lwork);
    cuDoubleComplex* work;
    CHECK(cudaMalloc((void**)&work, sizeof(cuDoubleComplex) * lwork));

    // get W
    int* info;
    CHECK(cudaMalloc((void**)&info, sizeof(int)));
    cusolverDnZheevd(handle, jobz, uplo, N, A, N, W, work, lwork, info);
    cudaMemcpy(W_cpu, W, sizeof(double) * N, cudaMemcpyDeviceToHost);

    // free
    cusolverDnDestroy(handle);
    MY_FREE(A_cpu);
    CHECK(cudaFree(A));
    CHECK(cudaFree(W));
    CHECK(cudaFree(work));
    CHECK(cudaFree(info));
}


void eig_hermitian_Jacobi(size_t N, double* AR, double* AI, double* W_cpu)
{
    // get A
    size_t N2 = N * N;
    cuDoubleComplex *A, *A_cpu; 
    MY_MALLOC(A_cpu, cuDoubleComplex, N2);
    CHECK(cudaMalloc((void**)&A, sizeof(cuDoubleComplex) * N2));
    for (size_t n = 0; n < N2; ++n) { A_cpu[n].x = AR[n]; A_cpu[n].y = AI[n]; }
    CHECK(cudaMemcpy(A, A_cpu, sizeof(cuDoubleComplex) * N2, 
        cudaMemcpyHostToDevice));

    // define W
    double* W; CHECK(cudaMalloc((void**)&W, sizeof(double) * N));

    // get handle
    cusolverDnHandle_t handle = NULL;
    cusolverDnCreate(&handle);
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_NOVECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    // some parameters for the Jacobi method
    syevjInfo_t para = NULL;
    cusolverDnCreateSyevjInfo(&para);

    // get work
    int lwork = 0;
    cusolverDnZheevj_bufferSize(handle, jobz, uplo, N, A, N, W, &lwork, para);
    cuDoubleComplex* work;
    CHECK(cudaMalloc((void**)&work, sizeof(cuDoubleComplex) * lwork));

    // get W
    int* info;
    CHECK(cudaMalloc((void**)&info, sizeof(int)));
    cusolverDnZheevj(handle, jobz, uplo, N, A, N, W, work, lwork, info, para);
    CHECK(cudaMemcpy(W_cpu, W, sizeof(double) * N, cudaMemcpyDeviceToHost));

    // free
    cusolverDnDestroy(handle);
    cusolverDnDestroySyevjInfo(para);
    MY_FREE(A_cpu);
    CHECK(cudaFree(A));
    CHECK(cudaFree(W));
    CHECK(cudaFree(work));
    CHECK(cudaFree(info));
}


void eigenvectors_symmetric_Jacobi
(size_t N, double* A_cpu, double* W_cpu, double* eigenvectors_cpu)
{
    // get A
    size_t N2 = N * N;
    double *A; 
    CHECK(cudaMalloc((void**)&A, sizeof(double) * N2));
    CHECK(cudaMemcpy(A, A_cpu, sizeof(double) * N2, cudaMemcpyHostToDevice));

    // define W
    double* W; CHECK(cudaMalloc((void**)&W, sizeof(double) * N));

    // get handle
    cusolverDnHandle_t handle = NULL;
    cusolverDnCreate(&handle);
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    // some parameters for the Jacobi method
    syevjInfo_t para = NULL;
    cusolverDnCreateSyevjInfo(&para);

    // get work
    int lwork = 0;
    cusolverDnDsyevj_bufferSize(handle, jobz, uplo, N, A, N, W, &lwork, para);
    double* work;
    CHECK(cudaMalloc((void**)&work, sizeof(double) * lwork));

    // get W
    int* info;
    CHECK(cudaMalloc((void**)&info, sizeof(int)));
    cusolverDnDsyevj(handle, jobz, uplo, N, A, N, W, work, lwork, info, para);
    CHECK(cudaMemcpy(W_cpu, W, sizeof(double) * N, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(eigenvectors_cpu, A, sizeof(double)*N*N, 
        cudaMemcpyDeviceToHost));

    // free
    cusolverDnDestroy(handle);
    cusolverDnDestroySyevjInfo(para);
    CHECK(cudaFree(W));
    CHECK(cudaFree(A));
    CHECK(cudaFree(work));
    CHECK(cudaFree(info));
}


void eig_hermitian_Jacobi_batch
(size_t N, size_t batch_size, double* AR, double* AI, double* W_cpu)
{
    // get A
    size_t M = N * N * batch_size;
    cuDoubleComplex *A, *A_cpu; 
    MY_MALLOC(A_cpu, cuDoubleComplex, M);
    CHECK(cudaMalloc((void**)&A, sizeof(cuDoubleComplex) * M));
    for (size_t n = 0; n < M; ++n) { A_cpu[n].x = AR[n]; A_cpu[n].y = AI[n]; }
    CHECK(cudaMemcpy(A, A_cpu, sizeof(cuDoubleComplex) * M, 
        cudaMemcpyHostToDevice));

    // define W
    double* W; CHECK(cudaMalloc((void**)&W, sizeof(double) * N * batch_size));

    // get handle
    cusolverDnHandle_t handle = NULL;
    cusolverDnCreate(&handle);
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_NOVECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    // some parameters for the Jacobi method
    syevjInfo_t para = NULL;
    cusolverDnCreateSyevjInfo(&para);

    // get work
    int lwork = 0;
    cusolverDnZheevjBatched_bufferSize
    (handle, jobz, uplo, N, A, N, W, &lwork, para, batch_size);
    cuDoubleComplex* work;
    CHECK(cudaMalloc((void**)&work, sizeof(cuDoubleComplex) * lwork));

    // get W
    int* info;
    CHECK(cudaMalloc((void**)&info, sizeof(int) * batch_size));
    cusolverDnZheevjBatched
    (handle, jobz, uplo, N, A, N, W, work, lwork, info, para, batch_size);
    cudaMemcpy(W_cpu, W, sizeof(double)*N*batch_size, cudaMemcpyDeviceToHost);

    // free
    cusolverDnDestroy(handle);
    cusolverDnDestroySyevjInfo(para);
    MY_FREE(A_cpu);
    CHECK(cudaFree(A));
    CHECK(cudaFree(W));
    CHECK(cudaFree(work));
    CHECK(cudaFree(info));
}


