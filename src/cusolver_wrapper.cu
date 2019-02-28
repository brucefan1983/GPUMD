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


void eig_hermitian_QR(int N, double* AR, double* AI, double* W_cpu)
{
    // get A
    int N2 = N * N;
    cuDoubleComplex *A, *A_cpu; 
    MY_MALLOC(A_cpu, cuDoubleComplex, N2);
    CHECK(cudaMalloc((void**)&A, sizeof(cuDoubleComplex) * N2));
    for (int n = 0; n < N2; ++n) { A_cpu[n].x = AR[n]; A_cpu[n].y = AI[n]; }
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


void eig_hermitian_Jacobi(int N, double* AR, double* AI, double* W_cpu)
{
    // get A
    int N2 = N * N;
    cuDoubleComplex *A, *A_cpu; 
    MY_MALLOC(A_cpu, cuDoubleComplex, N2);
    CHECK(cudaMalloc((void**)&A, sizeof(cuDoubleComplex) * N2));
    for (int n = 0; n < N2; ++n) { A_cpu[n].x = AR[n]; A_cpu[n].y = AI[n]; }
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
    cudaMemcpy(W_cpu, W, sizeof(double) * N, cudaMemcpyDeviceToHost);

    // free
    cusolverDnDestroy(handle);
    cusolverDnDestroySyevjInfo(para);
    MY_FREE(A_cpu);
    CHECK(cudaFree(A));
    CHECK(cudaFree(W));
    CHECK(cudaFree(work));
    CHECK(cudaFree(info));
}


void eig_hermitian_Jacobi_batch
(int N, int batch_size, double* AR, double* AI, double* W_cpu)
{
    // get A
    int M = N * N * batch_size;
    cuDoubleComplex *A, *A_cpu; 
    MY_MALLOC(A_cpu, cuDoubleComplex, M);
    CHECK(cudaMalloc((void**)&A, sizeof(cuDoubleComplex) * M));
    for (int n = 0; n < M; ++n) { A_cpu[n].x = AR[n]; A_cpu[n].y = AI[n]; }
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


