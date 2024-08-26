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

#pragma once

void eig_hermitian_QR(size_t, double*, double*, double*);
void eig_hermitian_Jacobi(size_t, double*, double*, double*);
void eig_hermitian_Jacobi_batch(size_t, size_t, double*, double*, double*);
void eigenvectors_symmetric_Jacobi(size_t N, double* A, double* W_cpu, double* eigenvectors_cpu);
