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

#pragma once
#include "common.cuh"

class Vector
{
public:
  Vector(int n);
  Vector(Vector& original);
  ~Vector();

  void add(Vector& other);
  void copy(Vector& other);
  void apply_sz(Vector& other);
  void copy_from_host(real* other_real, real* other_imag);
  void copy_to_host(real* target_real, real* target_imag);
  void swap(Vector& other);
  void inner_product_1(int, Vector& other, Vector& target, int offset);
  void inner_product_2(int, int, Vector& target);

  real* real_part;
  real* imag_part;

private:
  void initialize_gpu(int n);
  void initialize_cpu(int n);
  int n;
  int array_size;
};
