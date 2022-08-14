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

class Box
{
public:
  int pbc_x = 1;                      // pbc_x = 1 means periodic in the x-direction
  int pbc_y = 1;                      // pbc_y = 1 means periodic in the y-direction
  int pbc_z = 1;                      // pbc_z = 1 means periodic in the z-direction
  int triclinic = 0;                  // triclinic = 1 means the box is non-orthogonal
  double cpu_h[18];                   // the box data
  double thickness_x = 0.0;           // thickness perpendicular to (b x c)
  double thickness_y = 0.0;           // thickness perpendicular to (c x a)
  double thickness_z = 0.0;           // thickness perpendicular to (a x b)
  void update_triclinic();            // update the triclinic member
  double get_area(const int d) const; // get the area of one face
  double get_volume(void) const;      // get the volume of the box
  void get_inverse(void);             // get the inverse box matrix
  bool get_num_bins(const double rc, int num_bins[]); // get the number of bins in each direction
};

inline __host__ __device__ void apply_mic(const Box& box, double& x12, double& y12, double& z12)
{
  if (box.triclinic == 0) // orthogonal box
  {
    if (box.pbc_x == 1 && x12 < -box.cpu_h[3]) {
      x12 += box.cpu_h[0];
    } else if (box.pbc_x == 1 && x12 > +box.cpu_h[3]) {
      x12 -= box.cpu_h[0];
    }
    if (box.pbc_y == 1 && y12 < -box.cpu_h[4]) {
      y12 += box.cpu_h[1];
    } else if (box.pbc_y == 1 && y12 > +box.cpu_h[4]) {
      y12 -= box.cpu_h[1];
    }
    if (box.pbc_z == 1 && z12 < -box.cpu_h[5]) {
      z12 += box.cpu_h[2];
    } else if (box.pbc_z == 1 && z12 > +box.cpu_h[5]) {
      z12 -= box.cpu_h[2];
    }
  } else // triclinic box
  {
    double sx12 = box.cpu_h[9] * x12 + box.cpu_h[10] * y12 + box.cpu_h[11] * z12;
    double sy12 = box.cpu_h[12] * x12 + box.cpu_h[13] * y12 + box.cpu_h[14] * z12;
    double sz12 = box.cpu_h[15] * x12 + box.cpu_h[16] * y12 + box.cpu_h[17] * z12;
    if (box.pbc_x == 1)
      sx12 -= nearbyint(sx12);
    if (box.pbc_y == 1)
      sy12 -= nearbyint(sy12);
    if (box.pbc_z == 1)
      sz12 -= nearbyint(sz12);
    x12 = box.cpu_h[0] * sx12 + box.cpu_h[1] * sy12 + box.cpu_h[2] * sz12;
    y12 = box.cpu_h[3] * sx12 + box.cpu_h[4] * sy12 + box.cpu_h[5] * sz12;
    z12 = box.cpu_h[6] * sx12 + box.cpu_h[7] * sy12 + box.cpu_h[8] * sz12;
  }
}
