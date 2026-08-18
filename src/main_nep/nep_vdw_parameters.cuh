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

namespace
{
const int max_elem_vdw = 94;

// sqrt(C6) in sqrt(eV * A^6) based on TS
const float c6_ref_sqrt[max_elem_vdw] = {
  1.97076845f, 0.93401822f, 28.78837510f, 11.30800023f, 7.71064063f, 5.27681515f, 3.80265220f, 3.05310135f,
  2.38504950f, 1.95249199f, 30.49184606f, 19.35586982f, 17.76216432f, 13.49984686f, 10.51392286f, 8.94811015f,
  7.51838377f, 6.19846864f, 48.25520057f, 36.42949924f, 28.74683341f, 24.97636149f, 22.29669973f, 18.96606208f,
  18.16136392f, 16.97080492f, 15.61381408f, 14.92909006f, 12.29530150f, 13.02681040f, 17.25017851f, 14.54388906f,
  12.12401530f, 11.20181948f, 9.83867080f, 8.79997469f, 52.94337691f, 43.52197080f, 34.29694653f, 31.66381601f,
  27.47802088f, 24.79303107f, 28.82850971f, 19.08781649f, 16.74038121f, 9.70106024f, 14.23242065f, 16.43418379f,
  20.55430312f, 18.73493444f, 16.56675866f, 15.38248553f, 15.16733527f, 13.07031336f, 62.71342220f, 58.49820145f,
  48.17774682f, 47.07259162f, 48.34699255f, 48.32789390f, 47.94887203f, 47.07487644f, 45.80767446f, 40.76813048f,
  43.20787737f, 42.22789509f, 41.19402888f, 40.34521550f, 39.23896944f, 37.77055278f, 37.64592342f, 27.59941962f,
  24.68663956f, 22.50914105f, 20.60009656f, 18.88191435f, 14.64827973f, 14.40145027f, 13.34403137f, 15.30459902f,
  20.70483221f, 20.40775873f, 18.47127934f, 17.81121168f, 16.53441029f, 15.27783161f, 50.24160390f, 53.84047507f,
  46.40833796f, 49.17841153f, 37.61114702f, 33.49057677f, 38.71081351f, 35.56862233f
};

}
