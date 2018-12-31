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




#ifndef VAC_H
#define VAC_H




class VAC
{
public:
    int compute;         // 1 means you want to do this computation
    int sample_interval; // sample interval for velocity
    int Nc;              // number of correlation points
    real omega_max;    // maximal angular frequency for phonons
    void preprocess_vac(Parameters*, CPU_Data*, Atom*);
    void sample_vac(int step, Parameters*, CPU_Data*, Atom*);
    void postprocess_vac(char*, Parameters*, CPU_Data*, Atom*);

    real *vx_all;
    real *vy_all;
    real *vz_all;
private:
    void find_vac_rdc_dos
    (char *input_dir, Parameters *para, CPU_Data *cpu_data, Atom *atom);
};




#endif




