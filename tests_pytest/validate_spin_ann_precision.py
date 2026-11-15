#!/usr/bin/env python3
# Copyright 2017 Zheyong Fan and GPUMD development team
# This file is part of GPUMD.
# GPUMD is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# GPUMD is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License
# along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.

"""Check the spin-private one-layer ANN accumulator against an independent long-double oracle.

This isolates ANN cancellation and its energy derivative from descriptor rounding.
Use validate_spin_training.py --check-jit for the native/JIT end-to-end contract.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = r'''
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include "force/nep_spin/spin_math.cuh"
constexpr int D=64, H=30, P=H*D+2*H+1;
__global__ void evaluate(float* p, float* q, double* result) {
  double energy=0, gradient[D]={};
  const float *w0=p, *b0=w0+H*D, *w1=b0+H;
  nep_spin::apply_ann_one_layer(D,H,w0,b0,w1,w1+H,q,energy,gradient);
  result[0]=energy;
  for(int i=0;i<D;i++) result[i+1]=gradient[i];
}
long double oracle(const float* p, const long double* q) {
  const float *w0=p, *b0=w0+H*D, *w1=b0+H;
  long double energy=0;
  for(int n=0;n<H;n++) {
    long double z=-b0[n];
    for(int d=0;d<D;d++) z+=(long double)w0[n*D+d]*q[d];
    energy+=(long double)w1[n]*std::tanh(z);
  }
  return energy-w1[H];
}
#define CHECK(call) do { auto err=(call); if(err!=cudaSuccess) { \
  std::fprintf(stderr,"%s\n",cudaGetErrorString(err)); return 2; } } while(0)
int main() {
  float p[P],q[D];long double refq[D];
  for(int i=0;i<D;i++)q[i]=.5f+.2f*std::cos(.31*i);
  for(int i=0;i<P;i++)p[i]=.25f*std::sin(.17*(i+1));
  p[P-1]=32.125f;
  float *dp,*dq;double *dr,result[D+1],energies[2];long double references[2];
  CHECK(cudaMalloc(&dp,sizeof(p)));CHECK(cudaMalloc(&dq,sizeof(q)));
  CHECK(cudaMalloc(&dr,sizeof(result)));CHECK(cudaMemcpy(dp,p,sizeof(p),cudaMemcpyHostToDevice));
  double max_energy_error=0,max_gradient_error=0;
  for(int state=0;state<2;state++) {
    if(state)q[3]+=1e-5f;
    for(int d=0;d<D;d++)refq[d]=q[d];
    CHECK(cudaMemcpy(dq,q,sizeof(q),cudaMemcpyHostToDevice));
    evaluate<<<1,1>>>(dp,dq,dr);
    CHECK(cudaGetLastError());CHECK(cudaMemcpy(result,dr,sizeof(result),cudaMemcpyDeviceToHost));
    energies[state]=result[0];references[state]=oracle(p,refq);
    max_energy_error=std::fmax(max_energy_error,(double)std::fabs(result[0]-references[state]));
    for(int d=0;d<D;d++) {
      const long double h=1e-5L,old=refq[d];
      refq[d]=old+h;long double plus=oracle(p,refq);
      refq[d]=old-h;long double minus=oracle(p,refq);refq[d]=old;
      max_gradient_error=std::fmax(max_gradient_error,(double)std::fabs(result[d+1]-(plus-minus)/(2*h)));
    }
  }
  const double difference_error=(double)std::fabs(
    (energies[1]-energies[0])-(references[1]-references[0]));
  CHECK(cudaFree(dp));CHECK(cudaFree(dq));CHECK(cudaFree(dr));
  std::printf("energy %.3e, energy_difference %.3e, derivative %.3e\n",
    max_energy_error,difference_error,max_gradient_error);
  return max_energy_error<1e-11 && difference_error<1e-12 && max_gradient_error<1e-9?0:1;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nvcc', default='nvcc')
    parser.add_argument('--arch', default='sm_60')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='gpumd-spin-ann-precision-') as tmp:
        source = Path(tmp) / 'test.cu'
        binary = Path(tmp) / 'test'
        source.write_text(SOURCE)
        subprocess.run([args.nvcc, '-std=c++14', '-O3', '-DDEBUG', '-arch='+args.arch,
                        '-I'+str(ROOT/'src'), str(source), '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    main()
