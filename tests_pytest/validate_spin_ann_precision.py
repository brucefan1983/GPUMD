#!/usr/bin/env python3
"""Check the shared Spin ANN accumulator against an independent long-double oracle.

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
#include "utilities/nep_utilities.cuh"
constexpr int D=64, H=30, H2=7;
constexpr int P=H*D+H+H+H*H2+H2+H2+1;
__global__ void evaluate(float* p, float* q, double* result, int layers) {
  double e=0, g[D]={};
  const float *w0=p, *b0=w0+H*D, *w1=b0+H;
  if(layers==1) apply_ann_one_layer(D,H,w0,b0,w1,w1+H,q,e,g);
  else {
    const float *b1=w1+H*H2, *w2=b1+H2;
    apply_ann_two_layers(D,H,H2,w0,b0,w1,b1,w2,w2+H2,q,e,g);
  }
  result[0]=e;
  for(int i=0;i<D;i++) result[i+1]=g[i];
}
long double oracle(const float* p, const long double* q, int layers) {
  const float *w0=p, *b0=w0+H*D, *w1=b0+H;
  long double x[H], e=0;
  for(int n=0;n<H;n++) {
    long double z=-b0[n];
    for(int d=0;d<D;d++) z+=(long double)w0[n*D+d]*q[d];
    x[n]=std::tanh(z);
  }
  if(layers==1) {
    for(int n=0;n<H;n++) e+=(long double)w1[n]*x[n];
    return e-w1[H];
  }
  const float *b1=w1+H*H2, *w2=b1+H2;
  for(int n=0;n<H2;n++) {
    long double z=-b1[n];
    for(int m=0;m<H;m++) z+=(long double)w1[n*H+m]*x[m];
    e+=(long double)w2[n]*std::tanh(z);
  }
  return e-w2[H2];
}
#define CHECK(call) do { auto err=(call); if(err!=cudaSuccess) { \
  std::fprintf(stderr,"%s\n",cudaGetErrorString(err)); return 2; } } while(0)
int main() {
  float p[P], q[D]; long double refq[D];
  for(int i=0;i<D;i++) q[i]=0.5f+0.2f*std::cos(0.31*i);
  float *dp,*dq; double *dr,r[D+1];
  CHECK(cudaMalloc(&dp,sizeof(p))); CHECK(cudaMalloc(&dq,sizeof(q)));
  CHECK(cudaMalloc(&dr,sizeof(r)));
  double max_energy_error=0,max_gradient_error=0,max_difference_error=0;
  for(int layers=1;layers<=2;layers++) {
    for(int i=0;i<P;i++) p[i]=0.25f*std::sin(0.17*(i+1));
    // A large shared offset makes float atomic energies lose small responses.
    int bias=(layers==1 ? H*D+2*H : H*D+H+H*H2+2*H2);
    p[bias]=32.125f;
    CHECK(cudaMemcpy(dp,p,sizeof(p),cudaMemcpyHostToDevice));
    double energies[2]; long double references[2];
    for(int state=0;state<2;state++) {
      if(state) q[3]+=1e-5f;
      for(int i=0;i<D;i++) refq[i]=q[i];
      CHECK(cudaMemcpy(dq,q,sizeof(q),cudaMemcpyHostToDevice));
      evaluate<<<1,1>>>(dp,dq,dr,layers);
      CHECK(cudaGetLastError()); CHECK(cudaMemcpy(r,dr,sizeof(r),cudaMemcpyDeviceToHost));
      energies[state]=r[0]; references[state]=oracle(p,refq,layers);
      max_energy_error=std::fmax(max_energy_error,(double)std::fabs(r[0]-references[state]));
      for(int d=0;d<D;d++) {
        const long double h=1e-5L,old=refq[d];
        refq[d]=old+h;long double plus=oracle(p,refq,layers);
        refq[d]=old-h;long double minus=oracle(p,refq,layers);
        refq[d]=old;
        max_gradient_error=std::fmax(max_gradient_error,(double)std::fabs(r[d+1]-(plus-minus)/(2*h)));
      }
    }
    max_difference_error=std::fmax(max_difference_error,(double)std::fabs(
      (energies[1]-energies[0])-(references[1]-references[0])));
  }
  CHECK(cudaFree(dp));CHECK(cudaFree(dq));CHECK(cudaFree(dr));
  std::printf("energy %.3e, energy_difference %.3e, derivative %.3e\n",max_energy_error,max_difference_error,max_gradient_error);
  return max_energy_error<1e-11 && max_difference_error<1e-12 && max_gradient_error<1e-9 ? 0 : 1;
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
        subprocess.run([args.nvcc, '-std=c++14', '-O3', '-arch='+args.arch,
                        '-I'+str(ROOT/'src'), str(source), '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    main()
