"""Run the production midpoint kernel on the host with GPU indexing stubbed."""
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def test_midpoint_wraps_all_images(tmp_path):
    step = (ROOT/'src/integrate/spin/step.cu').read_text()
    kernel = re.split(r'__global__\s+void\s+midpoint_position\s*\(', step, maxsplit=1)[1].split('// Save each Action', 1)[0]
    box = (ROOT/'src/model/box.cu').read_text()
    inverse = 'void Box::get_inverse(' + box.split('void Box::get_inverse(', 1)[1].split('\nbool Box::get_num_bins', 1)[0]
    # Only get_inverse is needed; following Box definitions are excluded.
    inverse = inverse.split('\nvoid Box::', 1)[0]
    program = r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#define __host__
#define __device__
#include "model/box.cuh"
struct Index { int x; } blockIdx{0}, blockDim{128}, threadIdx{0};
''' + inverse + '\nvoid midpoint_position(' + kernel + r'''
int main() {
  constexpr int n=2;
  for(bool skew:{false,true}) for(bool deform:{false,true})
  for(int mask:{0,5,7}) for(int image:{-37,-5,0,5,29}) for(bool crossing:{false,true}) {
    Box box;
    box.pbc_x=mask&1; box.pbc_y=(mask>>1)&1; box.pbc_z=(mask>>2)&1;
    box.is_orthogonal=!skew;
    const double a[9]={20,skew?2.:0,skew?.5:0,0,22,skew?1.:0,0,0,24};
    double b[9];
    for(int k=0;k<9;k++) {
      b[k]=a[k]+(deform && k%4==0?.4:0);
      box.cpu_h[k]=.5*(a[k]+b[k]);
    }
    box.get_inverse();
    double start[3*n],end[3*n],out[3*n],expected[3*n];
    for(int atom=0;atom<n;atom++) {
      const int images[3]={atom?-image:image,2*image,-image};
      for(int d=0;d<3;d++) {
        const double r=d==0?(atom?3.:1.):2.+d;
        // A physical x crossing, also independent of accumulated image history.
        const double mid=(crossing && !skew && d==0 && atom==0)?box.cpu_h[0]:r;
        start[d*n+atom]=mid-.1;end[d*n+atom]=mid+.1;expected[d*n+atom]=mid;
        for(int k=0;k<3;k++) {
          start[d*n+atom]+=a[3*d+k]*images[k];
          end[d*n+atom]+=b[3*d+k]*images[k];
          if(!(mask&(1<<k))) expected[d*n+atom]+=box.cpu_h[3*d+k]*images[k];
        }
        if(crossing && !skew && d==0 && atom==0 && (mask&1)) expected[d*n+atom]=0;
      }
    }
    for(int i=0;i<128;i++) { threadIdx.x=i;midpoint_position(n,box,start,end,out); }
    for(int atom=0;atom<n;atom++) {
      for(int d=0;d<3;d++) if(mask&(1<<d)) {
        double fraction=0;
        for(int k=0;k<3;k++) fraction+=box.cpu_h[9+3*d+k]*out[k*n+atom];
        assert(fraction>=-1e-12 && fraction<=1+1e-12);
      }
      double dx=out[atom]-expected[atom],dy=out[n+atom]-expected[n+atom],dz=out[2*n+atom]-expected[2*n+atom];
      apply_mic(box,dx,dy,dz); // Either representative of an exact face is equivalent.
      if(std::max({std::abs(dx),std::abs(dy),std::abs(dz)})>=2e-11) {
        std::fprintf(stderr,"skew=%d deform=%d pbc=%d image=%d crossing=%d atom=%d\n",skew,deform,mask,image,crossing,atom);
        return 1;
      }
    }
    if(mask==7) {
      double dx=out[0]-out[1],dy=out[n]-out[n+1],dz=out[2*n]-out[2*n+1];
      apply_mic(box,dx,dy,dz);
      assert(dx*dx+dy*dy+dz*dz<36); // The neighbor must survive every image history.
    }
  }
}
'''
    source, binary = tmp_path/'midpoint.cpp', tmp_path/'midpoint'
    source.write_text(program)
    subprocess.run(['c++','-std=c++14','-O2','-I'+str(ROOT/'src'),str(source),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as directory:
        test_midpoint_wraps_all_images(Path(directory))
    print('Midpoint wrapping: 120 orthogonal/triclinic/deforming/PBC/image cases passed')
