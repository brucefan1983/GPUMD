"""Host-only checks of the production chain integrator against independent RK4."""
from pathlib import Path
import shutil
import subprocess
import tempfile


def test_nhc_stability(tmp_path):
    source = (Path(__file__).parents[1] / 'src/integrate/ensemble_nhc.cu').read_text()
    chain = source[source.index('static double nhc('):source.index('void Ensemble_NHC::integrate_nvt_nhc_1(')]
    program = r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#define PRINT_INPUT_ERROR(message) throw std::runtime_error(message)
using std::exp;
''' + chain + r'''
// q[0:4], p[4:8], twice kinetic energy: the continuous NHC equations.
void derivative(const double* y, const double* mass, double kt, double dof, double* out) {
  for (int i=0;i<4;++i) {
    out[i]=y[4+i]/mass[i];
    out[4+i]=(i ? y[3+i]*y[3+i]/mass[i-1]-kt : y[8]-dof*kt)
             -(i<3 ? y[4+i]*y[5+i]/mass[i+1] : 0);
  }
  out[8]=-2*y[8]*y[4]/mass[0];
}
void rk4(double* y, const double* mass, double kt, double dof, double dt) {
  double k[4][9],t[9];
  derivative(y,mass,kt,dof,k[0]);
  for(int stage=1;stage<4;++stage) {
    for(int i=0;i<9;++i)t[i]=y[i]+dt*(stage==3?1:.5)*k[stage-1][i];
    derivative(t,mass,kt,dof,k[stage]);
  }
  for(int i=0;i<9;++i)y[i]+=dt*(k[0][i]+2*k[1][i]+2*k[2][i]+k[3][i])/6;
}
double energy(const double* y,const double* mass,double kt,double dof) {
  double e=y[8]/2+dof*kt*y[0];
  for(int i=0;i<4;++i)e+=y[4+i]*y[4+i]/mass[i]/2+(i?kt*y[i]:0);
  return e;
}
int main() {
  const double kb=8.617343e-5,kt=100*kb,dt=.5/10.18051;
  for(int n : {4,5575680}) for(double startT : {30.,55.,100.}) {
    const double dof=3.*n,m=kt*std::pow(dt*100,2);
    double mass[4]={dof*m,m,m,m},y[9]={0,0,0,0,1,-1,1,-1,dof*kb*startT},ref[9];
    std::copy(y,y+9,ref);
    double initial=energy(y,mass,kt,dof),drift=0;
    for(int half=0;half<2400;++half) {
      double f=nhc(4,y,y+4,mass,y[8],kt,dof,dt/2);y[8]*=f*f;
      assert(std::isfinite(energy(y,mass,kt,dof)));
      drift=std::max(drift,std::abs(energy(y,mass,kt,dof)-initial));
      for(int sub=0;sub<512;++sub)rk4(ref,mass,kt,dof,dt/1024);
    }
    double deltaT=std::abs(y[8]-ref[8])/dof/kb;
    printf("n %d T0 %g Tfinal %.12g delta_RK4 %.6g energy_drift %.6g\n",n,startT,y[8]/dof/kb,deltaT,drift);
    assert(deltaT<1e-5);
    assert(drift<.005);
  }
}
'''
    cpp = tmp_path / 'chain.cpp'
    cpp.write_text(program)
    binary = tmp_path / 'chain'
    compiler = shutil.which('c++')
    assert compiler, 'C++ compiler required for the host NHC check'
    subprocess.run([compiler, '-std=c++14', '-O2', str(cpp), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as temporary:
        test_nhc_stability(Path(temporary))
