"""Host check of production MTTK temperature-chain stepping against independent RK4."""
from pathlib import Path
import shutil
import subprocess
import tempfile


def test_mttk_chain_second_order(tmp_path):
    source = (Path(__file__).parents[1] / 'src/integrate/ensemble_mttk.cu').read_text()
    start = source.index('void Ensemble_MTTK::nhc_temp_integrate(')
    method = source[start:source.index('void Ensemble_MTTK::nhc_press_integrate(', start)]
    program = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
struct Group {};
struct Box {};
struct Atom { double velocity_per_atom=1.7; };
template<class T> struct GPU_Vector {};
void scale_velocity_global(double factor,double& v) { v*=factor; }
class Ensemble_MTTK {
public:
 int tchain=3;
 double Q[3]={},eta_dot[4]={},eta_dotdot[3]={};
 double kB=1,t_target=1,t_freq=1,temperature_dof=3;
 double dt2,dt4,dt8,t_current,factor_eta;
 double find_current_temperature(const std::vector<Group>&,const Box&,Atom& a,GPU_Vector<double>&)
 { return a.velocity_per_atom*a.velocity_per_atom; }
 void nhc_temp_integrate(const std::vector<Group>&,const Box&,Atom&,GPU_Vector<double>&);
};
''' + method + r'''
using State=std::array<double,4>;
State rhs(State y) {
 return {-y[1]*y[0], y[0]*y[0]-1-y[2]*y[1],
         3*y[1]*y[1]-1-y[3]*y[2], y[2]*y[2]-1};
}
State add(State a,State b,double h) {for(int i=0;i<4;++i)a[i]+=h*b[i];return a;}
State reference() {
 State y={1.7,0,0,0};double h=1e-5;
 for(int step=0;step<100000;++step) {
  auto a=rhs(y),b=rhs(add(y,a,h/2)),c=rhs(add(y,b,h/2)),d=rhs(add(y,c,h));
  for(int i=0;i<4;++i)y[i]+=h*(a[i]+2*b[i]+2*c[i]+d[i])/6;
 }
 return y;
}
int main() {
 auto ref=reference();double previous=0;
 for(int steps:{20,40,80}) {
  double h=1./steps;Ensemble_MTTK m;Atom atom;Box box;GPU_Vector<double> thermo;
  m.dt2=h;m.dt4=h/2;m.dt8=h/4;
  for(int k=0;k<steps;++k)m.nhc_temp_integrate({},box,atom,thermo);
  State y={atom.velocity_per_atom,m.eta_dot[0],m.eta_dot[1],m.eta_dot[2]};double error=0;
  for(int i=0;i<4;++i)error=std::max(error,std::abs(y[i]-ref[i]));
  printf("steps %d max_error %.12e halving_ratio %.8f\n",steps,error,previous/error);
  if(previous)assert(previous/error>3.5 && previous/error<4.5);
  previous=error;
 }
}
'''
    compiler = shutil.which('c++')
    assert compiler, 'a C++ compiler is required'
    cpp = tmp_path / 'chain.cpp'
    cpp.write_text(program)
    binary = tmp_path / 'chain'
    subprocess.run([compiler, '-std=c++14', '-O2', str(cpp), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='mttk-chain-') as directory:
        test_mttk_chain_second_order(Path(directory))
