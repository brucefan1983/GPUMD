"""Host regression for the production Spin cutoff parser's 3 A lower bound."""
from pathlib import Path
import subprocess
import tempfile


def test_spin_cutoff_float_range(tmp_path):
    source = (Path(__file__).parents[1]/'src/main_nep/nep_spin/spin_parameters.cu').read_text()
    parser = source.split('void Spin_Parameters::parse_spin_cutoff(', 1)[1]
    parser = 'void Spin_Parameters::parse_spin_cutoff(' + parser.split(
        '\nvoid Spin_Parameters::parse_spin_type(', 1)[0]
    program = r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>
#define PRINT_INPUT_ERROR(message) throw std::runtime_error(message)
bool is_valid_real(const char* text, double* result) {
  char* end; *result=std::strtod(text,&end); return end!=text && *end=='\0';
}
struct Spin_Parameters {
  bool is_spin_cutoff_set=false, is_type_set=true;
  int num_types=2;
  std::vector<float> spin_cutoff_by_type;
  float spin_cutoff=0;
  void parse_spin_cutoff(const char**,int);
};
''' + parser + r'''
int main() {
  for(const char* value:{"2.999999","1e-100","0","-0","-1","nan","inf","1e39"}) {
    for(int position=0;position<3;position++) {
      const char* tokens[]={"spin_cutoff","6","6"};
      tokens[position==2?2:1]=value;
      Spin_Parameters model; bool rejected=false;
      try { model.parse_spin_cutoff(tokens,position==0?2:3); }
      catch(const std::runtime_error&) { rejected=true; }
      assert(rejected);
    }
  }
  for(const char* value:{"3","4","6"}) {
    for(int position=0;position<3;position++) {
      const char* tokens[]={"spin_cutoff","6","6"};
      tokens[position==2?2:1]=value;
      Spin_Parameters model; model.parse_spin_cutoff(tokens,position==0?2:3);
      const float expected=static_cast<float>(std::strtod(value,nullptr));
      assert(model.spin_cutoff_by_type.size()==2);
      for(int i=0;i<2;i++) {
        const float actual=model.spin_cutoff_by_type[i];
        assert(std::isfinite(actual) && actual>0);
        assert(actual==(position==0 || i==position-1?expected:6.0f));
      }
      assert(model.spin_cutoff==*std::max_element(model.spin_cutoff_by_type.begin(),model.spin_cutoff_by_type.end()));
    }
  }
}
'''
    cpp, binary = tmp_path/'cutoff.cpp', tmp_path/'cutoff'
    cpp.write_text(program)
    subprocess.run(['c++','-std=c++14','-O2','-Wall','-Wextra','-Werror',str(cpp),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as directory:
        test_spin_cutoff_float_range(Path(directory))
    print('Spin cutoff: scalar/per-type inputs enforce the 3 A lower bound')
