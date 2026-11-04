// Standalone CPU-only parser regression test:
// c++ -std=c++14 -Isrc/main_gpumd tests_regression/rerun_reader_test.cpp -o /tmp/rerun_reader_test
// /tmp/rerun_reader_test
#include "rerun_reader.hpp"
#include <cassert>
#include <cstdio>
#include <fstream>
#include <stdexcept>

int main()
{
  const char* path = "rerun_reader_test.xyz";
  {
    std::ofstream out(path);
    out << "2\n";
    out << "Time=20 pbc=\"T F T\" Lattice=\"5 0 0 0 6 0 0 0 7\" "
           "Properties=species:S:1:force:R:3:pos:R:3\n";
    out << "Ga 0 0 0 1 2 3\n";
    out << "Pd 0 0 0 4 5 6\n";
    out << "2\n";
    out << "Time=30 pbc=\"T F T\" Lattice=\"5 0 0 0 6 0 0 0 7\" "
           "Properties=species:S:1:pos:R:3\n";
    out << "Ga 2 3 4\n";
    out << "Pd 5 6 7\n";
  }
  gpumd_rerun::Reader reader(path);
  gpumd_rerun::Frame frame;
  assert(reader.next(frame));
  assert(frame.time_fs == 20);
  assert(frame.positions[0] == 1 && frame.positions[1] == 4);
  assert(frame.positions[2] == 2 && frame.positions[3] == 5);
  assert(frame.positions[4] == 3 && frame.positions[5] == 6);
  assert(frame.pbc[1] == 0);
  assert(gpumd_rerun::time_to_step(frame.time_fs, 1.0) == 20);
  assert(reader.next(frame));
  assert(frame.time_fs == 30 && frame.positions[0] == 2);
  assert(!reader.next(frame));
  std::remove(path);
  return 0;
}
