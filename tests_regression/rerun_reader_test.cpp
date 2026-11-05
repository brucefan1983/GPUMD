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
    out << "Time=300 Step=300 pbc=\"T F T\" Lattice=\"5 0 0 0 6 0 0 0 7\" "
           "Properties=species:S:1:pos:R:3\n";
    out << "Ga 2 3 4\n";
    out << "Pd 5 6 7\n";
    out << "2\n";
    out << "Time=400 Step=400 pbc=\"T T T\" Lattice=\"5 0 0 0 6 0 0 0 7\" "
           "Properties=id:I:1:species:S:1:pos:R:3:vel:R:3\n";
    out << "2 Pd 8 9 10 0.01 0.02 0.03\n";
    out << "1 Ga 2 3 4 0.04 0.05 0.06\n";
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
  assert(frame.time_fs == 300 && frame.positions[0] == 2);
  assert(frame.has_step && frame.step == 300);
  assert(gpumd_rerun::frame_step(frame, 1.0) == 300);
  assert(reader.next(frame));
  assert(frame.has_step && frame.step == 400);
  assert(frame.species[0] == "Ga" && frame.species[1] == "Pd");
  assert(frame.positions[0] == 2 && frame.positions[1] == 8);
  assert(frame.positions[2] == 3 && frame.positions[3] == 9);
  assert(frame.velocities.size() == 6);
  assert(std::abs(frame.velocities[0] - 0.04) < 1.0e-12);
  assert(std::abs(frame.velocities[1] - 0.01) < 1.0e-12);
  assert(std::abs(frame.velocities[2] - 0.05) < 1.0e-12);
  assert(!reader.next(frame));

  // The fixed step-to-time mapping must agree before PLUMED receives a frame.
  gpumd_rerun::Frame bad_clock;
  bad_clock.has_step = true;
  bad_clock.step = 300;
  bad_clock.time_fs = 30;
  bool bad_time_rejected = false;
  try {
    (void)gpumd_rerun::frame_step(bad_clock, 1.0);
  } catch (const std::runtime_error&) {
    bad_time_rejected = true;
  }
  assert(bad_time_rejected);

  gpumd_rerun::Selection selector;
  selector.first = 15;
  selector.last = 300;
  selector.every = 100;
  selector.skip = 2;
  // The first in-range frame is kept regardless of its modulo.
  assert(selector.accepts(20));
  assert(!selector.accepts(100));
  assert(selector.accepts(200));
  assert(!selector.accepts(300));
  assert(!selector.accepts(400));

  gpumd_rerun::Selection by_step;
  by_step.first = 100;
  by_step.last = 300;
  by_step.every = 100;
  assert(!by_step.accepts(20));
  assert(by_step.accepts(100));
  assert(by_step.accepts(200));
  assert(by_step.accepts(300));
  // Numeric parser must reject silent partial parsing such as 1.0garbage.
  bool rejected = false;
  try {
    gpumd_rerun::parse_finite_double("1.0junk", "coordinate");
  } catch (const std::exception&) {
    rejected = true;
  }
  assert(rejected);

  rejected = false;
  try {
    gpumd_rerun::parse_finite_double("nan", "coordinate");
  } catch (const std::exception&) {
    rejected = true;
  }
  assert(rejected);

  // LAMMPS-like selection: every checks timestep, skip counts candidates.
  gpumd_rerun::Selection sparse;
  sparse.every = 100;
  sparse.skip = 2;
  assert(sparse.accepts(25));   // first eligible snapshot always accepted
  assert(!sparse.accepts(50));  // not on every-grid
  assert(!sparse.accepts(100)); // candidate 2, removed by skip
  assert(sparse.accepts(200));  // candidate 3
  assert(!sparse.accepts(300)); // candidate 4

  std::remove(path);
  return 0;
}
