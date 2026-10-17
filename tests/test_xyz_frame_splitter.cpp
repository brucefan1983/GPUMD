// Host-only regression test: g++ -std=c++11 tests/test_xyz_frame_splitter.cpp -o /tmp/test_xyz_splitter
#include "../src/measure/xyz_frame_splitter.h"
#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static int count_frames(const std::string& filename)
{
  std::ifstream f(filename);
  int count = 0;
  std::string atom_count, comment, atom;
  while (std::getline(f, atom_count)) {
    assert(atom_count == "1");
    assert(bool(std::getline(f, comment)));
    assert(bool(std::getline(f, atom)));
    assert(atom == "Mo 0 0 0");
    ++count;
  }
  return count;
}

static std::vector<std::string> simulate_run(XYZFrameSplitter& splitter, int frames)
{
  std::vector<std::string> created;
  FILE* fid = nullptr;
  for (int i = 0; i < frames; ++i) {
    if (splitter.needs_new_file()) {
      if (fid) std::fclose(fid);
      const auto path = splitter.next_filename();
      fid = std::fopen(path.c_str(), "w");
      assert(fid);
      created.push_back(path);
    }
    std::fprintf(fid, "1\nFrame\nMo 0 0 0\n");
    std::fflush(fid);
    splitter.frame_written();
  }
  if (fid) std::fclose(fid);
  return created;
}

int main()
{
  XYZFrameSplitter s;
  s.reset("split_test.xyz", 3);
  auto first = simulate_run(s, 8);
  assert(first.size() == 3);
  assert(first[0] == "split_test_0001.xyz");
  assert(first[1] == "split_test_0002.xyz");
  assert(first[2] == "split_test_0003.xyz");
  assert(count_frames(first[0]) == 3);
  assert(count_frames(first[1]) == 3);
  assert(count_frames(first[2]) == 2);

  // A new run with the same base name must not overwrite previously saved frames.
  s.reset("split_test.xyz", 3);
  auto second = simulate_run(s, 4);
  assert(second.size() == 2);
  assert(second[0] == "split_test_0004.xyz");
  assert(second[1] == "split_test_0005.xyz");
  assert(count_frames(first[0]) == 3);
  assert(count_frames(first[2]) == 2);
  assert(count_frames(second[0]) == 3);
  assert(count_frames(second[1]) == 1);

  // The intended production use: exactly 10000 frames in each complete chunk.
  XYZFrameSplitter large;
  large.reset("tenk.xyz", 10000);
  auto tenk = simulate_run(large, 20001);
  assert(tenk.size() == 3);
  assert(count_frames(tenk[0]) == 10000);
  assert(count_frames(tenk[1]) == 10000);
  assert(count_frames(tenk[2]) == 1);

  // No extension, filename with multiple dots, and hidden filename.
  XYZFrameSplitter other;
  other.reset("noext", 1);
  assert(other.next_filename() == "noext_0001");
  other.reset("a.b.xyz", 1);
  assert(other.next_filename() == "a.b_0001.xyz");
  other.reset(".hidden", 1);
  assert(other.next_filename() == ".hidden_0001");

  // Skip pre-existing chunks even when the numbering has a gap.
  std::remove("split_test_0003.xyz");
  s.reset("split_test.xyz", 1);
  assert(s.next_filename() == "split_test_0003.xyz");

  for (const auto& name : first) std::remove(name.c_str());
  for (const auto& name : second) std::remove(name.c_str());
  for (const auto& name : tenk) std::remove(name.c_str());
  std::cout << "PASS: 10000+10000+1 frames; 3/3/2 then 3/1; numbering; no overwrite\n";
}
