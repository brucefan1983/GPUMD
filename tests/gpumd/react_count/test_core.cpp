// Host-only regression test for the core of GPUMD compute_react_count.
// This includes the exact same .cu file with GPUMD headers disabled.
#define REACT_COUNT_CORE_ONLY
#include "../../../src/measure/compute_react_count.cu"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace
{

std::vector<double> parse_lattice(const std::string& line)
{
  const std::string key = "Lattice=\"";
  const size_t pos = line.find(key);
  if (pos == std::string::npos)
    throw std::runtime_error("missing Lattice");
  const size_t end = line.find('"', pos + key.size());
  std::string text = line.substr(pos + key.size(), end - pos - key.size());
  std::stringstream ss(text);
  std::vector<double> out;
  double value = 0.0;
  while (ss >> value)
    out.push_back(value);
  if (out.size() != 9)
    throw std::runtime_error("bad Lattice");
  return out;
}

} // namespace

int main(int argc, char** argv)
{
  if (argc < 3 || argc > 7) {
    std::fprintf(
      stderr, "usage: %s model.xyz output_base [reax|mutual] [start] [end] [restart]\n", argv[0]);
    return 1;
  }

  int arg = 3;
  std::string perception = "reax";
  if (arg < argc && (std::string(argv[arg]) == "reax" || std::string(argv[arg]) == "mutual"))
    perception = argv[arg++];
  const long long start_frame = arg < argc ? std::stoll(argv[arg++]) : 1;
  const long long end_frame = arg < argc ? std::stoll(argv[arg++]) : 0;
  const std::string restart_file = arg < argc ? argv[arg++] : "";
  if (arg != argc) {
    std::fprintf(stderr, "too many arguments\n");
    return 1;
  }

  std::ifstream input(argv[1]);
  if (!input) {
    std::fprintf(stderr, "cannot open %s\n", argv[1]);
    return 1;
  }

  react_count::Config cfg;
  cfg.output_base = argv[2];
  cfg.cutoff = 1.2;
  cfg.max_neigh = 10;
  cfg.events = true;
  cfg.order = {"C", "H", "O"};
  cfg.make_elements({"C", "H", "O"}, {}, {});
  cfg.restart_file = restart_file;
  if (!restart_file.empty())
    cfg.restart_interval = 50;
  if (perception == "mutual")
    cfg.perception = 1;

  std::map<std::string, int> type_of = {{"C", 0}, {"H", 1}, {"O", 2}};
  react_count::Engine engine(cfg);

  std::string line;
  long long frame = 0;
  bool opened = false;
  while (std::getline(input, line)) {
    if (line.empty())
      continue;
    const int n = std::stoi(line);
    std::string comment;
    if (!std::getline(input, comment))
      break;
    const std::vector<double> lattice = parse_lattice(comment);

    std::vector<int> types(n, 0);
    std::vector<double> positions(static_cast<size_t>(n) * 3, 0.0);
    for (int i = 0; i < n; ++i) {
      std::string atom_line;
      if (!std::getline(input, atom_line))
        throw std::runtime_error("truncated frame");
      std::stringstream ss(atom_line);
      std::string symbol;
      double x = 0.0, y = 0.0, z = 0.0;
      ss >> symbol >> x >> y >> z;
      auto it = type_of.find(symbol);
      if (it == type_of.end())
        throw std::runtime_error("unknown element " + symbol);
      types[i] = it->second;
      positions[3 * i] = x;
      positions[3 * i + 1] = y;
      positions[3 * i + 2] = z;
    }

    react_count::CoreBox box;
    box.set_lattice(lattice.data(), 1, 1, 1);
    ++frame;
    if (!opened) {
      engine.open(n);
      opened = true;
    }
    if (frame >= start_frame && (end_frame == 0 || frame <= end_frame))
      engine.step(frame, n, types.data(), positions.data(), box);
  }

  engine.close();
  std::printf(
    "processed frames %lld..%lld of %lld; wrote %s.{species,bond,coord,events}.csv\n",
    start_frame,
    end_frame == 0 ? frame : end_frame,
    frame,
    argv[2]);
  return 0;
}
