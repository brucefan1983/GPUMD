/*
  Copyright 2026 GPUMD contributors.
  This file is part of GPUMD and is distributed under the LGPL-3.0-or-later.
*/
#pragma once

#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpumd_rerun {

struct Frame {
  std::vector<std::string> species;
  std::vector<double> positions; // GPUMD structure of arrays: x[N], y[N], z[N]
  std::array<double, 9> lattice{}; // GPUMD column-vector matrix
  std::array<int, 3> pbc{{1, 1, 1}};
  double time_fs = 0.0;
  bool has_step = false;
  long long step = 0;
};

inline std::string attribute(const std::string& line, const std::string& key)
{
  const std::string prefix = key + "=";
  // Match complete metadata keys, not suffixes of unrelated attributes.
  size_t start = line.find(prefix);
  while (start != std::string::npos && start > 0 &&
         line[start - 1] != ' ' && line[start - 1] != '\t') {
    start = line.find(prefix, start + 1);
  }
  if (start == std::string::npos) {
    throw std::runtime_error("rerun: missing XYZ attribute " + key);
  }
  size_t begin = start + prefix.size();
  if (begin >= line.size()) {
    throw std::runtime_error("rerun: empty XYZ attribute " + key);
  }
  if (line[begin] == '"') {
    const size_t end = line.find('"', ++begin);
    if (end == std::string::npos) {
      throw std::runtime_error("rerun: unterminated XYZ attribute " + key);
    }
    return line.substr(begin, end - begin);
  }
  const size_t end = line.find_first_of(" \t\r\n", begin);
  return line.substr(begin, end - begin);
}

inline std::vector<std::string> split(const std::string& text, char delim)
{
  std::vector<std::string> out;
  std::istringstream in(text);
  std::string token;
  while (std::getline(in, token, delim)) out.push_back(token);
  return out;
}

class Reader {
public:
  explicit Reader(const std::string& path) : stream_(path)
  {
    if (!stream_) throw std::runtime_error("rerun: cannot open " + path);
  }

  bool next(Frame& frame)
  {
    std::string line;
    if (!std::getline(stream_, line)) return false;
    while (line.empty() && std::getline(stream_, line)) {}
    if (line.empty()) return false;
    int n = 0;
    {
      std::istringstream in(line);
      std::string trailing;
      if (!(in >> n) || n <= 0 || (in >> trailing))
        throw std::runtime_error("rerun: invalid XYZ atom count");
    }
    if (!std::getline(stream_, line))
      throw std::runtime_error("rerun: truncated XYZ comment line");

    frame.has_step = false; // A reused Frame must not retain the previous step.
    frame.time_fs = std::stod(attribute(line, "Time"));
    if (!std::isfinite(frame.time_fs))
      throw std::runtime_error("rerun: nonfinite XYZ Time");

    // Prefer an explicit integer Step when supplied; time-based reconstruction
    // cannot recover variable-timestep trajectories unambiguously.
    for (const std::string& step_key : {"Step", "step", "Timestep"}) {
      try {
        const std::string value = attribute(line, step_key);
        size_t used = 0;
        const long long step = std::stoll(value, &used);
        if (used != value.size() || step < 0)
          throw std::runtime_error("rerun: invalid integer timestep");
        frame.step = step;
        frame.has_step = true;
        break;
      } catch (const std::invalid_argument&) {
        throw std::runtime_error("rerun: invalid integer timestep");
      } catch (const std::out_of_range&) {
        throw std::runtime_error("rerun: integer timestep is out of range");
      } catch (const std::runtime_error& error) {
        // A missing key is normal; a malformed present key is not.
        if (std::string(error.what()) != "rerun: missing XYZ attribute " + step_key)
          throw;
      }
    }
    std::istringstream cell(attribute(line, "Lattice"));
    // Extended XYZ writes a_x,a_y,a_z,b_x,...; GPUMD uses columns.
    for (int i = 0; i < 9; ++i) {
      if (!(cell >> frame.lattice[(i % 3) * 3 + i / 3]))
        throw std::runtime_error("rerun: invalid Lattice");
    }
    double extra;
    if (cell >> extra) throw std::runtime_error("rerun: Lattice must have 9 entries");
    std::istringstream bc_stream(attribute(line, "pbc"));
    std::vector<std::string> bc;
    std::string bc_item;
    while (bc_stream >> bc_item) bc.push_back(bc_item);
    if (bc.size() != 3) throw std::runtime_error("rerun: pbc must have 3 entries");
    for (int i = 0; i < 3; ++i) {
      if (bc[i] == "T" || bc[i] == "t") frame.pbc[i] = 1;
      else if (bc[i] == "F" || bc[i] == "f") frame.pbc[i] = 0;
      else throw std::runtime_error("rerun: invalid pbc flag");
    }

    auto props = split(attribute(line, "Properties"), ':');
    if (props.size() % 3 != 0) throw std::runtime_error("rerun: malformed Properties");
    int species_offset = -1, pos_offset = -1, width = 0;
    for (size_t i = 0; i < props.size(); i += 3) {
      int count = std::stoi(props[i + 2]);
      if (count <= 0) throw std::runtime_error("rerun: invalid property width");
      if (props[i] == "species" && count == 1) species_offset = width;
      if (props[i] == "pos" && count == 3) pos_offset = width;
      width += count;
    }
    if (species_offset < 0 || pos_offset < 0)
      throw std::runtime_error("rerun: require species:S:1 and pos:R:3");

    frame.species.resize(n);
    frame.positions.resize(3 * n);
    for (int i = 0; i < n; ++i) {
      if (!std::getline(stream_, line))
        throw std::runtime_error("rerun: truncated XYZ atom data");
      std::istringstream in(line);
      std::vector<std::string> fields;
      std::string field;
      while (in >> field) fields.push_back(field);
      if (static_cast<int>(fields.size()) != width)
        throw std::runtime_error("rerun: wrong number of XYZ columns");
      frame.species[i] = fields[species_offset];
      for (int d = 0; d < 3; ++d) {
        frame.positions[i + n * d] = std::stod(fields[pos_offset + d]);
        if (!std::isfinite(frame.positions[i + n * d]))
          throw std::runtime_error("rerun: nonfinite coordinates");
      }
    }
    return true;
  }

private:
  std::ifstream stream_;
};

inline long long time_to_step(double time_fs, double dt_fs)
{
  if (!(dt_fs > 0.0) || !std::isfinite(dt_fs))
    throw std::runtime_error("rerun: time_step must be positive");
  const double value = time_fs / dt_fs;
  if (!std::isfinite(value) || std::abs(value) > 9.0e15)
    throw std::runtime_error("rerun: timestep out of range");
  const long long step = std::llround(value);
  if (std::abs(value - static_cast<double>(step)) > 1.0e-5)
    throw std::runtime_error("rerun: Time is not a multiple of time_step");
  return step;
}


inline long long frame_step(const Frame& frame, double dt_fs)
{
  return frame.has_step ? frame.step : time_to_step(frame.time_fs, dt_fs);
}

// The selection policy is independent from file I/O and GPU evaluation.
struct Selection {
  long long first = 0;
  long long last = std::numeric_limits<long long>::max();
  long long every = 0;
  long long skip = 1;
  long long candidates = 0;

  bool accepts(long long step)
  {
    if (step < first || step > last) return false;
    // Like LAMMPS, preserve the first eligible snapshot even when it is
    // not divisible by 'every'. The skip counter counts eligible snapshots.
    if (every > 0 && candidates > 0 && step % every != 0) return false;
    const bool selected = (candidates % skip == 0);
    ++candidates;
    return selected;
  }
};

} // namespace gpumd_rerun
