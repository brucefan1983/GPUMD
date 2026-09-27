/*
    Copyright 2017 Zheyong Fan and GPUMD development team.
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
*/

/*----------------------------------------------------------------------------80
Standalone one-shot Cartesian Hessian diagnostic command. Two modes:
  compute_hessian analytic  -> analytic NEP Hessian
  compute_hessian fd        -> central finite-difference Hessian (any potential)
Both can optionally solve phonons from the resulting Hessian.
------------------------------------------------------------------------------*/

#include "nep_hessian_command.cuh"

#include "force/force.cuh"
#include "force/nep.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include "nep_analytic_hessian.cuh"
#include "analytic_phonon.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#ifdef USE_HIP
#include <hipsolver/hipsolver.h>
#else
#include <cusolverDn.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <cerrno>
#include <limits.h>
#include <sys/stat.h>
#include <unistd.h>

namespace
{

constexpr double kTiny = std::numeric_limits<double>::min();

// realpath alone cannot resolve a dangling output symlink. Resolve its target
// before canonicalizing the existing parent of a not-yet-created output file.
std::string canonical_output_path(const std::string& path, int depth = 0)
{
  if (depth > 40) throw std::runtime_error("Too many output path symlinks.");
  char resolved[PATH_MAX];
  if (realpath(path.c_str(), resolved)) return resolved;
  struct stat entry;
  if (lstat(path.c_str(), &entry) == 0) {
    if (!S_ISLNK(entry.st_mode))
      throw std::runtime_error("Cannot resolve Hessian path: " + path);
    char target[PATH_MAX];
    const ssize_t size = readlink(path.c_str(), target, sizeof(target));
    if (size <= 0 || size >= static_cast<ssize_t>(sizeof(target)))
      throw std::runtime_error("Cannot resolve Hessian symlink: " + path);
    const std::string link(target, size);
    const auto slash = path.find_last_of('/');
    const std::string parent = slash == std::string::npos ? "." : path.substr(0, slash);
    return canonical_output_path(link[0] == '/' ? link : parent + "/" + link, depth + 1);
  }
  if (errno != ENOENT) throw std::runtime_error("Cannot inspect Hessian path: " + path);
  const auto slash = path.find_last_of('/');
  const std::string parent = slash == std::string::npos ? "." :
    (slash == 0 ? "/" : path.substr(0, slash));
  const std::string name = slash == std::string::npos ? path : path.substr(slash + 1);
  if (name.empty() || !realpath(parent.c_str(), resolved))
    throw std::runtime_error("Cannot resolve Hessian output parent: " + path);
  return std::string(resolved) + "/" + name;
}

bool same_file(const std::string& left, const std::string& right)
{
  if (canonical_output_path(left) == canonical_output_path(right)) return true;
  struct stat a, b;
  return stat(left.c_str(), &a) == 0 && stat(right.c_str(), &b) == 0 &&
    a.st_dev == b.st_dev && a.st_ino == b.st_ino;
}

class SHA256
{
public:
  SHA256() { reset(); }

  void update(const unsigned char* data, size_t length)
  {
    for (size_t i = 0; i < length; ++i) {
      buffer_[buffer_size_++] = data[i];
      if (buffer_size_ == 64) {
        transform(buffer_.data());
        buffer_size_ = 0;
      }
      ++bytes_;
    }
  }

  std::string final_hex()
  {
    const std::uint64_t bits = bytes_ * 8;
    const unsigned char one = 0x80;
    update(&one, 1);
    const unsigned char zero = 0;
    while (buffer_size_ != 56)
      update(&zero, 1);
    unsigned char length_bytes[8];
    for (int i = 0; i < 8; ++i)
      length_bytes[i] = static_cast<unsigned char>(bits >> (56 - 8 * i));
    update(length_bytes, 8);
    std::string result(64, '0');
    static const char digits[] = "0123456789abcdef";
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 8; ++j) {
        const unsigned nibble = (state_[i] >> (4 * (7 - j))) & 0xfU;
        result[8 * i + j] = digits[nibble];
      }
    }
    return result;
  }

private:
  void reset()
  {
    state_ = {0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
              0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
    bytes_ = 0;
    buffer_size_ = 0;
  }

  static std::uint32_t rotr(std::uint32_t value, unsigned count)
  {
    return (value >> count) | (value << (32 - count));
  }

  void transform(const unsigned char* block)
  {
    static const std::uint32_t k[] = {
      0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU,
      0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U,
      0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U,
      0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
      0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U,
      0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
      0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
      0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
      0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U,
      0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U, 0x1e376c08U,
      0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU,
      0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
      0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};
    std::uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
      w[i] = (static_cast<std::uint32_t>(block[4 * i]) << 24) |
             (static_cast<std::uint32_t>(block[4 * i + 1]) << 16) |
             (static_cast<std::uint32_t>(block[4 * i + 2]) << 8) |
             static_cast<std::uint32_t>(block[4 * i + 3]);
    }
    for (int i = 16; i < 64; ++i) {
      const std::uint32_t s0 =
        rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const std::uint32_t s1 =
        rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    std::uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
    std::uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
    for (int i = 0; i < 64; ++i) {
      const std::uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const std::uint32_t ch = (e & f) ^ ((~e) & g);
      const std::uint32_t temp1 = h + s1 + ch + k[i] + w[i];
      const std::uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temp2 = s0 + maj;
      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }
    state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
    state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
  }

  std::array<std::uint32_t, 8> state_;
  std::array<unsigned char, 64> buffer_;
  size_t buffer_size_ = 0;
  std::uint64_t bytes_ = 0;
};

std::string trim(const std::string& value)
{
  size_t begin = 0;
  size_t end = value.size();
  while (begin < end &&
         std::isspace(static_cast<unsigned char>(value[begin])))
    ++begin;
  while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1])))
    --end;
  return value.substr(begin, end - begin);
}

std::string read_first_line(const std::string& path)
{
  std::ifstream input(path);
  std::string line;
  std::getline(input, line);
  return trim(line);
}

std::string parent_directory(const std::string& path)
{
  const std::string::size_type end = path.find_last_not_of("/\\");
  if (end == std::string::npos)
    return std::string();
  const std::string::size_type separator = path.find_last_of("/\\", end);
  if (separator == std::string::npos)
    return std::string();
  if (separator == 0)
    return path.substr(0, 1);
  return path.substr(0, separator);
}

std::string join_path(const std::string& left, const std::string& right)
{
  if (left.empty())
    return right;
  if (left.back() == '/' || left.back() == '\\')
    return left + right;
  return left + "/" + right;
}

bool path_exists(const std::string& path)
{
  std::ifstream input(path);
  return input.good();
}

std::string resolve_git_directory(const std::string& worktree)
{
  const std::string entry = join_path(worktree, ".git");
  std::ifstream as_file(entry);
  if (!as_file.good())
    return std::string();
  std::string line;
  std::getline(as_file, line);
  line = trim(line);
  const std::string prefix = "gitdir:";
  if (line.compare(0, prefix.size(), prefix) != 0)
    return entry;
  std::string git_directory = trim(line.substr(prefix.size()));
  if (!git_directory.empty() && git_directory.front() != '/' && git_directory.front() != '\\')
    git_directory = join_path(worktree, git_directory);
  return git_directory;
}

std::string find_git_worktree(std::string root)
{
  if (root.empty())
    root = ".";
#ifdef GPUMD_SOURCE_ROOT
  if (root == ".")
    root = GPUMD_SOURCE_ROOT;
#endif
  for (int depth = 0; depth < 32; ++depth) {
    if (path_exists(join_path(root, ".git")))
      return root;
    const std::string parent = parent_directory(root);
    if (parent.empty() || parent == root)
      break;
    root = parent;
  }
  return std::string();
}

std::string read_git_commit(const std::string& worktree)
{
  const std::string git_path = resolve_git_directory(worktree);
  if (git_path.empty())
    return std::string();
  std::string head = read_first_line(join_path(git_path, "HEAD"));
  const std::string prefix = "ref:";
  if (head.compare(0, prefix.size(), prefix) != 0)
    return head;
  const std::string ref = trim(head.substr(prefix.size()));
  std::string commit = read_first_line(join_path(git_path, ref));
  if (commit.size() >= 40)
    return commit;
  std::ifstream packed(join_path(git_path, "packed-refs"));
  std::string line;
  while (std::getline(packed, line)) {
    if (line.size() > ref.size() + 1 &&
        line.compare(line.size() - ref.size(), ref.size(), ref) == 0)
      return trim(line.substr(0, line.size() - ref.size() - 1));
  }
  return std::string();
}

std::string shell_quote(const std::string& value)
{
  std::string result = "'";
  for (char c : value) {
    if (c == '\'')
      result += "'\\''";
    else
      result += c;
  }
  result += "'";
  return result;
}

std::string read_git_dirty(const std::string& worktree)
{
  if (worktree.empty())
    return std::string();
  const std::string command = "git -C " + shell_quote(worktree) +
                              " status --porcelain --untracked-files=all 2>/dev/null";
#ifdef _WIN32
  FILE* process = _popen(command.c_str(), "r");
#else
  FILE* process = popen(command.c_str(), "r");
#endif
  if (process == nullptr)
    return std::string();
  std::string line;
  char buffer[256];
  while (std::fgets(buffer, sizeof(buffer), process) != nullptr)
    line += buffer;
#ifdef _WIN32
  const int status = _pclose(process);
#else
  const int status = pclose(process);
#endif
  if (status != 0)
    return std::string();
  return trim(line).empty() ? "false" : "true";
}

std::string sha256_file(const std::string& path)
{
  std::ifstream input(path, std::ios::binary);
  if (!input.is_open())
    return std::string();
  SHA256 digest;
  std::array<unsigned char, 65536> buffer;
  while (input) {
    input.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
    const std::streamsize count = input.gcount();
    if (count > 0)
      digest.update(buffer.data(), static_cast<size_t>(count));
  }
  if (input.bad())
    return std::string();
  return digest.final_hex();
}

std::vector<std::string> find_potential_paths()
{
  std::vector<std::string> paths;
  std::ifstream input("run.in");
  std::string line;
  while (std::getline(input, line)) {
    const auto tokens = get_tokens(line);
    if (tokens.size() >= 2 && tokens[0] == "potential")
      paths.push_back(tokens[1]);
  }
  return paths;
}

std::string json_escape(const std::string& value)
{
  std::string result;
  for (unsigned char c : value) {
    switch (c) {
      case '"': result += "\\\""; break;
      case '\\': result += "\\\\"; break;
      case '\b': result += "\\b"; break;
      case '\f': result += "\\f"; break;
      case '\n': result += "\\n"; break;
      case '\r': result += "\\r"; break;
      case '\t': result += "\\t"; break;
      default:
        if (c < 0x20) {
          char buffer[8];
          std::snprintf(buffer, sizeof(buffer), "\\u%04x", c);
          result += buffer;
        } else {
          result += static_cast<char>(c);
        }
    }
  }
  return result;
}

struct MatrixDiagnostics
{
  bool finite = true;
  double max_abs = 0.0;
  double frobenius = 0.0;
  double asymmetry_max = 0.0;
  double asymmetry_frobenius = 0.0;
  double translation_max = 0.0;
  double asymmetry_max_relative = 0.0;
  double asymmetry_frobenius_relative = 0.0;
  double translation_relative = 0.0;
};

MatrixDiagnostics inspect_matrix(
  const std::vector<double>& matrix, int N, bool translation)
{
  MatrixDiagnostics result;
  const int N3 = 3 * N;
  if (matrix.size() != static_cast<size_t>(N3) * N3) {
    result.finite = false;
    return result;
  }
  double asymmetry_sum = 0.0;
  for (int row = 0; row < N3; ++row) {
    for (int column = 0; column < N3; ++column) {
      const double value = matrix[row * N3 + column];
      if (!std::isfinite(value))
        result.finite = false;
      const double square = value * value;
      result.frobenius += square;
      result.max_abs = std::max(result.max_abs, std::abs(value));
      const double difference =
        value - matrix[column * N3 + row];
      result.asymmetry_max = std::max(result.asymmetry_max, std::abs(difference));
      asymmetry_sum += difference * difference;
    }
  }
  result.frobenius = std::sqrt(result.frobenius);
  result.asymmetry_frobenius = std::sqrt(asymmetry_sum);
  if (translation) {
    for (int atom_i = 0; atom_i < N; ++atom_i) {
      for (int axis_i = 0; axis_i < 3; ++axis_i) {
        for (int axis_j = 0; axis_j < 3; ++axis_j) {
          double row_sum = 0.0;
          for (int atom_j = 0; atom_j < N; ++atom_j) {
            row_sum += matrix[(axis_i * N + atom_i) * N3 + axis_j * N + atom_j];
          }
          result.translation_max =
            std::max(result.translation_max, std::abs(row_sum));
        }
      }
    }
  }
  const double scale = std::max(result.max_abs, kTiny);
  result.asymmetry_max_relative = result.asymmetry_max / scale;
  result.asymmetry_frobenius_relative =
    result.asymmetry_frobenius / std::max(result.frobenius, kTiny);
  result.translation_relative = result.translation_max / scale;
  return result;
}

std::vector<double> symmetrize(const std::vector<double>& matrix)
{
  std::vector<double> result(matrix.size());
  const size_t dimension =
    static_cast<size_t>(std::sqrt(static_cast<double>(matrix.size())));
  for (size_t row = 0; row < dimension; ++row)
    for (size_t column = 0; column < dimension; ++column)
      result[row * dimension + column] =
        0.5 * (matrix[row * dimension + column] +
               matrix[column * dimension + row]);
  return result;
}

void write_matrix(
  const std::string& path, const std::vector<double>& matrix, int N,
  const std::string& result_kind)
{
  std::ofstream output(path);
  if (!output.is_open())
    PRINT_INPUT_ERROR(("Cannot open Hessian output file: " + path).c_str());
  output << std::scientific << std::setprecision(17);
  output << "# coordinate_order=soa\n";
  output << "# matrix_order=row_major\n";
  output << "# definition=minus_force_jacobian\n";
  output << "# unit=eV/A^2\n";
  output << "# N=" << N << "\n";
  output << "# result=" << result_kind << "\n";
  const int N3 = 3 * N;
  for (int row = 0; row < N3; ++row) {
    for (int column = 0; column < N3; ++column)
      output << (column ? " " : "") << matrix[row * N3 + column];
    output << "\n";
  }
}

void write_matrix_market(
  const std::string& path,
  const sparse_hessian::BlockPattern& pattern,
  const std::vector<double>& values)
{
  if (values.size() != pattern.value_count())
    PRINT_INPUT_ERROR("Sparse Hessian output has inconsistent dimensions.");
  size_t nonzero_count = 0;
  for (const double value : values)
    if (value != 0.0) ++nonzero_count;
  std::ofstream output(path);
  if (!output.is_open())
    PRINT_INPUT_ERROR(("Cannot open Hessian output file: " + path).c_str());
  const int dimension = 3 * pattern.atoms;
  output << "%%MatrixMarket matrix coordinate real general\n"
         << "% coordinate_order=soa\n"
         << "% definition=minus_force_jacobian\n"
         << "% unit=eV/A^2\n"
         << "% atom_block_size=3\n"
         << "% N=" << pattern.atoms << '\n'
         << dimension << ' ' << dimension << ' ' << nonzero_count << '\n'
         << std::scientific << std::setprecision(17);
  for (int atom = 0; atom < pattern.atoms; ++atom)
    for (int block = pattern.row_offsets[atom];
         block < pattern.row_offsets[atom + 1]; ++block) {
      const int other = pattern.columns[block];
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
          const double value = values[static_cast<size_t>(block) * 9 + a * 3 + b];
          if (value == 0.0) continue;
          output << a * pattern.atoms + atom + 1 << ' '
                 << b * pattern.atoms + other + 1 << ' ' << value << '\n';
        }
    }
  output.close();
  if (!output)
    PRINT_INPUT_ERROR("Failed to write sparse Hessian output.");
}

std::string nullable_string(const std::string& value)
{
  return value.empty() ? std::string("null") : ("\"" + json_escape(value) + "\"");
}

std::string nullable_sha256(const std::string& path)
{
  const std::string digest = sha256_file(path);
  return digest.empty() ? std::string("null") : ("\"" + digest + "\"");
}

void write_diagnostics(
  std::ofstream& output, const MatrixDiagnostics& diagnostics)
{
  output << "      {\n"
         << "        \"finite\": " << (diagnostics.finite ? "true" : "false") << ",\n"
         << "        \"max_abs\": " << diagnostics.max_abs << ",\n"
         << "        \"frobenius\": " << diagnostics.frobenius << ",\n"
         << "        \"asymmetry_max\": " << diagnostics.asymmetry_max << ",\n"
         << "        \"asymmetry_frobenius\": " << diagnostics.asymmetry_frobenius << ",\n"
         << "        \"asymmetry_max_relative\": " << diagnostics.asymmetry_max_relative << ",\n"
         << "        \"asymmetry_frobenius_relative\": "
         << diagnostics.asymmetry_frobenius_relative << ",\n"
         << "        \"translation_max\": " << diagnostics.translation_max << ",\n"
         << "        \"translation_relative\": " << diagnostics.translation_relative << "\n"
         << "      }";
}


struct OutputPaths
{
  std::string matrix;
  std::string raw;
};

void write_metadata(
  const std::string& path, Force& force, Box& box, Atom& atom,
  const std::vector<double>& raw, const std::vector<double>& sym,
  const std::string& method, double displacement,
  double potential_energy, const std::vector<double>& cartesian_force,
  const OutputPaths& outputs)
{
  const int N = atom.number_of_atoms;
  const MatrixDiagnostics raw_diagnostics = inspect_matrix(raw, N, true);
  const MatrixDiagnostics sym_diagnostics = inspect_matrix(sym, N, true);
  const std::vector<std::string> potential_paths = find_potential_paths();
  const std::string git_worktree = find_git_worktree(std::string());
  const std::string git_commit = read_git_commit(git_worktree);
  const std::string git_dirty = read_git_dirty(git_worktree);

  const int device = 0;
  gpuDeviceProp properties;
  gpuError_t device_error = gpuGetDeviceProperties(&properties, device);

  std::ofstream output(path);
  if (!output.is_open())
    PRINT_INPUT_ERROR(("Cannot open metadata output file: " + path).c_str());
  output << std::setprecision(17);

  output << "{\n";
  output << "  \"schema_version\": 3,\n";
  output << "  \"source\": {\n";
  output << "    \"git_commit\": " << nullable_string(git_commit) << ",\n";
  output << "    \"git_dirty\": "
         << (git_dirty.empty() ? "null" : git_dirty) << "\n";
  output << "  },\n";

  output << "  \"inputs\": {\n";
  output << "    \"structure_path\": \"model.xyz\",\n";
  output << "    \"structure_sha256\": " << nullable_sha256("model.xyz") << ",\n";
  output << "    \"potentials\": [";
  for (size_t i = 0; i < potential_paths.size(); ++i) {
    output << (i ? "," : "") << "\n      {\"path\": \""
           << json_escape(potential_paths[i]) << "\", \"sha256\": "
           << nullable_sha256(potential_paths[i]) << "}";
  }
  output << (potential_paths.empty() ? "" : "\n    ") << "],\n";
  output << "    \"number_of_potentials\": " << force.get_number_of_potentials()
         << "\n";
  output << "  },\n";

  output << "  \"device\": {\n";
  output << "    \"device_index\": " << device << ",\n";
  output << "    \"name\": "
         << (device_error == gpuSuccess
               ? "\"" + json_escape(properties.name) + "\""
               : "null")
         << ",\n";
  output << "    \"compute_capability\": ";
  if (device_error == gpuSuccess)
    output << "[" << properties.major << ", " << properties.minor << "]";
  else
    output << "null";
  output << "\n";
  output << "  },\n";

  output << "  \"system\": {\n";
  output << "    \"N\": " << N << ",\n";
  output << "    \"atom_order\": [";
  for (int i = 0; i < N; ++i) {
    if (i)
      output << ", ";
    output << "\"" << json_escape(atom.cpu_atom_symbol[i]) << "\"";
  }
  output << "],\n";
  output << "    \"types\": [";
  for (int i = 0; i < N; ++i) {
    if (i)
      output << ", ";
    output << atom.cpu_type[i];
  }
  output << "],\n";
  output << "    \"pbc\": [" << box.pbc_x << ", " << box.pbc_y << ", "
         << box.pbc_z << "],\n";
  output << "    \"cell_row_major\": [";
  for (int i = 0; i < 9; ++i) {
    if (i)
      output << ", ";
    output << box.cpu_h[i];
  }
  output << "],\n";
  output << "    \"volume\": " << box.get_volume() << "\n";
  output << "  },\n";

  output << "  \"matrix\": {\n";
  output << "    \"coordinate_order\": \"soa\",\n";
  output << "    \"matrix_order\": \"row_major\",\n";
  output << "    \"definition\": \"minus_force_jacobian\",\n";
  output << "    \"unit\": \"eV/A^2\",\n";
  output << "    \"method\": \"" << json_escape(method) << "\"\n";
  output << "  },\n";

  output << "  \"finite_difference\": {\n";
  output << "    \"method\": \"central\",\n";
  output << "    \"epsilon_A\": "
         << (method == "fd" ? displacement : 0.0) << "\n";
  output << "  },\n";

  output << "  \"analytic\": {\n";
  output << "    \"status\": "
         << (method == "analytic" ? "\"success\"" : "null") << ",\n";
  output << "    \"potential_energy_eV\": " << potential_energy << ",\n";
  output << "    \"force_soa\": [";
  for (size_t i = 0; i < cartesian_force.size(); ++i) {
    if (i)
      output << ", ";
    output << cartesian_force[i];
  }
  output << "]\n";
  output << "  },\n";

  output << "  \"diagnostics\": {\n";
  output << "    \"raw\":";
  output << "\n";
  write_diagnostics(output, raw_diagnostics);
  output << ",\n    \"symmetrized\":";
  output << "\n";
  write_diagnostics(output, sym_diagnostics);
  output << "\n  },\n";

  output << "  \"outputs\": {\n";
  output << "    \"matrix\": " << nullable_string(outputs.matrix) << ",\n";
  output << "    \"raw\": " << nullable_string(outputs.raw) << "\n";
  output << "  }\n";
  output << "}\n";
}

struct PhononSolver
{
  gpusolverDnHandle_t handle = nullptr;
  PhononSolver()
  {
    if (gpusolverDnCreate(&handle) != 0)
      throw std::runtime_error("cannot create phonon eigensolver");
  }
  ~PhononSolver() { gpusolverDnDestroy(handle); }
  void check(int status)
  {
    if (status != 0) throw std::runtime_error("phonon eigensolver API failed");
  }
};

void solve_phonon_matrix(
  const std::vector<double>& real, const std::vector<double>& imaginary,
  int dim, bool gamma, std::vector<double>& values, std::vector<double>& vectors)
{
  PhononSolver solver;
  GPU_Vector<double> w(dim);
  GPU_Vector<int> info(1);
  int lwork = 0;
  if (gamma) {
    GPU_Vector<double> a(real.size());
    a.copy_from_host(real.data());
    solver.check(gpusolverDnDsyevd_bufferSize(
      solver.handle, GPUSOLVER_EIG_MODE_VECTOR, GPUSOLVER_FILL_MODE_LOWER,
      dim, a.data(), dim, w.data(), &lwork));
    if (lwork <= 0) throw std::runtime_error("invalid phonon eigensolver workspace");
    GPU_Vector<double> work(lwork);
    solver.check(gpusolverDnDsyevd(
      solver.handle, GPUSOLVER_EIG_MODE_VECTOR, GPUSOLVER_FILL_MODE_LOWER,
      dim, a.data(), dim, w.data(), work.data(), lwork, info.data()));
    vectors.resize(real.size());
    a.copy_to_host(vectors.data());
  } else {
    std::vector<gpuDoubleComplex> host(real.size());
    for (size_t i = 0; i < host.size(); ++i) {
      host[i].x = real[i]; host[i].y = imaginary[i];
    }
    GPU_Vector<gpuDoubleComplex> a(host.size());
    a.copy_from_host(host.data());
    solver.check(gpusolverDnZheevd_bufferSize(
      solver.handle, GPUSOLVER_EIG_MODE_NOVECTOR, GPUSOLVER_FILL_MODE_LOWER,
      dim, a.data(), dim, w.data(), &lwork));
    if (lwork <= 0) throw std::runtime_error("invalid phonon eigensolver workspace");
    GPU_Vector<gpuDoubleComplex> work(lwork);
    solver.check(gpusolverDnZheevd(
      solver.handle, GPUSOLVER_EIG_MODE_NOVECTOR, GPUSOLVER_FILL_MODE_LOWER,
      dim, a.data(), dim, w.data(), work.data(), lwork, info.data()));
  }
  int result = -1;
  info.copy_to_host(&result);
  if (result != 0) throw std::runtime_error("phonon eigensolver did not converge");
  values.resize(dim);
  w.copy_to_host(values.data());
  for (double value : values)
    if (!std::isfinite(value)) throw std::runtime_error("nonfinite phonon eigenvalue");
  for (double value : vectors)
    if (!std::isfinite(value)) throw std::runtime_error("nonfinite phonon eigenvector");
}

double phonon_interaction_range(Force& force)
{
  if (force.get_number_of_potentials() <= 0)
    throw std::runtime_error("compute_hessian requires at least one potential");
  double cutoff = 0.0;
  for (int i = 0; i < force.get_number_of_potentials(); ++i)
    cutoff = std::max(cutoff, force.get_potential(i).rc);
  const double range = cutoff * 2.0;
  if (!std::isfinite(range) || range <= 0)
    throw std::runtime_error("cannot determine the phonon interaction range");
  return range;
}

void write_phonons(
  const std::vector<double>& hessian, const std::vector<double>& position,
  const std::vector<double>& mass, const analytic_phonon::Cell& cell,
  const analytic_phonon::Counts& counts, const analytic_phonon::KPath& path)
{
  const size_t basis = mass.size()/analytic_phonon::cell_count(counts);
  const int dim = static_cast<int>(3*basis);
  const auto blocks = analytic_phonon::fold(hessian, mass.size(), counts);
  const bool gamma = path.points.size() == 1 &&
    std::abs(path.points[0].fractional[0]) < 1e-12 &&
    std::abs(path.points[0].fractional[1]) < 1e-12 &&
    std::abs(path.points[0].fractional[2]) < 1e-12;
  std::ofstream dfile("D.out"), wfile("omega2.out");
  if (!dfile || !wfile) throw std::runtime_error("cannot open phonon output files");
  dfile << std::setprecision(17);
  wfile << std::setprecision(17) << "#";
  for (double tick : path.ticks) wfile << " " << tick;
  for (const auto& label : path.labels) wfile << " " << label;
  wfile << "\n# path_distance_A^-1 omega_squared_ps^-2; f_THz=sqrt(omega_squared)/(2*pi)\n";
  const double conversion = 1e6/(TIME_UNIT_CONVERSION*TIME_UNIT_CONVERSION);
  for (const auto& point : path.points) {
    std::vector<double> real, imaginary, values, vectors;
    analytic_phonon::dynamical_matrix(
      blocks, position, mass, cell, counts, point.fractional, real, imaginary);
    solve_phonon_matrix(real, imaginary, dim, gamma, values, vectors);
    // Write the original matrix, not the eigensolver's overwritten workspace.
    for (int row = 0; row < dim; ++row) {
      for (int col = 0; col < dim; ++col) dfile << real[row+col*dim] << " ";
      if (!gamma)
        for (int col = 0; col < dim; ++col) dfile << imaginary[row+col*dim] << " ";
      dfile << "\n";
    }
    wfile << point.distance;
    for (double value : values) wfile << " " << value*conversion;
    wfile << "\n";
    if (gamma) {
      std::ofstream eigen("eigenvector.out", std::ios::binary);
      if (!eigen) throw std::runtime_error("cannot open eigenvector.out");
      for (double value : values) {
        const float converted = static_cast<float>(value*conversion);
        if (!std::isfinite(converted)) throw std::runtime_error("eigenvalue exceeds float output range");
        eigen.write(reinterpret_cast<const char*>(&converted), sizeof(float));
      }
      for (int mode = 0; mode < dim; ++mode)
        for (int a = 0; a < 3; ++a)
          for (size_t b = 0; b < basis; ++b) {
            const float value = static_cast<float>(vectors[3*b+a+mode*dim]);
            eigen.write(reinterpret_cast<const char*>(&value), sizeof(float));
          }
      eigen.close();
      if (!eigen) throw std::runtime_error("failed to write eigenvector.out");
    }
  }
  dfile.close(); wfile.close();
  if (!dfile || !wfile) throw std::runtime_error("failed to write phonon output");
  printf("Hessian phonons: basis=%zu kpoints=%zu Gamma_eigenvectors=%s.\n",
    basis, path.points.size(), gamma ? "yes" : "no");
}


// Build the phonon Cell and KPath for a gamma or dispersion request. The
// returned request carries valid data only when phonon_mode != "none".
struct PhononRequest
{
  analytic_phonon::Cell cell{};
  analytic_phonon::KPath path;
};

PhononRequest build_phonon_request(
  const std::string& phonon_mode, const std::string& kpoints_file,
  int kpoint_intervals, const std::array<int, 3>& supercell,
  Force& force, Box& box, const std::vector<double>& position,
  const std::vector<double>& mass, const std::vector<int>& types,
  const std::vector<Group>& group)
{
  PhononRequest request;
  if (phonon_mode == "none") return request;
  std::copy(box.cpu_h, box.cpu_h + 9, request.cell.h.begin());
  std::copy(box.cpu_h + 9, box.cpu_h + 18, request.cell.inverse.begin());
  request.cell.periodic = {{box.pbc_x, box.pbc_y, box.pbc_z}};
  const size_t basis = analytic_phonon::validate_mapping(
    request.cell, supercell, position, mass, types);
  for (const auto& grouping : group)
    for (size_t i = basis; i < grouping.cpu_label.size(); ++i)
      if (grouping.cpu_label[i] != grouping.cpu_label[i % basis])
        throw std::runtime_error(
          "supercell group labels do not repeat the reference cell");
  if (phonon_mode == "gamma") {
    request.path.points.push_back({{{0, 0, 0}}, 0});
    request.path.ticks.push_back(0);
    request.path.labels.push_back("G");
  } else {
    request.path = analytic_phonon::read_path(
      kpoints_file, kpoint_intervals, request.cell, supercell);
    analytic_phonon::validate_wavevectors(
      request.path, request.cell, supercell, phonon_interaction_range(force));
  }
  return request;
}

void write_structure_snapshot(
  const std::string& path, Force& force, Box& box, Atom& atom,
  const std::vector<double>& position, const std::vector<double>& forces)
{
  double interaction_range = 0.0;
  try { interaction_range = phonon_interaction_range(force); }
  catch (const std::exception& error) { PRINT_INPUT_ERROR(error.what()); }
  const int N = atom.number_of_atoms;
  std::ofstream snapshot(path);
  if (!snapshot) PRINT_INPUT_ERROR("Cannot open Hessian structure_output.");
  snapshot << std::setprecision(17) << N << "\nLattice=\"";
  for (int a = 0; a < 3; ++a)
    for (int d = 0; d < 3; ++d)
      snapshot << ((a || d) ? " " : "") << box.cpu_h[3 * d + a];
  snapshot << "\" Properties=species:S:1:pos:R:3:mass:R:1:forces:R:3 pbc=\""
           << (box.pbc_x ? "T" : "F") << " " << (box.pbc_y ? "T" : "F") << " "
           << (box.pbc_z ? "T" : "F") << "\" hessian_interaction_range_A="
           << interaction_range << "\n";
  for (int i = 0; i < N; ++i) {
    snapshot << atom.cpu_atom_symbol[i];
    for (int d = 0; d < 3; ++d) snapshot << " " << position[i + d * N];
    snapshot << " " << atom.cpu_mass[i];
    for (int d = 0; d < 3; ++d) snapshot << " " << forces[i + d * N];
    snapshot << "\n";
  }
  snapshot.close();
  if (!snapshot) PRINT_INPUT_ERROR("Failed to write Hessian structure_output.");
}

// Restore the caller's complete force state on scope exit, including when a
// displacement evaluation fails. GPU_Vector owns each saved device buffer.
class ScopedAtomStateBackup
{
public:
  ScopedAtomStateBackup(
    Force& force, Box& box, Atom& atom, const std::vector<Group>& group)
    : force_(force), box_(box), atom_(atom), group_(group),
      position_(atom.position_per_atom.size()),
      force_data_(atom.force_per_atom.size()),
      potential_(atom.potential_per_atom.size()),
      virial_(atom.virial_per_atom.size())
  {
    position_.copy_from_device(atom.position_per_atom.data());
    force_data_.copy_from_device(atom.force_per_atom.data());
    potential_.copy_from_device(atom.potential_per_atom.data());
    virial_.copy_from_device(atom.virial_per_atom.data());
  }

  ~ScopedAtomStateBackup()
  {
    atom_.position_per_atom.copy_from_device(position_.data());
    atom_.force_per_atom.copy_from_device(force_data_.data());
    atom_.potential_per_atom.copy_from_device(potential_.data());
    atom_.virial_per_atom.copy_from_device(virial_.data());
    force_.compute(
      box_, atom_.position_per_atom, atom_.type, group_,
      atom_.potential_per_atom, atom_.force_per_atom, atom_.virial_per_atom);
    CHECK(gpuDeviceSynchronize());
    // Preserve the exact caller positions even if force.compute applies PBC.
    atom_.position_per_atom.copy_from_device(position_.data());
  }

private:
  Force& force_;
  Box& box_;
  Atom& atom_;
  const std::vector<Group>& group_;
  GPU_Vector<double> position_;
  GPU_Vector<double> force_data_;
  GPU_Vector<double> potential_;
  GPU_Vector<double> virial_;
};

// Central finite-difference Hessian in the SoA 3N x 3N layout. This works for
// any potential the force engine supports (unlike the analytic provider), and
// uses the same H_ij = -dF_i/dr_j convention.
std::vector<double> compute_finite_difference_hessian(
  Force& force, Box& box, Atom& atom, std::vector<Group>& group,
  const std::vector<double>& position, int N, double displacement,
  std::vector<double>& baseline_force, double& potential_energy)
{
  const int N3 = 3 * N;
  std::vector<double> hessian(static_cast<size_t>(N3) * N3, 0.0);
  ScopedAtomStateBackup atom_backup(force, box, atom, group);

  // Baseline geometry: refresh forces and record energy/forces for snapshots.
  atom.position_per_atom.copy_from_host(position.data());
  force.compute(
    box, atom.position_per_atom, atom.type, group, atom.potential_per_atom,
    atom.force_per_atom, atom.virial_per_atom);
  CHECK(gpuDeviceSynchronize());
  baseline_force.resize(N3);
  atom.force_per_atom.copy_to_host(baseline_force.data());
  std::vector<double> potential(N);
  atom.potential_per_atom.copy_to_host(potential.data());
  potential_energy = std::accumulate(potential.begin(), potential.end(), 0.0);

  std::vector<double> displacement_buffer(position);
  for (int column = 0; column < N3; ++column) {
    std::vector<double> force_plus(N3), force_minus(N3);
    displacement_buffer[column] = position[column] + displacement;
    atom.position_per_atom.copy_from_host(displacement_buffer.data());
    force.compute(
      box, atom.position_per_atom, atom.type, group, atom.potential_per_atom,
      atom.force_per_atom, atom.virial_per_atom);
    CHECK(gpuDeviceSynchronize());
    atom.force_per_atom.copy_to_host(force_plus.data());

    displacement_buffer[column] = position[column] - displacement;
    atom.position_per_atom.copy_from_host(displacement_buffer.data());
    force.compute(
      box, atom.position_per_atom, atom.type, group, atom.potential_per_atom,
      atom.force_per_atom, atom.virial_per_atom);
    CHECK(gpuDeviceSynchronize());
    atom.force_per_atom.copy_to_host(force_minus.data());

    displacement_buffer[column] = position[column];
    for (int row = 0; row < N3; ++row)
      hessian[static_cast<size_t>(row) * N3 + column] =
        -(force_plus[row] - force_minus[row]) / (2.0 * displacement);
  }

  return hessian;
}

} // namespace

void NEP_Hessian_Command::parse(const char** param, int num_param)
{
  if (num_param < 2 ||
      (std::strcmp(param[1], "analytic") != 0 &&
       std::strcmp(param[1], "fd") != 0)) {
    PRINT_INPUT_ERROR(
      "compute_hessian requires a method: 'analytic' or 'fd'.");
  }
  mode_ = param[1];
  const bool analytic_mode = (mode_ == "analytic");

  bool phonon_options = false, path_options = false;
  for (int i = 2; i + 1 < num_param; i += 2) {
    const std::string key = param[i];
    const std::string value = param[i + 1];
    if (key == "phonon") {
      if (value != "none" && value != "gamma" && value != "dispersion")
        PRINT_INPUT_ERROR("phonon must be none, gamma, or dispersion.");
      phonon_mode_ = value;
    } else if (key == "supercell") {
      phonon_options = true;
      try { supercell_ = analytic_phonon::parse_counts(value); }
      catch (const std::exception& error) { PRINT_INPUT_ERROR(error.what()); }
    } else if (key == "kpoints") {
      phonon_options = path_options = true;
      kpoints_file_ = value;
    } else if (key == "kpoint_intervals") {
      phonon_options = path_options = true;
      if (!is_valid_int(value.c_str(), &kpoint_intervals_) || kpoint_intervals_ < 1)
        PRINT_INPUT_ERROR("kpoint_intervals must be a positive integer.");
    } else if (key == "output") {
      output_ = value;
    } else if (key == "metadata") {
      metadata_output_ = value;
    } else if (key == "structure_output") {
      structure_output_ = value;
    } else if (key == "output_format") {
      if (!analytic_mode)
        PRINT_INPUT_ERROR("output_format is only available for 'compute_hessian analytic'.");
      if (value != "dense" && value != "sparse")
        PRINT_INPUT_ERROR("output_format must be dense or sparse.");
      output_format_ = value;
    } else if (key == "raw_output") {
      if (!analytic_mode)
        PRINT_INPUT_ERROR("raw_output is only available for 'compute_hessian analytic'.");
      raw_output_ = value;
    } else if (key == "displacement") {
      if (analytic_mode)
        PRINT_INPUT_ERROR("displacement is only available for 'compute_hessian fd'.");
      if (!is_valid_real(value.c_str(), &displacement_) ||
          !std::isfinite(displacement_) || displacement_ <= 0.0) {
        PRINT_INPUT_ERROR("Hessian displacement must be finite and positive.");
      }
    } else {
      PRINT_INPUT_ERROR(
        "Unknown compute_hessian parameter. Supported parameters are "
        "output, metadata, structure_output, phonon, supercell, kpoints, and "
        "kpoint_intervals (both modes), plus output_format and raw_output "
        "(analytic only) and displacement (fd only).");
    }
  }
  if (num_param % 2 != 0) {
    PRINT_INPUT_ERROR(
      "compute_hessian parameters must be given as key-value pairs.");
  }
  if (phonon_options && phonon_mode_ == "none")
    PRINT_INPUT_ERROR("Phonon options require phonon gamma or dispersion.");
  if (path_options && phonon_mode_ != "dispersion")
    PRINT_INPUT_ERROR("kpoints and kpoint_intervals require phonon dispersion.");
  if (output_format_ == "sparse" &&
      (!raw_output_.empty() || !metadata_output_.empty()))
    PRINT_INPUT_ERROR(
      "sparse output currently supports the analytic matrix and optional structure snapshot; raw_output and metadata are unavailable.");
  try {
    std::vector<std::string> outputs;
    for (const auto& path : {output_, raw_output_, metadata_output_, structure_output_})
      if (!path.empty()) outputs.push_back(path);
    if (phonon_mode_ != "none") {
      for (const std::string name : {"D.out", "omega2.out", "eigenvector.out"}) {
        for (const auto& path : outputs)
          if (same_file(path, name))
            throw std::runtime_error("Hessian outputs must not overwrite reserved phonon output files.");
      }
      outputs.insert(outputs.end(), {"D.out", "omega2.out", "eigenvector.out"});
    }
    std::vector<std::string> inputs = {"model.xyz", "run.in"};
    if (phonon_mode_ == "dispersion") inputs.push_back(kpoints_file_);
    std::ifstream run_input("run.in");
    std::string line;
    while (std::getline(run_input, line)) {
      const auto tokens = get_tokens(line);
      if (tokens.empty() || tokens[0] != "potential") continue;
      for (size_t i = 1; i < tokens.size(); ++i) {
        struct stat entry;
        if (stat(tokens[i].c_str(), &entry) == 0 && S_ISREG(entry.st_mode))
          inputs.push_back(tokens[i]);
      }
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
      canonical_output_path(outputs[i]);
      for (const auto& input : inputs)
        if (same_file(outputs[i], input))
          throw std::runtime_error("Hessian outputs (including structure_output) must differ from input paths: " + outputs[i]);
      for (size_t j = 0; j < i; ++j)
        if (same_file(outputs[i], outputs[j]))
          throw std::runtime_error("Hessian output paths (including structure_output) must differ: " + outputs[i]);
    }
  } catch (const std::exception& error) {
    PRINT_INPUT_ERROR(error.what());
  }
}

void NEP_Hessian_Command::compute(
  Force& force, Box& box, Atom& atom, std::vector<Group>& group)
{
  if (mode_ == "analytic") {
    compute_analytic(force, box, atom, group);
  } else {
    compute_fd(force, box, atom, group);
  }
}

void NEP_Hessian_Command::compute_analytic(
  Force& force, Box& box, Atom& atom, std::vector<Group>& group)
{
  if (output_format_ == "sparse") {
    const int sparse_atom_count = atom.number_of_atoms;
    if (sparse_atom_count <= 0 ||
        sparse_atom_count > std::numeric_limits<int>::max() / 3 ||
        atom.position_per_atom.size() != static_cast<size_t>(3 * sparse_atom_count))
      PRINT_INPUT_ERROR("Sparse compute_hessian requires a valid, consistent system.");
    sparse_hessian::BlockPattern pattern;
    std::vector<double> sparse_values;
    std::vector<double> analytic_force;
    std::vector<double> position(static_cast<size_t>(3) * sparse_atom_count);
    atom.position_per_atom.copy_to_host(position.data());
    for (const double coordinate : position)
      if (!std::isfinite(coordinate))
        PRINT_INPUT_ERROR("Sparse compute_hessian found a non-finite coordinate.");
    double potential_energy = 0.0;
    if (!compute_nep_analytic_hessian(
          force, box, atom, position, sparse_atom_count, sparse_values,
          analytic_force, potential_energy, nullptr, &group, &pattern)) {
      PRINT_INPUT_ERROR(
        "Sparse analytic Hessian is unsupported for this potential or configuration.");
    }
    try {
      pattern.symmetrize(sparse_values);
    } catch (const std::exception& error) {
      PRINT_INPUT_ERROR(error.what());
    }
    if (phonon_mode_ != "none") {
      try {
        std::vector<double> dense = pattern.dense_soa(sparse_values);
        const auto request = build_phonon_request(
          phonon_mode_, kpoints_file_, kpoint_intervals_, supercell_, force,
          box, position, atom.cpu_mass, atom.cpu_type, group);
        write_phonons(
          dense, position, atom.cpu_mass, request.cell, supercell_, request.path);
      } catch (const std::exception& error) {
        PRINT_INPUT_ERROR(error.what());
      }
    }
    write_matrix_market(output_, pattern, sparse_values);
    if (!structure_output_.empty())
      write_structure_snapshot(
        structure_output_, force, box, atom, position, analytic_force);
    printf("Sparse analytic Hessian: N=%d blocks=%zu scalar_values=%zu\n",
      sparse_atom_count, pattern.block_count(), sparse_values.size());
    return;
  }

  const int N = atom.number_of_atoms;
  if (N <= 0 || N > std::numeric_limits<int>::max() / 3) {
    PRINT_INPUT_ERROR("compute_hessian requires a valid positive system.");
  }
  const int N3 = 3 * N;
  if (atom.position_per_atom.size() != static_cast<size_t>(N3)) {
    PRINT_INPUT_ERROR("compute_hessian has inconsistent Atom positions.");
  }

  std::vector<double> position(N3);
  atom.position_per_atom.copy_to_host(position.data());
  for (double value : position) {
    if (!std::isfinite(value)) {
      PRINT_INPUT_ERROR("compute_hessian found a non-finite coordinate.");
    }
  }

  PhononRequest request;
  if (phonon_mode_ != "none") {
    try {
      request = build_phonon_request(
        phonon_mode_, kpoints_file_, kpoint_intervals_, supercell_, force, box,
        position, atom.cpu_mass, atom.cpu_type, group);
    } catch (const std::exception& error) {
      PRINT_INPUT_ERROR(error.what());
    }
  }

  std::vector<double> analytic_raw;
  std::vector<double> analytic_force;
  double potential_energy = 0.0;
  if (!compute_nep_analytic_hessian(
        force, box, atom, position, N, analytic_raw, analytic_force,
        potential_energy, nullptr, &group)) {
    PRINT_INPUT_ERROR(
      "compute_hessian analytic is not supported for this potential/model or "
      "configuration. Use 'compute_hessian fd' for a finite-difference "
      "Hessian.");
  }

  const std::vector<double> symmetrized = symmetrize(analytic_raw);
  if (phonon_mode_ != "none") {
    try {
      write_phonons(
        symmetrized, position, atom.cpu_mass, request.cell, supercell_,
        request.path);
    } catch (const std::exception& error) {
      PRINT_INPUT_ERROR(error.what());
    }
  }
  if (!raw_output_.empty())
    write_matrix(raw_output_, analytic_raw, N, "analytic_raw");
  write_matrix(output_, symmetrized, N, "analytic_symmetrized");
  if (!structure_output_.empty())
    write_structure_snapshot(
      structure_output_, force, box, atom, position, analytic_force);

  OutputPaths paths;
  paths.matrix = output_;
  paths.raw = raw_output_;
  if (!metadata_output_.empty()) {
    write_metadata(
      metadata_output_, force, box, atom, analytic_raw, symmetrized,
      "analytic", displacement_, potential_energy, analytic_force, paths);
  }

  printf(
    "Hessian command: method=analytic N=%d raw_asymmetry_max=%.12g "
    "raw_translation_max=%.12g.\n",
    N, inspect_matrix(analytic_raw, N, true).asymmetry_max,
    inspect_matrix(analytic_raw, N, true).translation_max);
}

void NEP_Hessian_Command::compute_fd(
  Force& force, Box& box, Atom& atom, std::vector<Group>& group)
{
  const int N = atom.number_of_atoms;
  if (N <= 0 || N > std::numeric_limits<int>::max() / 3) {
    PRINT_INPUT_ERROR("compute_hessian requires a valid positive system.");
  }
  const int N3 = 3 * N;
  if (atom.position_per_atom.size() != static_cast<size_t>(N3)) {
    PRINT_INPUT_ERROR("compute_hessian has inconsistent Atom positions.");
  }

  std::vector<double> position(N3);
  atom.position_per_atom.copy_to_host(position.data());
  for (double value : position) {
    if (!std::isfinite(value)) {
      PRINT_INPUT_ERROR("compute_hessian found a non-finite coordinate.");
    }
  }

  PhononRequest request;
  if (phonon_mode_ != "none") {
    try {
      request = build_phonon_request(
        phonon_mode_, kpoints_file_, kpoint_intervals_, supercell_, force, box,
        position, atom.cpu_mass, atom.cpu_type, group);
    } catch (const std::exception& error) {
      PRINT_INPUT_ERROR(error.what());
    }
  }

  std::vector<double> baseline_force;
  double potential_energy = 0.0;
  std::vector<double> fd_raw;
  try {
    fd_raw = compute_finite_difference_hessian(
      force, box, atom, group, position, N, displacement_, baseline_force,
      potential_energy);
  } catch (const std::exception& error) {
    PRINT_INPUT_ERROR(error.what());
  }

  const MatrixDiagnostics fd_check = inspect_matrix(fd_raw, N, false);
  if (!fd_check.finite) {
    PRINT_INPUT_ERROR("compute_hessian fd produced a non-finite Hessian.");
  }

  const std::vector<double> symmetrized = symmetrize(fd_raw);
  if (phonon_mode_ != "none") {
    try {
      write_phonons(
        symmetrized, position, atom.cpu_mass, request.cell, supercell_,
        request.path);
    } catch (const std::exception& error) {
      PRINT_INPUT_ERROR(error.what());
    }
  }
  write_matrix(output_, symmetrized, N, "finite_difference_symmetrized");
  if (!structure_output_.empty())
    write_structure_snapshot(
      structure_output_, force, box, atom, position, baseline_force);

  OutputPaths paths;
  paths.matrix = output_;
  if (!metadata_output_.empty()) {
    write_metadata(
      metadata_output_, force, box, atom, fd_raw, symmetrized, "fd",
      displacement_, potential_energy, baseline_force, paths);
  }

  printf(
    "Hessian command: method=fd N=%d displacement=%.17g "
    "raw_asymmetry_max=%.12g raw_translation_max=%.12g.\n",
    N, displacement_, inspect_matrix(fd_raw, N, true).asymmetry_max,
    inspect_matrix(fd_raw, N, true).translation_max);
}
