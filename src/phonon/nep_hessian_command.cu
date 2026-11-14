/*
    Copyright 2017 Zheyong Fan and GPUMD development team.
    This file is part of GPUMD, licensed under GNU GPL version 3 or later.
*/
#include "nep_hessian_command.cuh"
#include "nep_analytic_hessian.cuh"
#include "nep_hessian_model.cuh"
#include "force/force.cuh"
#include "model/atom.cuh"
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sys/stat.h>
#include <unistd.h>

namespace {

std::string canonical_output_path(const std::string& path, int depth = 0)
{
  if (depth > 40) PRINT_INPUT_ERROR("Too many output path symlinks.");
  char resolved[PATH_MAX];
  if (realpath(path.c_str(), resolved)) return resolved;
  struct stat entry;
  if (lstat(path.c_str(), &entry) == 0) {
    if (!S_ISLNK(entry.st_mode))
      PRINT_INPUT_ERROR(("Cannot resolve Hessian path: " + path).c_str());
    char target[PATH_MAX];
    const ssize_t size = readlink(path.c_str(), target, sizeof(target));
    if (size <= 0 || size >= static_cast<ssize_t>(sizeof(target)))
      PRINT_INPUT_ERROR(("Cannot resolve Hessian symlink: " + path).c_str());
    const std::string link(target, size);
    const auto slash = path.find_last_of('/');
    const std::string parent = slash == std::string::npos ? "." : path.substr(0, slash);
    return canonical_output_path(link[0] == '/' ? link : parent + "/" + link, depth + 1);
  }
  if (errno != ENOENT) PRINT_INPUT_ERROR(("Cannot inspect Hessian path: " + path).c_str());
  const auto slash = path.find_last_of('/');
  const std::string parent = slash == std::string::npos ? "." :
    (slash == 0 ? "/" : path.substr(0, slash));
  const std::string name = slash == std::string::npos ? path : path.substr(slash + 1);
  if (name.empty() || !realpath(parent.c_str(), resolved))
    PRINT_INPUT_ERROR(("Cannot resolve Hessian output parent: " + path).c_str());
  return std::string(resolved) + "/" + name;
}

bool same_file(const std::string& left, const std::string& right)
{
  if (canonical_output_path(left) == canonical_output_path(right)) return true;
  struct stat a, b;
  return stat(left.c_str(), &a) == 0 && stat(right.c_str(), &b) == 0 &&
    a.st_dev == b.st_dev && a.st_ino == b.st_ino;
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
  output.close();
  if (!output)
    PRINT_INPUT_ERROR(("Failed to write Hessian output file: " + path).c_str());
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

void write_diagnostics(
  std::ofstream& output, const nep_hessian::Diagnostics& diagnostics)
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
  const std::string& path, Box& box, Atom& atom,
  const nep_hessian::Diagnostics& raw_diagnostics,
  const nep_hessian::Diagnostics& sym_diagnostics,
  const std::string& method,
  double potential_energy, const std::vector<double>& cartesian_force,
  const OutputPaths& outputs)
{
  const int N = atom.number_of_atoms;
  std::ofstream output(path);
  if (!output.is_open())
    PRINT_INPUT_ERROR(("Cannot open metadata output file: " + path).c_str());
  output << std::setprecision(17);

  output << "{\n";
  output << "  \"schema_version\": 4,\n";
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
  output.close();
  if (!output) PRINT_INPUT_ERROR("Failed to write Hessian metadata.");
}

void write_structure_snapshot(
  const std::string& path, Force& force, Box& box, Atom& atom,
  const std::vector<double>& position, const std::vector<double>& forces)
{
  const double interaction_range = 2.0 * force.get_potential(0).rc;
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

} // namespace

void NEP_Hessian_Command::parse(const std::vector<std::string>& tokens)
{
  const int num_param = static_cast<int>(tokens.size());
  if (num_param < 2 || tokens[1] != "analytic") {
    PRINT_INPUT_ERROR(
      "compute_hessian requires the method 'analytic'. Use 'compute_phonon' "
      "for a finite-difference Hessian.");
  }

  for (int i = 2; i + 1 < num_param; i += 2) {
    const std::string key = tokens[i];
    const std::string value = tokens[i + 1];
    if (key == "output") {
      output_ = value;
    } else if (key == "metadata") {
      metadata_output_ = value;
    } else if (key == "structure_output") {
      structure_output_ = value;
    } else if (key == "output_format") {
      if (value != "dense" && value != "sparse")
        PRINT_INPUT_ERROR("output_format must be dense or sparse.");
      output_format_ = value;
    } else if (key == "raw_output") {
      raw_output_ = value;
    } else {
      PRINT_INPUT_ERROR(
        "Unknown compute_hessian parameter. Supported parameters are "
        "output, metadata, structure_output, output_format, and raw_output.");
    }
  }
  if (num_param % 2 != 0) {
    PRINT_INPUT_ERROR(
      "compute_hessian parameters must be given as key-value pairs.");
  }
  if (output_format_ == "sparse" &&
      (!raw_output_.empty() || !metadata_output_.empty()))
    PRINT_INPUT_ERROR(
      "sparse output currently supports the analytic matrix and optional structure snapshot; raw_output and metadata are unavailable.");
  {
    std::vector<std::string> outputs;
    for (const auto& path : {output_, raw_output_, metadata_output_, structure_output_})
      if (!path.empty()) outputs.push_back(path);
    std::vector<std::string> inputs = {"model.xyz", "run.in"};
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
          PRINT_INPUT_ERROR(("Hessian outputs (including structure_output) must differ from input paths: " + outputs[i]).c_str());
      for (size_t j = 0; j < i; ++j)
        if (same_file(outputs[i], outputs[j]))
          PRINT_INPUT_ERROR(("Hessian output paths (including structure_output) must differ: " + outputs[i]).c_str());
    }
  }
}

void NEP_Hessian_Command::compute(
  Force& force, Box& box, Atom& atom, const std::string& potential_filename)
{
  if (force.get_number_of_potentials() != 1)
    PRINT_INPUT_ERROR("compute_hessian requires exactly one potential.");
  if (potential_filename.empty())
    PRINT_INPUT_ERROR("compute_hessian could not locate the potential file.");

  NEP_Hessian_Model model;
  model.initialize(potential_filename.c_str());
  const int N = atom.number_of_atoms;
  if (N <= 0 || N > std::numeric_limits<int>::max() / 3 ||
      atom.position_per_atom.size() != static_cast<size_t>(3) * N)
    PRINT_INPUT_ERROR("compute_hessian requires a valid, consistent system.");
  std::vector<double> position(static_cast<size_t>(3) * N);
  atom.position_per_atom.copy_to_host(position.data());
  for (double value : position)
    if (!std::isfinite(value))
      PRINT_INPUT_ERROR("compute_hessian found a non-finite coordinate.");

  sparse_hessian::BlockPattern pattern;
  const bool sparse = output_format_ == "sparse";
  std::vector<double> matrix, analytic_force;
  nep_hessian::Diagnostics raw_diagnostics;
  double potential_energy = 0.0;
  if (!compute_nep_analytic_hessian(
        force, model, box, atom, position, N, matrix, analytic_force,
        potential_energy, raw_diagnostics, sparse ? &pattern : nullptr)) {
    PRINT_INPUT_ERROR("Analytic Hessian is unsupported or failed validation.");
  }
  if (sparse) {
    pattern.symmetrize(matrix);
    write_matrix_market(output_, pattern, matrix);
  } else {
    if (!raw_output_.empty()) write_matrix(raw_output_, matrix, N, "analytic_raw");
    // Preserve raw diagnostics and any requested raw file, then reuse storage.
    const size_t dimension = static_cast<size_t>(3) * N;
    for (size_t row = 0; row < dimension; ++row)
      for (size_t column = row + 1; column < dimension; ++column) {
        const double average = 0.5 *
          (matrix[row * dimension + column] + matrix[column * dimension + row]);
        matrix[row * dimension + column] = average;
        matrix[column * dimension + row] = average;
      }
    write_matrix(output_, matrix, N, "analytic_symmetrized");
  }
  if (!structure_output_.empty())
    write_structure_snapshot(structure_output_, force, box, atom, position, analytic_force);
  if (!metadata_output_.empty()) {
    const auto sym_diagnostics = nep_hessian::inspect(matrix, N);
    write_metadata(metadata_output_, box, atom, raw_diagnostics, sym_diagnostics,
      "analytic", potential_energy, analytic_force, {output_, raw_output_});
  }
  printf("Analytic Hessian: N=%d format=%s raw_asymmetry_max=%.12g "
         "raw_translation_max=%.12g.\n", N, output_format_.c_str(),
         raw_diagnostics.asymmetry_max, raw_diagnostics.translation_max);
}
