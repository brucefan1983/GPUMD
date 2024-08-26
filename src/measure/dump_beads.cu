/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

/*-----------------------------------------------------------------------------------------------100
Dump bead data in PIMD-related run
--------------------------------------------------------------------------------------------------*/

#include "dump_beads.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/read_file.cuh"

void Dump_Beads::parse(const char** param, int num_param)
{
  dump_ = true;
  printf("Dump data for beads in PIMD-related runs.\n");

  if (num_param != 4) {
    PRINT_INPUT_ERROR("dump_beads should have 3 parameters.\n");
  }

  if (!is_valid_int(param[1], &dump_interval_)) {
    PRINT_INPUT_ERROR("dump interval should be an integer.");
  }
  if (dump_interval_ <= 0) {
    PRINT_INPUT_ERROR("dump interval should > 0.");
  }

  printf("    every %d steps.\n", dump_interval_);

  if (!is_valid_int(param[2], &has_velocity_)) {
    PRINT_INPUT_ERROR("has_velocity should be an integer.");
  }
  if (has_velocity_ == 0) {
    printf("    without velocity data.\n");
  } else {
    printf("    with velocity data.\n");
  }

  if (!is_valid_int(param[3], &has_force_)) {
    PRINT_INPUT_ERROR("has_force should be an integer.");
  }
  if (has_force_ == 0) {
    printf("    without force data.\n");
  } else {
    printf("    with force data.\n");
  }
}

void Dump_Beads::preprocess(const int number_of_atoms, const int number_of_beads)
{
  if (dump_) {
    if (number_of_beads == 0) {
      PRINT_INPUT_ERROR("Cannot use dump_beads for non-PIMD-related runs.");
    }
    number_of_beads_ = number_of_beads;
    fid_.resize(number_of_beads_);
    for (int k = 0; k < number_of_beads_; ++k) {
      std::string filename = "beads_dump_" + std::to_string(k) + ".xyz";
      fid_[k] = my_fopen(filename.c_str(), "a");
    }
    cpu_position_.resize(number_of_atoms * 3);
    if (has_velocity_) {
      cpu_velocity_.resize(number_of_atoms * 3);
    }
    if (has_force_) {
      cpu_force_.resize(number_of_atoms * 3);
    }
  }
}

void Dump_Beads::output_line2(FILE* fid, const double time, const Box& box)
{
  // time
  fprintf(fid, "Time=%.8f", time * TIME_UNIT_CONVERSION); // output time is in units of fs

  // PBC
  fprintf(
    fid, " pbc=\"%c %c %c\"", box.pbc_x ? 'T' : 'F', box.pbc_y ? 'T' : 'F', box.pbc_z ? 'T' : 'F');

  // box
  if (box.triclinic == 0) {
    fprintf(
      fid,
      " Lattice=\"%.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f\"",
      box.cpu_h[0],
      0.0,
      0.0,
      0.0,
      box.cpu_h[1],
      0.0,
      0.0,
      0.0,
      box.cpu_h[2]);
  } else {
    fprintf(
      fid,
      " Lattice=\"%.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f\"",
      box.cpu_h[0],
      box.cpu_h[3],
      box.cpu_h[6],
      box.cpu_h[1],
      box.cpu_h[4],
      box.cpu_h[7],
      box.cpu_h[2],
      box.cpu_h[5],
      box.cpu_h[8]);
  }

  // Properties
  fprintf(fid, " Properties=species:S:1:pos:R:3");

  if (has_velocity_) {
    fprintf(fid, ":vel:R:3");
  }
  if (has_force_) {
    fprintf(fid, ":forces:R:3");
  }

  // Over
  fprintf(fid, "\n");
}

void Dump_Beads::process(const int step, const double global_time, const Box& box, Atom& atom)
{
  if (!dump_)
    return;
  if ((step + 1) % dump_interval_ != 0)
    return;

  const int num_atoms_total = atom.position_per_atom.size() / 3;

  for (int k = 0; k < atom.number_of_beads; ++k) {

    atom.position_beads[k].copy_to_host(cpu_position_.data());
    if (has_velocity_) {
      atom.velocity_beads[k].copy_to_host(cpu_velocity_.data());
    }
    if (has_force_) {
      atom.force_beads[k].copy_to_host(cpu_force_.data());
    }

    // line 1
    fprintf(fid_[k], "%d\n", num_atoms_total);

    // line 2
    output_line2(fid_[k], global_time, box);

    // other lines
    for (int n = 0; n < num_atoms_total; n++) {
      fprintf(fid_[k], "%s", atom.cpu_atom_symbol[n].c_str());
      for (int d = 0; d < 3; ++d) {
        fprintf(fid_[k], " %.8f", cpu_position_[n + num_atoms_total * d]);
      }
      if (has_velocity_) {
        const double natural_to_A_per_fs = 1.0 / TIME_UNIT_CONVERSION;
        for (int d = 0; d < 3; ++d) {
          fprintf(fid_[k], " %.8f", cpu_velocity_[n + num_atoms_total * d] * natural_to_A_per_fs);
        }
      }
      if (has_force_) {
        for (int d = 0; d < 3; ++d) {
          fprintf(fid_[k], " %.8f", cpu_force_[n + num_atoms_total * d]);
        }
      }
      fprintf(fid_[k], "\n");
    }

    fflush(fid_[k]);
  }
}

void Dump_Beads::postprocess()
{
  if (dump_) {
    for (int k = 0; k < number_of_beads_; ++k) {
      fclose(fid_[k]);
    }
    dump_ = false;
  }
}
