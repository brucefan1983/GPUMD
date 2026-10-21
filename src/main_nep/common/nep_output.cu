/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/

#include "nep_output.cuh"
#include "nep_training.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"

void write_nep_structure_predictions(
  bool is_stress,
  int num_components,
  FILE* fid,
  float* prediction,
  float* reference,
  Dataset& dataset,
  const int nc)
{
  for (int n = 0; n < num_components; ++n) {
    int offset = n * dataset.N + dataset.Na_sum_cpu[nc];
    float data_nc = 0.0f;
    for (int m = 0; m < dataset.Na_cpu[nc]; ++m) {
      data_nc += prediction[offset + m];
    }
    if (!is_stress) {
      fprintf(fid, "%g ", data_nc / dataset.Na_cpu[nc]);
    } else {
      fprintf(fid, "%g ", data_nc / dataset.structures[nc].volume * PRESSURE_UNIT_CONVERSION);
    }
  }
  for (int n = 0; n < num_components; ++n) {
    float ref_value = reference[n * dataset.Nc + nc];
    if (is_stress) {
      if (ref_value > -1e5) {
        ref_value *= dataset.Na_cpu[nc] / dataset.structures[nc].volume * PRESSURE_UNIT_CONVERSION;
      }
    }
    if (n == num_components - 1) {
      fprintf(fid, "%g\n", ref_value);
    } else {
      fprintf(fid, "%g ", ref_value);
    }
  }
}

void write_nep_atomic_predictions(
  int num_components,
  FILE* fid,
  float* prediction,
  float* reference,
  Dataset& dataset,
  const int nc)
{
  int offset = dataset.Na_sum_cpu[nc];
  for (int m = 0; m < dataset.structures[nc].num_atom; ++m) {
    for (int n = 0; n < num_components; ++n) {
      int index = n * dataset.N + offset + m;
      fprintf(fid, "%g ", prediction[index]);
    }
    for (int n = 0; n < num_components; ++n) {
      float ref_value = reference[n * dataset.N + offset + m];
      if (n == num_components - 1) {
        fprintf(fid, "%g\n", ref_value);
      } else {
        fprintf(fid, "%g ", ref_value);
      }
    }
  }
}

static void update_energy_force_virial(
  FILE* fid_energy,
  FILE* fid_force,
  FILE* fid_virial,
  FILE* fid_stress,
  Dataset& dataset,
  const int nc)
{
  int offset = dataset.Na_sum_cpu[nc];
  for (int m = 0; m < dataset.structures[nc].num_atom; ++m) {
    int n = offset + m;
    fprintf(
      fid_force,
      "%g %g %g %g %g %g\n",
      dataset.force_cpu[n],
      dataset.force_cpu[n + dataset.N],
      dataset.force_cpu[n + dataset.N * 2],
      dataset.force_ref_cpu[n],
      dataset.force_ref_cpu[n + dataset.N],
      dataset.force_ref_cpu[n + dataset.N * 2]);
  }

  double energy = 0.0;
  for (int m = 0; m < dataset.Na_cpu[nc]; ++m) {
    energy += dataset.energy_cpu[offset + m];
  }
  fprintf(fid_energy, "%.17g %.17g\n", energy / dataset.Na_cpu[nc], dataset.energy_ref_cpu[nc]);

  write_nep_structure_predictions(
    false, 6, fid_virial, dataset.virial_cpu.data(), dataset.virial_ref_cpu.data(), dataset, nc);
  write_nep_structure_predictions(
    true, 6, fid_stress, dataset.virial_cpu.data(), dataset.virial_ref_cpu.data(), dataset, nc);
}

void print_nep_energy_header(bool has_ediff)
{
  printf(
    "%-8s %-11s %-11s %-11s %-13s %-13s %-13s %-13s %-13s %-13s",
    "Step", "Total-Loss", "L1Reg-Loss", "L2Reg-Loss",
    "RMSE-E-Train", "RMSE-F-Train", "RMSE-V-Train", "RMSE-E-Test", "RMSE-F-Test", "RMSE-V-Test");
  if (has_ediff) {
    printf(" %-13s %-13s", "RMSE-D-Train", "RMSE-D-Test");
  }
  printf("\n");
}

void print_nep_charge_header(bool has_ediff)
{
  printf(
    "%-8s %-9s %-9s %-9s %-9s %-9s %-9s %-9s %-9s %-9s %-9s %-9s %-9s %-9s",
    "Step", "Total", "L1Reg", "L2Reg", "E-Train", "F-Train", "V-Train", "Q-Train", "Z-Train",
    "E-Test", "F-Test", "V-Test", "Q-Test", "Z-Test");
  if (has_ediff) {
    printf(" %-9s %-9s", "D-Train", "D-Test");
  }
  printf("\n");
}

void write_nep_energy_error(FILE* file, const TrainingReport& report,
  const NEP_Energy_Error& train, const NEP_Energy_Error& test, bool has_ediff)
{
  fprintf(file,
    "%-8d %-11.5f %-11.5f %-11.5f %-13.5f %-13.5f %-13.5f %-13.5f %-13.5f %-13.5f",
    report.generation + 1, report.total, report.l1, report.l2,
    train.energy, train.force, train.virial, test.energy, test.force, test.virial);
  if (has_ediff) {
    fprintf(file, " %-13.5f %-13.5f", train.ediff, test.ediff);
  }
  fprintf(file, "\n");
  fflush(file);
}

void write_nep_charge_error(FILE* file, const TrainingReport& report,
  const NEP_Energy_Error& train, const NEP_Energy_Error& test,
  float charge_train, float bec_train, float charge_test, float bec_test, bool has_ediff)
{
  fprintf(file,
    "%-8d %-9.5f %-9.5f %-9.5f %-9.5f %-9.5f %-9.5f %-9.5f %-9.5f %-9.5f %-9.5f %-9.5f "
    "%-9.5f %-9.5f",
    report.generation + 1, report.total, report.l1, report.l2,
    train.energy, train.force, train.virial, charge_train, bec_train,
    test.energy, test.force, test.virial, charge_test, bec_test);
  if (has_ediff) {
    fprintf(file, " %-9.5f %-9.5f", train.ediff, test.ediff);
  }
  fprintf(file, "\n");
  fflush(file);
}

void copy_nep_energy_predictions(Dataset& data)
{
  data.energy.copy_to_host(data.energy_cpu.data());
  data.virial.copy_to_host(data.virial_cpu.data());
  data.force.copy_to_host(data.force_cpu.data());
}

static FILE* open_prediction_file(const char* quantity, const std::string& label)
{
  return my_fopen((std::string(quantity) + "_" + label + ".out").c_str(), "w");
}

void write_nep_energy_predictions(
  const std::string& label, const std::vector<std::pair<Dataset*, int>>& structures)
{
  FILE* fid_force = open_prediction_file("force", label);
  FILE* fid_energy = open_prediction_file("energy", label);
  FILE* fid_virial = open_prediction_file("virial", label);
  FILE* fid_stress = open_prediction_file("stress", label);
  for (const auto& structure : structures) {
    update_energy_force_virial(
      fid_energy, fid_force, fid_virial, fid_stress, *structure.first, structure.second);
  }
  fclose(fid_energy);
  fclose(fid_force);
  fclose(fid_virial);
  fclose(fid_stress);
}
