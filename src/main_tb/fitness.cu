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

#include "fitness.cuh"
#include "parameters.cuh"
#include "utilities/error.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <string>

Fitness::Data::Data(const Parameters& para, const DatasetCollection& data) : dataset_(data)
{
  models_.reserve(data.size());
  eigenvalues_.resize(data.size());
  for (int s = 0; s < data.size(); ++s) {
    models_.emplace_back(new TB(para, data[s]));
  }
}

Fitness::Fitness(
  const Parameters& para, const DatasetCollection& dataset, const DatasetCollection* test)
  : para_(para), train_(para, dataset), batch_(para, dataset)
{
  num_reference_bands_ = para_.band_max - para_.band_min;
  active_num_reference_bands_ = num_reference_bands_;
  active_band_weight_sum_ = para_.band_weight_sum;

  if (test && test->size()) {
    test_.reset(new Data(para, *test));
  }
  if (!para_.prediction) {
    initialize_global_scaler();
  }
}

void Fitness::initialize_parameters(std::vector<double>& parameters) const
{
  TB::initialize_parameters(para_, parameters);
}

void Fitness::initialize_global_scaler()
{
  std::vector<float> global_max_abs(para_.dim, 0.0f);
  for (int s = 0; s < train_.dataset_.size(); ++s) {
    std::vector<float> local_max_abs;
    train_.models_[s]->get_reference_descriptor_max_abs(local_max_abs);
    if (static_cast<int>(local_max_abs.size()) != para_.dim) {
      PRINT_INPUT_ERROR("Reference descriptor scaler has the wrong dimension.");
    }
    for (int d = 0; d < para_.dim; ++d) {
      global_max_abs[d] = std::max(global_max_abs[d], local_max_abs[d]);
    }
  }

  std::vector<float> global_q_scaler(para_.dim, 1.0f);
  for (int d = 0; d < para_.dim; ++d) {
    const double max_abs = global_max_abs[d];
    if (max_abs > 1.0e-12) {
      global_q_scaler[d] = static_cast<float>(1.0 / max_abs);
    }
  }
  set_q_scaler(global_q_scaler);
}

void Fitness::set_q_scaler(const std::vector<float>& scaler)
{
  for (auto& model : train_.models_) {
    model->set_q_scaler(scaler);
  }
  if (test_) {
    for (auto& model : test_->models_) {
      model->set_q_scaler(scaler);
    }
  }
}

void Fitness::set_active_band_count(const int count)
{
  if (count <= 0 || count > num_reference_bands_) {
    PRINT_INPUT_ERROR("Active TB band count is outside the selected band range.");
  }
  active_num_reference_bands_ = count;
  active_band_weight_sum_ = 0.0;
  for (int n = 0; n < active_num_reference_bands_; ++n) {
    active_band_weight_sum_ += para_.band_weights[n];
  }
  if (!(active_band_weight_sum_ > 0.0)) {
    PRINT_INPUT_ERROR("Active TB band curriculum stage has zero total band weight.");
  }
}

double Fitness::band_mse_from_eigenvalues(
  const Data& context,
  const int structure_index,
  const std::vector<float>& eigenvalues,
  const std::vector<int>& k_indices,
  const int band_count,
  const double band_weight_sum) const
{
  const Dataset& data = context.dataset_[structure_index];
  const int num_orbitals = context.models_[structure_index]->num_orbitals();
  const std::size_t expected = k_indices.size() * num_orbitals;
  if (eigenvalues.size() != expected) {
    PRINT_INPUT_ERROR("The mini-batch TB eigenvalue array has the wrong size.");
  }

  double weighted_squared_error = 0.0;
  for (std::size_t kb = 0; kb < k_indices.size(); ++kb) {
    const int k = k_indices[kb];
    const std::size_t offset = kb * num_orbitals;
    for (int n = 0; n < band_count; ++n) {
      const double difference = static_cast<double>(eigenvalues[offset + n]) - data.bands[k][n];
      const double weight = para_.band_weights[n];
      weighted_squared_error += weight * difference * difference;
    }
  }
  const double denominator = static_cast<double>(k_indices.size()) * band_weight_sum;
  if (!(denominator > 0.0)) {
    PRINT_INPUT_ERROR("Cannot compute weighted band loss with zero total weight.");
  }
  return weighted_squared_error / denominator;
}

double Fitness::evaluate_dispersion(
  const Data& context,
  const int structure_index,
  const std::vector<float>& eigenvalues,
  const std::vector<int>& k_indices,
  const int band_count,
  double& gradient_denominator,
  std::vector<double>* gradient,
  const double prefactor) const
{
  if (!gradient) {
    gradient_denominator = 0.0;
  }
  if (para_.dispersion_weight <= 0.0 || k_indices.size() < 2) {
    return 0.0;
  }

  const Dataset& data = context.dataset_[structure_index];
  const int num_orbitals = context.models_[structure_index]->num_orbitals();

  double numerator = 0.0;
  double denominator = 0.0;
  constexpr double dk_tol = 1.0e-10;
  const bool infinite_sigma = std::isinf(para_.dispersion_sigma);
  const double sigma = para_.dispersion_sigma;

  for (std::size_t a = 0; a < k_indices.size(); ++a) {
    const int ka = k_indices[a];
    const auto& kca = data.kpoints_cart[ka];
    const std::size_t offset_a = a * num_orbitals;
    for (std::size_t b = a + 1; b < k_indices.size(); ++b) {
      const int kb = k_indices[b];
      const auto& kcb = data.kpoints_cart[kb];
      const double dx = kcb[0] - kca[0];
      const double dy = kcb[1] - kca[1];
      const double dz = kcb[2] - kca[2];
      const double dk2 = dx * dx + dy * dy + dz * dz;
      if (dk2 <= dk_tol * dk_tol) {
        continue;
      }
      const double dk = std::sqrt(dk2);
      const double locality = infinite_sigma ? 1.0 : std::exp(-dk2 / (sigma * sigma));
      if (locality < 1.0e-30) {
        continue;
      }
      const double pair_weight = dk2 * locality;
      const std::size_t offset_b = b * num_orbitals;

      for (int n = 0; n < band_count; ++n) {
        const double band_weight = para_.band_weights[n];
        if (band_weight <= 0.0) {
          continue;
        }
        const double tb_delta = static_cast<double>(eigenvalues[offset_b + n]) -
                                static_cast<double>(eigenvalues[offset_a + n]);
        const double ref_delta = data.bands[kb][n] - data.bands[ka][n];
        const double slope_error = (tb_delta - ref_delta) / dk;
        if (gradient) {
          const double contribution = prefactor * band_weight * dk * locality * slope_error;
          (*gradient)[offset_b + n] += contribution;
          (*gradient)[offset_a + n] -= contribution;
        } else {
          const double weight = band_weight * pair_weight;
          numerator += weight * slope_error * slope_error;
          denominator += weight;
          gradient_denominator += band_weight * dk2 * locality;
        }
      }
    }
  }

  if (gradient) {
    return 0.0;
  }
  if (!(denominator > 0.0)) {
    PRINT_INPUT_ERROR("Dispersion loss is enabled but the selected k batch has no distinct "
                      "pair with nonzero weight. "
                      "Increase k_batch or dispersion sigma.");
  }
  return numerator / denominator;
}

void Fitness::accumulate_eigenvalue_gradient(
  const Data& context,
  const int structure_index,
  const std::vector<float>& eigenvalues,
  const std::vector<int>& k_indices,
  const double band_rmse,
  const double dispersion_rmse,
  double dispersion_denominator,
  const int structure_count,
  std::vector<double>& eigenvalue_gradient) const
{
  const Dataset& data = context.dataset_[structure_index];
  const int num_orbitals = context.models_[structure_index]->num_orbitals();
  eigenvalue_gradient.assign(k_indices.size() * num_orbitals, 0.0);
  const double inv_structures = 1.0 / static_cast<double>(structure_count);

  if (band_rmse > 1.0e-20) {
    const double denom = static_cast<double>(k_indices.size()) * active_band_weight_sum_;
    const double prefactor = inv_structures / band_rmse / denom;
    for (std::size_t kb = 0; kb < k_indices.size(); ++kb) {
      const int k = k_indices[kb];
      const std::size_t offset = kb * num_orbitals;
      for (int n = 0; n < active_num_reference_bands_; ++n) {
        const double weight = para_.band_weights[n];
        if (weight <= 0.0) {
          continue;
        }
        const double diff = static_cast<double>(eigenvalues[offset + n]) - data.bands[k][n];
        eigenvalue_gradient[offset + n] += prefactor * weight * diff;
      }
    }
  }

  if (para_.dispersion_weight <= 0.0 || dispersion_rmse <= 1.0e-20 || k_indices.size() < 2) {
    return;
  }
  if (!(dispersion_denominator > 0.0)) {
    PRINT_INPUT_ERROR("TB dispersion-gradient denominator is zero.");
  }
  const double rmse_prefactor =
    para_.dispersion_weight * inv_structures / dispersion_rmse / dispersion_denominator;
  evaluate_dispersion(
    context,
    structure_index,
    eigenvalues,
    k_indices,
    active_num_reference_bands_,
    dispersion_denominator,
    &eigenvalue_gradient,
    rmse_prefactor);
}

double Fitness::evaluate_data(
  Data& context,
  const double* parameters,
  Fitness::Breakdown* breakdown,
  std::vector<double>* gradient,
  const bool full)
{
  const bool has_reference = context.dataset_.has_reference();
  if (!has_reference && (!full || gradient)) {
    PRINT_INPUT_ERROR("Training loss requires reference bands.");
  }
  const int ns =
    full ? context.dataset_.size() : static_cast<int>(batch_.structure_indices().size());
  const int band_count = full ? num_reference_bands_ : active_num_reference_bands_;
  const double band_weight_sum = full ? para_.band_weight_sum : active_band_weight_sum_;
  std::vector<double> dispersion_denominators(context.dataset_.size(), 0.0);
  double sum_structure_band_mse = 0.0;
  double sum_structure_dispersion_mse = 0.0;

  for (int i = 0; i < ns; ++i) {
    const int s = full ? i : batch_.structure_indices()[i];
    std::vector<int> all_k(full ? context.dataset_[s].kpoints.size() : 0);
    std::iota(all_k.begin(), all_k.end(), 0);
    const auto& k_indices = full ? all_k : batch_.k_indices(s);

    if (full) {
      context.models_[s]->evaluate_all_bands(parameters, context.eigenvalues_[s]);
    } else {
      context.models_[s]->evaluate_bands(
        parameters, k_indices, context.eigenvalues_[s], gradient != nullptr);
    }
    if (has_reference) {
      sum_structure_band_mse += band_mse_from_eigenvalues(
        context, s, context.eigenvalues_[s], k_indices, band_count, band_weight_sum);
      if (para_.dispersion_weight > 0.0) {
        sum_structure_dispersion_mse += evaluate_dispersion(
          context, s, context.eigenvalues_[s], k_indices, band_count, dispersion_denominators[s]);
      }
    }
  }

  const double inv_structures = 1.0 / static_cast<double>(ns);
  Fitness::Breakdown loss;
  loss.band_rmse = std::sqrt(sum_structure_band_mse * inv_structures);
  loss.dispersion_rmse =
    para_.dispersion_weight > 0.0 ? std::sqrt(sum_structure_dispersion_mse * inv_structures) : 0.0;
  loss.data_loss = loss.band_rmse + para_.dispersion_weight * loss.dispersion_rmse;

  if (gradient) {
    gradient->assign(para_.number_of_variables, 0.0);
    for (int s : batch_.structure_indices()) {
      std::vector<double> grad_eigenvalues;
      accumulate_eigenvalue_gradient(
        context,
        s,
        context.eigenvalues_[s],
        batch_.k_indices(s),
        loss.band_rmse,
        loss.dispersion_rmse,
        dispersion_denominators[s],
        ns,
        grad_eigenvalues);
      std::vector<float> local_gradient;
      context.models_[s]->backward(batch_.k_indices(s), grad_eigenvalues, local_gradient);
      if (local_gradient.size() != gradient->size()) {
        PRINT_INPUT_ERROR("TB local gradient has wrong size.");
      }
      for (std::size_t v = 0; v < gradient->size(); ++v) {
        (*gradient)[v] += local_gradient[v];
      }
    }
  }
  if (breakdown) {
    *breakdown = loss;
  }
  return loss.data_loss;
}

void Fitness::write_bands(
  const Data& context,
  std::ostream& output,
  const int structure_index,
  const std::vector<float>& eigenvalues) const
{
  const Dataset& data = context.dataset_[structure_index];
  const int num_orbitals = context.models_[structure_index]->num_orbitals();
  const int nk = static_cast<int>(data.kpoints.size());
  if (eigenvalues.size() != static_cast<std::size_t>(nk) * num_orbitals) {
    PRINT_INPUT_ERROR("Full-k eigenvalues are required when writing band outputs.");
  }

  for (int k = 0; k < nk; ++k) {
    const auto& kp = data.kpoints[k];
    output << structure_index << " " << k << " " << kp[0] << " " << kp[1] << " " << kp[2];
    const std::size_t offset = static_cast<std::size_t>(k) * num_orbitals;
    for (int n = 0; n < num_reference_bands_; ++n) {
      output << " " << eigenvalues[offset + n];
    }
    for (int n = 0; n < num_reference_bands_ && !data.bands.empty(); ++n) {
      output << " " << data.bands[k][n];
    }
    output << "\n";
  }
}

void Fitness::write_outputs(Data& context, const char* label, const double* parameters)
{
  Breakdown loss;
  evaluate_data(context, parameters, &loss, nullptr, true);
  if (context.dataset_.has_reference()) {
    printf(
      "Final %s: band RMSE = %.9g eV, dispersion RMSE = %.9g eV Angstrom, loss = %.9g eV.\n",
      label,
      loss.band_rmse,
      loss.dispersion_rmse,
      loss.data_loss);
  }
  const std::string band_file = std::string("band_") + label + ".out";
  std::ofstream band_output(band_file);
  if (!band_output.is_open()) {
    PRINT_INPUT_ERROR(("Failed to open " + band_file + " for writing.").c_str());
  }

  band_output << "# band_weights";
  for (int n = 0; n < num_reference_bands_; ++n) {
    band_output << " " << (para_.band_min + n) << ":" << para_.band_weights[n];
  }
  band_output << "\n";
  band_output << "# structure kpoint kx ky kz";
  for (int n = 0; n < num_reference_bands_; ++n) {
    band_output << " tb_" << n;
  }
  for (int n = 0; n < num_reference_bands_ && context.dataset_.has_reference(); ++n) {
    band_output << " ref_" << (para_.band_min + n);
  }
  band_output << "\n";
  band_output << std::scientific << std::setprecision(16);

  for (int s = 0; s < context.dataset_.size(); ++s) {
    write_bands(context, band_output, s, context.eigenvalues_[s]);
  }
}

void Fitness::write_final_outputs(const double* parameters)
{
  write_outputs(train_, "train", parameters);
  if (test_) {
    write_outputs(*test_, "test", parameters);
  }
  train_.models_.front()->write_model_file("tb.txt", parameters);
  printf("Final model saved to tb.txt.\n");
}

void Fitness::predict()
{
  std::vector<double> parameters;
  std::vector<float> scaler;
  train_.models_.front()->read_model_file("tb.txt", parameters, scaler);
  set_q_scaler(scaler);
  write_outputs(train_, "train", parameters.data());
  printf("Prediction completed using tb.txt.\n");
}

int Fitness::curriculum_epoch_count() const
{
  if (para_.band_curriculum <= 0) {
    return 0;
  }
  const int num_bands = para_.band_max - para_.band_min;
  if (num_bands > std::numeric_limits<int>::max() / para_.band_curriculum) {
    PRINT_INPUT_ERROR("band_curriculum produces too many curriculum epochs.");
  }
  return num_bands * para_.band_curriculum;
}

bool Fitness::begin_epoch(const int epoch)
{
  const int count = para_.band_curriculum <= 0
                      ? num_reference_bands_
                      : std::min(num_reference_bands_, 1 + (epoch - 1) / para_.band_curriculum);
  set_active_band_count(count);
  epoch_loss_sum_ = epoch_band_sum_ = epoch_dispersion_sum_ = 0.0;
  epoch_steps_ = 0;
  return epoch > curriculum_epoch_count();
}

void Fitness::compute(const double* parameters, std::vector<double>& gradient)
{
  batch_.next();
  Fitness::Breakdown breakdown;
  evaluate_with_gradient(parameters, gradient, &breakdown);
  epoch_loss_sum_ += breakdown.data_loss;
  epoch_band_sum_ += breakdown.band_rmse;
  epoch_dispersion_sum_ += breakdown.dispersion_rmse;
  ++epoch_steps_;
}

void Fitness::print_training_header(const int steps_per_epoch, const int plateau_patience) const
{
  std::size_t min_k = batch_.k_indices(0).size(), max_k = min_k;
  for (int s = 1; s < train_.dataset_.size(); ++s) {
    const std::size_t nk = batch_.k_indices(s).size();
    min_k = std::min(min_k, nk);
    max_k = std::max(max_k, nk);
  }
  print_line_1();
  printf("Training\n");
  print_line_2();
  printf("  actual batch structures/k     = %zu / %zu", batch_.structure_indices().size(), min_k);
  if (max_k != min_k) {
    printf("-%zu", max_k);
  }
  printf("\n  estimated steps/epoch         = %d\n", steps_per_epoch);
  if (!para_.lr_scheduler_explicit) {
    printf("  resolved plateau patience     = %d\n", plateau_patience);
  }
  if (para_.band_curriculum > 0) {
    printf(
      "  curriculum ends at epoch      = %d (plateau starts afterward)\n",
      curriculum_epoch_count());
  }

  std::ofstream o("loss.out");
  if (!o.is_open()) {
    PRINT_INPUT_ERROR("Failed to open loss.out.");
  }
  o << "# epoch learning_rate train_batch_loss mean_abs_parameter band_rmse_train_batch "
       "dispersion_rmse_train_batch";
  if (test_) {
    o << " band_rmse_test dispersion_rmse_test";
  }
  o << " epoch_seconds\n";
  printf(
    "\n%8s%12s%14s%14s%14s%14s",
    "Epoch",
    "LR",
    "Batch-Loss",
    "|param|",
    "Band-Batch",
    "Disp-Batch");
  if (test_) {
    printf("%14s%14s", "Band-Test", "Disp-Test");
  }
  printf("%11s\n", "Sec");
}

double Fitness::report_epoch(
  const int epoch,
  const double* parameters,
  const double learning_rate,
  const double mean_abs_parameter,
  const double epoch_seconds)
{
  const double average_total_loss = epoch_loss_sum_ / epoch_steps_;
  const double average_band_rmse = epoch_band_sum_ / epoch_steps_;
  const double average_dispersion_rmse = epoch_dispersion_sum_ / epoch_steps_;
  std::ofstream o("loss.out", std::ios::app);
  if (!o.is_open()) {
    PRINT_INPUT_ERROR("Failed to append loss.out.");
  }
  o << std::scientific << std::setprecision(16) << epoch << " " << learning_rate << " "
    << average_total_loss << " " << mean_abs_parameter << " " << average_band_rmse << " "
    << average_dispersion_rmse;
  printf(
    "%8d%12.3e%14.5e%14.5e%14.5e%14.5e",
    epoch,
    learning_rate,
    average_total_loss,
    mean_abs_parameter,
    average_band_rmse,
    average_dispersion_rmse);
  if (test_) {
    Breakdown loss;
    evaluate_data(*test_, parameters, &loss, nullptr, true);
    o << " " << loss.band_rmse << " " << loss.dispersion_rmse;
    printf("%14.5e%14.5e", loss.band_rmse, loss.dispersion_rmse);
  }
  o << " " << epoch_seconds << "\n";
  printf("%11.3f\n", epoch_seconds);
  return average_total_loss;
}
