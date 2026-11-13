// CPU regression harness. test_remd_tp_host.py inserts the named production
// definitions at the markers below. No parser or SCR implementation is copied.
// GPU storage/events and exchange velocity/pressure operations are synchronous
// test doubles; actual CUDA execution is covered by test_remd_tp.py.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "utilities/common.cuh"
#define PRINT_INPUT_ERROR(message) throw std::runtime_error(message)
#define __device__
#define __global__
// Execute the scalar BDP scaling kernel for one synthetic velocity carrying K.
struct Kernel_Index { int x; };
const Kernel_Index blockIdx{0}, blockDim{1}, threadIdx{0};
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template<class T> using GPU_Pinned_Vector = std::vector<T>;
struct GPU_Event { explicit GPU_Event(int) {} };
// @production declarations
struct Atom {
  int number_of_atoms = 2;
  std::vector<std::string> cpu_atom_symbol = {"C", "C"};
  std::vector<double> cpu_mass = {12, 12};
  std::vector<double> cpu_position_per_atom = std::vector<double>(6);
  std::vector<double> cpu_velocity_per_atom = std::vector<double>(6);
};
struct Thermostat { double temperature = 300; };
struct Replica_Slot {
  int replica_id = 0, temperature_label = 0, pressure_refreshes = 0;
  double potential_energy = 0, target_temperature = 300;
  Box box;
  Atom atom;
  std::unique_ptr<SCR_Barostat> barostat{new SCR_Barostat(0)};
  std::unique_ptr<Thermostat> thermostat{new Thermostat};
};
namespace replica {
void validate_xyz_pbc(const std::string&, const Box&, const char*);
void refresh_barostat_pressure(Replica_Slot& slot) {
  // Test double: the production CUDA helper is tested by the GPU suite.
  ++slot.pressure_refreshes;
  for (int d = 0; d < 3; ++d) {
    double kinetic = 0;
    for (int i = 0; i < slot.atom.number_of_atoms; ++i) {
      double v = slot.atom.cpu_velocity_per_atom[i + d * slot.atom.number_of_atoms];
      kinetic += slot.atom.cpu_mass[i] * v * v;
    }
    slot.barostat->host_thermo[2 + d] = kinetic / slot.box.get_volume();
  }
  slot.barostat->pending = true;
}
}
struct Config {
  int replicas = 2, exchange_interval = 10, num_temperatures = 2, num_pressures = 1;
  std::string resume_prefix;
  std::vector<double> temperatures = {300, 400}, pressure_table = {0, 0};
};
struct Runtime {
  std::vector<Replica_Slot*> data;
  std::vector<Replica_Slot*>& slots() { return data; }
};
struct REMD_TP_Driver {
  Config config_;
  Barostat_Input barostat_input_;
  Atom source_atom_;
  Box source_box_;
  long long completed_steps_ = 100, exchange_count_ = 10;
  double time_step_ = 1;
  unsigned long long potential_hash_ = 123;
  std::vector<int> replica_to_label_ = {0, 1}, label_to_replica_ = {0, 1};
  std::mt19937 exchange_rng_{7};
  std::vector<std::string> restart_bdp_states_ = std::vector<std::string>(2);
  std::vector<std::string> restart_scr_states_ = std::vector<std::string>(2);
  std::vector<long long> attempts_[2] = {{0, 0}, {0, 0}}, accepts_[2] = {{0, 0}, {0, 0}};
  std::vector<double> probability_sum_[2] = {{0, 0}, {0, 0}};
  std::vector<std::pair<Direction, int>> phases_;
  Runtime runtime_;
  int num_t() const { return config_.num_temperatures; }
  int num_p() const { return config_.num_pressures; }
  int label_of(int t, int p) const { return p * num_t() + t; }
  double temperature_of(int label) const { return config_.temperatures[label % num_t()]; }
  double pressure_of(int label) const {
    return config_.pressure_table[(label % num_t()) * num_p() + label / num_t()];
  }
  double beta_of(int label) const { return 1 / (K_B * temperature_of(label)); }
  void read_restart_metadata();
  void read_restart_xyz(Replica_Slot&);
  void set_condition(Replica_Slot&, int);
  void build_phases();
  void attempt_pair(long long, Direction, int, int, FILE*);
  void attempt_exchanges(long long, FILE*);
  void scale_velocity(Replica_Slot& slot, double factor) {
    for (double& v : slot.atom.cpu_velocity_per_atom) v *= factor;
  }
};
// @production definitions
Box box(double length = 30) {
  Box b;
  std::fill(b.cpu_h, b.cpu_h + 18, 0);
  b.cpu_h[0] = b.cpu_h[4] = b.cpu_h[8] = length;
  b.get_inverse();
  return b;
}
void scr_moments() {
  SCR_Barostat b(0);
  b.interval = 10; b.tau_p = 1000; b.rng.seed(987);
  const double temperature = 300, volume = 27000, a = 0.0001;
  b.compressibility = a * volume / (K_B * temperature * b.interval / b.tau_p);
  double sum = 0, squares = 0;
  const int samples = 2000000;
  for (int i = 0; i < samples; ++i) {
    Box h = box();
    const double epsilon = 3 * std::log(b.update_box(h, temperature));
    sum += epsilon; squares += epsilon * epsilon;
  }
  // At Pint=P0, Eq. 5 has zero conditional mean and variance 2a.
  // The former extra kT/V term moves the mean by a, about ten standard errors.
  require(std::abs(sum / samples) < 6 * std::sqrt(2 * a / samples), "nonzero ln(V) drift at equilibrium pressure");
  require(std::abs(squares / samples / (2 * a) - 1) < .01, "incorrect SCR noise variance");
  std::cout << "mean_dlogV=" << sum / samples << " variance=" << squares / samples << '\n';
}
void scr_roundtrip() {
  SCR_Barostat a(0), b(0);
  a.interval = b.interval = 10; a.tau_p = b.tau_p = 1000;
  a.compressibility = b.compressibility = 50;
  a.normal(a.rng); // Leave the normal distribution's cached variate populated.
  a.step_counter = 100; a.pending = true;
  a.host_thermo = {300, -2, .001, .002, .003};
  require(b.import_state(a.export_state()), "SCR import failed");
  Box x = box(), y = box();
  for (int i = 0; i < 100; ++i)
    require(a.update_box(x, 300) == b.update_box(y, 300), "SCR continuation differs");
  require(a.export_state() == b.export_state(), "SCR RNG state differs");
  require(!b.import_state(a.export_state() + " trailing"), "SCR accepted trailing state data");
  a.step_counter = 101;
  require(!b.import_state(a.export_state()), "SCR accepted off-interval pending update");
}
void scr_directions() {
  for (int mask = 1; mask < 8; ++mask) {
    SCR_Barostat b(0);
    b.tau_p = 1000; b.interval = 10; b.compressibility = 10;
    Box h = box();
    for (int d = 0; d < 3; ++d) b.couple[d] = (mask >> d) & 1;
    double scale = b.update_box(h, 300);
    for (int d = 0; d < 3; ++d)
      require(h.cpu_h[d * 4] == 30 * (b.couple[d] ? scale : 1), "wrong coupled direction");
  }
}
void canonical_moments() {
  std::vector<double> mass = {1, 2, 3, 4, 5, 6, 7, 8}, v;
  double sum = 0, squares = 0;
  const int samples = 30000;
  for (int seed = 1; seed <= samples; ++seed) {
    generate_canonical_velocities(300, mass, v, seed);
    double kinetic = 0;
    for (int d = 0; d < 3; ++d) {
      double momentum = 0;
      for (int i = 0; i < 8; ++i) {
        momentum += mass[i] * v[i + 8 * d];
        kinetic += .5 * mass[i] * v[i + 8 * d] * v[i + 8 * d];
      }
      require(std::abs(momentum) < 1e-12, "nonzero COM momentum");
    }
    double reduced = kinetic / (K_B * 300);
    sum += reduced; squares += reduced * reduced;
  }
  const double mean = sum / samples, variance = squares / samples - mean * mean;
  require(std::abs(mean - 10.5) < .15, "wrong canonical kinetic mean (3N-3 DOF)");
  require(std::abs(variance - 10.5) < .6, "wrong canonical kinetic variance");
  std::cout << "K/kT mean=" << mean << " variance=" << variance << '\n';
}
double thermalize(double kinetic, int dof, std::mt19937& rng, SVR_Gaussian_Cache& cache) {
  SVR_Noise noise = draw_resamplekin_noise(dof, rng, cache);
  double thermo[1] = {2 * kinetic / (dof * K_B)};
  double vx = std::sqrt(2 * kinetic), vy = 0, vz = 0;
  gpu_resamplekin_and_scale(1, dof, 300, 10, noise.gaussian, noise.sum_of_squares,
                           thermo, &vx, &vy, &vz);
  return vx * vx / 2;
}
void bdp_moments() {
  std::mt19937 rng(19);
  SVR_Gaussian_Cache cache;
  double kinetic = 10.5 * K_B * 300, sum = 0, squares = 0;
  const int samples = 1000000;
  for (int i = 0; i < samples; ++i) {
    kinetic = thermalize(kinetic, 21, rng, cache);
    double reduced = kinetic / (K_B * 300);
    sum += reduced; squares += reduced * reduced;
  }
  double mean = sum / samples, variance = squares / samples - mean * mean;
  require(std::abs(mean - 10.5) < .15, "wrong BDP kinetic mean");
  require(std::abs(variance - 10.5) < .6, "wrong BDP kinetic variance");
  std::cout << "BDP K/kT mean=" << mean << " variance=" << variance << '\n';
}
void ideal_gas_moments() {
  // U=0 reduces isotropic SCR+BDP to a closed evolution of V and K.
  // This uses the actual SCR update, BDP kernel and noise generators; no GPU
  // forces, asynchronous queues or coordinate kernels are exercised here.
  constexpr int n = 16, dof = 3 * n - 3, block_size = 10000, blocks = 500;
  const double kT = K_B * 300, expected_v = 27000, expected_k = dof / 2.0;
  SCR_Barostat b(0);
  b.interval = 1; b.tau_p = 1000; b.target_pressure = n * kT / expected_v;
  b.compressibility = 1 / b.target_pressure; b.rng.seed(17);
  Box h = box();
  std::mt19937 rng(31); SVR_Gaussian_Cache cache;
  double kinetic = expected_k * kT;
  std::vector<double> moments[4];
  double totals[4] = {};
  const int warmup = 100000;
  for (int step = 0; step < warmup + block_size * blocks; ++step) {
    if (step > 0) {
      double scale = b.update_box(h, 300);
      kinetic /= scale * scale;
    }
    kinetic = thermalize(kinetic, dof, rng, cache);
    for (int d = 0; d < 3; ++d) b.host_thermo[2 + d] = 2 * kinetic / (3 * h.get_volume());
    if (step < warmup) continue;
    double v = h.get_volume() / expected_v, k = kinetic / kT;
    totals[0] += v; totals[1] += (v - 1) * (v - 1);
    totals[2] += k; totals[3] += (k - expected_k) * (k - expected_k);
    if ((step - warmup + 1) % block_size == 0) {
      for (int i = 0; i < 4; ++i) { moments[i].push_back(totals[i] / block_size); totals[i] = 0; }
    }
  }
  const double expected[4] = {1, 1.0 / n, expected_k, expected_k};
  for (int i = 0; i < 4; ++i) {
    double mean = 0, squares = 0;
    for (double x : moments[i]) { mean += x / blocks; squares += x * x / blocks; }
    double stderr = std::sqrt(std::max(0.0, squares - mean * mean) / (blocks - 1));
    std::cout << "ideal gas moment " << i << ": " << mean << " expected=" << expected[i]
              << " SE=" << stderr << '\n';
    require(std::abs(mean - expected[i]) < 6 * stderr + .01 * expected[i], "wrong ideal gas stationary moment");
  }
}
void exchanges() {
  REMD_TP_Driver d;
  Replica_Slot a, b;
  a.box = box(); b.box = box(); b.replica_id = 1;
  a.atom.cpu_velocity_per_atom = b.atom.cpu_velocity_per_atom = {1, -1, 2, -2, 3, -3};
  d.runtime_.data = {&a, &b};
  d.set_condition(a, 0); d.set_condition(b, 1);
  d.exchange_count_ = 0; d.build_phases();
  FILE* log = std::tmpfile(); require(log, "tmpfile failed");
  int updates = 0;
  for (int step = 1; step <= 101; ++step) {
    if (a.barostat->pending) { ++updates; a.barostat->pending = b.barostat->pending = false; }
    if (step % 10 == 0) {
      a.barostat->pending = b.barostat->pending = true;
      d.attempt_exchanges(step, log);
      require(a.barostat->pending && b.barostat->pending, "exchange dropped an SCR update");
    }
  }
  require(updates == 10 && a.pressure_refreshes == 10, "missing scheduled update or pressure refresh");
  require(d.replica_to_label_ == std::vector<int>({0, 1}), "wrong label mapping");
  require(d.accepts_[0][0] == 10, "zero enthalpy swap rejected");
  // Off-interval swaps must not create an extra barostat update.
  d.attempt_exchanges(110, log);
  require(!a.barostat->pending && a.pressure_refreshes == 10, "unscheduled SCR update");
  // Force a rejection and ensure velocities and pending pressure remain untouched.
  auto velocities = a.atom.cpu_velocity_per_atom;
  a.potential_energy = 1e8; b.potential_energy = -1e8; // a now has the hot label
  a.barostat->pending = b.barostat->pending = true;
  d.attempt_exchanges(120, log);
  require(a.atom.cpu_velocity_per_atom == velocities && a.pressure_refreshes == 10,
          "rejected swap changed state");
  std::fclose(log);
}
void phases() {
  REMD_TP_Driver d;
  d.config_.num_temperatures = d.config_.num_pressures = 3;
  d.build_phases();
  require(d.phases_ == std::vector<std::pair<Direction, int>>({
    {along_temperature, 0}, {along_pressure, 0}, {along_temperature, 1}, {along_pressure, 1}}),
    "wrong 2D exchange phases");
}
int main(int argc, char** argv) {
  try {
    require(argc >= 2, "missing test command");
    std::string command = argv[1];
    if (command == "scr_moments") scr_moments();
    else if (command == "scr_roundtrip") scr_roundtrip();
    else if (command == "scr_directions") scr_directions();
    else if (command == "canonical") canonical_moments();
    else if (command == "bdp") bdp_moments();
    else if (command == "ideal_gas") ideal_gas_moments();
    else if (command == "exchange") exchanges();
    else if (command == "phases") phases();
    else if (command == "rng") std::cout << std::mt19937(1);
    else if (command == "metadata" || command == "xyz") {
      require(argc == 3, "missing restart prefix");
      REMD_TP_Driver d; d.config_.resume_prefix = argv[2]; d.source_box_ = box();
      d.barostat_input_.temperature_coupling = 100;
      d.barostat_input_.bulk_modulus = 2; d.barostat_input_.pressure_coupling = 1000;
      d.barostat_input_.couple[0] = d.barostat_input_.couple[1] = 0;
      d.barostat_input_.couple_name = "z";
      if (command == "metadata") d.read_restart_metadata();
      else { Replica_Slot slot; slot.box = d.source_box_; d.read_restart_xyz(slot); }
    } else throw std::runtime_error("unknown command");
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
