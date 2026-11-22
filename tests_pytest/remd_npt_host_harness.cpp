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

// Direct tests of the production host-only utilities. No source extraction or driver doubles.
#include "integrate/replica/remd_utilities.cuh"
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>

using namespace replica;
void require(bool value)
{
  if (!value)
    throw std::runtime_error("assertion failed");
}
void close(double a, double b, double tolerance = 1e-11) { require(std::abs(a - b) < tolerance); }
REMD_Grid grid(int nt = 3, int np = 3)
{
  REMD_Grid g;
  g.is_npt = true;
  g.num_pressures = np;
  for (int t = 0; t < nt; ++t)
    g.temperatures.push_back(300 + 50 * t);
  for (int t = 0; t < nt; ++t)
    for (int p = 0; p < np; ++p)
      g.pressure_table.push_back(.001 * p + .0001 * t);
  require(g.validate(nt * np).empty());
  return g;
}
void nvt_schedule()
{
  auto g = grid(8, 1);
  g.is_npt = false;
  g.pressure_table.clear();
  for (int round = 0; round < 12; ++round) {
    const auto attempts = g.attempt_counts(round);
    for (size_t i = 0; i < attempts.size(); ++i)
      require(attempts[i] == (i % 2 == 0 ? (round + 1) / 2 : round / 2));
    const auto pairs = g.pairs(round);
    require(pairs.size() == (round % 2 ? 3 : 4));
    for (size_t i = 0; i < pairs.size(); ++i) {
      require(pairs[i].first == int(2 * i + round % 2));
      require(pairs[i].second == pairs[i].first + 1);
    }
  }
}
void two_replicas()
{
  for (int np : {1, 2}) {
    auto g = grid(np == 1 ? 2 : 1, np);
    for (int r = 0; r < 20; ++r) {
      auto p = g.pairs(r);
      require(p.size() == 1);
      require(p[0].first == 0 && p[0].second == 1);
    }
  }
}
void grid_schedule()
{
  auto g = grid();
  for (int round = 0; round < 4; ++round) {
    const auto pairs = g.pairs(round);
    require(pairs.size() == 3);
    for (const auto& pair : pairs) {
      require(pair.direction == round % 2);
      require(pair.second - pair.first == (round % 2 ? 3 : 1));
      require((round % 2 ? pair.first / 3 : pair.first % 3) == round / 2);
    }
  }
  std::vector<long long> counted(18, 0);
  for (int round = 0; round < 100; ++round) {
    require(g.attempt_counts(round) == counted);
    for (const auto& pair : g.pairs(round))
      ++counted[g.statistic_index(pair)];
  }
}
void pressure_schedule()
{
  auto g = grid(1, 5);
  for (int r = 0; r < 8; ++r)
    for (const auto& pair : g.pairs(r)) {
      require(pair.direction == 1);
      require(pair.first % 2 == r % 2);
      require(pair.second == pair.first + 1);
    }
}
void sheared()
{
  auto g = grid();
  for (int p = 0; p < 3; ++p)
    for (int t = 0; t < 3; ++t) {
      const int label = g.label(t, p);
      close(g.temperature(label), 300 + 50 * t);
      close(g.pressure(label), .001 * p + .0001 * t);
    }
}
void invalid_grid()
{
  auto original = grid();
  for (int which = 0; which < 9; ++which) {
    auto g = original;
    switch (which) {
      case 0:
        g.temperatures.clear();
        break;
      case 1:
        g.temperatures[0] = 0;
        break;
      case 2:
        g.temperatures[1] = g.temperatures[0];
        break;
      case 3:
        g.temperatures[0] = std::numeric_limits<double>::infinity();
        break;
      case 4:
        g.pressure_table.pop_back();
        break;
      case 5:
        g.pressure_table[0] = std::nan("");
        break;
      case 6:
        g.pressure_table[1] = g.pressure_table[0];
        break;
      case 7:
        g.num_pressures = 0;
        break;
      case 8:
        g.is_npt = false;
        break;
    }
    require(!g.validate(9).empty());
  }
  require(!original.validate(8).empty());
}
void logweights()
{
  std::mt19937 rng(19);
  std::uniform_real_distribution<double> u(0, 1);
  for (int i = 0; i < 1000; ++i) {
    double a = 250 + u(rng) * 100, b = 400 + u(rng) * 100, pa = u(rng) * .001, pb = u(rng) * .001;
    double ei = -5 + u(rng), ej = -5 + u(rng), vi = 1000 + u(rng) * 100, vj = 1000 + u(rng) * 100;
    double before = -(ei + pa * vi) / (K_B * a) - (ej + pb * vj) / (K_B * b);
    double after = -(ej + pa * vj) / (K_B * a) - (ei + pb * vi) / (K_B * b);
    close(remd_log_ratio(a, b, pa, pb, ei, ej, vi, vj), after - before);
  }
}
void pressure_swap()
{
  close(remd_log_ratio(300, 300, .001, .002, -5, -4, 1000, 1100), .1 / (K_B * 300));
}
void temperature_swap()
{
  close(
    remd_log_ratio(300, 400, .001, .001, -5, -4, 1000, 1100),
    (-1 - .1) * (1 / (K_B * 300) - 1 / (K_B * 400)));
}
void reverse_swap()
{
  double x = remd_log_ratio(300, 400, .001, .002, -5, -4, 1000, 1100);
  close(x, -remd_log_ratio(300, 400, .001, .002, -4, -5, 1100, 1000));
}
void probability()
{
  const double x = remd_log_ratio(300, 400, 0, 0, -5, -4.99, 1000, 1100);
  const double p = std::exp(x);
  require(p > 0 && p < 1);
  std::mt19937 rng(21);
  std::uniform_real_distribution<double> u(0, 1);
  int accepted = 0;
  for (int i = 0; i < 1000; ++i)
    accepted += u(rng) < p;
  require(accepted > 0 && accepted < 1000);
  require(std::exp(-10000.) == 0); // legitimate rejection, not a numeric error
}
void scr_moments()
{
  std::mt19937 rng(987);
  std::normal_distribution<double> normal;
  const double dt = .01, a = .0001, volume = 27000, compressibility = a * volume / (K_B * 300 * dt);
  double sum = 0, squares = 0;
  const int count = 2000000;
  for (int i = 0; i < count; ++i) {
    const double x =
      scr_log_volume_increment(300, volume, .001, .001, compressibility, dt, normal(rng));
    sum += x;
    squares += x * x;
  }
  require(std::abs(sum / count) < 6 * std::sqrt(2 * a / count));
  require(std::abs(squares / count / (2 * a) - 1) < .01);
  std::cout << "mean dlogV=" << sum / count << " variance=" << squares / count << '\n';
}
void scr_response() { close(scr_log_volume_increment(300, 27000, .002, .001, 10, .01, 0), -.0001); }
void scr_scaling()
{
  double a = scr_log_volume_increment(300, 27000, .001, .001, 10, .01, 1);
  double b = scr_log_volume_increment(300, 27000, .001, .001, 10, .04, 1);
  close(b, 2 * a);
  double c = scr_log_volume_increment(300, 6750, .001, .001, 10, .01, 1);
  close(c, 2 * a);
}
struct State {
  long long counter = 100;
  bool pending = true;
  double thermo[5] = {300, -5, .001, .002, .003};
  std::mt19937 rng{21};
  std::normal_distribution<double> normal;
  std::string text() const { return export_scr_state(counter, pending, thermo, rng, normal); }
  bool read(const std::string& text, int interval = 10)
  {
    return import_scr_state(text, interval, counter, pending, thermo, rng, normal);
  }
};
void state_roundtrip()
{
  State a, b;
  a.normal(a.rng);
  require(b.read(a.text()));
  require(a.text() == b.text());
  for (int i = 0; i < 100; ++i)
    close(a.normal(a.rng), b.normal(b.rng), 1e-100);
}
void bad_state()
{
  State a, b;
  require(!b.read(""));
  require(!b.read(a.text() + " trailing"));
  require(!b.read(a.text(), 0));
  a.counter = -1;
  require(!b.read(a.text()));
  a.counter = 101;
  require(!b.read(a.text()));
  a.counter = 100;
  a.thermo[2] = std::nan("");
  require(!b.read(a.text()));
  a.thermo[2] = .001;
  a.normal = std::normal_distribution<double>(1, 1);
  require(!b.read(a.text()));
}
void failed_state_preserves_original()
{
  State a;
  auto old = a.text();
  require(!a.read(old + " trailing"));
  require(a.text() == old);
}
int main(int argc, char** argv)
{
  const std::map<std::string, std::function<void()>> cases = {
    {"nvt_schedule", nvt_schedule},
    {"two_replicas", two_replicas},
    {"grid_schedule", grid_schedule},
    {"pressure_schedule", pressure_schedule},
    {"sheared", sheared},
    {"invalid_grid", invalid_grid},
    {"logweights", logweights},
    {"pressure_swap", pressure_swap},
    {"temperature_swap", temperature_swap},
    {"reverse_swap", reverse_swap},
    {"probability", probability},
    {"scr_moments", scr_moments},
    {"scr_response", scr_response},
    {"scr_scaling", scr_scaling},
    {"state_roundtrip", state_roundtrip},
    {"bad_state", bad_state},
    {"failed_state_preserves_original", failed_state_preserves_original}};
  try {
    require(argc == 2);
    cases.at(argv[1])();
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
