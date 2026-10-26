// SPDX-License-Identifier: LGPL-3.0-or-later
// Exercise the production driver without advancing atoms or allocating GPU arrays.
#include <memory>
#include <string>
#include <vector>
#include "integrate/ensemble.cuh"
#include "integrate/spin_integrator.cuh"
#define private public
#include "integrate/integrate.cuh"
#undef private
#include "force/force.cuh"
#include "model/atom.cuh"
#include <cmath>
#include <cstdio>
#include <limits>

class LatticeProbe final : public Ensemble
{
public:
  void compute1(double, int, int, const std::vector<Group>&, Box&, Atom&,
    GPU_Vector<double>&) override {}
  void compute2(double, int, int, const std::vector<Group>&, Box&, Atom&,
    GPU_Vector<double>&, Force&) override {}
};

class SpinProbe final : public Spin_Integrator
{
public:
  double first = 0.0, second = 0.0;
  void compute1(double, Atom&) override { first = temperature; }
  void compute2(double, Atom&) override { second = temperature; }
};

int main()
{
  int failures = 0;
  // A non-temperature-driving ensemble deliberately leaves its base target
  // unusable. The driver's initial-temperature cache must never read it.
  for (const auto type : {EnsembleType::NVT_BER, EnsembleType::MTTK}) {
    for (const bool with_spin : {false, true}) {
      if (with_spin && type == EnsembleType::MTTK) continue;
      Integrate integrate;
      integrate.type = type;
      integrate.temperature = integrate.temperature1 = 300.0;
      integrate.temperature2 = 600.0;
      integrate.ensemble_.reset(new LatticeProbe);
      integrate.ensemble_->temperature = std::numeric_limits<double>::quiet_NaN();
      if (with_spin) integrate.spin_integrator.reset(new SpinProbe);
      Atom atom;
      Box box = {};
      Force force;
      std::vector<Group> group;
      GPU_Vector<double> thermo;
      for (int phase = 1; phase <= 2; ++phase) {
        if (phase == 1) integrate.compute1(0.1, 2, 4, group, box, atom, thermo);
        else integrate.compute2(0.1, 2, 4, group, box, atom, thermo, force);
        if (integrate.get_temperature() != 300.0) {
          fprintf(stderr, "Initial temperature changed in phase %d (type %d, spin %d).\n",
            phase, int(type), int(with_spin));
          ++failures;
        }
        if (type == EnsembleType::NVT_BER && integrate.get_target_temperature_of_step() != 450.0)
          ++failures;
      }
      if (with_spin) {
        const auto& spin = static_cast<const SpinProbe&>(*integrate.spin_integrator);
        if (spin.first != 450.0 || spin.second != 450.0) ++failures;
      }
    }
  }
  printf("Temperature cache and Spin target checks: %s\n", failures ? "FAIL" : "PASS");
  return failures ? 1 : 0;
}
