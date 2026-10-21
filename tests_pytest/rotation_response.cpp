// CPU analytic oracle: no model, GPU, or fitted labels are used as references.
#include "main_nep/rotation_response.cuh"
#include <cassert>
#include <cstdio>

using Vec = std::array<double, 3>;

static Vec spin(double x, bool changing_radius)
{
  const double r = changing_radius ? 4.0 + 0.1*x + 0.02*x*x : 4.154;
  return {{r*std::cos(x), r*std::sin(x), 0.0}};
}

int main()
{
  const double pi = std::acos(-1.0);
  // Uniform and nonuniform grids, including one-sided endpoints.
  for (auto xs : {std::array<double,3>{{-pi/2, -pi/3, -pi/6}},
                  std::array<double,3>{{-0.4, 0.0, 0.7}}}) {
    for (bool radial : {false, true}) {
      for (double x : xs) {
        Vec tangent = {{0,0,0}}, cart = {{0,0,0}};
        const Vec current = spin(x, radial);
        for (int j = 0; j < 3; ++j) {
          const double a=xs[(j+1)%3], b=xs[(j+2)%3];
          const double w=(2*x-a-b)/((xs[j]-a)*(xs[j]-b));
          Vec delta;
          assert(fitness_spin::rotation_displacement(current, spin(xs[j],radial), delta));
          for (int d=0; d<3; ++d) {
            tangent[d] += w*delta[d];
            cart[d] += w*spin(xs[j],radial)[d];
          }
        }
        const double r=radial ? 4.0+0.1*x+0.02*x*x : 4.154;
        const double dr=radial ? 0.1+0.04*x : 0.0;
        Vec exact={{dr*std::cos(x)-r*std::sin(x), dr*std::sin(x)+r*std::cos(x),0}};
        for (int d=0;d<3;++d) assert(std::abs(tangent[d]-exact[d])<1e-12);
        if (!radial) {
          // Arbitrarily large longitudinal magnetic force must not create a
          // response for a constant-magnitude rotation.
          double projection=0;
          for (int d=0;d<3;++d) projection += tangent[d]*current[d];
          assert(std::abs(projection)<1e-11);
          if (x==xs.front()) {
            double old_projection=0;
            for (int d=0;d<3;++d) old_projection += cart[d]*current[d];
            assert(std::abs(old_projection)>0.01); // reproduces the old defect
          }
        }
      }
    }
  }
  Vec delta;
  assert(!fitness_spin::rotation_displacement({{1,0,0}},{{-1,0,0}},delta));
  assert(fitness_spin::rotation_displacement({{0,0,0}},{{1,2,3}},delta));
  assert((delta==Vec{{1,2,3}}));
  puts("rotation response analytic tests passed");
}
