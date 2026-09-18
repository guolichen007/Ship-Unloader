#include "support.hpp"
#include <random>
int main() { return test::run("G1", [](test::Metrics& m) {
  using namespace ship;
  const double eps = cfg::math_tolerance;
  const auto t = pose({1,2,3}, {0,0,std::acos(-1.0)/2});
  CHECK((t * Eigen::Vector3d(1,0,0) - Eigen::Vector3d(1,3,3)).norm() < eps);
  std::mt19937 rng(static_cast<unsigned>(cfg::seed));
  std::uniform_real_distribution<double> u(-2,2);
  double maximum = 0;
  for (int i=0; i<200; ++i) {
    const auto ab=pose({u(rng),u(rng),u(rng)}, {u(rng),u(rng),u(rng)});
    const auto bc=pose({u(rng),u(rng),u(rng)}, {u(rng),u(rng),u(rng)});
    const Eigen::Vector3d p(u(rng),u(rng),u(rng));
    const Transform ac = ab * bc;
    const double error = (ac*p - ab*(bc*p)).norm();
    maximum = std::max(maximum,error);
    CHECK(error < eps);
    CHECK((ab.matrix()*ab.inverse().matrix()-Eigen::Matrix4d::Identity()).norm() < eps);
    CHECK((ab.inverse()*(ab*p)-p).norm() < eps);
  }
  LocalOrigin origin{{500000,6000000,12}};
  Eigen::Vector3f local(.01f,.13f,-.27f);
  CHECK((to_local(to_world(local,origin),origin)-local).norm() < cfg::point_tolerance_m);
  CHECK(std::string(name(GateStatus::SITE_PENDING)) == "SITE_PENDING");
  m["random_composition_max_m"] = maximum;
  m["random_cases"] = 200;
}); }
