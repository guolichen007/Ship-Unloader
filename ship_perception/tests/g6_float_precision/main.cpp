#include "support.hpp"
int main() { return test::run("G6", [](test::Metrics& m) {
  using namespace ship;
  for(double base: cfg::world_origins_m) {
    LocalOrigin origin{{base,base,base}};
    double local_error=0,absolute_error=0;
    for(int i=1;i<=30;++i) {
      const Eigen::Vector3d delta(.01*i,.03*i,-.013*i);
      const Eigen::Vector3d world=origin.world_xyz+delta;
      const Eigen::Vector3f absolute=world.cast<float>();
      const Eigen::Vector3f local=to_local(world,origin);
      absolute_error=std::max(absolute_error,(absolute.cast<double>()-world).norm());
      local_error=std::max(local_error,(to_world(local,origin)-world).norm());
    }
    CHECK(local_error<cfg::local_precision_limit_m);
    CHECK(absolute_error>cfg::absolute_float_min_loss_m);
    const auto rotated=pose(origin.world_xyz,{.1,-.2,.3});
    const Eigen::Vector3f p(.13f,-.27f,.019f);
    const Eigen::Vector3d world=rotated*p.cast<double>();
    const Eigen::Vector3f roundtrip=(rotated.inverse()*world).cast<float>();
    CHECK((roundtrip-p).norm()<cfg::local_precision_limit_m);
    m["absolute_float_max_error_m_at_"+std::to_string(int(base))]=absolute_error;
    m["local_float_max_error_m_at_"+std::to_string(int(base))]=local_error;
  }
}); }
