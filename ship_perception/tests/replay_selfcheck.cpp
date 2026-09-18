#include "support.hpp"
#include <ship_perception/replay/replay.hpp>
int main() { return test::run("REPLAY_SELFCHECK", [](test::Metrics& m) {
  using namespace ship;
  const auto source=synthetic_ship();
  PoseFunction identity=[](std::int64_t){return Transform::Identity();};
  ReplayOptions o;
  auto scan=generate_scan(source,0,identity,identity,Sensor{},o,LocalOrigin{});
  const auto restored=reconstruct_truth(scan,identity,identity);
  CHECK(restored.size()==source.size());
  double max_error=0;
  for(std::size_t i=0;i<source.size();++i)
    max_error=std::max(max_error,(restored[i]-source[i].cast<double>()).norm());
  CHECK(max_error<cfg::point_tolerance_m);
  o.noise_sigma_m=cfg::noise_sigma_m;
  o.drop_probability=cfg::drop_probability;
  o.outlier_probability=cfg::outlier_probability;
  o.timestamp_jitter_ns=cfg::timestamp_jitter_ns;
  o.dynamic_points=static_cast<std::size_t>(cfg::dynamic_points);
  const auto a=generate_scan(source,0,identity,identity,Sensor{},o,LocalOrigin{});
  const auto b=generate_scan(source,0,identity,identity,Sensor{},o,LocalOrigin{});
  CHECK(a.frame.points.size()==b.frame.points.size());
  for(std::size_t i=0;i<a.frame.points.size();++i) {
    CHECK((a.frame.points[i].xyz-b.frame.points[i].xyz).norm()<cfg::point_tolerance_m);
    CHECK(a.frame.points[i].time_offset_ns==b.frame.points[i].time_offset_ns);
  }
  o.decimation=0;
  test::throws([&]{generate_scan(source,0,identity,identity,Sensor{},o,LocalOrigin{});});
  m["identity_max_error_m"]=max_error;
  m["source_points"]=double(source.size());
  m["fixed_seed_reproduced"]=1;
}); }
