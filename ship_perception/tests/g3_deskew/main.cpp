#include "support.hpp"
#include <ship_perception/core/deskew.hpp>
int main() { return test::run("G3", [](test::Metrics& m) {
  using namespace ship;
  const auto source=synthetic_ship();
  const PoseFunction identity=[](std::int64_t){return Transform::Identity();};
  // An analytic translation oracle independent of the generator.
  PoseFunction translate=[](std::int64_t t){return pose({double(t)*1e-9,0,0},{0,0,0});};
  Cloud raw{{Eigen::Vector3f(9,2,3),1,1000000000,1}};
  CHECK((deskew_crane(raw,0,0,translate,Transform::Identity())[0].xyz-
         Eigen::Vector3f(10,2,3)).norm()<cfg::point_tolerance_m);
  const std::int64_t stamp=1000000000;
  for(int scenario=0;scenario<3;++scenario) {
    Trajectory trajectory;
    if(scenario!=1) trajectory.drift_mps={2,-1,.5};
    if(scenario!=0) trajectory.rpy_amplitude_rad={.3,.2,.4};
    PoseFunction moving=[&](std::int64_t t){return trajectory.at(t);};
    ReplayOptions options;
    options.scan_duration_ns=cfg::scan_duration_ns;
    Sensor sensor;
    sensor.T_crane_lidar_true=pose({.7,-.2,.4},{.1,0,.2});
    sensor.T_crane_lidar_reported=sensor.T_crane_lidar_true;
    const auto scan=generate_scan(source,stamp,identity,moving,sensor,options,{});
    const auto restored=deskew_crane(scan.frame.points,stamp,stamp,moving,sensor.T_crane_lidar_reported);
    double sum=0;
    for(std::size_t i=0;i<source.size();++i) {
      const Eigen::Vector3d expected=moving(stamp).inverse()*source[i].cast<double>();
      sum+=(restored[i].xyz.cast<double>()-expected).squaredNorm();
    }
    const double rms=std::sqrt(sum/double(source.size()));
    CHECK(rms<cfg::point_tolerance_m);
    m["case_"+std::to_string(scenario)+"_rms_m"]=rms;
    options.timestamp_offset_ns=cfg::timestamp_offset_ns;
    const auto bad=generate_scan(source,stamp,identity,moving,sensor,options,{});
    const auto uncompensated=deskew_crane(bad.frame.points,stamp,stamp,moving,sensor.T_crane_lidar_reported);
    double bad_sum=0;
    for(std::size_t i=0;i<source.size();++i) {
      const Eigen::Vector3d expected=moving(stamp).inverse()*source[i].cast<double>();
      bad_sum+=(uncompensated[i].xyz.cast<double>()-expected).squaredNorm();
    }
    const double bad_rms=std::sqrt(bad_sum/double(source.size()));
    CHECK(bad_rms>cfg::fault_min_rms_m);
    m["case_"+std::to_string(scenario)+"_time_fault_rms_m"]=bad_rms;
  }
}); }
