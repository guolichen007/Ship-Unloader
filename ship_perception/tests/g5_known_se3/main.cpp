#include "support.hpp"
#include <ship_perception/replay/scenarios.hpp>
#include <ship_perception/sensor/timestamp.hpp>
#include <limits>
int main() { return test::run("G5", [](test::Metrics& m) {
  using namespace ship;
  LocalOrigin origin{{500000,6000000,10}};
  const auto trajectory=configured_trajectory(origin);
  PoseFunction moving=[&](std::int64_t t){return trajectory.at(t);};
  PoseFunction crane=[&](std::int64_t){return pose(origin.world_xyz,{0,0,0});};
  const auto matrix=scenario_matrix();
  CHECK(matrix.size()==11);
  std::ofstream csv("scenarios.csv");
  CHECK(bool(csv));
  csv << "scenario,frame,point_count,gt_reconstruction_rms_m,reported_reconstruction_rms_m,estimator_status,future_requirement,git_sha,config_hash,dataset_id,mode\n";
  double largest_fault=0;
  for(const auto& s:matrix) {
    CHECK(!s.estimator_implemented);
    if(s.id=="flat_deck_degeneracy") for(const auto& p:s.source) CHECK(p.z()==0);
    if(s.id=="missing_coaming") for(const auto& p:s.source) CHECK(p.z()<=0);
    if(s.id=="cargo_change") {
      const auto base=synthetic_ship();
      CHECK(s.source.size()==base.size());
      bool changed=false;
      for(std::size_t i=0;i<base.size();++i) if(base[i].z()<0) {
        CHECK(s.source[i].z()>base[i].z()); changed=true;
      }
      CHECK(changed);
    }
    if(s.id=="repeated_hatch_ambiguity") CHECK(s.source.size()==2*synthetic_ship().size());
    for(std::int64_t frame=0;frame<cfg::frames;++frame) {
      const auto stamp=checked_add(1000000000,frame*cfg::frame_period_ns);
      const auto scan=generate_scan(s.source,stamp,moving,crane,s.sensor,s.options,origin);
      CHECK((scan.ground_truth_ship.T_world_body.matrix()-moving(stamp).matrix()).norm()<cfg::math_tolerance);
      CHECK(scan.ground_truth_ship.timestamp_ns==stamp);
      if(s.id=="lost") CHECK(scan.frame.points.empty());
      else CHECK(!scan.frame.points.empty());
      if(s.id=="density_drop") CHECK(scan.frame.points.size()<s.source.size());
      if(s.id=="transient_cluster") CHECK(scan.frame.points.size()==s.source.size()+std::size_t(cfg::dynamic_points));
      const auto truth=reconstruct_truth(scan,moving,crane);
      double sum=0,reported_sum=0; std::size_t n=0;
      for(std::size_t i=0;i<truth.size();++i) {
        const auto index=scan.source_indices[i];
        if(index==std::numeric_limits<std::size_t>::max()) continue;
        const Eigen::Vector3d expected=s.source[index].cast<double>();
        sum+=(truth[i]-expected).squaredNorm();
        const auto reported_t=point_time(stamp,scan.frame.points[i]);
        const Eigen::Vector3d reported=moving(reported_t).inverse()*crane(reported_t)*
            s.sensor.T_crane_lidar_reported*scan.frame.points[i].xyz.cast<double>();
        reported_sum+=(reported-expected).squaredNorm();
        ++n;
      }
      const double rms=n ? std::sqrt(sum/double(n)) : 0;
      const double fault=n ? std::sqrt(reported_sum/double(n)) : 0;
      CHECK(rms<cfg::point_tolerance_m);
      if(s.id=="timestamp_offset" || s.id=="extrinsic_perturbation") {
        CHECK(fault>cfg::fault_min_rms_m); largest_fault=std::max(largest_fault,fault);
      }
      csv << std::setprecision(17) << s.id << ',' << frame << ',' << scan.frame.points.size()
          << ',' << rms << ',' << fault << ",NOT_IMPLEMENTED," << s.future_requirement
          << ',' << cfg::git_sha << ',' << cfg::config_hash << ',' << cfg::dataset_id << ',' << cfg::mode << '\n';
    }
  }
  csv.close(); CHECK(bool(csv));
  // GT trajectory must actually excite each rotation axis and heave.
  const auto a=moving(0), b=moving(1000000000);
  CHECK((b.translation()-a.translation()).norm()>cfg::fault_min_rms_m);
  for(int axis=0;axis<3;++axis) CHECK(std::abs(b.linear().eulerAngles(2,1,0)[axis])>cfg::math_tolerance);
  CHECK(std::abs(b.translation().z()-a.translation().z())>cfg::fault_min_rms_m);
  m["scenarios"]=double(matrix.size());
  m["frames_per_scenario"]=double(cfg::frames);
  m["largest_injected_fault_rms_m"]=largest_fault;
  m["registration_implemented"]=0;
  m["observability_covariance_evaluated"]=0;
}); }
