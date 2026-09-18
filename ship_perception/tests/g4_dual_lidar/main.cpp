#include "support.hpp"
#include <ship_perception/core/deskew.hpp>
#include <set>
int main() { return test::run("G4", [](test::Metrics& m) {
  using namespace ship;
  const auto source=synthetic_ship();
  const PoseFunction identity=[](std::int64_t){return Transform::Identity();};
  Sensor first, second;
  second.id=2;
  second.T_crane_lidar_true=pose({cfg::lidar2_translation_m[0],cfg::lidar2_translation_m[1],cfg::lidar2_translation_m[2]},
      {cfg::lidar2_rpy_rad[0],cfg::lidar2_rpy_rad[1],cfg::lidar2_rpy_rad[2]});
  second.T_crane_lidar_reported=second.T_crane_lidar_true;
  ReplayOptions a,b;
  a.noise_sigma_m=cfg::noise_sigma_m;
  b.noise_sigma_m=cfg::noise_sigma_m*2;
  b.drop_probability=cfg::drop_probability;
  b.seed=static_cast<unsigned>(cfg::seed+1);
  const auto s1=generate_scan(source,0,identity,identity,first,a,{});
  const auto s2=generate_scan(source,0,identity,identity,second,b,{});
  const auto c1=deskew_crane(s1.frame.points,0,0,identity,first.T_crane_lidar_reported);
  const auto c2=deskew_crane(s2.frame.points,0,0,identity,second.T_crane_lidar_reported);
  const auto fused=fuse_aligned(c1,c2);
  CHECK(fused.size()==c1.size()+c2.size());
  CHECK(c2.size()<c1.size() && !c2.empty());
  double sum=0;
  for(std::size_t i=0;i<c2.size();++i)
    sum+=(c2[i].xyz.cast<double>()-source[s2.source_indices[i]].cast<double>()).squaredNorm();
  const double rms=std::sqrt(sum/double(c2.size()));
  CHECK(rms<cfg::noisy_rms_limit_m);
  CHECK(c1.front().lidar_id==1 && c2.front().lidar_id==2);
  m["alignment_rms_m"]=rms;
  m["lidar1_point_count"]=double(c1.size());
  m["lidar2_point_count"]=double(c2.size());
  m["point_count"]=double(fused.size());
  m["overlap_fraction_of_source"]=double(c2.size())/double(source.size());
  m["transform_matrix_error"]=(second.T_crane_lidar_reported.matrix()-second.T_crane_lidar_true.matrix()).norm();
  Transform wrong=second.T_crane_lidar_reported;
  wrong.translation().x()+=cfg::extrinsic_error_m;
  const auto incorrect=deskew_crane(s2.frame.points,0,0,identity,wrong);
  double bad_sum=0;
  for(std::size_t i=0;i<incorrect.size();++i)
    bad_sum+=(incorrect[i].xyz.cast<double>()-source[s2.source_indices[i]].cast<double>()).squaredNorm();
  CHECK(std::sqrt(bad_sum/double(incorrect.size()))>cfg::fault_min_rms_m);
  m["perturbed_alignment_rms_m"]=std::sqrt(bad_sum/double(incorrect.size()));
  // Non-overlap is represented honestly, never treated as evidence of registration.
  a.noise_sigma_m=0; b=a;
  a.occlude=true; a.occlusion_min={0,-100,-100}; a.occlusion_max={100,100,100};
  b.occlude=true; b.occlusion_min={-100,-100,-100}; b.occlusion_max={0,100,100};
  const auto left=generate_scan(source,0,identity,identity,first,a,{});
  const auto right=generate_scan(source,0,identity,identity,second,b,{});
  CHECK(!left.source_indices.empty() && !right.source_indices.empty());
  std::set<std::size_t> indices(left.source_indices.begin(),left.source_indices.end());
  for(auto i:right.source_indices) CHECK(indices.count(i)==0);
  m["no_overlap_branch_overlap"]=0;
}); }
