#pragma once
#include <ship_perception/replay/replay.hpp>
#include <ship_perception/sensor/timestamp.hpp>

namespace ship {
// Fast stage: compensate crane motion only, output in crane coordinates at t_ref.
// Ship-motion map-update compensation is deliberately outside M0.
inline Cloud deskew_crane(const Cloud& points, std::int64_t frame_stamp,
    std::int64_t t_ref, const PoseFunction& T_world_crane, const Transform& T_crane_lidar) {
  const Transform T_ref_world=T_world_crane(t_ref).inverse();
  Cloud result;
  result.reserve(points.size());
  for(auto p: points) {
    const Eigen::Vector3d corrected=T_ref_world*T_world_crane(point_time(frame_stamp,p))*
                                    T_crane_lidar*p.xyz.cast<double>();
    p.xyz=corrected.cast<float>();
    if(!p.xyz.allFinite()) throw std::invalid_argument("nonfinite deskew point");
    result.push_back(p);
  }
  return result;
}
// Both inputs must already be expressed in the same crane reference frame.
inline Cloud fuse_aligned(const Cloud& lidar1,const Cloud& lidar2) {
  Cloud fused=lidar1;
  fused.insert(fused.end(),lidar2.begin(),lidar2.end());
  return fused;
}
} // namespace ship
