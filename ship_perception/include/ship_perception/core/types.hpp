#pragma once
#include <Eigen/Geometry>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ship {
// T_A_B maps B to A. World/frame transforms are always double precision.
using Transform = Eigen::Isometry3d;
struct TimedPoint {
  Eigen::Vector3f xyz = Eigen::Vector3f::Zero();
  float intensity = 0.0f;
  std::int64_t time_offset_ns = 0; // relative to frame stamp, may be negative
  std::uint16_t lidar_id = 0;
};
using Cloud = std::vector<TimedPoint>;
struct LocalOrigin { Eigen::Vector3d world_xyz = Eigen::Vector3d::Zero(); };
struct TimestampInfo {
  std::int64_t source_time_ns = 0;
  std::int64_t ingress_time_ns = 0;
  std::int64_t publish_time_ns = 0;
};
struct PoseStamped {
  std::int64_t timestamp_ns = 0;
  Transform T_world_body = Transform::Identity();
};
struct Frame {
  Cloud points;
  LocalOrigin local_origin_world;
  TimestampInfo time;
};
enum class Mode { EVALUATION_MODE, REALTIME_MODE };
enum class GateStatus {
  PASS_SYNTHETIC, PASS_LAB, PASS_LAB_LOCAL, PASS_SITE, SITE_PENDING, FAIL
};
inline const char* name(GateStatus s) {
  switch (s) {
    case GateStatus::PASS_SYNTHETIC: return "PASS_SYNTHETIC";
    case GateStatus::PASS_LAB: return "PASS_LAB";
    case GateStatus::PASS_LAB_LOCAL: return "PASS_LAB_LOCAL";
    case GateStatus::PASS_SITE: return "PASS_SITE";
    case GateStatus::SITE_PENDING: return "SITE_PENDING";
    case GateStatus::FAIL: return "FAIL";
  }
  throw std::logic_error("invalid gate status");
}
inline Eigen::Vector3d to_world(const Eigen::Vector3f& p, const LocalOrigin& o) {
  return o.world_xyz + p.cast<double>();
}
inline Eigen::Vector3f to_local(const Eigen::Vector3d& p, const LocalOrigin& o) {
  return (p - o.world_xyz).cast<float>();
}
// This constructor follows Rz(yaw)*Ry(pitch)*Rx(roll), angles in radians.
inline Transform pose(const Eigen::Vector3d& translation,
                      const Eigen::Vector3d& roll_pitch_yaw) {
  Transform t = Transform::Identity();
  t.linear() = (Eigen::AngleAxisd(roll_pitch_yaw.z(), Eigen::Vector3d::UnitZ()) *
                Eigen::AngleAxisd(roll_pitch_yaw.y(), Eigen::Vector3d::UnitY()) *
                Eigen::AngleAxisd(roll_pitch_yaw.x(), Eigen::Vector3d::UnitX())).toRotationMatrix();
  t.translation() = translation;
  return t;
}
} // namespace ship
