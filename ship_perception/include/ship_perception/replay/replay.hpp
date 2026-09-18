#pragma once
#include <ship_perception/core/types.hpp>
#include <functional>
#include <string>

namespace ship {
using PoseFunction = std::function<Transform(std::int64_t)>;
struct Trajectory {
  LocalOrigin origin;
  Eigen::Vector3d drift_mps = Eigen::Vector3d::Zero();
  Eigen::Vector3d rpy_amplitude_rad = Eigen::Vector3d::Zero();
  double heave_m = 0;
  double frequency_hz = 0.3;
  double smooth_random_m = 0;
  std::uint32_t seed = 42;
  Transform at(std::int64_t time_ns) const;
};
struct ReplayOptions {
  std::uint32_t seed = 42;
  std::int64_t scan_duration_ns = 80000000;
  double noise_sigma_m = 0;
  double drop_probability = 0;
  double outlier_probability = 0;
  double outlier_scale_m = 2;
  std::size_t decimation = 1;
  std::size_t dynamic_points = 0;
  bool occlude = false;
  Eigen::Vector3d occlusion_min{-1,-1,-1};
  Eigen::Vector3d occlusion_max{1,1,1};
  std::int64_t timestamp_offset_ns = 0;
  std::int64_t timestamp_jitter_ns = 0;
};
struct Sensor {
  std::uint16_t id = 1;
  Transform T_crane_lidar_true = Transform::Identity();
  Transform T_crane_lidar_reported = Transform::Identity();
};
struct ReplayScan {
  Frame frame; // xyz in the LiDAR frame, not world coordinates
  std::vector<std::size_t> source_indices; // max<size_t> means dynamic point
  std::vector<std::int64_t> true_point_times_ns;
  PoseStamped ground_truth_ship;
  Sensor sensor;
};
std::vector<Eigen::Vector3f> synthetic_ship();
ReplayScan generate_scan(const std::vector<Eigen::Vector3f>& source,
    std::int64_t frame_stamp, const PoseFunction& T_world_ship,
    const PoseFunction& T_world_crane, const Sensor& sensor,
    const ReplayOptions& options, const LocalOrigin& origin);
// Reconstruct observations into the GT ship frame (a generator self-check, NOT registration).
std::vector<Eigen::Vector3d> reconstruct_truth(const ReplayScan& scan,
    const PoseFunction& T_world_ship, const PoseFunction& T_world_crane);
} // namespace ship
