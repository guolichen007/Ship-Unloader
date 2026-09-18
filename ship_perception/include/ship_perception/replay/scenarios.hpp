#pragma once
#include <ship_perception/replay/replay.hpp>
#include <ship_perception/config.hpp>

namespace ship {
struct Scenario {
  std::string id;
  std::vector<Eigen::Vector3f> source;
  ReplayOptions options;
  Sensor sensor;
  // Test-oracle requirements for future M1/M3; these are not estimator outputs.
  std::string future_requirement;
  bool estimator_implemented = false;
};
inline std::vector<Scenario> scenario_matrix() {
  const std::vector<std::string> ids={"normal_6dof","flat_deck_degeneracy","missing_coaming",
    "cargo_change","transient_cluster","density_drop","timestamp_offset",
    "extrinsic_perturbation","yaw_roll_pitch_heave","lost","repeated_hatch_ambiguity"};
  std::vector<Scenario> matrix;
  for(const auto& id:ids) {
    Scenario s;
    s.id=id; s.source=synthetic_ship();
    s.options.seed=static_cast<unsigned>(cfg::seed);
    s.options.scan_duration_ns=cfg::scan_duration_ns;
    s.future_requirement="estimate_6dof_and_calibrated_uncertainty";
    if(id=="flat_deck_degeneracy") {
      for(auto& p:s.source) p.z()=0;
      s.future_requirement="XY_YAW_weak_covariance_increase_no_TRACKING";
    } else if(id=="missing_coaming") {
      std::vector<Eigen::Vector3f> kept;
      for(const auto& p:s.source) if(p.z()<=0) kept.push_back(p);
      s.source=kept;
      s.future_requirement="report_structural_support_loss";
    } else if(id=="cargo_change") {
      for(auto& p:s.source) if(p.z()<0) p.z()+=.7f;
      s.future_requirement="reject_cargo_as_rigid_tracking_reference";
    } else if(id=="transient_cluster") {
      s.options.dynamic_points=static_cast<std::size_t>(cfg::dynamic_points);
      s.future_requirement="reject_transient_cluster";
    } else if(id=="density_drop") {
      s.options.drop_probability=cfg::drop_probability;
      s.options.decimation=static_cast<std::size_t>(cfg::decimation);
      s.future_requirement="reduce_confidence_with_support";
    } else if(id=="timestamp_offset") {
      s.options.timestamp_offset_ns=cfg::timestamp_offset_ns;
      s.options.timestamp_jitter_ns=cfg::timestamp_jitter_ns;
      s.future_requirement="detect_timing_fault";
    } else if(id=="extrinsic_perturbation") {
      s.sensor.T_crane_lidar_reported=pose({cfg::extrinsic_error_m,0,0},{0,0,cfg::extrinsic_error_rad});
      s.future_requirement="detect_calibration_fault";
    } else if(id=="lost") {
      s.options.drop_probability=1;
      s.future_requirement="LOST_no_valid_pose";
    } else if(id=="repeated_hatch_ambiguity") {
      // Two identical structures with a known longitudinal displacement.
      const auto base=s.source;
      for(auto p:base) { p.x()+=16.f; s.source.push_back(p); }
      s.future_requirement="reject_ambiguous_relocalization_without_external_evidence";
    }
    matrix.push_back(s);
  }
  return matrix;
}
inline Trajectory configured_trajectory(const LocalOrigin& origin) {
  Trajectory t;
  t.origin=origin;
  t.seed=static_cast<unsigned>(cfg::seed);
  t.drift_mps={cfg::drift_mps[0],cfg::drift_mps[1],cfg::drift_mps[2]};
  t.rpy_amplitude_rad={cfg::rpy_amplitude_rad[0],cfg::rpy_amplitude_rad[1],cfg::rpy_amplitude_rad[2]};
  t.heave_m=cfg::heave_m;
  t.frequency_hz=cfg::frequency_hz;
  t.smooth_random_m=cfg::smooth_random_m;
  return t;
}
} // namespace ship
