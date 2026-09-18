#include <ship_perception/tracking/tracker.hpp>
#include <ship_perception/sensor/timestamp.hpp>
#include <ship_perception/timing/latency.hpp>

namespace ship { namespace v14 {
PreparedObservation prepare_observation(const TrackingInput& in,const Config& c) {
  if(!in.T_W_C) throw std::invalid_argument("MISSING_CRANE_POSE");
  PreparedObservation out;out.T_W_C=in.T_W_C(in.reference_ns);
  if(!healthy_transform(out.T_W_C,c)) throw std::invalid_argument("INVALID_CRANE_POSE");
  for(const auto& sensor:in.sensors) {
    if(!healthy_transform(sensor.T_C_L,c)) throw std::invalid_argument("INVALID_EXTRINSIC");
    for(const auto& point:sensor.points) {
      if(!point.xyz.allFinite()) throw std::invalid_argument("NONFINITE_SENSOR_POINT");
      const auto time=checked_add(sensor.stamp_ns,point.time_offset_ns);
      const Transform crane=in.T_W_C(time);
      if(!healthy_transform(crane,c)) throw std::invalid_argument("INVALID_CRANE_SAMPLE");
      const Transform aligned=out.T_W_C.inverse()*crane*sensor.T_C_L;
      const Eigen::Vector3d p=aligned*point.xyz.cast<double>();
      if(!p.allFinite() || p.cwiseAbs().maxCoeff()>c.registration.max_local_coordinate_m) throw std::invalid_argument("INVALID_LOCAL_POINT");
      out.points.push_back(p.cast<float>());
      out.rays.push_back({aligned.translation(),p});
    }
  }
  return out;
}
ShipTracker::ShipTracker(std::unique_ptr<RegistrationBackend> b,Method m,const Config& c):
  config_(c),method_(m),backend_(std::move(b)),map_(c) {
  if(!backend_) throw std::invalid_argument("缺少配准后端");
}
TrackingResult ShipTracker::process(const TrackingInput& input) {
  const auto start=monotonic_ns();
  TrackingResult out;out.T_W_B=last_pose_;out.map=map_.stats();out.initialized=initialized_;
  auto target=map_.snapshot();out.tracking_target_revision=target?target->revision:0;
  auto finish=[&]() {
    if(!out.valid && out.registration.failure_reason.empty()) out.registration.failure_reason=out.failure_reason;
    out.map=map_.stats();out.total_ms=double(monotonic_ns()-start)*1e-6;return out;
  };
  try {
    if(initialized_ && input.frame_id<=last_frame_) {out.failure_reason="NON_MONOTONIC_FRAME";return finish();}
    const auto observation=prepare_observation(input,config_);
    out.prepare_ms=double(monotonic_ns()-start)*1e-6;
    if(observation.points.size()<std::size_t(config_.registration.min_points)) {out.failure_reason="INSUFFICIENT_OBSERVATION";return finish();}
    if(!initialized_) {
      Transform provisional=Transform::Identity();
      provisional.linear()=input.bootstrap_R_W_B*pose(Eigen::Vector3d::Zero(),
        {config_.tracking.bootstrap_roll_rad,config_.tracking.bootstrap_pitch_rad,config_.tracking.bootstrap_yaw_rad}).linear();
      if(!healthy_transform(provisional,config_)) {out.failure_reason="INVALID_BOOTSTRAP_PRIOR";return finish();}
      Eigen::Vector3d centroid=Eigen::Vector3d::Zero();for(const auto& p:observation.points) centroid+=p.cast<double>();
      centroid/=double(observation.points.size());provisional.translation()=observation.T_W_C*centroid;
      out.T_B_C=provisional.inverse()*observation.T_W_C;
      Points initial;for(const auto& p:observation.points) initial.push_back((out.T_B_C*p.cast<double>()).cast<float>());
      map_.initialize(initial);last_pose_=provisional;last_frame_=input.frame_id;initialized_=true;
      out.T_W_B=last_pose_;out.valid=true;out.initialized=true;return finish();
    }
    if(target->points.size()<std::size_t(config_.registration.min_points)) {
      out.needs_reinitialization=true;out.failure_reason="REFERENCE_SUPPORT_LOST";return finish();
    }
    const Transform initial=last_pose_.inverse()*observation.T_W_C;
    RegistrationRequest request{observation.points,target,initial,method_};
    auto result=backend_->align(request);
    // 不信任适配器的 valid 标志；复核健康、共同质量以及工程跳变。
    const bool claimed_valid=result.valid;
    const bool math_failure=result.mathematical_failure;
    const std::string original_reason=result.failure_reason;
    PointIndex index(target->points);result=validate_result(result,request,config_,index);
    if(!claimed_valid || math_failure || !result.backend_executed) {
      result.valid=false;result.mathematical_failure=result.mathematical_failure||math_failure;
      if(!original_reason.empty()) result.failure_reason=original_reason;
      else if(result.failure_reason.empty()) result.failure_reason="BACKEND_REJECTED";
    }
    out.registration=result;out.registration_ms=result.elapsed_ms;
    if(!result.valid) {out.failure_reason=result.failure_reason;return finish();}
    const Transform pose=observation.T_W_C*result.T_target_source.inverse();
    const Transform step=last_pose_.inverse()*pose;
    if(!healthy_transform(pose,config_) || step.translation().norm()>config_.tracking.max_step_translation_m ||
       rotation_degrees(step.linear())>config_.tracking.max_step_rotation_rad*180/std::acos(-1.0)) {
      out.registration.valid=false;out.failure_reason="MOTION_JUMP";return finish();
    }
    Points in_body; std::vector<Ray> rays;
    for(const auto& p:observation.points) in_body.push_back((result.T_target_source*p.cast<double>()).cast<float>());
    for(const auto& ray:observation.rays) rays.push_back({result.T_target_source*ray.origin,result.T_target_source*ray.endpoint});
    const auto map_start=monotonic_ns();
    map_.commit(input.frame_id,in_body,rays,result);
    out.map_ms=double(monotonic_ns()-map_start)*1e-6;
    last_pose_=pose;last_frame_=input.frame_id;out.T_W_B=pose;out.T_B_C=result.T_target_source;out.valid=true;
    if(map_.stats().active<std::size_t(config_.registration.min_points)) out.needs_reinitialization=true;
  } catch(const std::exception& e) {out.failure_reason=e.what();}
  return finish();
}
}} // namespace ship::v14
