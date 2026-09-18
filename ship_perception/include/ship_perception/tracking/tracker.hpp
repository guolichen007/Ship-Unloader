#pragma once
#include <ship_perception/mapping/reference_map.hpp>
#include <functional>

namespace ship { namespace v14 {
struct SensorObservation {
  Cloud points;
  std::int64_t stamp_ns=0;
  Transform T_C_L=Transform::Identity();
};
struct TrackingInput {
  std::uint64_t frame_id=0;
  std::int64_t reference_ns=0;
  std::vector<SensorObservation> sensors;
  std::function<Transform(std::int64_t)> T_W_C;
  Eigen::Matrix3d bootstrap_R_W_B=Eigen::Matrix3d::Identity();
};
struct PreparedObservation {
  Points points;
  std::vector<Ray> rays;
  Transform T_W_C=Transform::Identity();
};
PreparedObservation prepare_observation(const TrackingInput& input,const Config& config);
struct TrackingResult {
  bool valid=false, initialized=false, needs_reinitialization=false;
  Transform T_W_B=Transform::Identity(), T_B_C=Transform::Identity();
  RegistrationResult registration;
  MapStats map;
  std::uint64_t tracking_target_revision=0;
  double prepare_ms=0, registration_ms=0, map_ms=0, total_ms=0;
  std::string failure_reason;
};
class ShipTracker {
 public:
  ShipTracker(std::unique_ptr<RegistrationBackend> backend,Method method,const Config& config=Config{});
  TrackingResult process(const TrackingInput& input);
  const TrackingReferenceMap& map() const {return map_;}
 private:
  Config config_;
  Method method_;
  std::unique_ptr<RegistrationBackend> backend_;
  TrackingReferenceMap map_;
  bool initialized_=false;
  std::uint64_t last_frame_=0;
  Transform last_pose_=Transform::Identity();
};
}} // namespace ship::v14
