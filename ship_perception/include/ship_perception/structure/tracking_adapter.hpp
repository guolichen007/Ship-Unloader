#pragma once
#include <ship_perception/structure/types.hpp>
#include <ship_perception/tracking/tracker.hpp>

namespace ship { namespace v15 {
// Separate bridge target: Geometry Core itself never links a registration backend.
// Runtime input contains no scoring labels and accumulates dynamic returns too.
class ShipFrameAccumulator {
 public:
  explicit ShipFrameAccumulator(const Config& config=Config{}):config_(config){}
  bool append(const v14::TrackingInput& input,const v14::TrackingResult& result,
              const v14::Config& tracking_config=v14::Config{});
  const ShipFrameCloud& cloud() const {return cloud_;}
  const std::string& failure_reason() const {return failure_reason_;}
 private:
  Config config_;
  ShipFrameCloud cloud_;
  std::optional<std::uint64_t> last_frame_;
  std::string failure_reason_;
};
}} // namespace ship::v15
