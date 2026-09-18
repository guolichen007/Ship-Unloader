#pragma once
#include <ship_perception/core/types.hpp>
#include <limits>
#include <type_traits>

namespace ship {
enum class TimeUnit { Nanoseconds, Microseconds };
inline std::int64_t checked_add(std::int64_t a, std::int64_t b) {
  const auto hi = std::numeric_limits<std::int64_t>::max();
  const auto lo = std::numeric_limits<std::int64_t>::min();
  if ((b > 0 && a > hi-b) || (b < 0 && a < lo-b))
    throw std::overflow_error("timestamp addition overflow");
  return a+b;
}
inline std::int64_t checked_sub(std::int64_t a, std::int64_t b) {
  const auto hi = std::numeric_limits<std::int64_t>::max();
  const auto lo = std::numeric_limits<std::int64_t>::min();
  if ((b > 0 && a < lo+b) || (b < 0 && a > hi+b))
    throw std::overflow_error("timestamp subtraction overflow");
  return a-b;
}
template<class Integer> std::int64_t normalize_offset(Integer raw, TimeUnit unit) {
  static_assert(std::is_integral<Integer>::value && sizeof(Integer)<=8, "integer offset required");
  const auto hi=std::numeric_limits<std::int64_t>::max();
  const auto lo=std::numeric_limits<std::int64_t>::min();
  if constexpr (std::is_unsigned<Integer>::value) {
    if (raw > static_cast<std::uint64_t>(hi)) throw std::overflow_error("unsigned offset overflow");
  }
  const auto value = static_cast<std::int64_t>(raw);
  std::int64_t scale = 1;
  switch(unit) {
    case TimeUnit::Nanoseconds: break;
    case TimeUnit::Microseconds: scale=1000; break;
    default: throw std::invalid_argument("unknown time unit");
  }
  if (value > hi/scale || value < lo/scale) throw std::overflow_error("time unit overflow");
  return value*scale;
}
inline std::int64_t point_time(std::int64_t frame_stamp, const TimedPoint& p) {
  return checked_add(frame_stamp,p.time_offset_ns);
}
// Vendor-agnostic adapter preserves input order; it never silently sorts a scan.
template<class Integer> Cloud adapt_points(const std::vector<Eigen::Vector3f>& points,
    const std::vector<Integer>& offsets, TimeUnit unit, std::int64_t stamp, std::uint16_t id) {
  if (points.size()!=offsets.size()) throw std::invalid_argument("offset/point count mismatch");
  Cloud result;
  result.reserve(points.size());
  for (std::size_t i=0; i<points.size(); ++i) {
    if (!points[i].allFinite()) throw std::invalid_argument("nonfinite point");
    TimedPoint p{points[i],0.0f,normalize_offset(offsets[i],unit),id};
    (void)point_time(stamp,p); // reject absolute overflow at the boundary
    result.push_back(p);
  }
  return result;
}
inline bool ordered_times(const Cloud& points) {
  for(std::size_t i=1; i<points.size(); ++i)
    if(points[i].time_offset_ns < points[i-1].time_offset_ns) return false;
  return true;
}
// No vendor layouts or encoder transport have been guessed in M0.
struct SiteAdapterStatus {
  GateStatus vendor_timestamp = GateStatus::SITE_PENDING;
  GateStatus encoder = GateStatus::SITE_PENDING;
  GateStatus lidar_calibration = GateStatus::SITE_PENDING;
};
} // namespace ship
