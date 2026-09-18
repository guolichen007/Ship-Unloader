#pragma once
#include <ship_perception/sensor/timestamp.hpp>
#include <chrono>
#include <deque>
#include <utility>

namespace ship {
inline std::int64_t monotonic_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}
struct Latency {
  std::int64_t pipeline_latency_ns;
  std::int64_t age_ns;
};
// Only comparable timestamps are accepted. Real devices require clock mapping.
inline Latency latency(const TimestampInfo& t,bool same_clock_domain) {
  if(!same_clock_domain) throw std::invalid_argument("SITE_PENDING: clock mapping required");
  if(t.source_time_ns>t.ingress_time_ns || t.ingress_time_ns>t.publish_time_ns)
    throw std::invalid_argument("timestamps not source <= ingress <= publish");
  return {checked_sub(t.publish_time_ns,t.ingress_time_ns),
          checked_sub(t.publish_time_ns,t.source_time_ns)};
}
// Single-consumer M0 reference queue; caller supplies synchronization if shared.
// Evaluation keeps every frame in FIFO order. Realtime is bounded/latest wins.
template<class T> class FrameQueue {
 public:
  FrameQueue(Mode mode,std::size_t capacity):mode_(mode),capacity_(capacity) {
    if(capacity==0) throw std::invalid_argument("zero queue capacity");
  }
  void push(T value) {
    if(mode_==Mode::REALTIME_MODE && queue_.size()==capacity_) {
      queue_.pop_front(); ++dropped_;
    }
    queue_.push_back(std::move(value));
  }
  T pop() {
    if(queue_.empty()) throw std::out_of_range("empty frame queue");
    if(mode_==Mode::REALTIME_MODE) {
      while(queue_.size()>1) {queue_.pop_front(); ++dropped_;}
    }
    T value=std::move(queue_.front()); queue_.pop_front(); return value;
  }
  std::size_t size() const {return queue_.size();}
  std::size_t dropped() const {return dropped_;}
 private:
  Mode mode_;
  std::size_t capacity_;
  std::size_t dropped_=0;
  std::deque<T> queue_;
};
} // namespace ship
