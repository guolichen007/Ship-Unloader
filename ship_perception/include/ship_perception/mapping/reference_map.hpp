#pragma once
#include <ship_perception/registration/backend.hpp>
#include <array>
#include <deque>
#include <map>

namespace ship { namespace v14 {
enum class AnchorState { ACTIVE, SUSPECT, QUARANTINED };
struct MapStats {
  std::size_t active=0, suspect=0, quarantined=0, stable=0, candidates=0;
  std::uint64_t revision=0, candidate_revision=0;
};
// 射线的端点是正常传感器返回值；没有隐藏标签或评分对应关系。
struct Ray { Eigen::Vector3d origin, endpoint; };
class TrackingReferenceMap {
 public:
  explicit TrackingReferenceMap(const Config& config=Config{}):config_(config) {}
  void initialize(const Points& points);
  std::shared_ptr<const TargetSnapshot> snapshot() const { return snapshot_; }
  MapStats stats() const;
  // rejected 事务必须无副作用；frame_id 必须递增。
  bool commit(std::uint64_t frame_id, const Points& points, const std::vector<Ray>& rays,
              const RegistrationResult& accepted);
 private:
  using Key=std::array<std::int64_t,3>;
  struct Cell {
    Eigen::Vector3d point=Eigen::Vector3d::Zero();
    bool bootstrap=true, stable=false;
    AnchorState state=AnchorState::ACTIVE;
    std::size_t support=1;
    std::uint64_t last_support=0;
    std::deque<std::uint64_t> conflicts;
  };
  struct Candidate {
    Eigen::Vector3d mean=Eigen::Vector3d::Zero(), m2=Eigen::Vector3d::Zero();
    std::size_t support=0;
    std::uint64_t last_frame=0;
  };
  Key key(const Eigen::Vector3d& p) const;
  void rebuild();
  Config config_;
  std::map<Key,Cell> cells_;
  std::map<Key,Candidate> candidates_;
  std::shared_ptr<const TargetSnapshot> snapshot_;
  std::uint64_t revision_=0, candidate_revision_=0, last_frame_=0;
};
}} // namespace ship::v14
