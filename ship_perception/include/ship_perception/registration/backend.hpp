#pragma once
#include <ship_perception/core/types.hpp>
#include <ship_perception/v14_config.hpp>
#include <memory>
#include <string>

namespace ship { namespace v14 {
using Points = std::vector<Eigen::Vector3f>;
enum class Method { GICP, VGICP, PCL_GICP };
const char* method_name(Method method);

struct TargetSnapshot {
  TargetSnapshot(Points p, std::uint64_t rev):points(std::move(p)),revision(rev) {}
  const Points points;
  const std::uint64_t revision;
};
struct RegistrationRequest {
  const Points& source;
  std::shared_ptr<const TargetSnapshot> target;
  Transform initial_guess = Transform::Identity(); // source -> target
  Method method = Method::GICP;
};
struct RegistrationResult {
  Transform T_target_source = Transform::Identity();
  // small_gicp 的右扰动排列为 [旋转, 平移]；不是已标定协方差。
  Eigen::Matrix<double,6,6> H = Eigen::Matrix<double,6,6>::Zero();
  bool hessian_available = false;
  bool objective_available = false;
  bool iterations_available = false;
  bool quality_available = false;
  bool backend_executed = false;
  bool converged = false;
  bool valid = false;
  bool mathematical_failure = false;
  double rmse = 0, overlap_ratio = 0, fitness_score = 0;
  double elapsed_ms = 0;
  double raw_objective = 0;
  std::size_t source_points = 0, target_points = 0, inliers = 0, iterations = 0;
  std::string failure_reason;
};

// 仅输入局部点云，公共接口不含第三方点类型。
class PointIndex {
 public:
  explicit PointIndex(const Points& points);
  ~PointIndex();
  PointIndex(PointIndex&&) noexcept;
  PointIndex& operator=(PointIndex&&) noexcept;
  bool nearest(const Eigen::Vector3d& query, std::size_t& index, double& squared_distance) const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
bool healthy_transform(const Transform& t, const Config& config, std::string* reason=nullptr);
double rotation_degrees(const Eigen::Matrix3d& r);
Points preprocess(const Points& points, const Config& config);
RegistrationResult validate_result(RegistrationResult result, const RegistrationRequest& request,
                                   const Config& config, const PointIndex& target_index);
class RegistrationBackend {
 public:
  virtual ~RegistrationBackend() = default;
  virtual RegistrationResult align(const RegistrationRequest& request) = 0;
};
class SmallGicpBackend final : public RegistrationBackend {
 public:
  explicit SmallGicpBackend(const Config& config = Config{});
  ~SmallGicpBackend();
  RegistrationResult align(const RegistrationRequest& request) override;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
std::unique_ptr<RegistrationBackend> make_pcl_backend(const Config& config);
bool pcl_backend_available();
}} // namespace ship::v14
