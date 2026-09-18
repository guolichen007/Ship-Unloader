#include <ship_perception/registration/backend.hpp>
#include <ship_perception/timing/latency.hpp>
#include <small_gicp/ann/kdtree.hpp>
#include <small_gicp/ann/gaussian_voxelmap.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/registration/registration.hpp>
#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/util/normal_estimation.hpp>

namespace ship { namespace v14 {
struct SmallGicpBackend::Impl {
  explicit Impl(const Config& c):config(c) {}
  Config config;
  std::shared_ptr<const TargetSnapshot> target;
  small_gicp::PointCloud::Ptr cloud;
  std::shared_ptr<small_gicp::KdTree<small_gicp::PointCloud>> tree;
  small_gicp::GaussianVoxelMap::Ptr gaussian;
  std::unique_ptr<PointIndex> quality_tree;
};
SmallGicpBackend::SmallGicpBackend(const Config& c):impl_(std::make_unique<Impl>(c)) {}
SmallGicpBackend::~SmallGicpBackend()=default;
RegistrationResult SmallGicpBackend::align(const RegistrationRequest& q) {
  const auto start=monotonic_ns();
  RegistrationResult result;
  const auto& c=impl_->config;
  if(q.method!=Method::GICP && q.method!=Method::VGICP) {result.failure_reason="UNSUPPORTED_METHOD";return result;}
  if(!healthy_transform(q.initial_guess,c,&result.failure_reason)) return result;
  if(!q.target) {result.failure_reason="MISSING_TARGET";return result;}
  try {
    const Points source=preprocess(q.source,c);
    if(source.size()<std::size_t(c.registration.min_points)) {result.failure_reason="INSUFFICIENT_SOURCE";return result;}
    if(impl_->target!=q.target) {
      const Points target=preprocess(q.target->points,c);
      if(target.size()<std::size_t(c.registration.min_points)) {result.failure_reason="INSUFFICIENT_TARGET";return result;}
      impl_->cloud=std::make_shared<small_gicp::PointCloud>(target);
      impl_->tree=std::make_shared<small_gicp::KdTree<small_gicp::PointCloud>>(impl_->cloud);
      small_gicp::estimate_normals_covariances(*impl_->cloud,*impl_->tree,int(c.registration.neighbors));
      impl_->quality_tree=std::make_unique<PointIndex>(q.target->points);
      impl_->gaussian.reset();
      impl_->target=q.target;
    }
    small_gicp::PointCloud source_cloud(source);
    small_gicp::UnsafeKdTree<small_gicp::PointCloud> source_tree(source_cloud);
    small_gicp::estimate_normals_covariances(source_cloud,source_tree,int(c.registration.neighbors));
    small_gicp::Registration<small_gicp::GICPFactor,small_gicp::SerialReduction> engine;
    engine.optimizer.max_iterations=int(c.registration.max_iterations);
    engine.rejector.max_dist_sq=c.registration.correspondence_m*c.registration.correspondence_m;
    engine.criteria.translation_eps=c.registration.translation_epsilon_m;
    engine.criteria.rotation_eps=c.registration.rotation_epsilon_rad;
    if(q.method==Method::VGICP && !impl_->gaussian) {
      impl_->gaussian=std::make_shared<small_gicp::GaussianVoxelMap>(c.registration.voxel_m);
      impl_->gaussian->insert(*impl_->cloud);
    }
    result.backend_executed=true;
    const auto raw=q.method==Method::GICP?
      engine.align(*impl_->cloud,source_cloud,*impl_->tree,q.initial_guess):
      engine.align(*impl_->gaussian,source_cloud,*impl_->gaussian,q.initial_guess);
    result.T_target_source=raw.T_target_source;
    result.H=raw.H; result.hessian_available=true;
    result.raw_objective=raw.error;result.objective_available=true;
    result.iterations=raw.iterations;result.iterations_available=true;
    result.converged=raw.converged;
    result=validate_result(result,q,c,*impl_->quality_tree);
  } catch(const std::exception& e) {
    result.valid=false;result.failure_reason=std::string("BACKEND_ERROR: ")+e.what();
  }
  result.elapsed_ms=double(monotonic_ns()-start)*1e-6;
  return result;
}
}} // namespace ship::v14
