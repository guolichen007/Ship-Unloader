#include <ship_perception/registration/backend.hpp>
#include <small_gicp/ann/kdtree.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <map>

namespace ship { namespace v14 {
const char* method_name(Method m) {
  switch(m) {case Method::GICP:return "GICP";case Method::VGICP:return "VGICP";case Method::PCL_GICP:return "PCL_GICP";}
  throw std::invalid_argument("未知配准方法");
}
struct PointIndex::Impl {
  explicit Impl(const Points& p):cloud(std::make_shared<small_gicp::PointCloud>(p)) {
    if(!p.empty()) tree=std::make_unique<small_gicp::KdTree<small_gicp::PointCloud>>(cloud);
  }
  small_gicp::PointCloud::Ptr cloud;
  std::unique_ptr<small_gicp::KdTree<small_gicp::PointCloud>> tree;
};
PointIndex::PointIndex(const Points& p):impl_(std::make_unique<Impl>(p)) {}
PointIndex::~PointIndex()=default;
PointIndex::PointIndex(PointIndex&&) noexcept=default;
PointIndex& PointIndex::operator=(PointIndex&&) noexcept=default;
bool PointIndex::nearest(const Eigen::Vector3d& q,std::size_t& i,double& d) const {
  if(!impl_->tree || !q.allFinite()) return false;
  return impl_->tree->nearest_neighbor_search(Eigen::Vector4d(q.x(),q.y(),q.z(),1),&i,&d)>0;
}
bool healthy_transform(const Transform& t,const Config& c,std::string* reason) {
  std::string error;
  if(!t.matrix().allFinite()) error="NONFINITE_TRANSFORM";
  else if((t.matrix().row(3)-Eigen::RowVector4d(0,0,0,1)).norm()>c.registration.homogeneous_tolerance) error="INVALID_HOMOGENEOUS_ROW";
  else if(std::abs(t.linear().determinant()-1)>c.registration.determinant_tolerance) error="INVALID_DETERMINANT";
  else if((t.linear().transpose()*t.linear()-Eigen::Matrix3d::Identity()).norm()>c.registration.orthogonality_tolerance) error="NON_ORTHOGONAL_ROTATION";
  if(reason) *reason=error;
  return error.empty();
}
double rotation_degrees(const Eigen::Matrix3d& r) {
  return std::acos(std::max(-1.0,std::min(1.0,(r.trace()-1)*.5)))*180/std::acos(-1.0);
}
Points preprocess(const Points& input,const Config& c) {
  if(!std::isfinite(c.registration.downsample_m) || c.registration.downsample_m<=0)
    throw std::invalid_argument("无效降采样参数");
  struct Voxel {Eigen::Vector3d sum=Eigen::Vector3d::Zero(); std::size_t count=0;};
  std::map<std::array<std::int64_t,3>,Voxel> voxels;
  for(const auto& p:input) {
    if(!p.allFinite() || p.cwiseAbs().maxCoeff()>c.registration.max_local_coordinate_m)
      throw std::invalid_argument("非法或非局部点坐标");
    std::array<std::int64_t,3> key;
    for(int a=0;a<3;++a) key[a]=static_cast<std::int64_t>(std::floor(double(p[a])/c.registration.downsample_m));
    auto& v=voxels[key]; v.sum+=p.cast<double>(); ++v.count;
  }
  Points out; out.reserve(voxels.size());
  for(const auto& kv:voxels) out.push_back((kv.second.sum/double(kv.second.count)).cast<float>());
  return out;
}
RegistrationResult validate_result(RegistrationResult r,const RegistrationRequest& q,const Config& c,const PointIndex& target) {
  r.valid=false; r.quality_available=false;
  std::string reason;
  if(!healthy_transform(r.T_target_source,c,&reason) ||
     (r.hessian_available && !r.H.allFinite()) ||
     (r.objective_available && !std::isfinite(r.raw_objective)) ||
     !std::isfinite(r.rmse) || !std::isfinite(r.overlap_ratio) || !std::isfinite(r.fitness_score) ||
     !std::isfinite(r.elapsed_ms) || r.elapsed_ms<0) {
    r.mathematical_failure=true;
    r.failure_reason=reason.empty()?"NONFINITE_BACKEND_METRIC":reason;
    r.T_target_source=Transform::Identity(); r.H.setZero(); r.hessian_available=false;
    r.rmse=0; r.fitness_score=0; r.overlap_ratio=0; r.elapsed_ms=0;
    r.raw_objective=0;r.objective_available=false;
    return r;
  }
  double sum=0; r.inliers=0;
  for(const auto& p:q.source) {
    std::size_t i=0; double d=0;
    if(!p.allFinite()) {r.failure_reason="NONFINITE_SOURCE";return r;}
    if(target.nearest(r.T_target_source*p.cast<double>(),i,d) && d<=c.registration.quality_distance_m*c.registration.quality_distance_m) {sum+=d;++r.inliers;}
  }
  r.source_points=q.source.size(); r.target_points=q.target?q.target->points.size():0;
  r.overlap_ratio=q.source.empty()?0:double(r.inliers)/double(q.source.size());
  if(r.inliers) {r.fitness_score=sum/double(r.inliers);r.rmse=std::sqrt(r.fitness_score);r.quality_available=true;}
  if(!std::isfinite(r.rmse) || !std::isfinite(r.fitness_score)) {r.mathematical_failure=true;r.failure_reason="NONFINITE_QUALITY";return r;}
  if(!r.converged) r.failure_reason="NOT_CONVERGED";
  else if(r.inliers<std::size_t(c.registration.min_points)) r.failure_reason="INSUFFICIENT_INLIERS";
  else if(r.overlap_ratio<c.registration.min_overlap) r.failure_reason="LOW_OVERLAP";
  else if(r.rmse>c.registration.max_rmse_m) r.failure_reason="HIGH_RESIDUAL";
  else {r.valid=true;r.failure_reason.clear();}
  return r;
}
}} // namespace ship::v14
