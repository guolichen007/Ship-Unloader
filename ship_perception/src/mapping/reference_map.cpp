#include <ship_perception/mapping/reference_map.hpp>
#include <cmath>

namespace ship { namespace v14 {
TrackingReferenceMap::Key TrackingReferenceMap::key(const Eigen::Vector3d& p) const {
  Key k; for(int a=0;a<3;++a) k[a]=static_cast<std::int64_t>(std::floor(p[a]/config_.tracking_map.voxel_m));
  return k;
}
void TrackingReferenceMap::rebuild() {
  Points p; p.reserve(cells_.size());
  for(const auto& kv:cells_) if(kv.second.state!=AnchorState::QUARANTINED) p.push_back(kv.second.point.cast<float>());
  snapshot_=std::make_shared<const TargetSnapshot>(std::move(p),++revision_);
}
void TrackingReferenceMap::initialize(const Points& points) {
  if(snapshot_) throw std::logic_error("地图已经初始化");
  // 先完成全部验证，再修改地图。
  for(const auto& p:points) if(!p.allFinite() || p.cwiseAbs().maxCoeff()>config_.registration.max_local_coordinate_m)
    throw std::invalid_argument("非法地图初始点");
  struct Aggregate {Eigen::Vector3d sum=Eigen::Vector3d::Zero();std::size_t count=0;};
  std::map<Key,Aggregate> voxels;
  for(const auto& p:points) {
    auto& voxel=voxels[key(p.cast<double>())];
    voxel.sum+=p.cast<double>();++voxel.count;
  }
  std::map<Key,Cell> initial;
  // 先聚合完整体素，再按有序 key 截断容量，避免点顺序影响代表点或保留集合。
  for(const auto& kv:voxels) {
    if(initial.size()>=std::size_t(config_.tracking_map.max_reference_voxels)) break;
    Cell cell;cell.point=kv.second.sum/double(kv.second.count);initial.emplace(kv.first,cell);
  }
  if(initial.size()<std::size_t(config_.registration.min_points)) throw std::invalid_argument("初始参考支撑不足");
  cells_=std::move(initial); rebuild();
}
MapStats TrackingReferenceMap::stats() const {
  MapStats s; s.revision=revision_; s.candidate_revision=candidate_revision_; s.candidates=candidates_.size();
  for(const auto& kv:cells_) {
    const auto& c=kv.second;
    if(c.state==AnchorState::QUARANTINED) ++s.quarantined;
    else {++s.active;if(c.state==AnchorState::SUSPECT) ++s.suspect;if(c.stable) ++s.stable;}
  }
  return s;
}
bool TrackingReferenceMap::commit(std::uint64_t frame,const Points& points,const std::vector<Ray>& rays,const RegistrationResult& r) {
  const auto& c=config_.tracking_map;
  if(!snapshot_ || frame<=last_frame_ || !r.valid || !r.converged || r.mathematical_failure ||
     !healthy_transform(r.T_target_source,config_) || (r.hessian_available && !r.H.allFinite()) ||
     !r.quality_available || !std::isfinite(r.rmse) || !std::isfinite(r.overlap_ratio) ||
     r.rmse>c.promotion_max_rmse_m || r.overlap_ratio<c.promotion_min_overlap) return false;
  for(const auto& p:points) if(!p.allFinite() || p.cwiseAbs().maxCoeff()>config_.registration.max_local_coordinate_m) return false;
  for(const auto& ray:rays) if(!ray.origin.allFinite() || !ray.endpoint.allFinite()) return false;
  if(points.empty()) return false;
  PointIndex observed(points);
  bool changed=false;
  for(auto& kv:cells_) {
    auto& cell=kv.second;
    if(cell.state==AnchorState::QUARANTINED) continue;
    std::size_t i=0; double d=0;
    const bool supported=observed.nearest(cell.point,i,d) && d<=c.support_distance_m*c.support_distance_m;
    if(supported) {
      cell.support=cell.last_support+1==frame?cell.support+1:1;
      cell.last_support=frame;
      cell.conflicts.clear(); cell.state=AnchorState::ACTIVE;
      if(cell.support>=std::size_t(c.min_support_frames)) cell.stable=true;
      continue;
    }
    // stable 只表示历史支持，不豁免自由空间冲突；晋升点也必须可被隔离。
    // 没有返回、遮挡、FOV 外均不计冲突；必须有射线穿越并更远返回。
    bool conflict=false;
    for(const auto& ray:rays) {
      const Eigen::Vector3d delta=ray.endpoint-ray.origin;
      const double length=delta.norm(); if(length<=c.ray_margin_m) continue;
      const Eigen::Vector3d direction=delta/length, offset=cell.point-ray.origin;
      const double along=offset.dot(direction);
      if(along>0 && along+c.ray_margin_m<length && (offset-along*direction).norm()<=c.ray_radius_m) {conflict=true;break;}
    }
    while(!cell.conflicts.empty() && cell.conflicts.front()+std::uint64_t(c.conflict_window_frames)<=frame) cell.conflicts.pop_front();
    if(conflict) cell.conflicts.push_back(frame);
    if(cell.conflicts.size()>=std::size_t(c.quarantine_min_conflict_frames)) {
      cell.state=AnchorState::QUARANTINED;changed=true;
    } else if(cell.conflicts.size()>=std::size_t(c.suspect_min_conflict_frames)) cell.state=AnchorState::SUSPECT;
  }
  std::map<Key,std::pair<Eigen::Vector3d,std::size_t>> frame_voxels;
  Eigen::Vector3d minimum=Eigen::Vector3d::Constant(config_.registration.max_local_coordinate_m);
  Eigen::Vector3d maximum=-minimum;
  for(const auto& p:snapshot_->points) {minimum=minimum.cwiseMin(p.cast<double>());maximum=maximum.cwiseMax(p.cast<double>());}
  for(const auto& p:points) {
    if((p.cast<double>().array()<minimum.array()-c.roi_margin_m).any() ||
       (p.cast<double>().array()>maximum.array()+c.roi_margin_m).any()) continue;
    const auto k=key(p.cast<double>()); auto cell=cells_.find(k);
    if(cell!=cells_.end() && cell->second.state!=AnchorState::QUARANTINED) continue;
    auto it=frame_voxels.find(k);
    if(it==frame_voxels.end()) frame_voxels.emplace(k,std::make_pair(p.cast<double>().eval(),std::size_t(1)));
    else {it->second.first+=p.cast<double>();++it->second.second;}
  }
  for(auto it=candidates_.begin();it!=candidates_.end();) {
    if(it->second.last_frame+std::uint64_t(c.candidate_ttl_frames)<frame) it=candidates_.erase(it); else ++it;
  }
  for(const auto& kv:frame_voxels) {
    auto it=candidates_.find(kv.first);
    if(it==candidates_.end()) {
      if(candidates_.size()>=std::size_t(c.max_candidate_voxels)) continue;
      it=candidates_.emplace(kv.first,Candidate{}).first;
    }
    auto& candidate=it->second;
    if(candidate.last_frame+1!=frame) candidate=Candidate{};
    const Eigen::Vector3d p=kv.second.first/double(kv.second.second);
    ++candidate.support; const Eigen::Vector3d delta=p-candidate.mean;
    candidate.mean+=delta/double(candidate.support);
    candidate.m2+=delta.cwiseProduct(p-candidate.mean);candidate.last_frame=frame;
    const double variance=candidate.support>1?candidate.m2.sum()/double(candidate.support-1):0;
    if(candidate.support>=std::size_t(c.min_support_frames) && variance<=c.position_std_max_m*c.position_std_max_m &&
       (cells_.count(kv.first) || cells_.size()<std::size_t(c.max_reference_voxels))) {
      Cell cell;cell.point=candidate.mean;cell.bootstrap=false;cell.stable=true;
      cell.support=candidate.support;cell.last_support=frame;cells_[kv.first]=cell;
      candidates_.erase(it);changed=true;
    }
  }
  last_frame_=frame;++candidate_revision_;
  if(changed) rebuild();
  return true;
}
}} // namespace ship::v14
