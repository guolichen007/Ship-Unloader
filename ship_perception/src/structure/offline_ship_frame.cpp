#include <ship_perception/structure/offline_ship_frame.hpp>
#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <cstdlib>
#include <iostream>
#include <random>
#include <Eigen/Eigenvalues>

namespace ship { namespace v15 {
DeckPlaneCandidate detect_deck(const Points& raw,const Config& c) {
  DeckPlaneCandidate best;
  if(raw.size()<std::size_t(c.geometry.min_plane_points))return best;
  const auto p=voxelize(raw,c.geometry.candidate_voxel_m);
  if(p.size()>std::size_t(c.geometry.max_candidate_points))throw std::runtime_error("CANDIDATE_CAPACITY_EXCEEDED");
  const auto ns=normals(p,c);
  const double cosine=std::cos(c.frame.up_cluster_deg*std::acos(-1.)/180.);
  std::vector<PlaneFit> hypotheses;std::mt19937 rng(42);
  // Stratify plane proposals over spatial levels so a sparse deck is not lost
  // merely because cargo/floor points dominate random sampling.
  Eigen::Vector3d mean=Eigen::Vector3d::Zero();for(const auto& q:p)mean+=q.cast<double>();mean/=double(p.size());
  Eigen::Matrix3d cov=Eigen::Matrix3d::Zero();for(const auto& q:p){const Eigen::Vector3d d=q.cast<double>()-mean;cov+=d*d.transpose();}
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> axes(cov);
  std::vector<PlaneFit> stratified;
  for(int axis=0;axis<3;++axis){
    Eigen::Vector3d up=axes.eigenvectors().col(axis),sum=Eigen::Vector3d::Zero();
    for(const auto& n:ns)if(n.valid && std::abs(n.direction.dot(up))>cosine)sum+=n.direction*(n.direction.dot(up)>0?1.:-1.);
    if(sum.norm()<1e-9)continue;up=sum.normalized();Eigen::Index major;up.cwiseAbs().maxCoeff(&major);if(up[major]<0)up=-up;
    std::map<int,std::vector<std::size_t>> levels;
    for(std::size_t i=0;i<p.size();++i)if(ns[i].valid && std::abs(ns[i].direction.dot(up))>cosine)levels[int(std::floor(up.dot(p[i].cast<double>())/c.geometry.coarse_voxel_m))].push_back(i);
    for(auto& level:levels){auto fit=fit_plane(p,level.second,up,c);if(fit.valid)stratified.push_back(std::move(fit));}
  }
  for(int trial=0;trial<c.geometry.ransac_iterations;++trial){
    const std::size_t a=rng()%p.size(),b=rng()%p.size(),d=rng()%p.size();
    Eigen::Vector3d up=(p[b]-p[a]).cast<double>().cross((p[d]-p[a]).cast<double>());
    if(up.norm()<c.geometry.candidate_voxel_m)continue;up.normalize();
    Eigen::Index major;up.cwiseAbs().maxCoeff(&major);if(up[major]<0)up=-up;
    double offset=-up.dot(p[a].cast<double>());std::vector<std::size_t> ids;
    for(std::size_t i=0;i<p.size();++i)if(ns[i].valid && std::abs(ns[i].direction.dot(up))>cosine && std::abs(up.dot(p[i].cast<double>())+offset)<c.geometry.plane_inlier_m)ids.push_back(i);
    if(ids.size()<std::size_t(c.geometry.min_plane_points))continue;
    auto fit=fit_plane(p,ids,up,c);if(!fit.valid)continue;
    bool duplicate=false;for(auto& h:hypotheses)if(h.plane.normal.dot(fit.plane.normal)>cosine && std::abs(h.plane.offset-fit.plane.offset)<c.roi.support_band_m){if(fit.inliers.size()>h.inliers.size())h=fit;duplicate=true;break;}
    if(!duplicate)hypotheses.push_back(std::move(fit));
  }
  std::stable_sort(hypotheses.begin(),hypotheses.end(),[](const PlaneFit& a,const PlaneFit& b){return a.inliers.size()>b.inliers.size();});
  if(hypotheses.size()>std::size_t(c.geometry.max_plane_hypotheses))hypotheses.resize(c.geometry.max_plane_hypotheses);
  hypotheses.insert(hypotheses.end(),stratified.begin(),stratified.end());
  double second_score=0;Plane second_plane;
  for(auto fit:hypotheses)for(double sign:{1.,-1.}) {
      fit.plane.normal*=sign;fit.plane.offset*=sign;
      Points support;for(auto i:fit.inliers)support.push_back(p[i]);
      const auto frame=plane_frame(fit.plane,support);Points aligned;aligned.reserve(p.size());for(const auto& v:p)aligned.push_back((frame*v.cast<double>()).cast<float>());
      const auto grid=height_grid(aligned,c);auto openings=find_openings(grid,0,c);if(openings.empty())continue;
      std::size_t support_cells=0;for(const auto& kv:grid)if(std::abs(kv.second.median)<c.roi.support_band_m)++support_cells;
      if(support_cells<std::size_t(c.roi.min_deck_support_cells))continue;
      double enclosure=0,area=0;for(const auto& o:openings){enclosure=std::max(enclosure,o.enclosure);area+=o.cells.size();}
      // Every term is bounded. Opening and enclosure are necessary, not just weights.
      const double support_score=std::min(1.,double(support_cells)/std::max(1.,area));
      const double normal_score=std::max(0.,1-fit.p95/c.geometry.plane_inlier_m);
      const double continuity=std::min(1.,double(support_cells)/std::max(1.,double(grid.size())*.25));
      const double opening_score=std::min(1.,area/(double(support_cells)+1));
      std::size_t above=0,below=0;for(const auto& q:aligned){if(q.z()>c.roi.opening_drop_m)++above;if(q.z()<-c.roi.opening_drop_m)++below;}
      const double cavity_direction=double(below)/std::max(1.,double(above+below));
      const double score=(c.roi.support_weight*support_score+c.roi.normal_weight*normal_score+c.roi.continuity_weight*continuity+c.roi.opening_weight*opening_score+c.roi.enclosure_weight*enclosure)*cavity_direction;
      if(std::getenv("SHIP_V15_DIAGNOSTICS"))std::cerr<<"DECK_CANDIDATE score="<<score<<" normal="<<fit.plane.normal.transpose()<<" offset="<<fit.plane.offset<<" support="<<fit.inliers.size()<<" enclosure="<<enclosure<<" area="<<area<<"\n";
      if(score>best.quality_score) {
        second_score=best.quality_score;second_plane=best.plane;
        best.plane=fit.plane;best.support_count=fit.inliers.size();best.support_ratio=double(fit.inliers.size())/p.size();best.residual_p50_m=fit.p50;best.residual_p95_m=fit.p95;best.quality_score=score;best.valid=true;
        Eigen::Vector2d lo=Eigen::Vector2d::Constant(1e20),hi=-lo;
        for(auto id:fit.inliers){auto v=frame*p[id].cast<double>();lo=lo.cwiseMin(v.head<2>());hi=hi.cwiseMax(v.head<2>());}
        best.support_region.clear();for(const Eigen::Vector2d v:{lo,Eigen::Vector2d(hi.x(),lo.y()),hi,Eigen::Vector2d(lo.x(),hi.y())})best.support_region.push_back(frame.inverse()*Eigen::Vector3d(v.x(),v.y(),0));
      }else if(score>second_score && std::abs(fit.plane.offset-best.plane.offset)>c.roi.support_band_m){second_score=score;second_plane=fit.plane;}
  }
  if(best.valid && second_score>0 && best.quality_score-second_score<c.frame.candidate_score_gap &&
     (best.plane.normal.dot(second_plane.normal)<cosine || std::abs(best.plane.offset-second_plane.offset)>c.roi.opening_drop_m))best.valid=false;
  return best;
}
FrameResult OfflineShipFrameProvider::resolve(const Points& p) const {
  FrameResult out;
  if(p.empty()||p.size()>std::size_t(config_.geometry.max_input_points)){out.reason="INVALID_INPUT_SIZE";return out;}
  for(const auto& v:p)if(!v.allFinite()||v.cwiseAbs().maxCoeff()>config_.geometry.max_local_extent_m){out.reason="INVALID_LOCAL_POINT";return out;}
  out.deck=detect_deck(p,config_);if(!out.deck.valid){out.reason="FRAME_UNRESOLVED";return out;}
  Points support;for(const auto& v:p)if(std::abs(out.deck.plane.distance(v.cast<double>()))<=config_.geometry.plane_inlier_m)support.push_back(v);
  const auto t=plane_frame(out.deck.plane,support);if(!healthy(t)){out.reason="INVALID_FRAME";return out;}
  const double angle=std::acos(std::clamp(out.deck.plane.normal.z(),-1.,1.))*180/std::acos(-1.);
  out.cloud.input_alignment=angle<=config_.frame.aligned_angle_max_deg?InputAlignment::ALIGNED:InputAlignment::RAW;
  out.cloud.T_B_input=t;out.cloud.offline=true;out.cloud.points.reserve(p.size());for(const auto& v:p)out.cloud.points.push_back((t*v.cast<double>()).cast<float>());
  out.deck.plane={Eigen::Vector3d::UnitZ(),0};for(auto& v:out.deck.support_region)v=t*v;
  out.valid=true;return out;
}
}} // namespace ship::v15
