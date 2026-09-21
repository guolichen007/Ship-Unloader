#include <ship_perception/structure/offline_ship_frame.hpp>
#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <random>
#include <queue>
#include <set>
#include <Eigen/Eigenvalues>

namespace ship { namespace v15 {
namespace {
// Read-only diagnostics: no score or acceptance decision consumes these fields.
struct DeckAudit {
  std::size_t id, samples, cloud_size; Plane plane;
  const char* reason="NORMAL_MODE_UNSUPPORTED";
  static constexpr double unknown=std::numeric_limits<double>::quiet_NaN();
  double support_area=unknown, roi_area=unknown, support_score=unknown, normal_score=unknown;
  double opening_score=unknown, enclosure=unknown, interior=unknown, continuity=unknown, score=unknown;
  std::size_t openings_before=0, openings_after=0, exterior_openings=0;
  static std::string number(double value){if(!std::isfinite(value))return "null";
    std::ostringstream out;out<<std::setprecision(17)<<value;return out.str();}
  ~DeckAudit(){if(!std::getenv("SHIP_V15_DIAGNOSTICS"))return;
    std::cerr<<std::setprecision(17)<<"DECK_AUDIT {\"candidate_id\":"<<id
      <<",\"normal\":["<<plane.normal.x()<<","<<plane.normal.y()<<","<<plane.normal.z()<<"]"
      <<",\"offset\":"<<plane.offset<<",\"support_count\":"<<samples
      <<",\"support_ratio\":"<<double(samples)/cloud_size
      <<",\"support_area_m2\":"<<number(support_area)<<",\"roi_area_m2\":"<<number(roi_area)
      <<",\"support_score\":"<<number(support_score)<<",\"normal_score\":"<<number(normal_score)
      <<",\"opening_score\":"<<number(opening_score)<<",\"enclosure\":"<<number(enclosure)
      <<",\"interior_parallel_fraction\":"<<number(interior)<<",\"continuity_score\":"<<number(continuity)
      <<",\"openings_before_boundary_filter\":"<<openings_before
      <<",\"openings_after_boundary_filter\":"<<openings_after
      <<",\"exterior_connected_openings\":"<<exterior_openings
      <<",\"vertical_support\":null,\"background_penalty\":null"
      <<",\"total_score\":"<<number(score)<<",\"reason\":\""<<reason<<"\"}\n";
  }
};
}
DeckPlaneCandidate detect_deck(const Points& raw,const Config& c,bool fixed_ship_frame) {
  DeckPlaneCandidate best;
  if(raw.size()<std::size_t(c.geometry.min_plane_points))return best;
  const auto p=voxelize(raw,c.geometry.candidate_voxel_m,c.geometry.max_local_extent_m);
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
  std::vector<Eigen::Vector3d> up_proposals{Eigen::Vector3d::UnitX(),Eigen::Vector3d::UnitY(),Eigen::Vector3d::UnitZ()};
  for(int axis=0;axis<3;++axis)up_proposals.push_back(axes.eigenvectors().col(axis));
  std::map<std::array<int,3>,std::pair<Eigen::Vector3d,std::size_t>> normal_modes;
  const double angular_bin=std::sin(c.frame.normal_refine_deg*std::acos(-1.)/180.);
  for(const auto& normal:ns)if(normal.valid){auto n=normal.direction;Eigen::Index major;n.cwiseAbs().maxCoeff(&major);if(n[major]<0)n=-n;
    std::array<int,3> key;for(int i=0;i<3;++i)key[i]=int(std::round(n[i]/angular_bin));auto it=normal_modes.find(key);if(it==normal_modes.end())normal_modes.emplace(key,std::make_pair(n,std::size_t(1)));else{it->second.first+=n;++it->second.second;}}
  std::vector<std::pair<std::size_t,Eigen::Vector3d>> ranked_modes;for(auto& kv:normal_modes)ranked_modes.emplace_back(kv.second.second,kv.second.first.normalized());
  std::stable_sort(ranked_modes.begin(),ranked_modes.end(),[](const auto& a,const auto& b){return a.first>b.first;});
  for(std::size_t i=0;i<std::min(ranked_modes.size(),std::size_t(c.frame.max_up_hypotheses));++i)up_proposals.push_back(ranked_modes[i].second);
  for(auto up:up_proposals){
    Eigen::Index major;up.cwiseAbs().maxCoeff(&major);if(up[major]<0)up=-up;
    std::map<int,std::vector<std::size_t>> levels;
    for(std::size_t i=0;i<p.size();++i)if(ns[i].valid && std::abs(ns[i].direction.dot(up))>cosine)levels[int(std::floor(up.dot(p[i].cast<double>())/c.geometry.coarse_voxel_m))].push_back(i);
    for(auto& level:levels){
      if(level.second.size()<std::size_t(c.geometry.min_plane_points))continue;
      std::vector<std::size_t> best_ids;
      for(int trial=0;trial<c.geometry.ransac_iterations;++trial){
        const auto a=level.second[rng()%level.second.size()],b=level.second[rng()%level.second.size()],d=level.second[rng()%level.second.size()];
        Eigen::Vector3d n=(p[b]-p[a]).cast<double>().cross((p[d]-p[a]).cast<double>());
        if(n.norm()<c.geometry.candidate_voxel_m)continue;n.normalize();if(n.dot(up)<0)n=-n;
        if(n.dot(up)<cosine)continue;const double offset=-n.dot(p[a].cast<double>());
        std::vector<std::size_t> ids;for(auto i:level.second)if(std::abs(n.dot(p[i].cast<double>())+offset)<c.geometry.plane_inlier_m)ids.push_back(i);
        if(ids.size()>best_ids.size())best_ids=std::move(ids);
      }
      auto fit=fit_plane(p,best_ids,up,c);
      // Regrow on the original candidate cloud: height bins propose a plane,
      // but must not clip an inclined deck at an arbitrary bin boundary.
      for(int pass=0;fit.valid&&pass<3;++pass){std::vector<std::size_t> ids;
        for(std::size_t i=0;i<p.size();++i)if(ns[i].valid&&std::abs(ns[i].direction.dot(fit.plane.normal))>cosine&&std::abs(fit.plane.distance(p[i].cast<double>()))<c.geometry.plane_inlier_m)ids.push_back(i);
        fit=fit_plane(p,ids,up,c);
      }
      if(fit.valid&&fit.plane.normal.dot(up)>cosine)stratified.push_back(std::move(fit));
    }
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
  // Merge duplicate fits by shared physical samples, not by a guessed world-Z
  // interval. Intersecting but distinct surfaces share only a thin strip.
  std::stable_sort(hypotheses.begin(),hypotheses.end(),[](const auto& a,const auto& b){return a.inliers.size()>b.inliers.size();});
  std::vector<PlaneFit> distinct;
  for(auto& hypothesis:hypotheses){bool duplicate=false;
    std::sort(hypothesis.inliers.begin(),hypothesis.inliers.end());
    for(const auto& prior:distinct){
      if(prior.plane.normal.dot(hypothesis.plane.normal)<std::cos(c.frame.normal_refine_deg*std::acos(-1.)/180.))continue;
      std::size_t i=0,j=0,shared=0;while(i<prior.inliers.size()&&j<hypothesis.inliers.size()){
        if(prior.inliers[i]==hypothesis.inliers[j]){++shared;++i;++j;}else if(prior.inliers[i]<hypothesis.inliers[j])++i;else ++j;
      }
      if(double(shared)/std::min(prior.inliers.size(),hypothesis.inliers.size())>=c.frame.same_plane_overlap_min){duplicate=true;break;}
    }
    if(!duplicate)distinct.push_back(std::move(hypothesis));
  }
  hypotheses=std::move(distinct);
  // Proposal tolerance is intentionally broad. Reusing its 10-degree cone
  // for scoring makes a sloped cargo plane look as globally coherent as the
  // Deck mode. Score actual normal consensus with the configured mode angle.
  const double consensus_cosine=std::cos(c.roi.deck_normal_mode_angle_deg*std::acos(-1.)/180.);
  std::size_t max_orientation_support=1;
  for(const auto& fit:hypotheses){std::size_t count=0;for(const auto& n:ns)if(n.valid&&std::abs(n.direction.dot(fit.plane.normal))>consensus_cosine)++count;max_orientation_support=std::max(max_orientation_support,count);}
  std::vector<DeckPlaneCandidate> scored;
  std::size_t audit_id=0;
  for(auto fit:hypotheses)for(double sign:{1.,-1.}) {
      fit.plane.normal*=sign;fit.plane.offset*=sign;
      DeckAudit audit{audit_id++,fit.inliers.size(),p.size(),fit.plane};
      // A production cloud already has the runtime Ship Frame up convention.
      // Offline XYZ has no such promise and must retain both sign hypotheses.
      if(fixed_ship_frame&&fit.plane.normal.z()<=0){audit.reason="FIXED_FRAME_UP_CONFLICT";continue;}
      bool normal_mode_supported=false;
      for(const auto& mode:ranked_modes){
        if(mode.first<ranked_modes.front().first*c.roi.deck_normal_mode_support_ratio)break;
        if(std::abs(mode.second.dot(fit.plane.normal))>=std::cos(c.roi.deck_normal_mode_angle_deg*std::acos(-1.)/180.))normal_mode_supported=true;
      }
      if(!normal_mode_supported)continue;
      audit.reason="INSUFFICIENT_CONNECTED_DECK_SUPPORT";
      Points support;for(auto i:fit.inliers)support.push_back(p[i]);
      const auto frame=plane_frame(fit.plane,support);Points aligned;aligned.reserve(p.size());for(const auto& v:p)aligned.push_back((frame*v.cast<double>()).cast<float>());
      // A coaming top consisting only of sparse line returns can have a large
      // convex hull and a perfect plane fit. It is not a measured Deck surface.
      // Require at least one locally filled 2D patch on this actual plane;
      // neither convex-hull area nor opening area can supply missing samples.
      std::set<CellKey> plane_cells;const double patch_cell=c.geometry.candidate_voxel_m;
      for(const auto& q:aligned)if(std::abs(q.z())<=c.geometry.plane_inlier_m)
        plane_cells.insert({int(std::floor(q.x()/patch_cell)),int(std::floor(q.y()/patch_cell))});
      const int patch_width=std::max(2,int(std::ceil(c.roi.deck_patch_width_m/patch_cell)));
      bool area_supported=false;
      for(const auto& key:plane_cells){std::size_t count=0;
        for(int x=0;x<patch_width;++x)for(int y=0;y<patch_width;++y)count+=plane_cells.count({key[0]+x,key[1]+y});
        if(double(count)/(patch_width*patch_width)>=c.roi.deck_patch_fill_min){area_supported=true;break;}
      }
      if(!area_supported){audit.reason="NO_MEASURED_TWO_DIMENSIONAL_DECK_PATCH";continue;}
      const auto full_grid=height_grid(aligned,c);
      std::set<CellKey> supported;
      for(std::size_t i=0;i<aligned.size();++i)if(std::abs(aligned[i].z())<c.roi.support_band_m && ns[i].valid && std::abs(ns[i].direction.dot(fit.plane.normal))>cosine){
        supported.insert({int(std::floor((aligned[i].x()-c.geometry.grid_phase_x_m)/c.geometry.coarse_voxel_m)),int(std::floor((aligned[i].y()-c.geometry.grid_phase_y_m)/c.geometry.coarse_voxel_m))});
      }
      std::vector<CellKey> largest;
      const int connection=int(std::ceil(c.roi.support_connectivity_m/c.geometry.coarse_voxel_m));
      while(!supported.empty()){
        std::vector<CellKey> component;std::queue<CellKey> pending;pending.push(*supported.begin());supported.erase(supported.begin());
        while(!pending.empty()){const auto key=pending.front();pending.pop();component.push_back(key);
          for(int dx=-connection;dx<=connection;++dx)for(int dy=-connection;dy<=connection;++dy){CellKey next{key[0]+dx,key[1]+dy};if(supported.erase(next))pending.push(next);}}
        if(component.size()>largest.size())largest=std::move(component);
      }
      if(largest.size()<std::size_t(c.roi.min_deck_support_cells))continue;
      audit.support_area=largest.size()*c.geometry.coarse_voxel_m*c.geometry.coarse_voxel_m;
      // A support cell occupies an area, not just its centre. A centre hull
      // cuts off its own boundary cells (and their raw samples), creating an
      // artificial unobserved slit around a one-cell-wide Deck rim.
      // Cell extent is proposal geometry only; final edges still use raw 3D.
      std::vector<Eigen::Vector2d> support_xy;
      const double half_cell=c.geometry.coarse_voxel_m*.5;
      for(const auto& key:largest)for(double dx:{-half_cell,half_cell})for(double dy:{-half_cell,half_cell})
        support_xy.push_back(full_grid.at(key).center+Eigen::Vector2d(dx,dy));
      const auto hull=convex_support_hull(std::move(support_xy));
      audit.reason="INSUFFICIENT_SUPPORT_HULL";audit.roi_area=polygon_area(hull);
      if(hull.size()<3||polygon_area(hull)<c.roi.opening_min_area_m2)continue;
      Eigen::Vector2d hull_lo=hull.front(),hull_hi=hull.front();
      for(const auto& q:hull){hull_lo=hull_lo.cwiseMin(q);hull_hi=hull_hi.cwiseMax(q);}
      audit.reason="INSUFFICIENT_DECK_SPAN";
      if((hull_hi-hull_lo).minCoeff()<c.roi.min_deck_span_m)continue;
      // The hull bounds a proposal ROI only. It never creates an observed edge.
      // Exterior low returns and cargo outside the supported deck domain must
      // not determine the sign or score of the deck hypothesis.
      HeightGrid grid;for(const auto& kv:full_grid)if(inside(kv.second.center,hull))grid.insert(kv);
      auto openings=find_openings(grid,0,c);
      audit.openings_before=openings.size();for(const auto& o:openings)if(o.touches_scan_boundary)++audit.exterior_openings;
      // An ROI cut cannot manufacture an enclosed cavity. With unknown offline
      // frame, a lower region reaching unobserved ROI space is insufficient to
      // establish deck/up. The fixed-frame path can retain partial structures.
      if(!fixed_ship_frame)openings.erase(std::remove_if(openings.begin(),openings.end(),[](const auto& opening){return opening.touches_scan_boundary;}),openings.end());
      double scoring_resolution=c.geometry.coarse_voxel_m;
      if(openings.empty()&&!fixed_ship_frame){
        // A narrow but observed rim may share the outermost L0 cell with
        // lower returns. Resolve that aliasing using actual L1 samples before
        // declaring a scan cut; absent samples stay UNKNOWN at both levels.
        Config fine=c;fine.geometry.coarse_voxel_m=c.geometry.candidate_voxel_m;
        const HeightGrid refined=height_grid(aligned,fine);
        auto retry=find_openings(refined,0,fine);
        retry.erase(std::remove_if(retry.begin(),retry.end(),[&](const auto& opening){
          if(opening.touches_scan_boundary)return true;
          for(const auto& key:opening.cells)if(!inside(refined.at(key).center,hull))return true;
          return false;
        }),retry.end());
        if(!retry.empty()){grid=std::move(refined);openings=std::move(retry);scoring_resolution=fine.geometry.coarse_voxel_m;}
      }
      audit.openings_after=openings.size();audit.reason="NO_ENCLOSED_LOWER_RETURN_COMPONENT";
      if(openings.empty())continue;
      std::size_t support_cells=0;for(const auto& kv:grid)if(std::abs(kv.second.median)<c.roi.support_band_m)++support_cells;
      audit.reason="INSUFFICIENT_MEDIAN_DECK_SUPPORT";
      if(support_cells*scoring_resolution*scoring_resolution<c.roi.min_deck_support_cells*c.geometry.coarse_voxel_m*c.geometry.coarse_voxel_m)continue;
      double enclosure=0,area=0;for(const auto& o:openings){enclosure+=o.enclosure*o.cells.size();area+=o.cells.size();}enclosure/=std::max(1.,area);
      audit.enclosure=enclosure;audit.reason="INSUFFICIENT_ENCLOSURE";
      if(enclosure<(fixed_ship_frame?c.roi.min_enclosure_ratio:c.roi.deck_min_enclosure_ratio))continue;
      std::size_t interior_count=0,interior_parallel=0;
      const double interior_cosine=std::cos(c.frame.normal_refine_deg*std::acos(-1.)/180.);
      for(const auto& opening:openings)for(const auto& key:opening.cells)for(auto id:grid.at(key).ids){
        ++interior_count;if(ns[id].valid&&std::abs(ns[id].direction.dot(fit.plane.normal))>interior_cosine)++interior_parallel;
      }
      const double interior_fraction=double(interior_parallel)/std::max(std::size_t(1),interior_count);
      // Every term is bounded. Opening and enclosure are necessary, not just weights.
      const double support_score=std::min(1.,double(support_cells)/std::max(1.,area));
      std::size_t orientation_support=0;for(const auto& n:ns)if(n.valid&&std::abs(n.direction.dot(fit.plane.normal))>consensus_cosine)++orientation_support;
      const double orientation_ratio=double(orientation_support)/max_orientation_support;
      const double normal_score=orientation_ratio*std::max(0.,1-fit.p95/c.geometry.plane_inlier_m);
      const double continuity=std::min(1.,double(support_cells)/std::max(1.,double(grid.size())*.25));
      const double opening_score=std::min(1.,area/std::max(1.,double(grid.size())));
      const double score=(c.roi.support_weight*support_score+c.roi.normal_weight*normal_score+c.roi.continuity_weight*continuity+c.roi.opening_weight*opening_score+c.roi.enclosure_weight*enclosure)*orientation_ratio*enclosure*opening_score;
      audit.support_score=support_score;audit.normal_score=normal_score;audit.opening_score=opening_score;
      audit.interior=interior_fraction;audit.continuity=continuity;audit.score=score;audit.reason="ELIGIBLE_FOR_RANKING";
      if(std::getenv("SHIP_V15_DIAGNOSTICS"))std::cerr<<"DECK_CANDIDATE score="<<score<<" normal="<<fit.plane.normal.transpose()<<" offset="<<fit.plane.offset<<" support="<<fit.inliers.size()<<" enclosure="<<enclosure<<" area="<<area<<" interior="<<interior_fraction<<"\n";
      {
        best.plane=fit.plane;best.support_count=fit.inliers.size();best.support_ratio=double(fit.inliers.size())/p.size();best.residual_p50_m=fit.p50;best.residual_p95_m=fit.p95;best.quality_score=score;best.valid=true;
        // Preserve the connected support ROI actually used to qualify this
        // Deck. An AABB over all coplanar inliers can reintroduce the wharf or
        // detached background after candidate selection.
        best.support_region.clear();for(const auto& v:hull)best.support_region.push_back(frame.inverse()*Eigen::Vector3d(v.x(),v.y(),0));
        scored.push_back(best);
      }
  }
  const auto selected=select_deck_candidate(std::move(scored),c);
  if(std::getenv("SHIP_V15_DIAGNOSTICS"))std::cerr<<std::setprecision(17)
    <<"DECK_SELECTION {\"candidate_count\":"<<audit_id<<",\"valid\":"<<(selected.valid?"true":"false")
    <<",\"normal\":["<<selected.plane.normal.x()<<","<<selected.plane.normal.y()<<","<<selected.plane.normal.z()
    <<"],\"offset\":"<<selected.plane.offset<<",\"score\":"<<selected.quality_score<<"}\n";
  return selected;
}
DeckPlaneCandidate select_deck_candidate(std::vector<DeckPlaneCandidate> candidates,const Config& c){
  candidates.erase(std::remove_if(candidates.begin(),candidates.end(),[](const auto& x){return !x.valid||!std::isfinite(x.quality_score);}),candidates.end());
  if(candidates.empty())return {};
  std::stable_sort(candidates.begin(),candidates.end(),[](const auto& a,const auto& b){return a.quality_score>b.quality_score;});
  auto best=candidates.front();
  for(std::size_t i=1;i<candidates.size();++i){const auto& alternative=candidates[i];
    const double relative_gap=(best.quality_score-alternative.quality_score)/std::max(1e-12,best.quality_score);
    if(relative_gap>=c.frame.candidate_score_gap)break;
    bool equivalent=best.plane.normal.dot(alternative.plane.normal)>std::cos(c.frame.normal_refine_deg*std::acos(-1.)/180.);
    for(const auto& q:best.support_region)if(std::abs(alternative.plane.distance(q))>c.roi.support_band_m)equivalent=false;
    for(const auto& q:alternative.support_region)if(std::abs(best.plane.distance(q))>c.roi.support_band_m)equivalent=false;
    if(!equivalent){best.valid=false;break;}
  }
  return best;
}
FrameResult OfflineShipFrameProvider::resolve(const Points& p) const {
  FrameResult out;
  if(p.empty()||p.size()>std::size_t(config_.geometry.max_input_points)){out.reason="INVALID_INPUT_SIZE";return out;}
  for(const auto& v:p)if(!v.allFinite()||v.cwiseAbs().maxCoeff()>config_.geometry.max_local_extent_m){out.reason="INVALID_LOCAL_POINT";return out;}
  out.deck=detect_deck(p,config_);if(!out.deck.valid){out.reason="FRAME_UNRESOLVED";return out;}
  Points support;for(const auto& v:p)if(std::abs(out.deck.plane.distance(v.cast<double>()))<=config_.geometry.plane_inlier_m)support.push_back(v);
  const auto t=plane_frame(out.deck.plane,support);if(!healthy(t)){out.reason="INVALID_FRAME";return out;}
  Eigen::Vector2d mean=Eigen::Vector2d::Zero();for(const auto& q:support)mean+=(t*q.cast<double>()).head<2>();mean/=double(support.size());
  Eigen::Matrix2d axis_cov=Eigen::Matrix2d::Zero();for(const auto& q:support){const Eigen::Vector2d d=(t*q.cast<double>()).head<2>()-mean;axis_cov+=d*d.transpose();}
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> axes(axis_cov);
  if(axes.info()!=Eigen::Success||axes.eigenvalues()[0]<=0||std::sqrt(axes.eigenvalues()[1]/axes.eigenvalues()[0])<config_.frame.longitudinal_ratio_min){out.reason="LONG_AXIS_UNRESOLVED";return out;}
  const double angle=std::acos(std::clamp(out.deck.plane.normal.z(),-1.,1.))*180/std::acos(-1.);
  out.cloud.input_alignment=angle<=config_.frame.aligned_angle_max_deg?InputAlignment::ALIGNED:InputAlignment::RAW;
  out.cloud.T_B_input=t;out.cloud.offline=true;out.cloud.points.reserve(p.size());for(const auto& v:p)out.cloud.points.push_back((t*v.cast<double>()).cast<float>());
  out.deck.plane={Eigen::Vector3d::UnitZ(),0};for(auto& v:out.deck.support_region)v=t*v;
  out.valid=true;return out;
}
}} // namespace ship::v15
