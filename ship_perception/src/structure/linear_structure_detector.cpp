#include <ship_perception/structure/linear_structure_detector.hpp>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <numeric>
#include <random>
#include <set>
#include <queue>

namespace ship { namespace v15 { namespace {
struct Line2 {Eigen::Vector2d origin=Eigen::Vector2d::Zero(),direction=Eigen::Vector2d::UnitX();double lo=0,hi=0,residual=0,normal_mad_rad=0,zlo=0,zhi=0;std::vector<std::size_t> ids;};
Line2 refine_line(const std::vector<Eigen::Vector2d>& points,const std::vector<std::size_t>& ids) {
  Line2 line;line.ids=ids;for(auto i:ids)line.origin+=points[i];line.origin/=double(ids.size());
  Eigen::Matrix2d cov=Eigen::Matrix2d::Zero();for(auto i:ids){const Eigen::Vector2d d=points[i]-line.origin;cov+=d*d.transpose();}
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eig(cov);line.direction=eig.eigenvectors().col(1);
  Eigen::Index major;line.direction.cwiseAbs().maxCoeff(&major);if(line.direction[major]<0)line.direction=-line.direction;
  line.lo=1e20;line.hi=-1e20;std::vector<double> residual;
  for(auto i:ids){const Eigen::Vector2d d=points[i]-line.origin;const double s=d.dot(line.direction);line.lo=std::min(line.lo,s);line.hi=std::max(line.hi,s);residual.push_back(std::abs(d.x()*line.direction.y()-d.y()*line.direction.x()));}
  line.residual=quantile(residual,.95);return line;
}
std::vector<Line2> lines(const std::vector<Eigen::Vector2d>& points,double distance,const Config& c,bool shared_corners=false) {
  std::vector<Line2> out;std::vector<std::size_t> remaining(points.size());std::iota(remaining.begin(),remaining.end(),0);std::mt19937 rng(42);
  while(remaining.size()>=std::size_t(c.boundary.line_min_points) && out.size()<std::size_t(c.geometry.max_plane_hypotheses)) {
    std::vector<std::size_t> best;
    for(int trial=0;trial<c.geometry.ransac_iterations;++trial){
      const auto a=remaining[rng()%remaining.size()],b=remaining[rng()%remaining.size()];const Eigen::Vector2d delta=points[b]-points[a];if(delta.norm()<c.boundary.line_min_length_m)continue;
      const Eigen::Vector2d direction=delta.normalized();std::vector<std::size_t> inliers;
      bool repeated=false;if(shared_corners)for(const auto& previous:out){const Eigen::Vector2d d=points[a]-previous.origin;
        if(std::abs(direction.dot(previous.direction))>=std::cos(c.boundary.merge_angle_deg*std::acos(-1.)/180.)&&
           std::abs(d.x()*previous.direction.y()-d.y()*previous.direction.x())<=distance)repeated=true;}
      if(repeated)continue;
      for(auto i:remaining){const Eigen::Vector2d d=points[i]-points[a];if(std::abs(d.x()*direction.y()-d.y()*direction.x())<=distance)inliers.push_back(i);}
      if(inliers.size()>best.size())best=std::move(inliers);
    }
    if(best.size()<std::size_t(c.boundary.line_min_points))break;
    auto line=refine_line(points,best);if(line.hi-line.lo>=c.boundary.line_min_length_m)out.push_back(line);
    // End samples can support two meeting edges. Consuming them all makes a
    // short side disappear after its long neighbours win RANSAC (especially
    // at a concave corner). Keep those samples; repeated-line rejection above
    // prevents selecting the same long side forever.
    std::set<std::size_t> consumed;for(auto i:best){const double s=(points[i]-line.origin).dot(line.direction);
      if(!shared_corners||(s>line.lo+distance&&s<line.hi-distance))consumed.insert(i);}
    if(consumed.empty())break;
    remaining.erase(std::remove_if(remaining.begin(),remaining.end(),[&](std::size_t i){return consumed.count(i)>0;}),remaining.end());
  }return out;
}
std::vector<std::size_t> nearby(const HeightGrid& grid,const Eigen::Vector2d& center,double radius,const Config& c) {
  std::vector<std::size_t> ids;const double s=c.geometry.coarse_voxel_m;
  const int x0=int(std::floor((center.x()-radius-c.geometry.grid_phase_x_m)/s)),x1=int(std::floor((center.x()+radius-c.geometry.grid_phase_x_m)/s));
  const int y0=int(std::floor((center.y()-radius-c.geometry.grid_phase_y_m)/s)),y1=int(std::floor((center.y()+radius-c.geometry.grid_phase_y_m)/s));
  for(int x=x0;x<=x1;++x)for(int y=y0;y<=y1;++y){auto it=grid.find({x,y});if(it!=grid.end())ids.insert(ids.end(),it->second.ids.begin(),it->second.ids.end());}
  return ids;
}
std::vector<Line2> contour_proposals(const Opening& opening,double resolution,const Config& c){
  // Trace oriented cell-edge chains instead of greedily consuming unordered
  // boundary points. Short and concave sides retain their own proposal even
  // when both adjoining long sides have much stronger support. These remain
  // L0/L1 proposals: every output boundary is independently fitted on raw 3D.
  const std::set<CellKey> cells(opening.cells.begin(),opening.cells.end());
  std::multimap<CellKey,CellKey> edges;
  for(const auto& k:cells){const int x=k[0],y=k[1];
    if(!cells.count({x,y-1}))edges.emplace(CellKey{x,y},CellKey{x+1,y});
    if(!cells.count({x+1,y}))edges.emplace(CellKey{x+1,y},CellKey{x+1,y+1});
    if(!cells.count({x,y+1}))edges.emplace(CellKey{x+1,y+1},CellKey{x,y+1});
    if(!cells.count({x-1,y}))edges.emplace(CellKey{x,y+1},CellKey{x,y});
  }
  std::vector<Line2> proposals;
  while(!edges.empty()){
    const auto start=edges.begin()->first;auto current=start;CellKey previous{start[0]-1,start[1]};
    std::vector<Eigen::Vector2d> chain;bool closed=false;
    while(true){
      chain.emplace_back(current[0]*resolution+c.geometry.grid_phase_x_m,current[1]*resolution+c.geometry.grid_phase_y_m);
      auto range=edges.equal_range(current);if(range.first==range.second)break;
      auto chosen=range.first;double best=-10;
      const Eigen::Vector2d incoming(current[0]-previous[0],current[1]-previous[1]);
      for(auto it=range.first;it!=range.second;++it){const Eigen::Vector2d outgoing(it->second[0]-current[0],it->second[1]-current[1]);
        const double angle=std::atan2(incoming.x()*outgoing.y()-incoming.y()*outgoing.x(),incoming.dot(outgoing));
        if(angle>best){best=angle;chosen=it;}
      }
      previous=current;current=chosen->second;edges.erase(chosen);
      if(current==start){closed=true;break;}
    }
    // Interior islands (e.g. cargo) are not outer Hatch boundaries. Their
    // absence here does not classify that space as empty or observed free.
    if(!closed||chain.size()<3||polygon_area(chain)<=0)continue;
    std::size_t opposite=1;for(std::size_t i=2;i<chain.size();++i)if((chain[i]-chain[0]).squaredNorm()>(chain[opposite]-chain[0]).squaredNorm())opposite=i;
    chain.push_back(chain[0]);std::set<std::size_t> retained{0,opposite,chain.size()-1};
    std::vector<std::pair<std::size_t,std::size_t>> pending{{0,opposite},{opposite,chain.size()-1}};
    while(!pending.empty()){const auto interval=pending.back();pending.pop_back();const auto a=interval.first,b=interval.second;
      if(b<=a+1)continue;const Eigen::Vector2d d=chain[b]-chain[a];double largest=resolution*.65;std::size_t split=b;
      for(std::size_t i=a+1;i<b;++i){const double u=std::clamp((chain[i]-chain[a]).dot(d)/std::max(1e-12,d.squaredNorm()),0.,1.);
        const double distance=(chain[i]-chain[a]-u*d).norm();if(distance>largest){largest=distance;split=i;}}
      if(split!=b){retained.insert(split);pending.emplace_back(a,split);pending.emplace_back(split,b);}
    }
    auto before=retained.begin();for(auto after=std::next(before);after!=retained.end();++after){
      const Eigen::Vector2d d=chain[*after]-chain[*before];
      if(d.norm()>=c.boundary.line_min_length_m){Line2 line;line.origin=chain[*before];line.direction=d.normalized();line.lo=0;line.hi=d.norm();proposals.push_back(line);}
      before=after;
    }
  }
  return proposals;
}
struct BreakPoint {Eigen::Vector2d xy;double residual=0;bool valid=false;};
BreakPoint profile_break(const Points& p,const HeightGrid& grid,const Eigen::Vector2d& q,const Eigen::Vector2d& tangent,const Eigen::Vector2d& inward,const Config& c) {
  BreakPoint result;const double step=c.boundary.profile_step_m;
  std::map<int,std::vector<double>> bins;
  for(auto id:nearby(grid,q,c.boundary.profile_half_length_m+c.boundary.profile_half_width_m,c)){
    const Eigen::Vector2d delta=p[id].head<2>().cast<double>()-q;
    if(std::abs(delta.dot(tangent))>c.boundary.profile_half_width_m || std::abs(delta.dot(inward))>c.boundary.profile_half_length_m)continue;
    if(p[id].z()>c.roi.max_height_above_deck_m || p[id].z()<-c.roi.max_opening_depth_m)continue;
    bins[int(std::floor(delta.dot(inward)/step))].push_back(p[id].z());
  }
  struct Sample {double x,z;std::size_t n;};std::vector<Sample> samples;
  for(const auto& kv:bins)samples.push_back({(kv.first+.5)*step,quantile(kv.second,c.boundary.profile_lower_quantile),kv.second.size()});
  if(samples.size()<std::size_t(c.boundary.profile_min_points))return result;
  // Search an observed deck-to-interior transition. No fixed low-height contour
  // is used as the final line; the slope breakpoint is fitted against deck z=0.
  for(std::size_t i=1;i<samples.size();++i){
    if(samples[i].z>=-c.boundary.profile_min_drop_m)continue;
    std::size_t platform=i;
    for(std::size_t j=i;j>0;){--j;if(samples[i].x-samples[j].x>c.boundary.profile_max_bracket_m+step*1e-6)break;
      if(std::abs(samples[j].z)<=c.roi.support_band_m){platform=j;break;}}
    if(platform==i)continue;
    // An observed raised coaming between Deck and a lower return is not a
    // deck-to-opening breakpoint. Interpolating across its top would produce
    // the coaming centre/outer edge; only independently resolved inner-face
    // evidence may determine that boundary.
    bool raised_barrier=false;for(std::size_t j=platform+1;j<i;++j)if(samples[j].z>c.roi.support_band_m)raised_barrier=true;
    if(raised_barrier)continue;
    std::size_t deck=0,low=0;for(std::size_t j=0;j<=platform;++j)if(std::abs(samples[j].z)<c.geometry.plane_inlier_m*2)++deck;
    for(std::size_t j=i;j<samples.size();++j)if(samples[j].z<-c.boundary.profile_min_drop_m)++low;
    if(deck<3 || low<3)continue;
    double last_flat=0,flat_span=0;std::vector<Sample> flat_run;
    auto inspect_flat=[&](){if(flat_run.size()<3)return;double mx=0,mz=0;for(const auto& s:flat_run){mx+=s.x;mz+=s.z;}mx/=flat_run.size();mz/=flat_run.size();double xx=0,xz=0;
      for(const auto& s:flat_run){xx+=(s.x-mx)*(s.x-mx);xz+=(s.x-mx)*(s.z-mz);}
      if(xx>1e-12&&std::abs(xz/xx)<=std::tan(c.boundary.profile_deck_slope_max_deg*std::acos(-1.)/180.))flat_span=std::max(flat_span,flat_run.back().x-flat_run.front().x);
    };
    for(std::size_t j=0;j<=platform;++j){
      if(std::abs(samples[j].z)>c.geometry.plane_inlier_m*2){inspect_flat();flat_run.clear();continue;}
      if(!flat_run.empty()&&samples[j].x-last_flat>c.boundary.profile_max_bracket_m+step*1e-6){inspect_flat();flat_run.clear();}
      last_flat=samples[j].x;flat_run.push_back(samples[j]);
    }
    inspect_flat();
    if(flat_span<c.boundary.profile_deck_support_span_m)continue;
    // Flat support must reach the putative break. A long flat region farther
    // away does not justify calling the middle of a smooth roll-off an edge.
    std::vector<Sample> terminal_deck;
    for(std::size_t j=0;j<=platform;++j)if(samples[platform].x-samples[j].x<=c.boundary.profile_deck_support_span_m+step*1e-6)terminal_deck.push_back(samples[j]);
    if(terminal_deck.size()<3)continue;
    double terminal_x=0,terminal_z=0;for(const auto& s:terminal_deck){terminal_x+=s.x;terminal_z+=s.z;}terminal_x/=terminal_deck.size();terminal_z/=terminal_deck.size();
    double terminal_xx=0,terminal_xz=0;for(const auto& s:terminal_deck){terminal_xx+=(s.x-terminal_x)*(s.x-terminal_x);terminal_xz+=(s.x-terminal_x)*(s.z-terminal_z);}
    if(terminal_xx<=1e-12||std::abs(terminal_xz/terminal_xx)>std::tan(c.boundary.profile_deck_slope_max_deg*std::acos(-1.)/180.))continue;
    const double gap=samples[i].x-samples[platform].x;if(gap>c.boundary.profile_max_bracket_m+step*1e-6)continue;
    double boundary=(samples[i].x+samples[platform].x)*.5,residual=step*.5;
    bool justified=(samples[platform].z-samples[i].z)/gap>=c.boundary.profile_step_slope_min;
    // A smooth descent can look like a step at a chosen height threshold.
    // Require the extrapolated deck intersection to remain stable over nested
    // measured portions of the descent. Curved/competing breaks are retained
    // as unobservable rather than promoted by a single favorable line fit.
    double floor_height=samples[i].z;for(std::size_t j=i;j<samples.size();++j)floor_height=std::min(floor_height,samples[j].z);
    std::vector<double> intersections;
    std::vector<Sample> descending;
    for(std::size_t j=i;j<samples.size()&&samples[j].x-samples[i].x<c.boundary.profile_half_length_m;++j){
      if(samples[j].z<=floor_height+c.boundary.profile_min_drop_m)break;
      descending.push_back(samples[j]);if(descending.size()<5)continue;
      double mx=0,mz=0;for(const auto& s:descending){mx+=s.x;mz+=s.z;}mx/=descending.size();mz/=descending.size();double xx=0,xz=0;
      for(const auto& s:descending){xx+=(s.x-mx)*(s.x-mx);xz+=(s.x-mx)*(s.z-mz);}
      if(xx>1e-12&&xz<0)intersections.push_back(mx-mz/(xz/xx));
    }
    if(intersections.size()>1&&*std::max_element(intersections.begin(),intersections.end())-*std::min_element(intersections.begin(),intersections.end())>c.boundary.profile_breakpoint_stability_m)return result;
    // A sloped transition has intermediate heights. Fit its first contiguous
    // descending portion, stopping before the interior floor plateau.
    std::vector<Sample> slope;
    for(std::size_t j=i;j<samples.size() && samples[j].x-samples[i].x<c.geometry.coarse_voxel_m;++j){
      if(j>i && samples[j].z>samples[j-1].z+c.geometry.plane_inlier_m)break;
      slope.push_back(samples[j]);
    }
    if(slope.size()>=3){
      double mx=0,mz=0;for(const auto& s:slope){mx+=s.x;mz+=s.z;}mx/=slope.size();mz/=slope.size();double xx=0,xz=0;
      for(const auto& s:slope){xx+=(s.x-mx)*(s.x-mx);xz+=(s.x-mx)*(s.z-mz);}
      if(xx>1e-12 && xz<0){const double a=xz/xx,b=mz-a*mx;const double hit=-b/a;
        std::vector<double> errors;for(const auto& s:slope)errors.push_back(std::abs(s.z-a*s.x-b));
        const double r=quantile(errors,.95);
        if(r<=c.boundary.profile_max_residual_m && hit>=samples[platform].x-c.boundary.profile_max_bracket_m && hit<=samples[i].x){boundary=hit;residual=std::max(step*.5,r/std::max(1.,std::abs(a)));justified=true;}
      }
    }
    if(!justified)continue;
    result.xy=q+inward*boundary;result.residual=residual;result.valid=true;return result;
  }return result;
}
BoundarySegment segment(const Line2& line,const std::vector<Eigen::Vector2d>& points,const Config& c,const std::string& mechanism) {
  BoundarySegment edge;const auto a=line.origin+line.direction*line.lo,b=line.origin+line.direction*line.hi;
  edge.a={a.x(),a.y(),0};edge.b={b.x(),b.y(),0};edge.side=Side::INNER_OPENING_FACE;edge.visibility=Visibility::VISIBLE;edge.evidence_flags=OBSERVED_3D;edge.evidence_mechanism=mechanism;edge.support_count=line.ids.size();
  std::vector<double> positions,errors;for(auto i:line.ids){const auto d=points[i]-line.origin;positions.push_back(d.dot(line.direction));errors.push_back(std::abs(d.x()*line.direction.y()-d.y()*line.direction.x()));}
  std::sort(positions.begin(),positions.end());double start=positions.front(),last=start;
  for(std::size_t i=1;i<positions.size();++i){if(positions[i]-last>c.boundary.max_support_gap_m){edge.support_intervals.emplace_back(start-line.lo,last-line.lo);start=positions[i];}last=positions[i];}
  edge.support_intervals.emplace_back(start-line.lo,last-line.lo);
  for(const auto& interval:edge.support_intervals)edge.observed_support_length+=interval.second-interval.first;
  edge.fit_residual_p50_m=quantile(errors,.5);edge.fit_residual_p95_m=quantile(errors,.95);
  edge.normal_uncertainty_rad=std::max(line.normal_mad_rad,std::atan2(edge.fit_residual_p95_m,std::max(c.geometry.refine_voxel_m,line.hi-line.lo)));
  edge.quality_score=std::clamp(edge.observed_support_length/std::max(c.geometry.refine_voxel_m,line.hi-line.lo)*(1-edge.fit_residual_p95_m/c.boundary.line_inlier_m),0.,1.);
  return edge;
}
} // namespace
StructureEvidence extract_structures(const Points& raw,const Config& c) {
  StructureEvidence result;const auto p=remove_isolated(voxelize(raw,c.geometry.refine_voxel_m,c.geometry.max_local_extent_m),c);const auto grid=height_grid(p,c);
  const auto coarse_candidates=find_openings(grid,0,c);
  // L0 can alias two sides of a thin divider into neighbouring low cells.
  // Refine the observed lower-return connectivity at L1 before topology;
  // never let a 0.5 m cell decide whether a physical separator exists.
  Config connectivity=c;connectivity.geometry.coarse_voxel_m=c.geometry.candidate_voxel_m;
  const auto connectivity_grid=height_grid(p,connectivity);
  const auto refined_candidates=find_openings(connectivity_grid,0,connectivity);
  auto candidates=coarse_candidates;
  const HeightGrid* proposal_grid=&grid;double proposal_resolution=c.geometry.coarse_voxel_m;
  std::set<CellKey> roi_cells;
  const int padding=int(std::ceil(c.roi.boundary_search_m/c.geometry.coarse_voxel_m));
  for(const auto* proposals:{&coarse_candidates,&refined_candidates})for(const auto& candidate:*proposals)for(const auto& b:candidate.boundary){
    CellKey k{int(std::floor((b.x()-c.geometry.grid_phase_x_m)/c.geometry.coarse_voxel_m)),int(std::floor((b.y()-c.geometry.grid_phase_y_m)/c.geometry.coarse_voxel_m))};
    for(int x=-padding;x<=padding;++x)for(int y=-padding;y<=padding;++y)roi_cells.insert({k[0]+x,k[1]+y});
  }
  // L1 normals remain three dimensional; independent of the coarse height map.
  const auto l1=voxelize(p,c.geometry.candidate_voxel_m,c.geometry.max_local_extent_m);const auto ns=normals(l1,c);
  std::vector<Eigen::Vector2d> vertical;std::vector<std::size_t> vertical_ids;
  for(std::size_t i=0;i<l1.size();++i){
    CellKey k{int(std::floor((l1[i].x()-c.geometry.grid_phase_x_m)/c.geometry.coarse_voxel_m)),int(std::floor((l1[i].y()-c.geometry.grid_phase_y_m)/c.geometry.coarse_voxel_m))};
    if(roi_cells.count(k) && ns[i].valid && std::abs(ns[i].direction.z())<std::sin(c.geometry.normal_angle_deg*std::acos(-1.)/180.) && l1[i].z()<=c.roi.max_height_above_deck_m && l1[i].z()>=-c.roi.max_opening_depth_m){vertical.push_back(l1[i].head<2>().cast<double>());vertical_ids.push_back(i);}
  }
  // Fit a plane in 3D before intersecting the datum. A tilted wall projects to
  // a thick XY strip; fitting a thin 2D line first would discard most of its
  // real support and make one face disappear depending on sampling order.
  std::vector<PlaneFit> face_planes;auto remaining=vertical_ids;std::mt19937 plane_rng(42);
  const double face_cosine=std::cos(c.geometry.normal_angle_deg*std::acos(-1.)/180.);
  while(remaining.size()>=std::size_t(c.geometry.min_plane_points)&&face_planes.size()<std::size_t(c.geometry.max_plane_hypotheses)){
    std::vector<std::size_t> best_ids;Eigen::Vector3d best_normal=Eigen::Vector3d::UnitX();
    for(int trial=0;trial<c.geometry.ransac_iterations;++trial){const auto sample=remaining[plane_rng()%remaining.size()];const auto normal=ns[sample].direction;
      const double offset=-normal.dot(l1[sample].cast<double>());std::vector<std::size_t> ids;
      for(auto id:remaining)if(std::abs(ns[id].direction.dot(normal))>=face_cosine&&std::abs(normal.dot(l1[id].cast<double>())+offset)<=c.geometry.plane_inlier_m)ids.push_back(id);
      if(ids.size()>best_ids.size()){best_ids=std::move(ids);best_normal=normal;}
    }
    if(best_ids.size()<std::size_t(c.geometry.min_plane_points))break;
    auto fit=fit_plane(l1,best_ids,best_normal,c);std::set<std::size_t> consumed(best_ids.begin(),best_ids.end());
    remaining.erase(std::remove_if(remaining.begin(),remaining.end(),[&](auto id){return consumed.count(id)>0;}),remaining.end());
    if(fit.valid)face_planes.push_back(std::move(fit));
  }
  std::vector<Line2> face_lines;std::vector<Eigen::Vector2d> face_support;
  for(const auto& fit:face_planes){
    Line2 face;
    std::vector<double> angular_deviations;
    for(auto id:fit.inliers)if(ns[id].valid)angular_deviations.push_back(std::acos(std::clamp(std::abs(ns[id].direction.dot(fit.plane.normal)),0.,1.)));
    face.normal_mad_rad=quantile(angular_deviations,.5);
    const Eigen::Vector2d n=fit.plane.normal.head<2>();if(n.norm()<std::cos(c.geometry.normal_angle_deg*std::acos(-1.)/180.))continue;
    double zlo=1e20,zhi=-1e20;for(auto i:fit.inliers){zlo=std::min(zlo,double(l1[i].z()));zhi=std::max(zhi,double(l1[i].z()));}
    // Mixed Deck/wall corner normals may truncate the L1 plane support. Check
    // datum reach on the recovered raw vertical columns below, not on those
    // deliberately excluded mixed-normal samples.
    face.zlo=1e20;face.zhi=-1e20;
    face.origin=-fit.plane.offset*n/n.squaredNorm();face.direction=Eigen::Vector2d(-n.y(),n.x()).normalized();face.lo=1e20;face.hi=-1e20;
    for(auto i:fit.inliers){double along=(l1[i].head<2>().cast<double>()-face.origin).dot(face.direction);face.lo=std::min(face.lo,along);face.hi=std::max(face.hi,along);}
    // Recover endpoints on L2 after the plane is fitted. Mixed corner normals
    // cannot truncate a visible wall, and a perpendicular wall's isolated
    // intersection cannot extend this wall to the outer coaming face.
    std::map<int,std::vector<std::size_t>> along_bins;
    for(std::size_t i=0;i<p.size();++i)if(std::abs(fit.plane.distance(p[i].cast<double>()))<=c.geometry.plane_inlier_m){
      const double along=(p[i].head<2>().cast<double>()-face.origin).dot(face.direction);
      if(along>=face.lo-c.boundary.side_probe_m&&along<=face.hi+c.boundary.side_probe_m)along_bins[int(std::floor(along/c.geometry.refine_voxel_m))].push_back(i);
    }
    for(auto it=along_bins.begin();it!=along_bins.end();){double lo=1e20,hi=-lo;for(auto id:it->second){lo=std::min(lo,double(p[id].z()));hi=std::max(hi,double(p[id].z()));}
      if(hi-lo<c.boundary.face_vertical_span_min_m)it=along_bins.erase(it);else ++it;
    }
    face.ids.clear();std::vector<int> run;
    auto commit_run=[&](){if(run.empty()||(run.back()-run.front()+1)*c.geometry.refine_voxel_m<c.boundary.line_min_length_m)return;
      for(auto bin:run)for(auto id:along_bins.at(bin)){face.ids.push_back(face_support.size());face_support.push_back(p[id].head<2>().cast<double>());
        face.zlo=std::min(face.zlo,double(p[id].z()));face.zhi=std::max(face.zhi,double(p[id].z()));}};
    for(const auto& bin:along_bins){if(!run.empty()&&bin.first-run.back()>2){commit_run();run.clear();}run.push_back(bin.first);}commit_run();
    if(face.ids.empty()||face.zlo>c.roi.support_band_m||face.zhi<-c.roi.support_band_m)continue;
    face.lo=1e20;face.hi=-1e20;for(auto i:face.ids){double along=(face_support[i]-face.origin).dot(face.direction);face.lo=std::min(face.lo,along);face.hi=std::max(face.hi,along);}
    face.residual=fit.p95;face_lines.push_back(face);
  }
  // Sparse sampling alone cannot split an opening. Adopt the finer proposal
  // only when two real below-datum faces also support a possible separator.
  bool separator_faces=false;
  for(std::size_t i=0;i<face_lines.size();++i)for(std::size_t j=i+1;j<face_lines.size();++j){
    const auto& a=face_lines[i];const auto& b=face_lines[j];
    if(a.zlo>-c.boundary.face_vertical_span_min_m||b.zlo>-c.boundary.face_vertical_span_min_m)continue;
    if(std::abs(a.direction.dot(b.direction))<std::cos(c.boundary.merge_angle_deg*std::acos(-1.)/180.))continue;
    const Eigen::Vector2d n(-a.direction.y(),a.direction.x());const double width=std::abs((a.origin-b.origin).dot(n));
    if(width>=c.geometry.candidate_voxel_m&&width<=c.boundary.beam_max_width_m)separator_faces=true;
  }
  if(separator_faces&&refined_candidates.size()>coarse_candidates.size()&&!coarse_candidates.empty()){
    candidates=refined_candidates;proposal_grid=&connectivity_grid;proposal_resolution=c.geometry.candidate_voxel_m;
  }
  for(const auto& opening:candidates){
    const std::set<CellKey> opening_cells(opening.cells.begin(),opening.cells.end());
    OpeningEvidence evidence;evidence.enclosure=opening.enclosure;evidence.area_m2=opening.cells.size()*proposal_resolution*proposal_resolution;
    for(const auto& key:opening.cells){evidence.center+=proposal_grid->at(key).center;evidence.observed_low_cells.push_back(proposal_grid->at(key).center);}evidence.center/=double(opening.cells.size());
    const auto proposals=contour_proposals(opening,proposal_resolution,c);
    for(const auto& proposal:proposals){
      Eigen::Vector2d inward(-proposal.direction.y(),proposal.direction.x());if(inward.dot(evidence.center-proposal.origin)<0)inward=-inward;
      std::vector<Eigen::Vector2d> breaks;
      for(double along=proposal.lo-c.roi.boundary_search_m;along<=proposal.hi+c.roi.boundary_search_m;along+=c.boundary.profile_step_m){
        const auto r=profile_break(p,grid,proposal.origin+proposal.direction*along,proposal.direction,inward,c);if(!r.valid)continue;
        if(proposal_grid==&connectivity_grid){
          // Do not borrow a collinear measured edge from the neighbouring
          // opening across a separator. Component identity is runtime XYZ
          // geometry, not a hidden scorer/source-point label.
          const Eigen::Vector2d q=r.xy+inward*c.boundary.side_probe_m;
          CellKey key{int(std::floor((q.x()-c.geometry.grid_phase_x_m)/proposal_resolution)),int(std::floor((q.y()-c.geometry.grid_phase_y_m)/proposal_resolution))};
          bool belongs=false;for(int dx=-1;dx<=1;++dx)for(int dy=-1;dy<=1;++dy)if(opening_cells.count({key[0]+dx,key[1]+dy}))belongs=true;
          if(!belongs)continue;
        }
        breaks.push_back(r.xy);
      }
      std::vector<BoundarySegment> surface;
      if(breaks.size()>=std::size_t(c.boundary.profile_min_sections))for(const auto& line:lines(breaks,c.boundary.line_inlier_m,c)){
        // Adjacent sliding profiles reuse the same raw samples. A handful of
        // overlapping profiles around one outlier are not independent support
        // for a new short structural edge. Long lines with separated corner
        // fragments remain available for explicit topology inference.
        if(line.hi-line.lo<2*c.boundary.profile_half_width_m*c.boundary.profile_min_sections)continue;
        auto edge=segment(line,breaks,c,"SURFACE_BREAK_3D");
        // Sliding cross-sections can produce a long fitted line from a few
        // isolated breaks. Require measured length spanning independent
        // section steps before calling it an observed structural edge.
        if(edge.observed_support_length<c.boundary.profile_step_m*c.boundary.profile_min_sections)continue;
        surface.push_back(std::move(edge));
      }
      const Line2* inner=nullptr;double best_distance=c.roi.boundary_search_m;
      for(const auto& face:face_lines){
        if(std::abs(face.direction.dot(proposal.direction))<std::cos(c.boundary.merge_angle_deg*std::acos(-1.)/180.))continue;
        const double distance=std::abs((face.origin-proposal.origin).dot(inward));
        if(distance>=best_distance)continue;
        // Probe real returns on both sides. The inner side must have lower returns
        // and the outer side observed deck support. No assumed coaming thickness.
        bool low=false,deck=false;double low_gap=c.boundary.side_probe_m;
        const Eigen::Vector2d face_middle=face.origin+face.direction*(face.lo+face.hi)*.5;
        // The inner gap tests adjacency to the opening, whereas the exterior
        // Deck search may cross a measured coaming top. They are not the same
        // distance: applying the inner probe radius to both sides silently
        // loses an otherwise visible inner face on a thick coaming.
        for(auto id:nearby(grid,face_middle,c.roi.boundary_search_m,c)){
          if(std::abs((p[id].head<2>().cast<double>()-face_middle).dot(face.direction))>c.boundary.side_probe_m)continue;
          const double signed_distance=(p[id].head<2>().cast<double>()-face.origin).dot(inward);
          if(signed_distance>c.geometry.plane_inlier_m && signed_distance<c.boundary.side_probe_m && p[id].z()<-c.roi.opening_drop_m){low=true;low_gap=std::min(low_gap,signed_distance);}
          if(signed_distance<-c.geometry.plane_inlier_m && signed_distance>-c.roi.boundary_search_m && std::abs(p[id].z())<c.roi.support_band_m)deck=true;
        }
        if(low&&deck&&low_gap<=c.boundary.side_inner_gap_max_m){inner=&face;best_distance=distance;}
      }
      if(inner){auto edge=segment(*inner,face_support,c,"INNER_FACE_3D");edge.fit_residual_p95_m=inner->residual;edge.quality_score=std::clamp(1-inner->residual/c.geometry.plane_inlier_m,0.,1.);bool conflict=false;
        for(const auto& s:surface)if(std::abs((s.a.head<2>()-inner->origin).dot(inward))>c.boundary.merge_distance_m+s.fit_residual_p95_m+edge.fit_residual_p95_m)conflict=true;
        if(conflict){edge.side=Side::SIDE_UNRESOLVED;edge.evidence_flags=NONE;edge.visibility=Visibility::UNCERTAIN;edge.quality_score=0;}
        evidence.boundaries.push_back(std::move(edge));
      }else evidence.boundaries.insert(evidence.boundaries.end(),surface.begin(),surface.end());
    }
    for(const auto& edge:evidence.boundaries)if(edge.side==Side::INNER_OPENING_FACE && edge.evidence_flags==OBSERVED_3D)
      result.primitives.push_back({PrimitiveKind::COAMING_OR_HOLD_WALL,edge,"OPENING_SIDE_STRUCTURAL_SUPPORT"});
    result.openings.push_back(std::move(evidence));
  }
  // A shared separator is supported by TWO distinct observed cavities and
  // full-depth faces, not by an empty stripe in the height image. This is
  // deliberately separate from a floating beam, which retains lower returns
  // underneath and must not split the opening.
  for(std::size_t i=0;i<result.openings.size();++i)for(std::size_t j=i+1;j<result.openings.size();++j){
    const auto& left=result.openings[i];const auto& right=result.openings[j];
    for(const auto& a:left.boundaries)for(const auto& b:right.boundaries){
      if(a.evidence_mechanism!="INNER_FACE_3D"||b.evidence_mechanism!="INNER_FACE_3D"||
         a.evidence_flags!=OBSERVED_3D||b.evidence_flags!=OBSERVED_3D)continue;
      const Eigen::Vector2d u=(a.b-a.a).head<2>().normalized(),v=(b.b-b.a).head<2>().normalized();
      if(std::abs(u.dot(v))<std::cos(c.boundary.merge_angle_deg*std::acos(-1.)/180.))continue;
      const Eigen::Vector2d n(-u.y(),u.x());
      const double distance=(b.a-a.a).head<2>().dot(n),width=std::abs(distance);
      if(width<c.geometry.candidate_voxel_m||width>c.boundary.beam_max_width_m)continue;
      const Eigen::Vector2d middle=a.a.head<2>()+n*distance*.5;
      if((left.center-middle).dot(n)*(right.center-middle).dot(n)>=0)continue;
      const double lo=std::max(0.,std::min((b.a-a.a).head<2>().dot(u),(b.b-a.a).head<2>().dot(u)));
      const double hi=std::min((a.b-a.a).norm(),std::max((b.a-a.a).head<2>().dot(u),(b.b-a.a).head<2>().dot(u)));
      if(hi-lo<c.boundary.beam_min_length_m||(hi-lo)/width<c.boundary.beam_min_aspect_ratio)continue;
      // Require multiple cross-sections with a measured top and continuous
      // vertical support down to the actual nearby interior return layer.
      std::vector<Eigen::Vector2d> supported_sections;
      for(double s=lo;s<=hi;s+=c.geometry.candidate_voxel_m){
        const Eigen::Vector2d center=middle+u*s;
        std::vector<double> depths;std::array<std::set<int>,2> face_heights;std::set<int> top_bins;
        for(auto id:nearby(grid,center,width+c.boundary.side_probe_m,c)){
          const Eigen::Vector2d d=p[id].head<2>().cast<double>()-center;
          if(std::abs(d.dot(u))>c.geometry.candidate_voxel_m)continue;
          const double across=d.dot(n),z=p[id].z();
          if(std::abs(across)>width*.5+c.geometry.plane_inlier_m&&std::abs(across)<width*.5+c.boundary.side_probe_m&&z<-c.roi.opening_drop_m)depths.push_back(z);
          for(int side=0;side<2;++side)if(std::abs(across-(side?1.:-1.)*width*.5)<c.geometry.plane_inlier_m)
            face_heights[side].insert(int(std::floor(z/c.geometry.candidate_voxel_m)));
          if(std::abs(across)<width*.5&&std::abs(z)<c.roi.support_band_m)
            top_bins.insert(int(std::floor((across+width*.5)/c.geometry.candidate_voxel_m)));
        }
        if(depths.empty())continue;const double floor=quantile(depths,.5);
        if(floor>-c.boundary.face_vertical_span_min_m)continue;
        bool full=true;
        for(double h=floor+c.geometry.candidate_voxel_m;h<0;h+=c.geometry.candidate_voxel_m){
          const int bin=int(std::floor(h/c.geometry.candidate_voxel_m));
          for(const auto& heights:face_heights)if(!heights.count(bin)&&!heights.count(bin+1))full=false;
        }
        for(double t=c.geometry.candidate_voxel_m;t<width-c.geometry.candidate_voxel_m;t+=c.geometry.candidate_voxel_m)
          if(!top_bins.count(int(std::floor(t/c.geometry.candidate_voxel_m))))full=false;
        if(full)supported_sections.push_back(center);
      }
      if(supported_sections.size()<std::size_t(c.boundary.line_min_points))continue;
      std::vector<std::size_t> ids(supported_sections.size());std::iota(ids.begin(),ids.end(),0);
      auto line=refine_line(supported_sections,ids);
      auto edge=segment(line,supported_sections,c,"PAIRED_FULL_DEPTH_SEPARATOR_FACES");edge.side=Side::SIDE_UNRESOLVED;
      if(edge.observed_support_length<(hi-lo)*c.boundary.min_edge_coverage)continue;
      result.primitives.push_back({PrimitiveKind::BEAM_OR_PARTITION,edge,"TWO_CAVITIES_WITH_MEASURED_FULL_DEPTH_SEPARATOR"});
    }
  }
  // Extract elevated planar strips only inside observed cavities. A strip must
  // connect to the enclosure at BOTH ends through actual vertical support;
  // merely lining up transient clusters never establishes a beam.
  for(const auto& opening:result.openings){
    std::set<CellKey> cavity;
    for(const auto& q:opening.observed_low_cells)cavity.insert({int(std::floor((q.x()-c.geometry.grid_phase_x_m)/c.geometry.coarse_voxel_m)),int(std::floor((q.y()-c.geometry.grid_phase_y_m)/c.geometry.coarse_voxel_m))});
    std::map<CellKey,std::vector<std::size_t>> elevated;
    for(std::size_t i=0;i<l1.size();++i){const auto& q=l1[i];
      if(q.z()<c.roi.opening_drop_m||q.z()>c.roi.max_height_above_deck_m||!ns[i].valid||std::abs(ns[i].direction.z())<std::cos(c.geometry.normal_angle_deg*std::acos(-1.)/180.))continue;
      CellKey key{int(std::floor((q.x()-c.geometry.grid_phase_x_m)/c.geometry.coarse_voxel_m)),int(std::floor((q.y()-c.geometry.grid_phase_y_m)/c.geometry.coarse_voxel_m))};
      if(cavity.count(key))elevated[key].push_back(i);
    }
    while(!elevated.empty()){
      std::queue<CellKey> pending;pending.push(elevated.begin()->first);std::vector<std::size_t> ids;
      std::set<CellKey> visited;
      while(!pending.empty()){auto key=pending.front();pending.pop();if(!visited.insert(key).second)continue;auto found=elevated.find(key);if(found==elevated.end())continue;
        ids.insert(ids.end(),found->second.begin(),found->second.end());elevated.erase(found);
        for(int dx=-1;dx<=1;++dx)for(int dy=-1;dy<=1;++dy)if(elevated.count({key[0]+dx,key[1]+dy}))pending.push({key[0]+dx,key[1]+dy});
      }
      if(ids.size()<std::size_t(c.geometry.min_plane_points))continue;
      std::vector<Eigen::Vector2d> xy;double height=0;for(auto id:ids){xy.push_back(l1[id].head<2>().cast<double>());height+=l1[id].z();}height/=ids.size();
      std::vector<std::size_t> local(xy.size());std::iota(local.begin(),local.end(),0);auto line=refine_line(xy,local);
      const double length=line.hi-line.lo;std::vector<double> widths;
      for(const auto& q:xy){const Eigen::Vector2d d=q-line.origin;widths.push_back(std::abs(d.x()*line.direction.y()-d.y()*line.direction.x()));}
      const double width=2*quantile(widths,.95);
      if(length<c.boundary.beam_min_length_m||width>c.boundary.beam_max_width_m||length/std::max(width,c.geometry.candidate_voxel_m)<c.boundary.beam_min_aspect_ratio)continue;
      // Normal estimation excludes mixed attachment corners. Recover the
      // measured top extent on L2 instead of reporting shortened beam ends.
      std::vector<Eigen::Vector2d> top_support;
      for(const auto& q:p){if(std::abs(q.z()-height)>c.geometry.plane_inlier_m)continue;
        const Eigen::Vector2d d=q.head<2>().cast<double>()-line.origin;const double s=d.dot(line.direction);
        if(s<line.lo-c.geometry.normal_radius_m||s>line.hi+c.geometry.normal_radius_m)continue;
        if(std::abs(d.x()*line.direction.y()-d.y()*line.direction.x())<=width*.5+c.geometry.refine_voxel_m)top_support.push_back(q.head<2>().cast<double>());
      }
      if(top_support.size()>=std::size_t(c.boundary.line_min_points)){
        xy=std::move(top_support);local.resize(xy.size());std::iota(local.begin(),local.end(),0);line=refine_line(xy,local);
      }
      auto edge=segment(line,xy,c,"ELEVATED_CAVITY_STRIP_3D");edge.a.z()=edge.b.z()=height;edge.side=Side::SIDE_UNRESOLVED;
      auto connected=[&](const Eigen::Vector3d& end){
        bool enclosure=false;for(const auto& boundary:opening.boundaries){const Eigen::Vector2d a=boundary.a.head<2>(),d=(boundary.b-boundary.a).head<2>();double u=std::clamp((end.head<2>()-a).dot(d)/std::max(1e-12,d.squaredNorm()),0.,1.);if((a+u*d-end.head<2>()).norm()<=c.boundary.side_probe_m)enclosure=true;}
        if(!enclosure)return false;
        std::set<int> heights;for(auto id:nearby(grid,end.head<2>(),c.boundary.side_probe_m,c))if((p[id].head<2>().cast<double>()-end.head<2>()).norm()<=c.boundary.side_probe_m && p[id].z()>=-c.roi.support_band_m && p[id].z()<=height+c.roi.support_band_m)heights.insert(int(std::floor(p[id].z()/c.geometry.candidate_voxel_m)));
        if(heights.empty())return false;
        for(double h=0;h<height;h+=c.geometry.candidate_voxel_m){int bin=int(std::floor(h/c.geometry.candidate_voxel_m));if(!heights.count(bin)&&!heights.count(bin+1))return false;}return true;
      };
      const bool confirmed=connected(edge.a)&&connected(edge.b)&&edge.observed_support_length>=(line.hi-line.lo)*c.boundary.min_edge_coverage;
      result.primitives.push_back({confirmed?PrimitiveKind::BEAM_OR_PARTITION:PrimitiveKind::STRUCTURAL_LINEAR_UNKNOWN,edge,confirmed?"LOWER_RETURNS_AND_TWO_SUPPORTED_ATTACHMENTS":"INCOMPLETE_ATTACHMENT_EVIDENCE"});
    }
  }
  return result;
}
}} // namespace ship::v15
