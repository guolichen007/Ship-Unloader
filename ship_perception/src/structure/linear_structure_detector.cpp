#include <ship_perception/structure/linear_structure_detector.hpp>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <numeric>
#include <random>
#include <set>

namespace ship { namespace v15 { namespace {
struct Line2 {Eigen::Vector2d origin=Eigen::Vector2d::Zero(),direction=Eigen::Vector2d::UnitX();double lo=0,hi=0,residual=0;std::vector<std::size_t> ids;};
Line2 refine_line(const std::vector<Eigen::Vector2d>& points,const std::vector<std::size_t>& ids) {
  Line2 line;line.ids=ids;for(auto i:ids)line.origin+=points[i];line.origin/=double(ids.size());
  Eigen::Matrix2d cov=Eigen::Matrix2d::Zero();for(auto i:ids){const Eigen::Vector2d d=points[i]-line.origin;cov+=d*d.transpose();}
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eig(cov);line.direction=eig.eigenvectors().col(1);
  Eigen::Index major;line.direction.cwiseAbs().maxCoeff(&major);if(line.direction[major]<0)line.direction=-line.direction;
  line.lo=1e20;line.hi=-1e20;std::vector<double> residual;
  for(auto i:ids){const Eigen::Vector2d d=points[i]-line.origin;const double s=d.dot(line.direction);line.lo=std::min(line.lo,s);line.hi=std::max(line.hi,s);residual.push_back(std::abs(d.x()*line.direction.y()-d.y()*line.direction.x()));}
  line.residual=quantile(residual,.95);return line;
}
std::vector<Line2> lines(const std::vector<Eigen::Vector2d>& points,double distance,const Config& c) {
  std::vector<Line2> out;std::vector<std::size_t> remaining(points.size());std::iota(remaining.begin(),remaining.end(),0);std::mt19937 rng(42);
  while(remaining.size()>=std::size_t(c.boundary.line_min_points) && out.size()<std::size_t(c.geometry.max_plane_hypotheses)) {
    std::vector<std::size_t> best;
    for(int trial=0;trial<c.geometry.ransac_iterations;++trial){
      const auto a=remaining[rng()%remaining.size()],b=remaining[rng()%remaining.size()];const Eigen::Vector2d delta=points[b]-points[a];if(delta.norm()<c.boundary.line_min_length_m)continue;
      const Eigen::Vector2d direction=delta.normalized();std::vector<std::size_t> inliers;
      for(auto i:remaining){const Eigen::Vector2d d=points[i]-points[a];if(std::abs(d.x()*direction.y()-d.y()*direction.x())<=distance)inliers.push_back(i);}
      if(inliers.size()>best.size())best=std::move(inliers);
    }
    if(best.size()<std::size_t(c.boundary.line_min_points))break;
    auto line=refine_line(points,best);if(line.hi-line.lo>=c.boundary.line_min_length_m)out.push_back(line);
    std::set<std::size_t> consumed(best.begin(),best.end());remaining.erase(std::remove_if(remaining.begin(),remaining.end(),[&](std::size_t i){return consumed.count(i)>0;}),remaining.end());
  }return out;
}
std::vector<std::size_t> nearby(const HeightGrid& grid,const Eigen::Vector2d& center,double radius,const Config& c) {
  std::vector<std::size_t> ids;const double s=c.geometry.coarse_voxel_m;
  const int x0=int(std::floor((center.x()-radius-c.geometry.grid_phase_x_m)/s)),x1=int(std::floor((center.x()+radius-c.geometry.grid_phase_x_m)/s));
  const int y0=int(std::floor((center.y()-radius-c.geometry.grid_phase_y_m)/s)),y1=int(std::floor((center.y()+radius-c.geometry.grid_phase_y_m)/s));
  for(int x=x0;x<=x1;++x)for(int y=y0;y<=y1;++y){auto it=grid.find({x,y});if(it!=grid.end())ids.insert(ids.end(),it->second.ids.begin(),it->second.ids.end());}
  return ids;
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
  for(const auto& kv:bins)samples.push_back({(kv.first+.5)*step,quantile(kv.second,.5),kv.second.size()});
  if(samples.size()<std::size_t(c.boundary.profile_min_points))return result;
  // Search an observed deck-to-interior transition. No fixed low-height contour
  // is used as the final line; the slope breakpoint is fitted against deck z=0.
  for(std::size_t i=1;i<samples.size();++i){
    if(samples[i].z>=-c.boundary.profile_min_drop_m || std::abs(samples[i-1].z)>c.roi.support_band_m)continue;
    std::size_t deck=0,low=0;for(std::size_t j=0;j<i;++j)if(std::abs(samples[j].z)<c.geometry.plane_inlier_m*2)++deck;
    for(std::size_t j=i;j<samples.size();++j)if(samples[j].z<-c.boundary.profile_min_drop_m)++low;
    if(deck<3 || low<3)continue;
    const double gap=samples[i].x-samples[i-1].x;if(gap>c.boundary.profile_max_bracket_m)continue;
    double boundary=(samples[i].x+samples[i-1].x)*.5,residual=step*.5;
    // A sloped transition has intermediate heights. Fit its first contiguous
    // descending portion, stopping before the interior floor plateau.
    std::vector<Sample> slope;
    for(std::size_t j=i;j<samples.size() && samples[j].x-samples[i].x<c.geometry.coarse_voxel_m;++j){
      if(j>i && samples[j].z>=samples[j-1].z-c.geometry.plane_inlier_m)break;
      slope.push_back(samples[j]);
    }
    if(slope.size()>=3){
      double mx=0,mz=0;for(const auto& s:slope){mx+=s.x;mz+=s.z;}mx/=slope.size();mz/=slope.size();double xx=0,xz=0;
      for(const auto& s:slope){xx+=(s.x-mx)*(s.x-mx);xz+=(s.x-mx)*(s.z-mz);}
      if(xx>1e-12 && xz<0){const double a=xz/xx,b=mz-a*mx;const double hit=-b/a;
        std::vector<double> errors;for(const auto& s:slope)errors.push_back(std::abs(s.z-a*s.x-b));
        const double r=quantile(errors,.95);
        if(r<=c.boundary.profile_max_residual_m && hit>=samples[i-1].x-c.boundary.profile_max_bracket_m && hit<=samples[i].x){boundary=hit;residual=std::max(step*.5,r/std::max(1.,std::abs(a)));}
        else continue;
      }
    }
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
  edge.normal_uncertainty_rad=std::atan2(edge.fit_residual_p95_m,std::max(c.geometry.refine_voxel_m,line.hi-line.lo));
  edge.quality_score=std::clamp(edge.observed_support_length/std::max(c.geometry.refine_voxel_m,line.hi-line.lo)*(1-edge.fit_residual_p95_m/c.boundary.line_inlier_m),0.,1.);
  return edge;
}
} // namespace
StructureEvidence extract_structures(const Points& raw,const Config& c) {
  StructureEvidence result;const auto p=voxelize(raw,c.geometry.refine_voxel_m);const auto grid=height_grid(p,c);
  const auto candidates=find_openings(grid,0,c);
  std::set<CellKey> roi_cells;
  const int padding=int(std::ceil(c.roi.boundary_search_m/c.geometry.coarse_voxel_m));
  for(const auto& candidate:candidates)for(const auto& b:candidate.boundary){
    CellKey k{int(std::floor((b.x()-c.geometry.grid_phase_x_m)/c.geometry.coarse_voxel_m)),int(std::floor((b.y()-c.geometry.grid_phase_y_m)/c.geometry.coarse_voxel_m))};
    for(int x=-padding;x<=padding;++x)for(int y=-padding;y<=padding;++y)roi_cells.insert({k[0]+x,k[1]+y});
  }
  // L1 normals remain three dimensional; independent of the coarse height map.
  const auto l1=voxelize(raw,c.geometry.candidate_voxel_m);const auto ns=normals(l1,c);
  std::vector<Eigen::Vector2d> vertical;std::vector<std::size_t> vertical_ids;
  for(std::size_t i=0;i<l1.size();++i){
    CellKey k{int(std::floor((l1[i].x()-c.geometry.grid_phase_x_m)/c.geometry.coarse_voxel_m)),int(std::floor((l1[i].y()-c.geometry.grid_phase_y_m)/c.geometry.coarse_voxel_m))};
    if(roi_cells.count(k) && ns[i].valid && std::abs(ns[i].direction.z())<std::sin(c.geometry.normal_angle_deg*std::acos(-1.)/180.) && l1[i].z()<=c.roi.max_height_above_deck_m && l1[i].z()>=-c.roi.max_opening_depth_m){vertical.push_back(l1[i].head<2>().cast<double>());vertical_ids.push_back(i);}
  }
  auto coarse_faces=lines(vertical,c.geometry.plane_inlier_m,c);std::vector<Line2> face_lines;
  for(auto face:coarse_faces){
    std::vector<std::size_t> ids;for(auto i:face.ids)ids.push_back(vertical_ids[i]);
    auto fit=fit_plane(l1,ids,Eigen::Vector3d(-face.direction.y(),face.direction.x(),0),c);if(!fit.valid)continue;
    const Eigen::Vector2d n=fit.plane.normal.head<2>();if(n.norm()<std::cos(c.geometry.normal_angle_deg*std::acos(-1.)/180.))continue;
    double zlo=1e20,zhi=-1e20;for(auto i:fit.inliers){zlo=std::min(zlo,double(l1[i].z()));zhi=std::max(zhi,double(l1[i].z()));}
    if(zlo>c.roi.support_band_m || zhi<-c.roi.support_band_m)continue; // no unbounded extrapolation to datum
    face.origin=-fit.plane.offset*n/n.squaredNorm();face.direction=Eigen::Vector2d(-n.y(),n.x()).normalized();face.lo=1e20;face.hi=-1e20;
    for(auto i:face.ids){double along=(vertical[i]-face.origin).dot(face.direction);face.lo=std::min(face.lo,along);face.hi=std::max(face.hi,along);}
    face.residual=fit.p95;face_lines.push_back(face);
  }
  for(const auto& opening:candidates){
    OpeningEvidence evidence;evidence.enclosure=opening.enclosure;evidence.area_m2=opening.cells.size()*c.geometry.coarse_voxel_m*c.geometry.coarse_voxel_m;
    for(const auto& key:opening.cells){evidence.center+=grid.at(key).center;evidence.observed_low_cells.push_back(grid.at(key).center);}evidence.center/=double(opening.cells.size());
    Config coarse=c;coarse.boundary.line_min_points=c.boundary.coarse_min_points;
    const auto proposals=lines(opening.boundary,c.geometry.coarse_voxel_m*.65,coarse);
    for(const auto& proposal:proposals){
      Eigen::Vector2d inward(-proposal.direction.y(),proposal.direction.x());if(inward.dot(evidence.center-proposal.origin)<0)inward=-inward;
      std::vector<Eigen::Vector2d> breaks;
      for(double along=proposal.lo-c.roi.boundary_search_m;along<=proposal.hi+c.roi.boundary_search_m;along+=c.boundary.profile_half_width_m){const auto r=profile_break(p,grid,proposal.origin+proposal.direction*along,proposal.direction,inward,c);if(r.valid)breaks.push_back(r.xy);}
      std::vector<BoundarySegment> surface;
      if(breaks.size()>=std::size_t(c.boundary.profile_min_sections))for(const auto& line:lines(breaks,c.boundary.line_inlier_m,c))surface.push_back(segment(line,breaks,c,"SURFACE_BREAK_3D"));
      const Line2* inner=nullptr;double best_distance=c.roi.boundary_search_m;
      for(const auto& face:face_lines){
        if(std::abs(face.direction.dot(proposal.direction))<std::cos(c.boundary.merge_angle_deg*std::acos(-1.)/180.))continue;
        const double distance=std::abs((face.origin-proposal.origin).dot(inward));
        if(distance>=best_distance)continue;
        // Probe real returns on both sides. The inner side must have lower returns
        // and the outer side observed deck support. No assumed coaming thickness.
        bool low=false,deck=false;
        for(auto id:nearby(grid,face.origin,c.boundary.side_probe_m*2,c)){
          const double signed_distance=(p[id].head<2>().cast<double>()-face.origin).dot(inward);
          if(signed_distance>c.geometry.plane_inlier_m && signed_distance<c.boundary.side_probe_m && p[id].z()<-c.roi.opening_drop_m)low=true;
          if(signed_distance<-c.geometry.plane_inlier_m && signed_distance>-c.boundary.side_probe_m && std::abs(p[id].z())<c.roi.support_band_m)deck=true;
        }
        if(low&&deck){inner=&face;best_distance=distance;}
      }
      if(inner){auto edge=segment(*inner,vertical,c,"INNER_FACE_3D");edge.fit_residual_p95_m=inner->residual;edge.quality_score=std::clamp(1-inner->residual/c.geometry.plane_inlier_m,0.,1.);bool conflict=false;
        for(const auto& s:surface)if(std::abs((s.a.head<2>()-inner->origin).dot(inward))>c.boundary.merge_distance_m+s.fit_residual_p95_m+edge.fit_residual_p95_m)conflict=true;
        if(conflict){edge.side=Side::SIDE_UNRESOLVED;edge.evidence_flags=NONE;edge.visibility=Visibility::UNCERTAIN;edge.quality_score=0;}
        evidence.boundaries.push_back(std::move(edge));
      }else evidence.boundaries.insert(evidence.boundaries.end(),surface.begin(),surface.end());
    }
    for(const auto& edge:evidence.boundaries)if(edge.side==Side::INNER_OPENING_FACE && edge.evidence_flags==OBSERVED_3D)
      result.primitives.push_back({PrimitiveKind::COAMING_OR_HOLD_WALL,edge,"OPENING_SIDE_STRUCTURAL_SUPPORT"});
    result.openings.push_back(std::move(evidence));
  }
  return result;
}
}} // namespace ship::v15
