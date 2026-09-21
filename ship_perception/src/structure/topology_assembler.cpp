#include <ship_perception/structure/topology_assembler.hpp>
#include <algorithm>
#include <cmath>

namespace ship { namespace v15 { namespace {
double cross(const Eigen::Vector2d& a,const Eigen::Vector2d& b){return a.x()*b.y()-a.y()*b.x();}
bool intersect(const BoundarySegment& a,const BoundarySegment& b,Eigen::Vector2d& point,double& conditioning,const Config& c){
  const Eigen::Vector2d u=(a.b-a.a).head<2>(),v=(b.b-b.a).head<2>();
  if(u.norm()<c.boundary.line_min_length_m||v.norm()<c.boundary.line_min_length_m)return false;
  const double denominator=cross(u,v);conditioning=std::abs(denominator)/(u.norm()*v.norm());
  if(conditioning<std::sin(c.boundary.min_intersection_angle_deg*std::acos(-1.)/180.))return false;
  point=a.a.head<2>()+u*(cross(b.a.head<2>()-a.a.head<2>(),v)/denominator);return point.allFinite();
}
bool segments_cross(const Eigen::Vector2d& a,const Eigen::Vector2d& b,const Eigen::Vector2d& c,const Eigen::Vector2d& d){
  return cross(b-a,c-a)*cross(b-a,d-a)<0 && cross(d-c,a-c)*cross(d-c,b-c)<0;
}
bool merge_observed(BoundarySegment& prior,const BoundarySegment& other,const Config& c){
  if(prior.side!=other.side||prior.evidence_flags!=other.evidence_flags||prior.evidence_mechanism!=other.evidence_mechanism)return false;
  const Eigen::Vector3d u=(prior.b-prior.a).normalized(),v=(other.b-other.a).normalized();
  if(std::abs(u.dot(v))<std::cos(c.boundary.merge_angle_deg*std::acos(-1.)/180.))return false;
  // Test the shorter measured fragment against the longer fit. Extrapolating
  // a short fragment over the entire long edge amplifies its angular noise
  // and makes duplicate suppression depend on sorting order.
  const auto& reference=prior.observed_support_length>=other.observed_support_length?prior:other;
  const auto& sample=prior.observed_support_length>=other.observed_support_length?other:prior;
  const Eigen::Vector3d reference_direction=(reference.b-reference.a).normalized();
  const double separation=std::max((sample.a-reference.a).cross(reference_direction).norm(),(sample.b-reference.a).cross(reference_direction).norm());
  const double prior_proxy=std::max(prior.fit_residual_p95_m,c.geometry.refine_voxel_m*.5);
  const double other_proxy=std::max(other.fit_residual_p95_m,c.geometry.refine_voxel_m*.5);
  if(separation>c.boundary.merge_distance_m+prior_proxy+other_proxy)return false;
  const double p0=(other.a-prior.a).dot(u),p1=(other.b-prior.a).dot(u),length=(prior.b-prior.a).norm();
  if(std::max(p0,p1)<0 || std::min(p0,p1)>length)return false;
  BoundarySegment merged=other.observed_support_length>prior.observed_support_length?other:prior;
  const Eigen::Vector3d origin=merged.a,direction=(merged.b-merged.a).normalized();
  std::vector<std::pair<double,double>> intervals;
  for(const auto* edge:std::array<const BoundarySegment*,2>{&prior,&other}){const Eigen::Vector3d along=(edge->b-edge->a).normalized();
    for(const auto& range:edge->support_intervals){double a=(edge->a+along*range.first-origin).dot(direction),b=(edge->a+along*range.second-origin).dot(direction);intervals.emplace_back(std::min(a,b),std::max(a,b));}}
  if(intervals.empty())return false;
  std::sort(intervals.begin(),intervals.end());merged.support_intervals.clear();
  for(const auto& range:intervals){if(!merged.support_intervals.empty()&&range.first<=merged.support_intervals.back().second+1e-9)merged.support_intervals.back().second=std::max(merged.support_intervals.back().second,range.second);else merged.support_intervals.push_back(range);}
  const double first=merged.support_intervals.front().first,last=merged.support_intervals.back().second;
  merged.a=origin+direction*first;merged.b=origin+direction*last;merged.observed_support_length=0;
  for(auto& range:merged.support_intervals){range.first-=first;range.second-=first;merged.observed_support_length+=range.second-range.first;}
  // The same raw samples may support two proposals. Do not claim their counts
  // are independent, and never turn the gap between intervals into observation.
  merged.support_count=std::max(prior.support_count,other.support_count);
  merged.fit_residual_p95_m=std::max({prior.fit_residual_p95_m,other.fit_residual_p95_m,separation});
  merged.normal_uncertainty_rad=std::max(prior.normal_uncertainty_rad,other.normal_uncertainty_rad);
  merged.quality_score=std::min(prior.quality_score,other.quality_score);prior=std::move(merged);return true;
}
} // namespace
std::vector<HatchModelCandidate> assemble_hatches(const StructureEvidence& evidence,const Config& c){
  std::vector<HatchModelCandidate> out;
  for(const auto& opening:evidence.openings){
    HatchModelCandidate hatch;hatch.center={opening.center.x(),opening.center.y(),0};
    hatch.boundaries=opening.boundaries;
    // Sorting establishes a local cycle proposal, not a rectangular template.
    std::stable_sort(hatch.boundaries.begin(),hatch.boundaries.end(),[&](const BoundarySegment& a,const BoundarySegment& b){
      const Eigen::Vector2d x=(a.a+a.b).head<2>()*.5-opening.center,y=(b.a+b.b).head<2>()*.5-opening.center;
      return std::atan2(x.y(),x.x())<std::atan2(y.y(),y.x());
    });
    std::vector<BoundarySegment> unique;
    for(const auto& edge:hatch.boundaries){bool duplicate=false;
      for(auto& prior:unique)if(merge_observed(prior,edge,c)){duplicate=true;break;}
      if(!duplicate)unique.push_back(edge);
    }hatch.boundaries=std::move(unique);
    // A single occluded span may be inferred only from independently refined
    // short boundary fragments at BOTH corners and their observed neighbours.
    // This uses neither an L0 grid line nor a rectangular/parallel template.
    std::vector<std::size_t> sparse;
    for(std::size_t i=0;i<hatch.boundaries.size();++i){const auto& e=hatch.boundaries[i];
      if((e.evidence_flags&OBSERVED_3D)&&e.observed_support_length<std::max(c.geometry.refine_voxel_m,(e.b-e.a).norm())*c.boundary.min_edge_coverage)sparse.push_back(i);
    }
    if(c.boundary.max_inferred_edges>=1&&sparse.size()==1&&hatch.boundaries.size()>=3){
      const auto i=sparse.front(),n=hatch.boundaries.size();auto& e=hatch.boundaries[i];const auto& previous=hatch.boundaries[(i+n-1)%n];const auto& next=hatch.boundaries[(i+1)%n];
      Eigen::Vector2d a,b;double ca=0,cb=0;
      bool justified=e.side==Side::INNER_OPENING_FACE&&previous.side==Side::INNER_OPENING_FACE&&next.side==Side::INNER_OPENING_FACE&&
        (previous.evidence_flags&OBSERVED_3D)&&(next.evidence_flags&OBSERVED_3D)&&e.support_intervals.size()==2&&
        intersect(previous,e,a,ca,c)&&intersect(e,next,b,cb,c);
      if(justified){
        const Eigen::Vector2d direction=(e.b-e.a).head<2>().normalized();const double length=(e.b-e.a).norm();
        const double ta=(a-e.a.head<2>()).dot(direction),tb=(b-e.a.head<2>()).dot(direction);
        auto supported_corner=[&](double t){for(const auto& interval:e.support_intervals)if(t>=interval.first-c.boundary.corner_join_m&&t<=interval.second+c.boundary.corner_join_m)return true;return false;};
        justified=std::abs(std::min(ta,tb))<=c.boundary.corner_join_m&&std::abs(std::max(ta,tb)-length)<=c.boundary.corner_join_m&&
          supported_corner(ta)&&supported_corner(tb)&&
          e.support_intervals.front().second-e.support_intervals.front().first>=c.boundary.profile_step_m*c.boundary.profile_min_sections&&
          e.support_intervals.back().second-e.support_intervals.back().first>=c.boundary.profile_step_m*c.boundary.profile_min_sections&&
          std::min((previous.a.head<2>()-a).norm(),(previous.b.head<2>()-a).norm())<=c.boundary.corner_join_m&&
          std::min((next.a.head<2>()-b).norm(),(next.b.head<2>()-b).norm())<=c.boundary.corner_join_m;
      }
      if(justified){e.evidence_flags=INFERRED_TOPOLOGY;e.visibility=Visibility::UNCERTAIN;e.evidence_mechanism="TWO_OBSERVED_CORNER_FRAGMENTS";
        e.extrapolation_length=(e.b-e.a).norm()-e.observed_support_length;e.quality_score*=e.observed_support_length/(e.b-e.a).norm();
        hatch.warnings.push_back("ONE_OCCLUDED_SPAN_INFERRED_FROM_TWO_REFINED_CORNERS");}
    }
    const auto n=hatch.boundaries.size();bool closed=n>=3;Polygon vertices(n);std::vector<double> conditions(n,0);
    for(std::size_t i=0;i<n;++i){auto& edge=hatch.boundaries[i];
      if(edge.side!=Side::INNER_OPENING_FACE){closed=false;hatch.warnings.push_back("BOUNDARY_SIDE_UNRESOLVED");}
      if(edge.evidence_flags&OBSERVED_3D)++hatch.observed_edge_count;else if(edge.evidence_flags&(INFERRED_PARALLEL|INFERRED_TOPOLOGY))++hatch.inferred_edge_count;else closed=false;
      Eigen::Vector2d hit;
      if(!intersect(edge,hatch.boundaries[(i+1)%n],hit,conditions[i],c)){closed=false;continue;}
      vertices[i]={hit.x(),hit.y(),0};
      if(std::min((edge.a.head<2>()-hit).norm(),(edge.b.head<2>()-hit).norm())>c.boundary.corner_join_m ||
         std::min((hatch.boundaries[(i+1)%n].a.head<2>()-hit).norm(),(hatch.boundaries[(i+1)%n].b.head<2>()-hit).norm())>c.boundary.corner_join_m){closed=false;continue;}
      auto& next=hatch.boundaries[(i+1)%n];
      if(edge.side==Side::INNER_OPENING_FACE&&next.side==Side::INNER_OPENING_FACE&&(edge.evidence_flags&OBSERVED_3D)&&(next.evidence_flags&OBSERVED_3D)){
        const double sampling_a=edge.support_count>1?edge.observed_support_length/(edge.support_count-1):c.geometry.refine_voxel_m;
        const double sampling_b=next.support_count>1?next.observed_support_length/(next.support_count-1):c.geometry.refine_voxel_m;
        const double extension_a=std::min((edge.a.head<2>()-hit).norm(),(edge.b.head<2>()-hit).norm());
        const double extension_b=std::min((next.a.head<2>()-hit).norm(),(next.b.head<2>()-hit).norm());
        const double proxy=std::max({edge.fit_residual_p95_m,next.fit_residual_p95_m,sampling_a*.5,sampling_b*.5,extension_a,extension_b})/conditions[i];
        auto bound=[&](BoundarySegment& e){auto& endpoint=(e.a.head<2>()-hit).norm()<=(e.b.head<2>()-hit).norm()?e.start:e.end;
          endpoint.bounded=true;endpoint.uncertainty_m=proxy;endpoint.termination_evidence="OBSERVED_EDGE_INTERSECTION";};
        bound(edge);bound(next);
      }
    }
    if(hatch.inferred_edge_count>std::size_t(c.boundary.max_inferred_edges))closed=false;
    if(closed){
      std::vector<Eigen::Vector2d> polygon;for(const auto& v:vertices)polygon.push_back(v.head<2>());
      if(std::abs(polygon_area(polygon))<c.roi.opening_min_area_m2 || !inside(opening.center,polygon))closed=false;
      for(std::size_t i=0;i<n;++i)for(std::size_t j=i+1;j<n;++j){if(j==i+1||(i==0&&j==n-1))continue;if(segments_cross(polygon[i],polygon[(i+1)%n],polygon[j],polygon[(j+1)%n]))closed=false;}
      std::size_t enclosed=0;for(const auto& q:opening.observed_low_cells)if(inside(q,polygon))++enclosed;
      if(!opening.observed_low_cells.empty() && double(enclosed)/opening.observed_low_cells.size()<c.boundary.min_edge_coverage)closed=false;
    }
    double total=0,supported=0,quality=0;
    for(const auto& edge:hatch.boundaries){total+=(edge.b-edge.a).norm();supported+=edge.observed_support_length;quality+=edge.quality_score;}
    hatch.completeness=total>0?std::min(1.,supported/total):0;hatch.quality_score=n?quality/n:0;
    if(closed){
      for(std::size_t i=0;i<n;++i){auto& edge=hatch.boundaries[i];const Eigen::Vector3d start=vertices[(i+n-1)%n],end=vertices[i];
        const double length=(end-start).norm();if(edge.evidence_flags&OBSERVED_3D && edge.observed_support_length/std::max(c.geometry.refine_voxel_m,length)<c.boundary.min_edge_coverage){closed=false;break;}
      }
    }
    if(closed){
      hatch.nominal_polygon=vertices;hatch.status=hatch.inferred_edge_count?HatchStatus::COMPLETE_WITH_INFERENCE:HatchStatus::COMPLETE_OBSERVED;
      for(std::size_t i=0;i<n;++i){auto& edge=hatch.boundaries[i];
        const Eigen::Vector3d start=vertices[(i+n-1)%n],end=vertices[i];
        const double start_extension=std::min((edge.a-start).norm(),(edge.b-start).norm()),end_extension=std::min((edge.a-end).norm(),(edge.b-end).norm());
        edge.extrapolation_length=start_extension+end_extension+((edge.evidence_flags&(INFERRED_PARALLEL|INFERRED_TOPOLOGY))?std::max(0.,(edge.b-edge.a).norm()-edge.observed_support_length):0.);
      }
    }else{hatch.status=HatchStatus::PARTIAL;hatch.nominal_polygon.reset();hatch.warnings.push_back("INSUFFICIENT_UNIQUE_CLOSURE_EVIDENCE");}
    Eigen::Vector3d lo=Eigen::Vector3d::Constant(1e20),hi=-lo;
    for(const auto& edge:hatch.boundaries){lo=lo.cwiseMin(edge.a).cwiseMin(edge.b);hi=hi.cwiseMax(edge.a).cwiseMax(edge.b);}if(n)hatch.extent=hi-lo;
    out.push_back(std::move(hatch));
  }
  std::stable_sort(out.begin(),out.end(),[](const HatchModelCandidate& a,const HatchModelCandidate& b){if(a.center.x()!=b.center.x())return a.center.x()<b.center.x();return a.center.y()<b.center.y();});
  for(std::size_t i=0;i<out.size();++i)out[i].candidate_id="candidate_"+std::to_string(i+1);
  return out;
}
}} // namespace ship::v15
