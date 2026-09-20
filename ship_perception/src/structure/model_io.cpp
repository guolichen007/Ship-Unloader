#include <ship_perception/structure/model_io.hpp>
#include <ship_perception/config.hpp>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace ship { namespace v15 { namespace {
void string(std::ostream& out,const std::string& value){out<<'"';for(unsigned char ch:value){if(ch=='"'||ch=='\\')out<<'\\'<<char(ch);else if(ch=='\n')out<<"\\n";else if(ch<32)throw std::runtime_error("JSON_CONTROL_CHARACTER");else out<<char(ch);}out<<'"';}
void number(std::ostream& out,double n){if(!std::isfinite(n))throw std::runtime_error("NONFINITE_MODEL_FIELD");out<<std::setprecision(17)<<n;}
void vector(std::ostream& out,const Eigen::Vector3d& v){out<<'[';for(int i=0;i<3;++i){if(i)out<<',';number(out,v[i]);}out<<']';}
void polygon(std::ostream& out,const Polygon& p){out<<'[';for(std::size_t i=0;i<p.size();++i){if(i)out<<',';vector(out,p[i]);}out<<']';}
void strings(std::ostream& out,const std::vector<std::string>& s){out<<'[';for(std::size_t i=0;i<s.size();++i){if(i)out<<',';string(out,s[i]);}out<<']';}
void endpoint(std::ostream& out,const Endpoint& e){out<<"{\"endpoint_bounded\":"<<(e.bounded?"true":"false")<<",\"endpoint_uncertainty_m\":";if(e.uncertainty_m){if(!e.bounded)throw std::runtime_error("UNBOUNDED_ENDPOINT_WITH_FINITE_UNCERTAINTY");number(out,*e.uncertainty_m);}else out<<"null";out<<",\"termination_evidence\":";string(out,e.termination_evidence);out<<'}';}
void boundary(std::ostream& out,const BoundarySegment& e){
  out<<"{\"a\":";vector(out,e.a);out<<",\"b\":";vector(out,e.b);
  out<<",\"side\":";string(out,e.side==Side::INNER_OPENING_FACE?"INNER_OPENING_FACE":e.side==Side::OUTER_FACE?"OUTER_FACE":"SIDE_UNRESOLVED");
  out<<",\"visibility\":";string(out,e.visibility==Visibility::VISIBLE?"VISIBLE":e.visibility==Visibility::OCCLUDED?"OCCLUDED":e.visibility==Visibility::UNOBSERVED?"UNOBSERVED":"UNCERTAIN");
  out<<",\"evidence_flags\":"<<e.evidence_flags<<",\"evidence_mechanism\":";string(out,e.evidence_mechanism);
  out<<",\"support_count\":"<<e.support_count<<",\"observed_support_length\":";number(out,e.observed_support_length);
  out<<",\"extrapolation_length\":";number(out,e.extrapolation_length);
  out<<",\"fit_residual_p50_m\":";number(out,e.fit_residual_p50_m);out<<",\"fit_residual_p95_m\":";number(out,e.fit_residual_p95_m);
  out<<",\"normal_uncertainty_rad\":";number(out,e.normal_uncertainty_rad);out<<",\"quality_score\":";number(out,e.quality_score);
  out<<",\"start\":";endpoint(out,e.start);out<<",\"end\":";endpoint(out,e.end);
  out<<",\"last_observed_ns\":";if(e.last_observed_ns)out<<*e.last_observed_ns;else out<<"null";
  out<<",\"support_intervals\":[";for(std::size_t i=0;i<e.support_intervals.size();++i){if(i)out<<',';out<<'[';number(out,e.support_intervals[i].first);out<<',';number(out,e.support_intervals[i].second);out<<']';}out<<"]}";
}
} // namespace
Points read_xyz_cache(const std::string& path,const Config& c){
  std::ifstream f(path,std::ios::binary);char magic[8];std::uint64_t n=0;f.read(magic,8);f.read(reinterpret_cast<char*>(&n),8);
  if(!f||std::memcmp(magic,"SXYZV15\0",8)!=0||n==0||n>std::uint64_t(c.geometry.max_input_points))throw std::runtime_error("INVALID_XYZ_CACHE_HEADER");
  Points p;p.reserve(std::size_t(n));for(std::uint64_t i=0;i<n;++i){float v[3];f.read(reinterpret_cast<char*>(v),12);Eigen::Vector3f q(v[0],v[1],v[2]);if(!f||!q.allFinite()||q.cwiseAbs().maxCoeff()>c.geometry.max_local_extent_m)throw std::runtime_error("INVALID_XYZ_CACHE_POINT");p.push_back(q);}
  if(f.peek()!=std::char_traits<char>::eof())throw std::runtime_error("XYZ_CACHE_TRAILING_BYTES");return p;
}
void write_xyz_cache(const Points& p,const std::string& path){std::ofstream f(path,std::ios::binary);f.write("SXYZV15\0",8);const std::uint64_t n=p.size();f.write(reinterpret_cast<const char*>(&n),8);for(const auto& q:p){if(!q.allFinite())throw std::runtime_error("NONFINITE_PCD");float v[3]={q.x()==0?0:q.x(),q.y()==0?0:q.y(),q.z()==0?0:q.z()};f.write(reinterpret_cast<const char*>(v),12);}if(!f)throw std::runtime_error("CACHE_WRITE_FAILED");}
void write_model(std::ostream& out,const StructuralModelCandidate& m){
  out<<"{\"schema_version\":";string(out,m.schema_version);out<<",\"software_git_sha\":";string(out,ship::cfg::git_sha);
  out<<",\"config_hash\":";string(out,config_hash);out<<",\"review_status\":\"UNREVIEWED\",\"control_ready\":false,\"canonical\":false,\"heading_semantics\":\"UNRESOLVED\",\"candidate_id_semantics\":\"LOCAL_TO_THIS_MODEL\"";
  out<<",\"frame_resolved\":"<<(m.frame_resolved?"true":"false")<<",\"input_alignment\":";string(out,m.input_alignment==InputAlignment::ALIGNED?"ALIGNED":m.input_alignment==InputAlignment::RAW?"RAW":"UNKNOWN");
  out<<",\"T_B_input\":";if(!m.frame_resolved)out<<"null";else{out<<'[';for(int i=0;i<4;++i){if(i)out<<',';out<<'[';for(int j=0;j<4;++j){if(j)out<<',';number(out,m.T_B_input.matrix()(i,j));}out<<']';}out<<']';}
  out<<",\"deck\":{\"valid\":"<<(m.deck.valid?"true":"false")<<",\"normal\":";if(m.deck.valid)vector(out,m.deck.plane.normal);else out<<"null";out<<",\"offset\":";if(m.deck.valid)number(out,m.deck.plane.offset);else out<<"null";
  out<<",\"support_region\":";polygon(out,m.deck.support_region);out<<",\"support_count\":"<<m.deck.support_count<<",\"support_ratio\":";number(out,m.deck.support_ratio);
  out<<",\"residual_p50_m\":";if(m.deck.valid)number(out,m.deck.residual_p50_m);else out<<"null";out<<",\"residual_p95_m\":";if(m.deck.valid)number(out,m.deck.residual_p95_m);else out<<"null";out<<",\"quality_score\":";number(out,m.deck.quality_score);out<<'}';
  out<<",\"hatches\":[";for(std::size_t i=0;i<m.hatches.size();++i){if(i)out<<',';const auto& h=m.hatches[i];out<<"{\"candidate_id\":";string(out,h.candidate_id);out<<",\"status\":";string(out,h.status==HatchStatus::COMPLETE_OBSERVED?"COMPLETE_OBSERVED":h.status==HatchStatus::COMPLETE_WITH_INFERENCE?"COMPLETE_WITH_INFERENCE":"PARTIAL");
    out<<",\"nominal_polygon\":";if(h.nominal_polygon)polygon(out,*h.nominal_polygon);else out<<"null";
    out<<",\"center\":";vector(out,h.center);out<<",\"extent\":";vector(out,h.extent);
    out<<",\"observed_edge_count\":"<<h.observed_edge_count<<",\"inferred_edge_count\":"<<h.inferred_edge_count<<",\"completeness\":";number(out,h.completeness);out<<",\"quality_score\":";number(out,h.quality_score);
    out<<",\"boundaries\":[";for(std::size_t j=0;j<h.boundaries.size();++j){if(j)out<<',';boundary(out,h.boundaries[j]);}out<<"],\"warnings\":";strings(out,h.warnings);out<<'}';
  }out<<"],\"structures\":[";
  for(std::size_t i=0;i<m.structures.size();++i){if(i)out<<',';const auto& p=m.structures[i];out<<"{\"kind\":";string(out,p.kind==PrimitiveKind::COAMING_OR_HOLD_WALL?"COAMING_OR_HOLD_WALL":p.kind==PrimitiveKind::BEAM_OR_PARTITION?"BEAM_OR_PARTITION":p.kind==PrimitiveKind::HULL_STRUCTURE?"HULL_STRUCTURE":"STRUCTURAL_LINEAR_UNKNOWN");out<<",\"reason\":";string(out,p.reason);out<<",\"segment\":";boundary(out,p.segment);out<<'}';}
  out<<"],\"warnings\":";strings(out,m.warnings);out<<"}\n";
}
}} // namespace ship::v15
