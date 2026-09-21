#include <ship_perception/structure/topology_assembler.hpp>
#include <iostream>
using namespace ship::v15;
int main(){try{
  if(quantile({3,1,2},0)!=1||quantile({3,1,2},1)!=3)throw std::runtime_error("QUANTILE_ENDPOINTS");
  {Points large{{2000,0,0}};
    if(voxelize(large,.05,3000).size()!=1)throw std::runtime_error("CONFIGURED_EXTENT_IGNORED");
    bool rejected=false;try{voxelize(large,.05,1000);}catch(const std::invalid_argument&){rejected=true;}
    if(!rejected)throw std::runtime_error("EXTENT_LIMIT_NOT_ENFORCED");
    rejected=false;try{voxelize(large,1e-30,3000);}catch(const std::invalid_argument&){rejected=true;}
    if(!rejected)throw std::runtime_error("VOXEL_INDEX_OVERFLOW_NOT_REJECTED");
  }
  // Unknown enclosed holes are not external FOV cuts, nor free-space evidence.
  {Config c;HeightGrid grid;
    for(int x=-10;x<=10;++x)for(int y=-10;y<=10;++y){
      const int radius=std::max(std::abs(x),std::abs(y));if(radius==4||radius==5)continue;
      HeightCell cell;cell.center={x*.5+.25,y*.5+.25};cell.lo=cell.hi=cell.median=radius<4?-3:0;grid[{x,y}]=cell;
    }
    auto enclosed=find_openings(grid,0,c);
    if(enclosed.size()!=1||enclosed[0].touches_scan_boundary)throw std::runtime_error("UNKNOWN_HOLE_IS_NOT_SCAN_CUT");
    for(int x=4;x<=10;++x)grid.erase({x,0});
    auto cut=find_openings(grid,0,c);
    if(cut.empty()||!cut[0].touches_scan_boundary)throw std::runtime_error("EXTERIOR_CUT_NOT_REPORTED");
  }
  Points p;
  for(int ix=-120;ix<=120;++ix)for(int iy=-80;iy<=80;++iy){double x=ix*.05,y=iy*.05;bool low=((x>-4&&x<-1)||(x>1&&x<4))&&std::abs(y)<2;p.emplace_back(float(x),float(y),low?-3.f:0.f);}
  const auto structures=extract_structures(p,Config{});std::cout<<"OPENINGS="<<structures.openings.size()<<"\n";
  if(structures.openings.size()!=2)throw std::runtime_error("MULTI_OPENING");
  for(const auto& o:structures.openings){std::cout<<"EDGES="<<o.boundaries.size()<<"\n";for(const auto& e:o.boundaries)std::cout<<e.evidence_mechanism<<" "<<e.a.transpose()<<" -> "<<e.b.transpose()<<" support="<<e.observed_support_length<<"\n";if(o.boundaries.size()<4)throw std::runtime_error("VERTICAL_BLIND_BOUNDARY");}
  const auto hatches=assemble_hatches(structures,Config{});
  if(hatches.size()!=2)throw std::runtime_error("HATCH_COUNT");
  for(const auto& h:hatches){std::cout<<"COMPLETE="<<(h.status==HatchStatus::COMPLETE_OBSERVED)<<"\n";if(h.status!=HatchStatus::COMPLETE_OBSERVED)throw std::runtime_error("HATCH_CLOSURE");}
  auto one_gap=structures;auto& gapped=one_gap.openings[0].boundaries.front();const double length=(gapped.b-gapped.a).norm();
  gapped.support_intervals={{0,.3},{length-.3,length}};gapped.observed_support_length=.6;
  const auto inferred=assemble_hatches(one_gap,Config{});
  if(inferred[0].status!=HatchStatus::COMPLETE_WITH_INFERENCE||inferred[0].inferred_edge_count!=1)throw std::runtime_error("UNIQUE_CORNER_CLOSURE_NOT_INFERRED");
  for(const auto& edge:inferred[0].boundaries)if(edge.evidence_flags&INFERRED_TOPOLOGY){
    if(edge.evidence_flags&OBSERVED_3D||edge.visibility!=Visibility::UNCERTAIN||edge.extrapolation_length<length-.61)throw std::runtime_error("INFERENCE_MASQUERADES_AS_OBSERVATION");}
  auto one_corner=one_gap;one_corner.openings[0].boundaries.front().support_intervals={{0,.3}};one_corner.openings[0].boundaries.front().observed_support_length=.3;
  if(assemble_hatches(one_corner,Config{})[0].status!=HatchStatus::PARTIAL)throw std::runtime_error("ONE_CORNER_INVENTED_CLOSURE");
  auto two_gaps=one_gap;auto& second=two_gaps.openings[0].boundaries[1];const double second_length=(second.b-second.a).norm();
  second.support_intervals={{0,.3},{second_length-.3,second_length}};second.observed_support_length=.6;
  if(assemble_hatches(two_gaps,Config{})[0].status!=HatchStatus::PARTIAL)throw std::runtime_error("TWO_MISSING_EDGES_INVENTED_CLOSURE");
  auto partial=structures;partial.openings[0].boundaries.resize(2);
  auto failed=assemble_hatches(partial,Config{});
  if(failed[0].status!=HatchStatus::PARTIAL || failed[0].nominal_polygon)throw std::runtime_error("FORCED_CLOSURE");
  // Two retained adjacent edges may have one real observed corner even
  // though the hatch is PARTIAL. Their other two endpoints remain unbounded.
  int bounded=0,unbounded=0;
  for(const auto& edge:failed[0].boundaries)for(const auto* endpoint:{&edge.start,&edge.end}){
    if(endpoint->bounded){++bounded;if(!endpoint->uncertainty_m || endpoint->termination_evidence!="OBSERVED_EDGE_INTERSECTION")throw std::runtime_error("INVALID_CORNER_EVIDENCE");}
    else{++unbounded;if(endpoint->uncertainty_m)throw std::runtime_error("FALSE_ENDPOINT_UNCERTAINTY");}
  }
  if(bounded!=2||unbounded!=2)throw std::runtime_error("FALSE_ENDPOINT_BOUND");
  std::cout<<"TOP_DOWN_ZERO_VERTICAL_FACES=PASS\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
