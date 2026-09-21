#include <ship_perception/structure/structural_recognizer.hpp>
#include <chrono>
namespace ship { namespace v15 {
namespace {using Clock=std::chrono::steady_clock;double elapsed(Clock::time_point start){return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}}
StructuralModelCandidate StructuralRecognizer::recognize_offline(const Points& p,RecognitionTimings* timings) const {
  if(timings)*timings={};const auto started=Clock::now();
  const auto frame=OfflineShipFrameProvider(config_).resolve(p);
  if(timings)timings->offline_frame_ms=elapsed(started);
  if(!frame.valid){StructuralModelCandidate out;out.warnings.push_back(frame.reason);return out;}
  return run(frame.cloud,frame.deck,timings);
}
StructuralModelCandidate StructuralRecognizer::recognize(const ShipFrameCloud& input,RecognitionTimings* timings) const {
  if(input.offline)throw std::invalid_argument("PRODUCTION_INPUT_MUST_NOT_REBOOTSTRAP");
  if(timings)*timings={};const auto started=Clock::now();const auto deck=detect_deck(input.points,config_,true);
  if(timings)timings->deck_ms=elapsed(started);
  return run(input,deck,timings);
}
StructuralModelCandidate StructuralRecognizer::run(const ShipFrameCloud& input,const DeckPlaneCandidate& deck,RecognitionTimings* timings) const {
  StructuralModelCandidate out;out.T_B_input=input.T_B_input;out.input_alignment=input.input_alignment;out.deck=deck;
  if(!healthy(input.T_B_input)){out.warnings.push_back("INVALID_INPUT_FRAME");return out;}
  if(!deck.valid){out.warnings.push_back("DECK_UNRESOLVED");return out;}
  out.frame_resolved=true;
  // This is only an auxiliary datum. Production B remains unchanged.
  // Its Z is the Deck normal and its X/Y preserve the Ship Frame longitudinal
  // axis; it must NOT re-derive XY from a full-cloud PCA, which would let
  // cargo/background rotate the structure coordinate frame.
  const auto T_D_B=ship_datum_frame(deck.plane);
  if(!healthy(T_D_B)){out.frame_resolved=false;out.warnings.push_back("DATUM_AXIS_UNRESOLVED");return out;}
  Points datum;datum.reserve(input.points.size());
  std::vector<Eigen::Vector2d> roi;for(const auto& vertex:deck.support_region)roi.push_back((T_D_B*vertex).head<2>());
  if(roi.size()<3){out.frame_resolved=false;out.warnings.push_back("STRUCTURE_ROI_UNRESOLVED");return out;}
  for(const auto& p:input.points){const Eigen::Vector3d q=T_D_B*p.cast<double>();
    if(inside(q.head<2>(),roi))datum.push_back(q.cast<float>());
  }
  const auto structure_start=Clock::now();auto evidence=extract_structures(datum,config_);if(timings)timings->structure_ms=elapsed(structure_start);
  const auto topology_start=Clock::now();out.hatches=assemble_hatches(evidence,config_);if(timings)timings->topology_ms=elapsed(topology_start);
  // Topology merges overlapping proposals while preserving support gaps.
  // Publish that same resolved edge set in the primitive table; otherwise a
  // clean hatch could still carry duplicate confirmed coaming primitives.
  for(auto& primitive:evidence.primitives)if(primitive.kind!=PrimitiveKind::COAMING_OR_HOLD_WALL)out.structures.push_back(std::move(primitive));
  for(const auto& hatch:out.hatches)for(const auto& edge:hatch.boundaries)
    if(edge.evidence_flags==OBSERVED_3D&&edge.side==Side::INNER_OPENING_FACE)
      out.structures.push_back({PrimitiveKind::COAMING_OR_HOLD_WALL,edge,"OPENING_SIDE_STRUCTURAL_SUPPORT"});
  const auto T_B_D=T_D_B.inverse();
  auto transform_edge=[&](BoundarySegment& e){e.a=T_B_D*e.a;e.b=T_B_D*e.b;};
  for(auto& h:out.hatches){h.center=T_B_D*h.center;if(h.nominal_polygon)for(auto& v:*h.nominal_polygon)v=T_B_D*v;
    Eigen::Vector3d lo=Eigen::Vector3d::Constant(1e20),hi=-lo;
    for(auto& e:h.boundaries){transform_edge(e);lo=lo.cwiseMin(e.a).cwiseMin(e.b);hi=hi.cwiseMax(e.a).cwiseMax(e.b);}
    if(!h.boundaries.empty())h.extent=hi-lo;
  }
  for(auto& p:out.structures)transform_edge(p.segment);
  return out;
}
}} // namespace ship::v15
