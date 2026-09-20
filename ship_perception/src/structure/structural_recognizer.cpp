#include <ship_perception/structure/structural_recognizer.hpp>
namespace ship { namespace v15 {
StructuralModelCandidate StructuralRecognizer::recognize_offline(const Points& p) const {
  const auto frame=OfflineShipFrameProvider(config_).resolve(p);
  if(!frame.valid){StructuralModelCandidate out;out.warnings.push_back(frame.reason);return out;}
  return run(frame.cloud,frame.deck);
}
StructuralModelCandidate StructuralRecognizer::recognize(const ShipFrameCloud& input) const {
  if(input.offline)throw std::invalid_argument("PRODUCTION_INPUT_MUST_NOT_REBOOTSTRAP");
  return run(input,detect_deck(input.points,config_,true));
}
StructuralModelCandidate StructuralRecognizer::run(const ShipFrameCloud& input,const DeckPlaneCandidate& deck) const {
  StructuralModelCandidate out;out.T_B_input=input.T_B_input;out.input_alignment=input.input_alignment;out.deck=deck;
  if(!healthy(input.T_B_input)){out.warnings.push_back("INVALID_INPUT_FRAME");return out;}
  if(!deck.valid){out.warnings.push_back("DECK_UNRESOLVED");return out;}
  out.frame_resolved=true;
  // This is only an auxiliary datum. Production B remains unchanged.
  const auto T_D_B=plane_frame(deck.plane,input.points);Points datum;datum.reserve(input.points.size());
  for(const auto& p:input.points)datum.push_back((T_D_B*p.cast<double>()).cast<float>());
  auto evidence=extract_structures(datum,config_);out.hatches=assemble_hatches(evidence,config_);out.structures=std::move(evidence.primitives);
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
