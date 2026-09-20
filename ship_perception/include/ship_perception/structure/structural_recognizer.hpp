#pragma once
#include <ship_perception/structure/topology_assembler.hpp>
namespace ship { namespace v15 {
class StructuralRecognizer {
 public:
  explicit StructuralRecognizer(const Config& c=Config{}):config_(c){}
  StructuralModelCandidate recognize(const ShipFrameCloud& input) const;
  StructuralModelCandidate recognize_offline(const Points& input) const;
 private: Config config_;
  StructuralModelCandidate run(const ShipFrameCloud& input,const DeckPlaneCandidate& deck) const;
};
}} // namespace ship::v15
