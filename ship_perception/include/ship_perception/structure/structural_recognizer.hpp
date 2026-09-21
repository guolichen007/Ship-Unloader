#pragma once
#include <ship_perception/structure/topology_assembler.hpp>
namespace ship { namespace v15 {
struct RecognitionTimings {double offline_frame_ms=0,deck_ms=0,structure_ms=0,topology_ms=0;};
class StructuralRecognizer {
 public:
  explicit StructuralRecognizer(const Config& c=Config{}):config_(c){}
  StructuralModelCandidate recognize(const ShipFrameCloud& input,RecognitionTimings* timings=nullptr) const;
  StructuralModelCandidate recognize_offline(const Points& input,RecognitionTimings* timings=nullptr) const;
 private: Config config_;
  StructuralModelCandidate run(const ShipFrameCloud& input,const DeckPlaneCandidate& deck,RecognitionTimings* timings) const;
};
}} // namespace ship::v15
