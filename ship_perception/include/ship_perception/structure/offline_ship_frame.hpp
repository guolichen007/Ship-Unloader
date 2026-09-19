#pragma once
#include <ship_perception/structure/geometry_fit.hpp>
namespace ship { namespace v15 {
DeckPlaneCandidate detect_deck(const Points& p,const Config& c);
class OfflineShipFrameProvider {
 public:
  explicit OfflineShipFrameProvider(const Config& c=Config{}):config_(c){}
  FrameResult resolve(const Points& p) const;
 private:Config config_;
};
}} // namespace ship::v15
