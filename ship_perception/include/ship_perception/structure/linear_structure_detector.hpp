#pragma once
#include <ship_perception/structure/offline_ship_frame.hpp>
namespace ship { namespace v15 {
struct OpeningEvidence {
  std::vector<BoundarySegment> boundaries;
  Eigen::Vector2d center=Eigen::Vector2d::Zero();
  std::vector<Eigen::Vector2d> observed_low_cells;
  double area_m2=0,enclosure=0;
};
struct StructureEvidence {
  std::vector<OpeningEvidence> openings;
  std::vector<StructuralPrimitive> primitives;
};
StructureEvidence extract_structures(const Points& deck_frame_points,const Config& config);
}} // namespace ship::v15
