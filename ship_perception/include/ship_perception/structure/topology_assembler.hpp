#pragma once
#include <ship_perception/structure/linear_structure_detector.hpp>
namespace ship { namespace v15 {
std::vector<HatchModelCandidate> assemble_hatches(const StructureEvidence& evidence,const Config& config);
}} // namespace ship::v15
