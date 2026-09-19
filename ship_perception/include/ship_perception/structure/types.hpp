#pragma once
#include <Eigen/Geometry>
#include <ship_perception/v15_config.hpp>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ship { namespace v15 {
using Points=std::vector<Eigen::Vector3f>;
using Transform=Eigen::Isometry3d;
using Polygon=std::vector<Eigen::Vector3d>;
enum class InputAlignment { ALIGNED, RAW, UNKNOWN };
enum class Side { INNER_OPENING_FACE, OUTER_FACE, SIDE_UNRESOLVED };
enum class Visibility { VISIBLE, OCCLUDED, UNOBSERVED, UNCERTAIN };
enum class HatchStatus { COMPLETE_OBSERVED, COMPLETE_WITH_INFERENCE, PARTIAL };
enum class PrimitiveKind { COAMING_OR_HOLD_WALL, BEAM_OR_PARTITION, STRUCTURAL_LINEAR_UNKNOWN, HULL_STRUCTURE };
enum Evidence : std::uint32_t { NONE=0, OBSERVED_3D=1, INFERRED_PARALLEL=2, INFERRED_TOPOLOGY=4, HISTORICAL=8 };
struct Plane {
  Eigen::Vector3d normal=Eigen::Vector3d::UnitZ(); double offset=0;
  double distance(const Eigen::Vector3d& p) const {return normal.dot(p)+offset;}
};
struct DeckPlaneCandidate {
  Plane plane; Polygon support_region;
  double support_ratio=0,residual_p50_m=0,residual_p95_m=0,quality_score=0;
  std::size_t support_count=0; bool valid=false;
};
struct Endpoint {
  bool bounded=false;
  std::optional<double> uncertainty_m;
  std::string termination_evidence="UNOBSERVED_TERMINATION";
};
struct BoundarySegment {
  Eigen::Vector3d a=Eigen::Vector3d::Zero(),b=Eigen::Vector3d::Zero();
  Side side=Side::SIDE_UNRESOLVED; Visibility visibility=Visibility::UNCERTAIN;
  std::uint32_t evidence_flags=NONE;
  std::string evidence_mechanism;
  std::vector<std::pair<double,double>> support_intervals;
  std::size_t support_count=0;
  double observed_support_length=0,extrapolation_length=0;
  double fit_residual_p50_m=0,fit_residual_p95_m=0,normal_uncertainty_rad=0,quality_score=0;
  Endpoint start,end; std::optional<std::int64_t> last_observed_ns;
};
struct StructuralPrimitive {
  PrimitiveKind kind=PrimitiveKind::STRUCTURAL_LINEAR_UNKNOWN;
  BoundarySegment segment;
  std::string reason;
};
struct HatchModelCandidate {
  std::string candidate_id;
  HatchStatus status=HatchStatus::PARTIAL;
  std::optional<Polygon> nominal_polygon;
  std::vector<BoundarySegment> boundaries;
  Eigen::Vector3d center=Eigen::Vector3d::Zero(),extent=Eigen::Vector3d::Zero();
  std::size_t observed_edge_count=0,inferred_edge_count=0;
  double completeness=0,quality_score=0;
  std::vector<std::string> warnings;
};
struct ShipFrameCloud {
  Points points;
  std::string frame_id="PROVISIONAL_SHIP_FRAME";
  Transform T_B_input=Transform::Identity();
  InputAlignment input_alignment=InputAlignment::UNKNOWN;
  bool offline=false;
  std::uint64_t accepted_frames=0;
};
struct StructuralModelCandidate {
  std::string schema_version="ship_perception.v15.model.1";
  std::string review_status="UNREVIEWED",heading_semantics="UNRESOLVED";
  bool control_ready=false,canonical=false,frame_resolved=false;
  InputAlignment input_alignment=InputAlignment::UNKNOWN;
  Transform T_B_input=Transform::Identity();
  DeckPlaneCandidate deck;
  std::vector<HatchModelCandidate> hatches;
  std::vector<StructuralPrimitive> structures;
  std::vector<std::string> warnings;
};
struct FrameResult {bool valid=false; ShipFrameCloud cloud; DeckPlaneCandidate deck; std::string reason;};
}} // namespace ship::v15
