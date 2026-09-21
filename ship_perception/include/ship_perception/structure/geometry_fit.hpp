#pragma once
#include <ship_perception/structure/types.hpp>
#include <map>
#include <array>
namespace ship { namespace v15 {
class LocalIndex {
 public:
  explicit LocalIndex(const Points& p); ~LocalIndex();
  std::vector<std::size_t> neighbors(const Eigen::Vector3d& q,int k,double radius) const;
 private: struct Impl; std::unique_ptr<Impl> impl_;
};
struct Normal {Eigen::Vector3d direction=Eigen::Vector3d::Zero();double planarity=0,spread=0;bool valid=false;};
struct PlaneFit {Plane plane;double p50=0,p95=0;std::vector<std::size_t> inliers;bool valid=false;};
double quantile(std::vector<double> values,double q);
bool healthy(const Transform& t);
Points voxelize(const Points& p,double size,double max_local_extent_m);
Points remove_isolated(const Points& p,const Config& config);
std::vector<Normal> normals(const Points& p,const Config& c);
PlaneFit fit_plane(const Points& p,const std::vector<std::size_t>& ids,const Eigen::Vector3d& initial,const Config& c);
Transform plane_frame(const Plane& plane,const Points& support);
// Structure datum D whose Z is the ship-frame Deck normal and whose X is the
// ship longitudinal axis projected onto the Deck plane (Y = Z x X). Origin is
// the orthogonal projection of the ship origin onto the Deck plane. It never
// reads the point cloud, so cargo/background cannot rotate the datum. A
// degenerate ship X vs Deck normal yields a non-healthy transform.
Transform ship_datum_frame(const Plane& deck_in_ship);
using CellKey=std::array<int,2>;
struct HeightCell {std::vector<std::size_t> ids;double lo=0,hi=0,median=0;Eigen::Vector2d center=Eigen::Vector2d::Zero();};
using HeightGrid=std::map<CellKey,HeightCell>;
HeightGrid height_grid(const Points& p,const Config& c);
struct Opening {std::vector<CellKey> cells;std::vector<Eigen::Vector2d> boundary;double enclosure=0;bool touches_scan_boundary=false;};
std::vector<Opening> find_openings(const HeightGrid& grid,double level,const Config& c);
double polygon_area(const std::vector<Eigen::Vector2d>& p);
bool inside(const Eigen::Vector2d& p,const std::vector<Eigen::Vector2d>& polygon);
std::vector<Eigen::Vector2d> convex_support_hull(std::vector<Eigen::Vector2d> points);
}} // namespace ship::v15
