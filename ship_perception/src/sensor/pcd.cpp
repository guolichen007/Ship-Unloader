#include <ship_perception/sensor/pcd.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
namespace ship {
std::vector<Eigen::Vector3f> load_local_pcd(const std::string& path) {
  pcl::PointCloud<pcl::PointXYZ> cloud;
  if(pcl::io::loadPCDFile<pcl::PointXYZ>(path,cloud)<0)
    throw std::runtime_error("cannot read local XYZ PCD: "+path);
  if(cloud.empty()) throw std::invalid_argument("empty input PCD");
  std::vector<Eigen::Vector3f> points;
  points.reserve(cloud.size());
  for(const auto& p:cloud) {
    Eigen::Vector3f point(p.x,p.y,p.z);
    if(!point.allFinite()) throw std::invalid_argument("nonfinite PCD point");
    points.push_back(point);
  }
  return points;
}
}
