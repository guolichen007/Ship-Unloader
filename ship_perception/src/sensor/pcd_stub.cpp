#include <ship_perception/sensor/pcd.hpp>
namespace ship {
std::vector<Eigen::Vector3f> load_local_pcd(const std::string&) {
  throw std::runtime_error("PCD I/O disabled: configure with M0_WITH_PCL=ON");
}
}
