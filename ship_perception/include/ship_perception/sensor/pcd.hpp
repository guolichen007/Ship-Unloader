#pragma once
#include <ship_perception/core/types.hpp>
#include <string>
namespace ship {
// Input PCD must already contain LOCAL coordinates. Its double world origin is
// supplied separately, never embedded in a float PCL point.
std::vector<Eigen::Vector3f> load_local_pcd(const std::string& path);
}
