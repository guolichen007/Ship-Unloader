#include "support.hpp"
#include <ship_perception/sensor/pcd.hpp>
#include <ship_perception/replay/replay.hpp>
int main(int argc,char** argv) { return test::run("PCD_SELFCHECK", [&](test::Metrics& m) {
  using namespace ship;
  CHECK(argc==2);
  const auto points=load_local_pcd(argv[1]);
  CHECK(points.size()==5);
  CHECK((points[4]-Eigen::Vector3f(.125f,-.25f,.5f)).norm()<cfg::point_tolerance_m);
  PoseFunction identity=[](std::int64_t){return Transform::Identity();};
  const auto scan=generate_scan(points,0,identity,identity,{},ReplayOptions{},{});
  const auto rebuilt=reconstruct_truth(scan,identity,identity);
  for(std::size_t i=0;i<points.size();++i)
    CHECK((rebuilt[i]-points[i].cast<double>()).norm()<cfg::point_tolerance_m);
  test::throws([]{load_local_pcd("file_that_does_not_exist.pcd");});
  m["input_points"]=double(points.size());
}, "local_xyz_fixture_v1"); }
