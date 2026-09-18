#include "support.hpp"
#include <ship_perception/sensor/timestamp.hpp>
#include <limits>
int main() { return test::run("G2", [](test::Metrics& m) {
  using namespace ship;
  const auto hi=std::numeric_limits<std::int64_t>::max();
  const auto lo=std::numeric_limits<std::int64_t>::min();
  CHECK(normalize_offset(-13,TimeUnit::Microseconds)==-13000);
  CHECK(normalize_offset(13u,TimeUnit::Nanoseconds)==13);
  CHECK(normalize_offset(lo,TimeUnit::Nanoseconds)==lo);
  CHECK(normalize_offset(hi,TimeUnit::Nanoseconds)==hi);
  CHECK(normalize_offset(hi/1000,TimeUnit::Microseconds)==(hi/1000)*1000);
  CHECK(normalize_offset(lo/1000,TimeUnit::Microseconds)==(lo/1000)*1000);
  test::throws([&]{normalize_offset(hi/1000+1,TimeUnit::Microseconds);});
  test::throws([&]{normalize_offset(lo/1000-1,TimeUnit::Microseconds);});
  test::throws([]{normalize_offset(std::numeric_limits<std::uint64_t>::max(),TimeUnit::Nanoseconds);});
  test::throws([&]{checked_add(hi,1);});
  test::throws([&]{checked_add(lo,-1);});
  test::throws([&]{checked_sub(hi,-1);});
  test::throws([&]{checked_sub(lo,1);});
  CHECK(checked_add(lo,hi)==-1);
  CHECK(checked_sub(lo,lo)==0);
  std::vector<Eigen::Vector3f> xyz(3,Eigen::Vector3f(1,2,3));
  auto unordered=adapt_points(xyz,std::vector<int>{1,-2,0},TimeUnit::Microseconds,10000,2);
  CHECK(!ordered_times(unordered));
  CHECK(point_time(10000,unordered[1])==8000);
  CHECK(unordered[0].time_offset_ns==1000 && unordered[0].lidar_id==2);
  auto ordered=adapt_points(xyz,std::vector<unsigned>{0,1,2},TimeUnit::Nanoseconds,10000,1);
  CHECK(ordered_times(ordered) && ordered.back().lidar_id==1);
  test::throws([&]{adapt_points(xyz,std::vector<int>{0},TimeUnit::Nanoseconds,0,1);});
  test::throws([&]{adapt_points(xyz,std::vector<int>{0,1,2},TimeUnit::Nanoseconds,hi,1);});
  CHECK(SiteAdapterStatus{}.vendor_timestamp==GateStatus::SITE_PENDING);
  m["unordered_preserved"] = 1;
  m["lidar_ids_tested"] = 2;
}); }
