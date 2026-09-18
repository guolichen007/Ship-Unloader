#include "support.hpp"
#include <ship_perception/timing/latency.hpp>
#include <ship_perception/core/deskew.hpp>
#include <algorithm>
int main() { return test::run("G7", [](test::Metrics& m) {
  using namespace ship;
  const auto known=latency({100,150,220},true);
  CHECK(known.pipeline_latency_ns==70 && known.age_ns==120);
  test::throws([]{latency({100,99,220},true);});
  test::throws([]{latency({100,150,149},true);});
  test::throws([]{latency({100,150,220},false);});
  test::throws([]{latency({std::numeric_limits<std::int64_t>::min(),0,std::numeric_limits<std::int64_t>::max()},true);});
  FrameQueue<int> eval(Mode::EVALUATION_MODE,std::size_t(cfg::queue_capacity));
  for(int i=0;i<10;++i) eval.push(i);
  CHECK(eval.size()==10 && eval.dropped()==0);
  for(int i=0;i<10;++i) CHECK(eval.pop()==i);
  FrameQueue<int> realtime(Mode::REALTIME_MODE,std::size_t(cfg::queue_capacity));
  for(int i=0;i<10;++i) realtime.push(i);
  CHECK(realtime.size()<=std::size_t(cfg::queue_capacity));
  CHECK(realtime.pop()==9 && realtime.dropped()==9);
  test::throws([&]{realtime.pop();});
  test::throws([]{FrameQueue<int> invalid(Mode::REALTIME_MODE,0);});
  std::ofstream csv("timing.csv"); CHECK(bool(csv));
  csv << "source_time_ns,ingress_time_ns,publish_time_ns,pipeline_latency_ns,age_ns,clock_domain,git_sha,config_hash,dataset_id,mode\n";
  const auto source=synthetic_ship();
  PoseFunction identity=[](std::int64_t){return Transform::Identity();};
  std::vector<std::int64_t> measured;
  std::size_t processed=0;
  for(std::int64_t i=0;i<cfg::latency_samples;++i) {
    TimestampInfo times;
    times.source_time_ns=monotonic_ns();
    auto scan=generate_scan(source,0,identity,identity,{},ReplayOptions{},{});
    times.ingress_time_ns=monotonic_ns();
    auto output=deskew_crane(scan.frame.points,0,0,identity,Transform::Identity());
    processed+=output.size();
    times.publish_time_ns=monotonic_ns();
    const auto l=latency(times,true);
    CHECK(l.age_ns>=l.pipeline_latency_ns && l.pipeline_latency_ns>=0);
    measured.push_back(l.pipeline_latency_ns);
    csv << times.source_time_ns << ',' << times.ingress_time_ns << ',' << times.publish_time_ns
        << ',' << l.pipeline_latency_ns << ',' << l.age_ns << ",steady_clock_host,"
        << cfg::git_sha << ',' << cfg::config_hash << ',' << cfg::dataset_id << ',' << cfg::mode << '\n';
  }
  csv.close(); CHECK(bool(csv));
  std::sort(measured.begin(),measured.end());
  auto percentile=[&](double q) {return double(measured[std::size_t(std::ceil(q*double(measured.size())))-1]);};
  m["p50_pipeline_ns"]=percentile(.50);
  m["p95_pipeline_ns"]=percentile(.95);
  m["p99_pipeline_ns"]=percentile(.99);
  m["samples"]=double(measured.size()); m["processed_points"]=double(processed);
  m["threads"]=1;
  m["evaluation_dropped_frames"]=double(eval.dropped());
  m["realtime_queue_capacity"]=double(cfg::queue_capacity);
  m["realtime_queue_test_dropped_frames"]=double(realtime.dropped());
  // No real-time deadline is asserted without a site latency budget.
}); }
