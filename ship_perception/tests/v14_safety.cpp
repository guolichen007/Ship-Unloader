#include <ship_perception/tracking/tracker.hpp>
#include <iostream>
#include <limits>
#define REQUIRE(condition) do {if(!(condition)) throw std::runtime_error(#condition);} while(false)
using namespace ship;
using namespace ship::v14;
struct Injected final:RegistrationBackend {
  RegistrationResult next;
  RegistrationResult align(const RegistrationRequest&) override {return next;}
};
RegistrationResult accepted() {
  RegistrationResult r;r.valid=true;r.converged=true;r.quality_available=true;r.backend_executed=true;
  r.rmse=.001;r.overlap_ratio=1;return r;
}
Points fixture() {
  Points points;for(int x=0;x<6;++x) for(int y=0;y<6;++y) points.emplace_back(float(x)*.3f,float(y)*.3f,0);
  return points;
}
int main() {
 try {
  Config config; const Points ground=fixture();Points anchors=ground;anchors.emplace_back(0,0,1);
  TrackingReferenceMap map(config);map.initialize(anchors);
  const auto frozen=map.snapshot();
  auto good=accepted();
  for(std::uint64_t frame=1;frame<=10;++frame) REQUIRE(map.commit(frame,ground,{},good));
  REQUIRE(map.stats().quarantined==0); // 未观测不能被当成冲突。
  const std::vector<Ray> free_space{{Eigen::Vector3d(0,0,3),Eigen::Vector3d(0,0,0)}};
  for(std::uint64_t frame=11;frame<=15;++frame) REQUIRE(map.commit(frame,ground,free_space,good));
  REQUIRE(map.stats().suspect==1);REQUIRE(map.stats().quarantined==0);
  REQUIRE(map.commit(16,anchors,{},good));REQUIRE(map.stats().suspect==0); // 支持恢复清嫌疑。
  for(std::uint64_t frame=17;frame<=26;++frame) REQUIRE(map.commit(frame,ground,free_space,good));
  REQUIRE(map.stats().quarantined==1);REQUIRE(map.snapshot()->points.size()==ground.size());
  REQUIRE(frozen->points.size()==anchors.size());REQUIRE(frozen->revision<map.snapshot()->revision);
  for(std::uint64_t frame=27;frame<=31;++frame) REQUIRE(map.commit(frame,anchors,{},good));
  REQUIRE(map.stats().quarantined==0); // 重新启用经过完整候选审核。
  const auto before=map.stats();const auto immutable=map.snapshot();auto bad=good;bad.valid=false;
  REQUIRE(!map.commit(32,ground,free_space,bad));
  bad=good;bad.T_target_source.matrix()(0,0)=2;REQUIRE(!map.commit(32,ground,free_space,bad));
  REQUIRE(map.stats().revision==before.revision);REQUIRE(map.stats().candidate_revision==before.candidate_revision);
  REQUIRE(map.snapshot()==immutable);
  // 只出现一帧的候选不能进入目标；不连续支持重新计数，过期释放容量。
  TrackingReferenceMap candidates(config);candidates.initialize(ground);
  Points extended=ground;extended.emplace_back(2.1f,.5f,1);
  REQUIRE(candidates.commit(1,extended,{},good));REQUIRE(candidates.stats().candidates==1);
  REQUIRE(candidates.snapshot()->points.size()==ground.size());
  REQUIRE(candidates.commit(3,extended,{},good));
  for(std::uint64_t f=4;f<=6;++f) REQUIRE(candidates.commit(f,extended,{},good));
  REQUIRE(candidates.snapshot()->points.size()==ground.size());
  REQUIRE(candidates.commit(7,extended,{},good));REQUIRE(candidates.snapshot()->points.size()==extended.size());
  config.tracking_map.max_reference_voxels=36;config.tracking_map.max_candidate_voxels=2;
  TrackingReferenceMap limited(config);limited.initialize(anchors);
  Points extras=ground;for(int i=0;i<20;++i) extras.emplace_back(float(i),5,1);
  for(std::uint64_t f=1;f<=6;++f) REQUIRE(limited.commit(f,extras,{},good));
  REQUIRE(limited.stats().active<=36);REQUIRE(limited.stats().candidates<=2);
  REQUIRE(limited.commit(30,ground,{},good));REQUIRE(limited.stats().candidates==0);
  // 健康检查：NaN、反射、非正交、齐次行和 Hessian 非有限值。
  for(int mode=0;mode<7;++mode) {
    auto inject=std::make_unique<Injected>();auto* control=inject.get();control->next=accepted();
    ShipTracker tracker(std::move(inject),Method::GICP);
    TrackingInput in;in.T_W_C=[](std::int64_t){return Transform::Identity();};
    SensorObservation sensor;for(const auto& p:ground) sensor.points.push_back({p});in.sensors.push_back(sensor);
    const auto first=tracker.process(in);REQUIRE(first.valid);
    control->next.T_target_source=first.T_B_C;
    if(mode==0) control->next.T_target_source.matrix()(0,0)=std::numeric_limits<double>::quiet_NaN();
    if(mode==1) control->next.T_target_source.linear()(0,0)=-1;
    if(mode==2) control->next.T_target_source.linear()(0,1)=.1;
    if(mode==3) control->next.T_target_source.matrix()(3,0)=.1;
    if(mode==4) {control->next.hessian_available=true;control->next.H(0,0)=std::numeric_limits<double>::infinity();}
    if(mode==5) control->next.rmse=std::numeric_limits<double>::quiet_NaN();
    if(mode==6) {control->next.objective_available=true;control->next.raw_objective=std::numeric_limits<double>::infinity();}
    const auto snap=tracker.map().snapshot();in.frame_id=1;const auto rejected=tracker.process(in);
    REQUIRE(!rejected.valid);REQUIRE(rejected.registration.mathematical_failure);
    REQUIRE(rejected.T_W_B.matrix()==first.T_W_B.matrix());REQUIRE(tracker.map().snapshot()==snap);
    REQUIRE(rejected.map.candidate_revision==0);
  }
  // 错误但合法 SE3 不能凭后端 valid 标志污染状态。
  auto inject=std::make_unique<Injected>();auto* control=inject.get();control->next=accepted();
  ShipTracker tracker(std::move(inject),Method::GICP);
  TrackingInput in;in.T_W_C=[](std::int64_t){return Transform::Identity();};SensorObservation sensor;
  for(const auto& p:ground) sensor.points.push_back({p});in.sensors.push_back(sensor);
  const auto first=tracker.process(in);control->next.T_target_source=first.T_B_C;
  control->next.T_target_source.translation().x()+=1.2;in.frame_id=1;
  const auto rejected=tracker.process(in);REQUIRE(!rejected.valid);REQUIRE(rejected.map.revision==first.map.revision);
  REQUIRE(rejected.map.candidate_revision==0);REQUIRE(rejected.T_W_B.matrix()==first.T_W_B.matrix());
  // 隔离后参考支撑不足：下一帧必须停止接受新位姿。
  Config sparse;sparse.registration.min_points=37;
  auto sparse_backend=std::make_unique<Injected>();auto* sparse_control=sparse_backend.get();sparse_control->next=accepted();
  ShipTracker sparse_tracker(std::move(sparse_backend),Method::GICP,sparse);
  TrackingInput sparse_in;sparse_in.T_W_C=[](std::int64_t){return Transform::Identity();};
  SensorObservation elevated;elevated.T_C_L.translation()=Eigen::Vector3d(0,0,3);
  for(const auto& p:anchors) elevated.points.push_back({(p-Eigen::Vector3f(0,0,3)).eval()});
  sparse_in.sensors.push_back(elevated);const auto sparse_first=sparse_tracker.process(sparse_in);REQUIRE(sparse_first.valid);
  sparse_control->next.T_target_source=sparse_first.T_B_C;
  sparse_in.sensors[0].points.pop_back();sparse_in.sensors[0].points.push_back(sparse_in.sensors[0].points[0]);
  TrackingResult removed;
  for(std::uint64_t f=1;f<=10;++f) {sparse_in.frame_id=f;removed=sparse_tracker.process(sparse_in);REQUIRE(removed.valid);}
  REQUIRE(removed.needs_reinitialization);REQUIRE(removed.map.active==36);
  sparse_in.frame_id=11;const auto unsupported=sparse_tracker.process(sparse_in);
  REQUIRE(!unsupported.valid);REQUIRE(unsupported.needs_reinitialization);REQUIRE(unsupported.map.revision==removed.map.revision);
  std::cout<<"PASS: snapshot, revision, quarantine, occlusion, recovery, promotion, capacity, invalid SE3/H, legal wrong pose\n";
  return 0;
 } catch(const std::exception& e) {std::cerr<<e.what()<<std::endl;return 1;}
}
