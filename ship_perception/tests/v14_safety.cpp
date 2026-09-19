#include <ship_perception/tracking/tracker.hpp>
#include <iostream>
#include <limits>
#include <algorithm>
#include <random>
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
void stable_lifecycle(bool promoted) {
  const Config config;const auto& c=config.tracking_map;
  const Points ground=fixture();Points structure=ground;structure.emplace_back(0,0,1);
  TrackingReferenceMap map(config);map.initialize(promoted?ground:structure);
  const auto good=accepted();std::uint64_t frame=0;
  for(std::int64_t i=0;i<c.min_support_frames;++i) REQUIRE(map.commit(++frame,structure,{},good));
  REQUIRE(map.stats().stable==structure.size());REQUIRE(map.stats().active==structure.size());
  const auto stable_snapshot=map.snapshot();
  auto contains_structure=[](const std::shared_ptr<const TargetSnapshot>& snapshot) {
    for(const auto& p:snapshot->points) if((p-Eigen::Vector3f(0,0,1)).norm()<1e-6f) return true;
    return false;
  };
  // 没看见和在旧点前方被遮挡，都不能产生自由空间冲突。
  const std::vector<Ray> occluded{{Eigen::Vector3d(0,0,3),Eigen::Vector3d(0,0,2)}};
  for(std::int64_t i=0;i<c.quarantine_min_conflict_frames+1;++i) {
    REQUIRE(map.commit(++frame,ground,i%2?occluded:std::vector<Ray>{},good));
    REQUIRE(map.stats().suspect==0);REQUIRE(map.stats().quarantined==0);REQUIRE(map.snapshot()==stable_snapshot);
  }
  const std::vector<Ray> conflict{{Eigen::Vector3d(0,0,3),Eigen::Vector3d(0,0,0)}};
  for(std::int64_t i=1;i<=c.quarantine_min_conflict_frames;++i) {
    REQUIRE(map.commit(++frame,ground,conflict,good));
    if(i>=c.suspect_min_conflict_frames && i<c.quarantine_min_conflict_frames) {
      REQUIRE(map.stats().suspect==1);REQUIRE(map.stats().quarantined==0);REQUIRE(map.snapshot()==stable_snapshot);
    }
  }
  REQUIRE(map.stats().quarantined==1);REQUIRE(map.stats().active==ground.size());
  REQUIRE(!contains_structure(map.snapshot()));REQUIRE(contains_structure(stable_snapshot));
  REQUIRE(map.snapshot()->revision>stable_snapshot->revision);
  const auto isolated=map.snapshot();
  for(std::int64_t i=1;i<c.min_support_frames;++i) {
    REQUIRE(map.commit(++frame,structure,{},good));
    REQUIRE(map.snapshot()==isolated);REQUIRE(map.stats().quarantined==1);
  }
  REQUIRE(map.commit(++frame,structure,{},good));
  REQUIRE(map.stats().quarantined==0);REQUIRE(map.stats().stable==structure.size());
  REQUIRE(contains_structure(map.snapshot()));REQUIRE(map.snapshot()->revision>isolated->revision);
  // 恢复后的晋升点仍非永久真理，可以再次隔离。
  for(std::int64_t i=0;i<c.quarantine_min_conflict_frames;++i) REQUIRE(map.commit(++frame,ground,conflict,good));
  REQUIRE(map.stats().quarantined==1);REQUIRE(!contains_structure(map.snapshot()));
  std::cout<<(promoted?"PROMOTED_STABLE_LIFECYCLE":"BOOTSTRAP_STABLE_LIFECYCLE")<<"=PASS\n";
}
void initialization_order_invariance() {
  Points points;
  for(int x=0;x<6;++x) for(int y=0;y<6;++y) for(float offset:{.01f,.03f,.08f})
    points.emplace_back(float(x)*.3f+offset,float(y)*.3f+offset,offset);
  Points shuffled=points;std::mt19937 rng(1337);std::shuffle(shuffled.begin(),shuffled.end(),rng);
  Config config;
  for(auto capacity:{config.tracking_map.max_reference_voxels,std::int64_t(30)}) {
    config.tracking_map.max_reference_voxels=capacity;
    TrackingReferenceMap original(config),reordered(config);original.initialize(points);reordered.initialize(shuffled);
    const auto a=original.snapshot(),b=reordered.snapshot();
    REQUIRE(a->revision==b->revision);REQUIRE(a->points.size()==b->points.size());
    REQUIRE(a->points.size()==std::min(std::size_t(36),std::size_t(capacity)));
    for(std::size_t i=0;i<a->points.size();++i) REQUIRE((a->points[i]-b->points[i]).norm()<1e-7f);
    const float centroid=(.01f+.03f+.08f)/3.f;
    REQUIRE((a->points.front()-Eigen::Vector3f::Constant(centroid)).norm()<1e-7f);
  }
  std::cout<<"INITIALIZATION_CENTROID_AND_SHUFFLE=PASS\n";
}
int main() {
 try {
  stable_lifecycle(false);stable_lifecycle(true);initialization_order_invariance();
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
  TrackingReferenceMap limited(config);limited.initialize(ground);
  Points extras=ground;for(int i=0;i<20;++i) extras.emplace_back(.05f+.11f*float(i),2,.5f);
  for(std::uint64_t f=1;f<=6;++f) REQUIRE(limited.commit(f,extras,{},good));
  REQUIRE(limited.stats().active==36);REQUIRE(limited.stats().candidates==2);
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
