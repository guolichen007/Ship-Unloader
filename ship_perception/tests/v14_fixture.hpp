#pragma once
// 本文件只链接验收程序。产品 TRACKER 完全不依赖此 GT_SCORER 通道。
#include <ship_perception/replay/replay.hpp>
#include <ship_perception/tracking/tracker.hpp>
#include <algorithm>
#include <numeric>
#include <random>
#include <set>
namespace validation {
using namespace ship;
using namespace ship::v14;
struct Scenario {std::string name;bool required=true;};
inline std::vector<Scenario> scenarios(bool full) {
  if(!full) return {{"NORMAL_6DOF"},{"NOISE"},{"TRANSIENT"},{"INITIAL_OFFSET"},{"INDEPENDENT_VIEW_SAMPLING"}};
  return {{"NORMAL_6DOF"},{"TRANSLATION"},{"YAW"},{"ROLL_PITCH"},{"COMBINED"},{"NOISE"},{"LOW_DENSITY"},
    {"OCCLUSION"},{"TRANSIENT"},{"INITIAL_OFFSET"},{"INDEPENDENT_VIEW_SAMPLING"},
    {"FLAT_DECK",false},{"TIME_DAMAGE",false},{"EXTRINSIC_DAMAGE",false},{"LOST",false},{"REPEATED_HATCH",false},{"MOTION_80MS",false}};
}
inline std::uint32_t stream_seed(const std::string& scenario,std::uint32_t run,std::uint64_t frame,std::uint16_t sensor,std::uint32_t channel) {
  std::uint64_t h=1469598103934665603ULL;
  for(unsigned char b:scenario) {h^=b;h*=1099511628211ULL;}
  for(auto v:{std::uint64_t(run),frame,std::uint64_t(sensor),std::uint64_t(channel)}) {h^=v+0x9e3779b97f4a7c15ULL+(h<<6)+(h>>2);}
  return std::uint32_t(h^(h>>32));
}
inline Points structure(const Config& c,bool flat=false,bool repeat=false) {
  Points p;const double s=c.replay.surface_spacing_m;
  for(double x=-4;x<=4;x+=s) for(double y=-2;y<=2;y+=s) p.emplace_back(float(x),float(y),flat?0.f:float(.025*std::sin(x)*std::cos(y)));
  if(flat) return p;
  for(double x=-4;x<=4;x+=s) for(double z=.16;z<=1.4;z+=s) {
    p.emplace_back(float(x),-2,float(z));p.emplace_back(float(x),2,float(z));
  }
  for(double y=-2;y<=2;y+=s) for(double z=.16;z<=1.0;z+=s) p.emplace_back(-4,float(y),float(z));
  for(double x=.6;x<2.0;x+=s) for(double y=-.8;y<.6;y+=s) {
    p.emplace_back(float(x),float(y),.75f);
    for(double z=.16;z<.75;z+=s) {
      if(x<.6+s) p.emplace_back(float(x),float(y),float(z));
      if(y<-.8+s) p.emplace_back(float(x),float(y),float(z));
    }
  }
  if(repeat) for(double x=-3;x<3;x+=1.0) for(double y=-1;y<=1;y+=s) for(double z=.2;z<=1.0;z+=s) p.emplace_back(float(x),float(y),float(z));
  return p;
}
inline Transform ship_truth(const Scenario& s,std::int64_t ns) {
  double t=double(ns)*1e-9; Eigen::Vector3d xyz(1000000+0.025*t,2000000+.035*std::sin(.25*t),10+.05*std::sin(.3*t));
  Eigen::Vector3d rpy(.008*std::sin(.2*t),.012*std::sin(.31*t),.03*std::sin(.23*t));
  if(s.name=="TRANSLATION") rpy.setZero();
  if(s.name=="YAW") {rpy.x()=0;rpy.y()=0;xyz={1000000,2000000,10};}
  if(s.name=="ROLL_PITCH") {rpy.z()=0;xyz={1000000,2000000,10};}
  if(s.name=="MOTION_80MS") {rpy*=6;xyz.z()+=.4*std::sin(t);}
  return pose(xyz,rpy);
}
inline Transform crane_pose(std::int64_t ns) {
  double t=double(ns)*1e-9;
  return pose({1000000+.15*std::sin(.13*t),2000000+.1*std::sin(.17*t),15+.1*std::sin(.3*t)},
              {.005*std::sin(.11*t),.005*std::sin(.21*t),.015*std::sin(.1*t)});
}
struct ScoringChannel {
  Transform T_W_S=Transform::Identity();
  std::vector<bool> static_labels;
  std::set<std::uint64_t> visible_identity;
  double injected_time_error_ns=0;
  Transform injected_extrinsic=Transform::Identity();
};
struct Packet {TrackingInput runtime;ScoringChannel scorer;};
inline Packet generate(const Scenario& scene,std::uint32_t seed,std::size_t frame,const Points& reference,const Config& c) {
  Packet packet;auto& in=packet.runtime;
  in.frame_id=frame;in.reference_ns=std::int64_t(frame)*c.replay.frame_period_ns;in.T_W_C=crane_pose;
  packet.scorer.T_W_S=ship_truth(scene,in.reference_ns);
  for(std::uint16_t sensor_id=1;sensor_id<=2;++sensor_id) {
    Sensor sensor;sensor.id=sensor_id;
    sensor.T_crane_lidar_true=pose({sensor_id==1?-1.5:1.5,0,-.3},{0,0,sensor_id==1?.05:-.05});
    sensor.T_crane_lidar_reported=sensor.T_crane_lidar_true;
    ReplayOptions o;o.seed=stream_seed(scene.name,seed,frame,sensor_id,0);
    o.independent_streams=true;o.noise_seed=stream_seed(scene.name,seed,frame,sensor_id,2);
    o.dropout_seed=stream_seed(scene.name,seed,frame,sensor_id,3);o.time_seed=stream_seed(scene.name,seed,frame,sensor_id,4);
    o.dynamic_seed=stream_seed(scene.name,seed,frame,sensor_id,5);
    o.scan_duration_ns=scene.name=="MOTION_80MS"?c.replay.diagnostic_scan_ns:0;
    o.drop_probability=scene.name=="LOW_DENSITY"?c.replay.density_drop:c.replay.ordinary_dropout;
    if(scene.name=="INDEPENDENT_VIEW_SAMPLING") o.drop_probability=c.replay.view_dropout;
    o.noise_sigma_m=c.replay.noise_sigma_m*(scene.name=="NOISE"?3:1);
    if(scene.name=="NOISE") o.outlier_probability=c.replay.outlier_probability;
    if(scene.name=="TRANSIENT") o.dynamic_points=std::size_t(c.replay.transient_points);
    if(scene.name=="OCCLUSION" && frame>0) {o.occlude=true;o.occlusion_min={-1,-2,-1};o.occlusion_max={1,0,2};}
    if(scene.name=="TIME_DAMAGE" && frame>0) o.timestamp_offset_ns=2000000000;
    if(scene.name=="EXTRINSIC_DAMAGE" && frame>0 && sensor_id==2) sensor.T_crane_lidar_reported.translation().x()+=1;
    if(scene.name=="LOST" && frame>=10 && frame<15) o.drop_probability=1;
    const LocalOrigin origin{Eigen::Vector3d(1000000,2000000,10)};
    auto scan=generate_scan(reference,in.reference_ns,[&](std::int64_t t){return ship_truth(scene,t);},crane_pose,sensor,o,origin);
    packet.scorer.injected_time_error_ns=double(o.timestamp_offset_ns);
    packet.scorer.injected_extrinsic=sensor.T_crane_lidar_true.inverse()*sensor.T_crane_lidar_reported;
    std::vector<std::size_t> order(scan.frame.points.size());std::iota(order.begin(),order.end(),0);
    std::mt19937 shuffle_rng(stream_seed(scene.name,seed,frame,sensor_id,1));std::shuffle(order.begin(),order.end(),shuffle_rng);
    SensorObservation observation;observation.stamp_ns=in.reference_ns;observation.T_C_L=sensor.T_crane_lidar_reported;
    for(auto i:order) {
      const auto id=scan.source_indices[i];
      if(scene.name=="INDEPENDENT_VIEW_SAMPLING" && id<reference.size()) {
        // 周期改变左右 FOV 边界；独立 dropout 和噪声由每帧每传感器随机流控制。
        const double margin=.08*(1+std::sin(double(frame)*.7+sensor_id));
        if(reference[id].x()<-4+margin || reference[id].x()>4-(.16-margin)) continue;
      }
      observation.points.push_back(scan.frame.points[i]);packet.scorer.static_labels.push_back(scan.static_structure[i]);
      if(scan.static_structure[i]) packet.scorer.visible_identity.insert((std::uint64_t(sensor_id)<<48)|id);
    }
    in.sensors.push_back(std::move(observation));
  }
  return packet;
}
inline double view_change(const std::set<std::uint64_t>& a,const std::set<std::uint64_t>& b) {
  std::vector<std::uint64_t> intersection;std::set_intersection(a.begin(),a.end(),b.begin(),b.end(),std::back_inserter(intersection));
  const auto count=a.size()+b.size()-intersection.size();return count?1-double(intersection.size())/double(count):0;
}
class PerturbedBackend final:public RegistrationBackend {
 public:
  PerturbedBackend(const Config& c,bool perturb):core_(c),config_(c),perturb_(perturb) {}
  RegistrationResult align(const RegistrationRequest& request) override {
    auto modified=request;
    if(perturb_ && first_) modified.initial_guess=pose({config_.replay.initial_offset_m,0,0},{0,0,config_.replay.initial_yaw_deg*std::acos(-1.0)/180})*request.initial_guess;
    first_=false;return core_.align(modified);
  }
 private:SmallGicpBackend core_;Config config_;bool perturb_,first_=true;
};
} // namespace validation
