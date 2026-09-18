#include <ship_perception/replay/replay.hpp>
#include <ship_perception/sensor/timestamp.hpp>
#include <cmath>
#include <limits>
#include <random>

namespace ship {
Transform Trajectory::at(std::int64_t time_ns) const {
  const double t = static_cast<double>(time_ns)*1e-9;
  const double w = 2.0*std::acos(-1.0)*frequency_hz;
  Eigen::Vector3d xyz = origin.world_xyz + drift_mps*t;
  xyz.z() += heave_m*std::sin(w*t);
  // A fixed seeded sum of sinusoids is smooth and independent of query order.
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> phase(-std::acos(-1.0),std::acos(-1.0));
  for(int axis=0; axis<3; ++axis) {
    const double p=phase(rng);
    xyz[axis] += smooth_random_m*(std::sin(w*0.7*t+p)-std::sin(p));
  }
  Eigen::Vector3d rpy;
  for(int axis=0; axis<3; ++axis) rpy[axis]=rpy_amplitude_rad[axis]*std::sin(w*t);
  return pose(xyz,rpy);
}
std::vector<Eigen::Vector3f> synthetic_ship() {
  std::vector<Eigen::Vector3f> p;
  for(int x=-12; x<=12; ++x) for(int y=-5; y<=5; ++y) {
    p.emplace_back(float(x)*.5f,float(y)*.5f,0.f); // deck
    if(std::abs(y)==5 || std::abs(x)==12)
      p.emplace_back(float(x)*.5f,float(y)*.5f,1.f); // coaming
    else if(x>-4 && x<4 && y>-3 && y<3)
      p.emplace_back(float(x)*.5f,float(y)*.5f,-.4f+.03f*float(x)); // cargo
  }
  return p;
}
ReplayScan generate_scan(const std::vector<Eigen::Vector3f>& source,
    std::int64_t stamp,const PoseFunction& ship_pose,const PoseFunction& crane_pose,
    const Sensor& sensor,const ReplayOptions& o,const LocalOrigin& origin) {
  if(o.scan_duration_ns<0 || o.timestamp_jitter_ns<0 || o.decimation==0 ||
     !std::isfinite(o.noise_sigma_m) || o.noise_sigma_m<0 ||
     !std::isfinite(o.drop_probability) || o.drop_probability<0 || o.drop_probability>1 ||
     !std::isfinite(o.outlier_probability) || o.outlier_probability<0 || o.outlier_probability>1 ||
     !std::isfinite(o.outlier_scale_m) || o.outlier_scale_m<0 ||
     !o.occlusion_min.allFinite() || !o.occlusion_max.allFinite() ||
     (o.occlusion_min.array()>o.occlusion_max.array()).any())
    throw std::invalid_argument("invalid replay options");
  ReplayScan result;
  result.sensor=sensor;
  result.frame.time.source_time_ns=stamp;
  result.frame.local_origin_world=origin;
  result.ground_truth_ship={stamp,ship_pose(stamp)};
  std::mt19937 rng(o.seed);
  std::mt19937 noise_rng(o.noise_seed), dropout_rng(o.dropout_seed), time_rng(o.time_seed), dynamic_rng(o.dynamic_seed);
  auto& measurement=o.independent_streams?noise_rng:rng;
  auto& dropout=o.independent_streams?dropout_rng:rng;
  auto& clock_rng=o.independent_streams?time_rng:rng;
  auto& dynamics=o.independent_streams?dynamic_rng:rng;
  std::uniform_real_distribution<double> unit(0,1), scatter(-1,1);
  std::normal_distribution<double> noise(0,1);
  std::uniform_int_distribution<std::int64_t> jitter(-o.timestamp_jitter_ns,o.timestamp_jitter_ns);
  auto append = [&](const Eigen::Vector3d& world,std::int64_t time,std::size_t index) {
    Eigen::Vector3d local=(crane_pose(time)*sensor.T_crane_lidar_true).inverse()*world;
    for(int axis=0;axis<3;++axis) local[axis]+=o.noise_sigma_m*noise(measurement);
    const bool outlier=unit(measurement)<o.outlier_probability;
    if(outlier)
      for(int axis=0;axis<3;++axis) local[axis]+=o.outlier_scale_m*scatter(measurement);
    const auto offset=checked_add(checked_add(checked_sub(time,stamp),o.timestamp_offset_ns),jitter(clock_rng));
    (void)checked_add(stamp,offset);
    const Eigen::Vector3f point=local.cast<float>();
    if(!point.allFinite()) throw std::overflow_error("nonfinite sensor point");
    result.frame.points.push_back({point,1.f,offset,sensor.id});
    result.source_indices.push_back(index);
    result.true_point_times_ns.push_back(time);
    result.static_structure.push_back(!outlier && index!=std::numeric_limits<std::size_t>::max());
  };
  for(std::size_t i=0;i<source.size();++i) {
    if(!source[i].allFinite()) throw std::invalid_argument("nonfinite source point");
    const Eigen::Vector3d p=source[i].cast<double>();
    if(i%o.decimation!=0 || unit(dropout)<o.drop_probability) continue;
    if(o.occlude && (p.array()>=o.occlusion_min.array()).all() &&
                     (p.array()<=o.occlusion_max.array()).all()) continue;
    const double fraction=source.size()>1 ? double(i)/double(source.size()-1) : 0;
    const auto time=checked_add(stamp,static_cast<std::int64_t>(fraction*double(o.scan_duration_ns)));
    append(ship_pose(time)*p,time,i);
  }
  for(std::size_t i=0;i<o.dynamic_points;++i) {
    // A compact moving cluster in the world, distinct from the static ship map.
    Eigen::Vector3d p(0,0,3);
    p.x()=static_cast<double>(stamp)*1e-9;
    for(int axis=0;axis<3;++axis) p[axis]+=.15*scatter(dynamics);
    append(origin.world_xyz+p,stamp,std::numeric_limits<std::size_t>::max());
  }
  return result;
}
std::vector<Eigen::Vector3d> reconstruct_truth(const ReplayScan& s,
    const PoseFunction& ship_pose,const PoseFunction& crane_pose) {
  if(s.frame.points.size()!=s.true_point_times_ns.size())
    throw std::invalid_argument("GT timestamp count mismatch");
  std::vector<Eigen::Vector3d> result;
  for(std::size_t i=0;i<s.frame.points.size();++i) {
    const auto t=s.true_point_times_ns[i];
    result.push_back(ship_pose(t).inverse()*crane_pose(t)*s.sensor.T_crane_lidar_true*
                     s.frame.points[i].xyz.cast<double>());
  }
  return result;
}
} // namespace ship
