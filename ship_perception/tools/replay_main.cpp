#include <ship_perception/replay/scenarios.hpp>
#include <ship_perception/sensor/pcd.hpp>
#include <ship_perception/sensor/timestamp.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

int main(int argc,char** argv) {
  using namespace ship;
  try {
    if(argc!=3) throw std::invalid_argument("usage: synthetic_replay <local.pcd|--synthetic> <new-output-directory>");
    const auto source=std::string(argv[1])=="--synthetic" ? synthetic_ship() : load_local_pcd(argv[1]);
    const std::filesystem::path dir(argv[2]);
    if(std::filesystem::exists(dir)) throw std::invalid_argument("output directory already exists");
    std::filesystem::create_directories(dir);
    LocalOrigin origin{{cfg::local_origin_world_m[0],cfg::local_origin_world_m[1],cfg::local_origin_world_m[2]}};
    const auto trajectory=configured_trajectory(origin);
    PoseFunction ship_pose=[&](std::int64_t t){return trajectory.at(t);};
    PoseFunction crane_pose=[&](std::int64_t){return pose(origin.world_xyz,{0,0,0});};
    ReplayOptions o;
    o.seed=static_cast<unsigned>(cfg::seed);
    o.scan_duration_ns=cfg::scan_duration_ns;
    o.noise_sigma_m=cfg::noise_sigma_m;
    o.drop_probability=cfg::drop_probability;
    o.outlier_probability=cfg::outlier_probability;
    o.outlier_scale_m=cfg::outlier_scale_m;
    o.dynamic_points=static_cast<std::size_t>(cfg::dynamic_points);
    o.decimation=static_cast<std::size_t>(cfg::decimation);
    o.timestamp_offset_ns=cfg::timestamp_offset_ns;
    o.timestamp_jitter_ns=cfg::timestamp_jitter_ns;
    o.occlude=cfg::occlusion_enabled;
    o.occlusion_min={cfg::occlusion_min_m[0],cfg::occlusion_min_m[1],cfg::occlusion_min_m[2]};
    o.occlusion_max={cfg::occlusion_max_m[0],cfg::occlusion_max_m[1],cfg::occlusion_max_m[2]};
    std::ofstream points(dir/"points.csv"), gt(dir/"gt.csv"), meta(dir/"metadata.json"), calibration(dir/"calibration.csv");
    if(!points || !gt || !meta || !calibration) throw std::runtime_error("cannot open replay artifacts");
    points << std::setprecision(17) << "frame,lidar_id,x,y,z,intensity,source_time_ns,time_offset_ns,true_point_time_ns,source_index\n";
    gt << std::setprecision(17) << "frame,timestamp_ns,tx,ty,tz,qx,qy,qz,qw\n";
    calibration << std::setprecision(17) << "lidar_id,kind,row,column,value\n";
    for(int id=1;id<=2;++id) {
      Sensor sensor; sensor.id=static_cast<std::uint16_t>(id);
      if(id==2) sensor.T_crane_lidar_true=pose(
        {cfg::lidar2_translation_m[0],cfg::lidar2_translation_m[1],cfg::lidar2_translation_m[2]},
        {cfg::lidar2_rpy_rad[0],cfg::lidar2_rpy_rad[1],cfg::lidar2_rpy_rad[2]});
      sensor.T_crane_lidar_reported=sensor.T_crane_lidar_true*
          pose({cfg::extrinsic_error_m,0,0},{0,0,cfg::extrinsic_error_rad});
      for(int r=0;r<4;++r) for(int c=0;c<4;++c) {
        calibration << id << ",true," << r << ',' << c << ',' << sensor.T_crane_lidar_true.matrix()(r,c) << '\n';
        calibration << id << ",reported," << r << ',' << c << ',' << sensor.T_crane_lidar_reported.matrix()(r,c) << '\n';
      }
      for(std::int64_t frame=0;frame<cfg::frames;++frame) {
        const auto stamp=frame*cfg::frame_period_ns;
        o.seed=static_cast<unsigned>(cfg::seed+frame*2+id);
        const auto scan=generate_scan(source,stamp,ship_pose,crane_pose,sensor,o,origin);
        for(std::size_t i=0;i<scan.frame.points.size();++i) {
          const auto& p=scan.frame.points[i];
          points << frame << ',' << p.lidar_id << ',' << p.xyz.x() << ',' << p.xyz.y() << ',' << p.xyz.z()
                 << ',' << p.intensity << ',' << stamp << ',' << p.time_offset_ns << ',' << scan.true_point_times_ns[i]
                 << ',' << (scan.source_indices[i]==std::numeric_limits<std::size_t>::max() ? -1LL : static_cast<long long>(scan.source_indices[i])) << '\n';
        }
        if(id==1) {
          const auto& t=scan.ground_truth_ship.T_world_body;
          const Eigen::Quaterniond q(t.linear());
          gt << frame << ',' << stamp << ',' << t.translation().x() << ',' << t.translation().y() << ',' << t.translation().z()
             << ',' << q.x() << ',' << q.y() << ',' << q.z() << ',' << q.w() << '\n';
        }
      }
    }
    meta << std::setprecision(17) << "{\"schema\":\"ship_perception.replay\",\"version\":1,\"git_sha\":\"" << cfg::git_sha
         << "\",\"config_hash\":\"" << cfg::config_hash << "\",\"mode\":\"" << cfg::mode
         << "\",\"calibration_version\":\"" << cfg::calibration_version << "\",\"local_origin_world\":["
         << origin.world_xyz.x() << ',' << origin.world_xyz.y() << ',' << origin.world_xyz.z()
         << "],\"point_frame\":\"lidar\",\"pose_frame\":\"T_world_ship\",\"source_points\":" << source.size()
         << ",\"site_status\":\"SITE_PENDING\"}\n";
    points.close(); gt.close(); meta.close(); calibration.close();
    if(!points || !gt || !meta || !calibration) throw std::runtime_error("replay artifact write failed");
    return 0;
  } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
