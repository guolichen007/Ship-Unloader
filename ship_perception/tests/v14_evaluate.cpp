#include "v14_fixture.hpp"
#include <ship_perception/config.hpp>
#include <ship_perception/timing/latency.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <cstring>
using namespace validation;
namespace fs=std::filesystem;
std::string request_id(const RegistrationRequest& q) {
  std::uint64_t hash=1469598103934665603ULL;
  auto bytes=[&](const void* data,std::size_t size) {const auto* p=static_cast<const unsigned char*>(data);for(std::size_t i=0;i<size;++i) {hash^=p[i];hash*=1099511628211ULL;}};
  for(const auto* points:{&q.source,&q.target->points}) {
    const std::uint64_t count=points->size();bytes(&count,sizeof(count));
    for(const auto& p:*points) for(int a=0;a<3;++a) {const float v=p[a];bytes(&v,sizeof(v));}
  }
  for(int i=0;i<4;++i) for(int j=0;j<4;++j) {const double v=q.initial_guess.matrix()(i,j);bytes(&v,sizeof(v));}
  std::ostringstream out;out<<std::hex<<hash;return out.str();
}
double percentile(std::vector<double> values,double q) {
  if(values.empty()) return 0;
  const auto index=std::size_t(std::ceil(q*double(values.size())))-1;
  std::nth_element(values.begin(),values.begin()+index,values.end());return values[index];
}
struct Series {
  std::vector<double> translation,rotation,static_error,times;
  std::size_t frames=0,valid=0,current_fail=0,longest_fail=0,executed=0,mathematical_fail=0;
  bool initialized=false,last_valid=false;double last_translation=0,last_rotation=0,worst_frame_static=0;
  void record(bool ok,double dt,double dr,const RegistrationResult& r) {
    ++frames;executed+=r.backend_executed?1:0;mathematical_fail+=r.mathematical_failure?1:0;
    last_valid=ok;last_translation=dt;last_rotation=dr;times.push_back(r.elapsed_ms);
    if(ok) {++valid;current_fail=0;translation.push_back(dt);rotation.push_back(dr);}
    else {++current_fail;longest_fail=std::max(longest_fail,current_fail);}
  }
};
struct Output {
  std::ofstream pose,quality,timing,map,summary;
  explicit Output(const fs::path& path):pose(path/"pose.csv"),quality(path/"quality.csv"),timing(path/"timing.csv"),map(path/"map.csv"),summary(path/"series.csv") {
    for(auto* file:{&pose,&quality,&timing,&map,&summary}) {*file<<std::setprecision(17);if(!*file) throw std::runtime_error("无法创建输出文件");}
    const std::string keys="scenario,seed,frame,method,lane,";
    pose<<keys<<"valid,initialization,translation_error_m,rotation_error_deg,static_p50_m,static_p95_m,static_max_m,static_samples,view_change_ratio,tx,ty,tz,qw,qx,qy,qz,gt_tx,gt_ty,gt_tz,gt_qw,gt_qx,gt_qy,gt_qz\n";
    quality<<keys<<"backend_executed,converged,valid,mathematical_failure,hessian_available,iterations_available,quality_available,source_points,target_points,inliers,iterations,overlap,rmse,fitness,reason,objective_available,raw_objective,request_fingerprint";
    for(int i=0;i<6;++i) for(int j=0;j<6;++j) quality<<",H"<<i<<j;
    quality<<"\n";
    timing<<keys<<"prepare_ms,registration_ms,map_ms,total_ms\n";
    map<<keys<<"tracking_target_revision,tracking_map_revision,candidate_revision,active,suspect,quarantined,stable,candidates,needs_reinitialization\n";
    summary<<"scenario,seed,method,lane,required,initialized,frames,valid,valid_ratio,max_consecutive_failures,backend_calls,mathematical_failures,translation_p95_m,rotation_p95_deg,last_valid,last_translation_m,last_rotation_deg,static_p50_m,static_p95_m,static_max_m,worst_frame_static_p95_m,static_samples,time_p50_ms,time_p95_ms\n";
  }
  static std::string key(const Scenario& s,std::uint32_t seed,std::size_t frame,Method method,const std::string& lane) {
    return s.name+","+std::to_string(seed)+","+std::to_string(frame)+","+method_name(method)+","+lane+",";
  }
  void result(const std::string& key,const RegistrationResult& r,const std::string& fingerprint="") {
    quality<<key<<r.backend_executed<<','<<r.converged<<','<<r.valid<<','<<r.mathematical_failure<<','<<r.hessian_available<<','<<r.iterations_available<<','<<r.quality_available<<','<<r.source_points<<','<<r.target_points<<','<<r.inliers<<','<<r.iterations<<','<<r.overlap_ratio<<','<<r.rmse<<','<<r.fitness_score<<','<<std::quoted(r.failure_reason);
    quality<<','<<r.objective_available<<','<<r.raw_objective<<','<<fingerprint;
    for(int i=0;i<6;++i) for(int j=0;j<6;++j) quality<<','<<r.H(i,j);quality<<'\n';
  }
  void pose_row(const std::string& key,bool valid,bool init,double dt,double dr,const std::vector<double>& distances,double view,const Transform& est,const Transform& truth) {
    pose<<key<<valid<<','<<init<<',';
    if(valid) pose<<dt<<','<<dr<<',';else pose<<",,";
    if(!distances.empty()) pose<<percentile(distances,.5)<<','<<percentile(distances,.95)<<','<<*std::max_element(distances.begin(),distances.end());else pose<<",,";
    pose<<','<<distances.size()<<','<<view;
    for(const auto* t:{&est,&truth}) {
      const Eigen::Quaterniond q(t->linear());pose<<','<<t->translation().x()<<','<<t->translation().y()<<','<<t->translation().z()<<','<<q.w()<<','<<q.x()<<','<<q.y()<<','<<q.z();
    }pose<<'\n';
  }
  void series(const Scenario& scene,std::uint32_t seed,Method method,const std::string& lane,const Series& s) {
    summary<<scene.name<<','<<seed<<','<<method_name(method)<<','<<lane<<','<<scene.required<<','<<s.initialized<<','<<s.frames<<','<<s.valid<<','<<(s.frames?double(s.valid)/double(s.frames):0)<<','<<s.longest_fail<<','<<s.executed<<','<<s.mathematical_fail<<','<<percentile(s.translation,.95)<<','<<percentile(s.rotation,.95)<<','<<s.last_valid<<','<<s.last_translation<<','<<s.last_rotation<<','<<percentile(s.static_error,.5)<<','<<percentile(s.static_error,.95)<<','<<(s.static_error.empty()?0:*std::max_element(s.static_error.begin(),s.static_error.end()))<<','<<s.worst_frame_static<<','<<s.static_error.size()<<','<<percentile(s.times,.5)<<','<<percentile(s.times,.95)<<'\n';
  }
};
void write_pcd(const fs::path& path,const Points& points) {
  std::ofstream f(path);f<<"# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH "<<points.size()<<"\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS "<<points.size()<<"\nDATA ascii\n"<<std::setprecision(9);
  for(const auto& p:points) f<<p.x()<<' '<<p.y()<<' '<<p.z()<<'\n';
}
void isolation(const Config& c) {
  Scenario scene{"NORMAL_6DOF"};const auto points=structure(c);
  ShipTracker a(std::make_unique<SmallGicpBackend>(c),Method::GICP,c),b(std::make_unique<SmallGicpBackend>(c),Method::GICP,c);
  for(std::size_t frame=0;frame<8;++frame) {
    auto original=generate(scene,42,frame,points,c),modified=original;
    modified.scorer.T_W_S=pose({-4,99,300},{.2,1.3,2});modified.scorer.static_labels.assign(modified.scorer.static_labels.size(),false);
    modified.scorer.visible_identity.clear();modified.scorer.injected_time_error_ns=1e12;modified.scorer.injected_extrinsic.translation().setConstant(999);
    const auto x=a.process(original.runtime),y=b.process(modified.runtime);
    if(x.valid!=y.valid || x.T_W_B.matrix()!=y.T_W_B.matrix() || x.T_B_C.matrix()!=y.T_B_C.matrix() ||
       x.map.revision!=y.map.revision || x.map.candidate_revision!=y.map.candidate_revision || a.map().snapshot()->points!=b.map().snapshot()->points)
      throw std::runtime_error("GT_ISOLATION_FAILED");
  }
}
void capture(const Config& c,Output& out,const fs::path& directory) {
  const Scenario scene{"CAPTURE_RANGE",false};const auto points=structure(c);
  auto first=generate({"NORMAL_6DOF"},42,0,points,c),second=generate({"NORMAL_6DOF"},42,1,points,c);
  const auto target=prepare_observation(first.runtime,c),source=prepare_observation(second.runtime,c);
  auto snapshot=std::make_shared<const TargetSnapshot>(target.points,1);
  const Transform initial=target.T_W_C.inverse()*source.T_W_C;
  const Transform truth=target.T_W_C.inverse()*first.scorer.T_W_S*second.scorer.T_W_S.inverse()*source.T_W_C;
  std::ofstream csv(directory/"capture_range.csv");csv<<"index,yaw_deg,x_m,y_m,z_m,method,valid,translation_error_m,rotation_error_deg\n";
  std::vector<std::pair<double,Eigen::Vector3d>> perturbations;
  for(double sign:{-1.,1.}) {
    for(double yaw:c.acceptance.capture_yaw_deg) perturbations.push_back({sign*yaw,Eigen::Vector3d::Zero()});
    for(int axis=0;axis<3;++axis) for(double distance:c.acceptance.capture_translation_m) {Eigen::Vector3d p=Eigen::Vector3d::Zero();p[axis]=sign*distance;perturbations.push_back({0,p});}
    for(std::size_t i=0;i<c.acceptance.capture_yaw_deg.size();++i) perturbations.push_back({sign*c.acceptance.capture_yaw_deg[i],Eigen::Vector3d::Constant(sign*c.acceptance.capture_translation_m.at(i))});
  }
  SmallGicpBackend backend(c);
  for(std::size_t i=0;i<perturbations.size();++i) for(Method method:{Method::GICP,Method::VGICP}) {
    const auto& p=perturbations[i];const Transform guess=pose(p.second,{0,0,p.first*std::acos(-1.0)/180})*initial;
    const auto result=backend.align({source.points,snapshot,guess,method});
    const Transform error=truth.inverse()*result.T_target_source;const double dt=error.translation().norm(),dr=rotation_degrees(error.linear());
    csv<<i<<','<<p.first<<','<<p.second.x()<<','<<p.second.y()<<','<<p.second.z()<<','<<method_name(method)<<','<<result.valid<<','<<dt<<','<<dr<<'\n';
    out.result(Output::key(scene,42,i,method,"capture"),result);
  }
  // 改变 provisional 首帧艏向只改变坐标自由度，不能宣称绝对艏向恢复。
  std::ofstream gauge(directory/"bootstrap_gauge.csv");gauge<<"yaw_prior_deg,valid,fixed_world_prior_error_deg,interpretation\n";
  for(double yaw:{-5.,-3.,-1.,1.,3.,5.}) {
    ShipTracker tracker(std::make_unique<SmallGicpBackend>(c),Method::GICP,c);
    auto in=first.runtime;in.bootstrap_R_W_B=Eigen::AngleAxisd(yaw*std::acos(-1.0)/180,Eigen::Vector3d::UnitZ()).toRotationMatrix();
    const auto result=tracker.process(in);gauge<<yaw<<','<<result.valid<<','<<rotation_degrees(result.T_W_B.linear())<<",GAUGE_ONLY_NOT_ABSOLUTE_RECOVERY\n";
  }
}
int main(int argc,char** argv) {
 try {
  if(argc<3 || (std::string(argv[1])!="quick" && std::string(argv[1])!="full")) throw std::invalid_argument("用法: v14_evaluate quick|full 输出目录 [单场景诊断过滤]");
  const bool full=std::string(argv[1])=="full";const fs::path directory=argv[2];fs::create_directories(directory);
  const Config c;const std::size_t frames=std::size_t(full?c.acceptance.full_frames:c.acceptance.quick_frames);
  Output out(directory);isolation(c);capture(c,out,directory);
  const auto seed_list=full?c.acceptance.seeds:std::vector<std::uint32_t>{42};
  std::ofstream meta(directory/"metadata.json");meta<<"{\"schema\":\"ship_perception.v14.acceptance\",\"profile\":\""<<argv[1]<<"\",\"compiled_sha\":\""<<ship::cfg::git_sha<<"\",\"dataset_id\":\"v14_asymmetric_surface_v1\",\"random_streams\":\"scenario/run/frame/sensor/channel\",\"mode\":\"EVALUATION_MODE\",\"config_hash\":\""<<v14_config_hash<<"\",\"frames\":"<<frames<<",\"gt_isolation\":true,\"pcl_available\":"<<(pcl_backend_available()?"true":"false")<<",\"filtered\":"<<(argc>3?"true":"false")<<",\"site_status\":\"SITE_PENDING\"}\n";
  for(const auto& scene:scenarios(full)) {
    if(argc>3 && scene.name!=argv[3]) continue;
    const Points reference=structure(c,scene.name=="FLAT_DECK",scene.name=="REPEATED_HATCH");
    write_pcd(directory/(scene.name+"_reference.pcd"),reference);
    for(const auto seed:seed_list) {
      std::array<std::unique_ptr<ShipTracker>,2> trackers;
      std::array<Series,2> closed,benchmark;
      std::array<Transform,2> gauge{Transform::Identity(),Transform::Identity()};
      std::array<std::unique_ptr<PointIndex>,2> frozen_reference;
      std::array<Points,2> accumulated;
      const std::array<Method,2> methods{Method::GICP,Method::VGICP};
      for(int j=0;j<2;++j) trackers[j]=std::make_unique<ShipTracker>(std::make_unique<PerturbedBackend>(c,scene.name=="INITIAL_OFFSET"),methods[j],c);
      SmallGicpBackend fair(c);PreparedObservation previous;Transform previous_truth=Transform::Identity();std::set<std::uint64_t> previous_identity;
      for(std::size_t frame=0;frame<frames;++frame) {
        const auto packet=generate(scene,seed,frame,reference,c);const auto prepared=prepare_observation(packet.runtime,c);
        const double view=frame?view_change(previous_identity,packet.scorer.visible_identity):0;
        if(scene.name=="INDEPENDENT_VIEW_SAMPLING" && frame && (view<c.acceptance.view_change_min || view>c.acceptance.view_change_max)) throw std::runtime_error("独立视角集合变化率未达到 20%-40%");
        for(int j=0;j<2;++j) {
          const auto result=trackers[j]->process(packet.runtime);auto& series=closed[j];
          if(frame==0 && result.valid) {
            gauge[j]=packet.scorer.T_W_S.inverse()*result.T_W_B;series.initialized=true;
            Points ref;for(const auto& p:reference) ref.push_back((gauge[j].inverse()*p.cast<double>()).cast<float>());
            frozen_reference[j]=std::make_unique<PointIndex>(ref);
          }
          const Transform truth=packet.scorer.T_W_S*gauge[j];const Transform error=truth.inverse()*result.T_W_B;
          const double dt=error.translation().norm(),dr=rotation_degrees(error.linear());std::vector<double> distances;
          if(result.valid && frozen_reference[j]) {
            if(prepared.points.size()!=packet.scorer.static_labels.size()) throw std::runtime_error("评分标签计数不一致");
            for(std::size_t i=0;i<prepared.points.size();++i) {
              const Eigen::Vector3d p=result.T_B_C*prepared.points[i].cast<double>();
              if(frame%10==0) accumulated[j].push_back(p.cast<float>()); // 包含动态观测，不能冒充语义静态地图。
              if(!packet.scorer.static_labels[i]) continue;
              std::size_t index=0;double d=0;if(frozen_reference[j]->nearest(p,index,d)) distances.push_back(std::sqrt(d));
            }
            series.static_error.insert(series.static_error.end(),distances.begin(),distances.end());
            series.worst_frame_static=std::max(series.worst_frame_static,percentile(distances,.95));
          }
          if(frame) series.record(result.valid,dt,dr,result.registration);
          const auto key=Output::key(scene,seed,frame,methods[j],"closed_loop");out.pose_row(key,result.valid,frame==0,dt,dr,distances,view,result.T_W_B,truth);
          out.result(key,result.registration);out.timing<<key<<result.prepare_ms<<','<<result.registration_ms<<','<<result.map_ms<<','<<result.total_ms<<'\n';
          out.map<<key<<result.tracking_target_revision<<','<<result.map.revision<<','<<result.map.candidate_revision<<','<<result.map.active<<','<<result.map.suspect<<','<<result.map.quarantined<<','<<result.map.stable<<','<<result.map.candidates<<','<<result.needs_reinitialization<<'\n';
        }
        if(frame) {
          const auto snapshot=std::make_shared<const TargetSnapshot>(previous.points,frame);
          Transform guess=previous.T_W_C.inverse()*prepared.T_W_C;
          if(scene.name=="INITIAL_OFFSET") guess=pose({c.replay.initial_offset_m,0,0},{0,0,c.replay.initial_yaw_deg*std::acos(-1.0)/180})*guess;
          const Transform truth=previous.T_W_C.inverse()*previous_truth*packet.scorer.T_W_S.inverse()*prepared.T_W_C;
          for(int j=0;j<2;++j) {
            const RegistrationRequest request{prepared.points,snapshot,guess,methods[j]};
            const auto fingerprint=request_id(request);
            const auto result=fair.align(request);const Transform error=truth.inverse()*result.T_target_source;
            const double dt=error.translation().norm(),dr=rotation_degrees(error.linear());benchmark[j].initialized=true;benchmark[j].record(result.valid,dt,dr,result);
            const auto key=Output::key(scene,seed,frame,methods[j],"benchmark");out.result(key,result,fingerprint);out.pose_row(key,result.valid,false,dt,dr,{},view,result.T_target_source,truth);
            out.timing<<key<<0<<','<<result.elapsed_ms<<",0,"<<result.elapsed_ms<<'\n';
          }
          if(frame==1 && pcl_backend_available()) {
            auto pcl=make_pcl_backend(c);const auto result=pcl->align({prepared.points,snapshot,guess,Method::PCL_GICP});const Transform error=truth.inverse()*result.T_target_source;
            const auto key=Output::key(scene,seed,frame,Method::PCL_GICP,"pcl_comparison");out.result(key,result);out.pose_row(key,result.valid,false,error.translation().norm(),rotation_degrees(error.linear()),{},view,result.T_target_source,truth);
          }
        }
        previous=prepared;previous_truth=packet.scorer.T_W_S;previous_identity=packet.scorer.visible_identity;
      }
      for(int j=0;j<2;++j) {
        out.series(scene,seed,methods[j],"closed_loop",closed[j]);out.series(scene,seed,methods[j],"benchmark",benchmark[j]);
        const auto prefix=scene.name+"_"+std::to_string(seed)+"_"+method_name(methods[j]);
        write_pcd(directory/(prefix+"_accumulated_stride10.pcd"),accumulated[j]);write_pcd(directory/(prefix+"_tracking_reference.pcd"),trackers[j]->map().snapshot()->points);
        std::cout<<prefix<<" valid="<<closed[j].valid<<'/'<<closed[j].frames<<" t95="<<percentile(closed[j].translation,.95)<<" r95="<<percentile(closed[j].rotation,.95)<<" static95="<<percentile(closed[j].static_error,.95)<<std::endl;
      }
    }
  }
  return 0;
 } catch(const std::exception& e) {std::cerr<<"V14_EVALUATION_ERROR: "<<e.what()<<std::endl;return 1;}
}
