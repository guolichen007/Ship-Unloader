// Validation-only producer/scorer. No GT type crosses the runtime API boundary.
#include "v14_fixture.hpp"
#include <ship_perception/structure/tracking_adapter.hpp>
#include <ship_perception/structure/structural_recognizer.hpp>
#include <ship_perception/structure/model_io.hpp>
#include <ship_perception/config.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

int main(int argc,char** argv){try{
  if(argc!=8)throw std::invalid_argument("usage: v15_integration reference.xyzbin GICP|VGICP frames seed CLEAN|DYNAMIC output_dir run_id");
  const auto reference=ship::v15::read_xyz_cache(argv[1]);
  const std::string method_name=argv[2],scene=argv[5];
  const auto method=method_name=="GICP"?ship::v14::Method::GICP:method_name=="VGICP"?ship::v14::Method::VGICP:throw std::invalid_argument("METHOD");
  const auto frames=std::stoul(argv[3]),seed=std::stoul(argv[4]);if(frames<2||frames>1000||(scene!="CLEAN"&&scene!="DYNAMIC"))throw std::invalid_argument("PROFILE");
  const std::filesystem::path output=argv[6];std::filesystem::create_directories(output);
  ship::v14::Config tracking_config;
  ship::v14::ShipTracker tracker(std::make_unique<ship::v14::SmallGicpBackend>(tracking_config),method,tracking_config);
  ship::v15::ShipFrameAccumulator accumulator;
  ship::Transform T_B_S=ship::Transform::Identity();bool gauge_initialized=false;
  std::size_t accepted=0,failed_run=0,max_failed_run=0,backend_calls=0;
  double worst_translation=0,worst_rotation=0;
  std::ofstream poses(output/"poses.csv");poses<<std::setprecision(17);
  poses<<"frame,valid,backend_executed,translation_error_m,rotation_error_deg,source_points,target_revision,map_revision,prepare_ms,registration_ms,map_ms,total_ms,failure_reason";
  for(int r=0;r<4;++r)for(int c=0;c<4;++c)poses<<",T_B_C_"<<r<<c;poses<<'\n';
  const validation::Scenario scenario{scene=="DYNAMIC"?"TRANSIENT":"NORMAL_6DOF"};
  for(std::size_t frame=0;frame<frames;++frame){
    auto packet=validation::generate(scenario,std::uint32_t(seed),frame,reference,tracking_config);
    const auto result=tracker.process(packet.runtime);
    if(result.registration.backend_executed)++backend_calls;
    if(frame==0&&result.valid){
      const auto prepared=ship::v14::prepare_observation(packet.runtime,tracking_config);
      T_B_S=result.T_B_C*prepared.T_W_C.inverse()*packet.scorer.T_W_S;gauge_initialized=true;
    }
    const bool appended=accumulator.append(packet.runtime,result,tracking_config);
    poses<<frame<<','<<appended<<','<<result.registration.backend_executed<<',';
    if(appended&&gauge_initialized){
      ++accepted;failed_run=0;
      const ship::Transform expected=packet.scorer.T_W_S*T_B_S.inverse();
      const double translation=(result.T_W_B.translation()-expected.translation()).norm();
      const double rotation=ship::v14::rotation_degrees(result.T_W_B.linear()*expected.linear().transpose());
      worst_translation=std::max(worst_translation,translation);worst_rotation=std::max(worst_rotation,rotation);
      poses<<translation<<','<<rotation;
    }else{max_failed_run=std::max(max_failed_run,++failed_run);poses<<',';}
    poses<<','<<result.registration.source_points<<','<<result.tracking_target_revision<<','<<result.map.revision
         <<','<<result.prepare_ms<<','<<result.registration_ms<<','<<result.map_ms<<','<<result.total_ms<<','
         <<(appended?"":accumulator.failure_reason());
    for(int r=0;r<4;++r)for(int c=0;c<4;++c)poses<<','<<result.T_B_C.matrix()(r,c);poses<<'\n';
  }
  if(!poses)throw std::runtime_error("POSE_WRITE_FAILED");
  ship::v15::StructuralRecognizer recognizer;ship::v15::RecognitionTimings stages;
  const auto model=recognizer.recognize(accumulator.cloud(),&stages);
  std::ofstream model_file(output/"model.json");ship::v15::write_model(model_file,model);
  ship::v15::write_xyz_cache(accumulator.cloud().points,(output/"accumulated.xyzbin").string());
  ship::v15::write_xyz_pcd(accumulator.cloud().points,(output/"accumulated.pcd").string());
  std::ofstream timing(output/"model.json.timing.json");timing<<"{\"offline_frame_ms\":"<<stages.offline_frame_ms<<",\"deck_ms\":"<<stages.deck_ms<<",\"structure_ms\":"<<stages.structure_ms<<",\"topology_ms\":"<<stages.topology_ms<<"}\n";
  std::ofstream meta(output/"integration.json");meta<<std::setprecision(17);
  meta<<"{\"schema\":\"ship_perception.v15.integration.1\",\"method\":\""<<method_name<<"\",\"scene\":\""<<scene<<"\",\"compiled_sha\":\""<<ship::cfg::git_sha
      <<"\",\"frames\":"<<frames<<",\"seed\":"<<seed<<",\"accepted_frames\":"<<accepted<<",\"backend_calls\":"<<backend_calls
      <<",\"max_consecutive_failures\":"<<max_failed_run<<",\"point_count\":"<<accumulator.cloud().points.size()
      <<",\"dynamic_points_filtered\":false,\"offline_rebootstrap\":false,\"gauge_initialized\":"<<(gauge_initialized?"true":"false")
      <<",\"worst_translation_m\":";
  if(accepted>1)meta<<worst_translation;else meta<<"null";
  meta<<",\"worst_rotation_deg\":";if(accepted>1)meta<<worst_rotation;else meta<<"null";
  meta<<",\"T_B_S_initial\":[";
  for(int r=0;r<4;++r){if(r)meta<<',';meta<<'[';for(int c=0;c<4;++c){if(c)meta<<',';meta<<T_B_S.matrix()(r,c);}meta<<']';}meta<<"]}\n";
  if(!model_file||!meta||!timing)throw std::runtime_error("ARTIFACT_WRITE_FAILED");
  std::cout<<method_name<<' '<<scene<<" ACCEPTED="<<accepted<<'/'<<frames<<" POINTS="<<accumulator.cloud().points.size()<<'\n';return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
