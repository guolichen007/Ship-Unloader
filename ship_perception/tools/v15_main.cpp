#include <ship_perception/structure/structural_recognizer.hpp>
#include <ship_perception/structure/model_io.hpp>
#include <ship_perception/sensor/pcd.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <chrono>
int main(int argc,char** argv){try{
  if(argc!=4){std::cerr<<"usage: v15_recognize offline|ship-frame|pcd-offline|pcd-cache input output\n";return 2;}
  const auto started=std::chrono::steady_clock::now();
  const std::string mode=argv[1];ship::v15::Points points;
  if(mode=="pcd-offline"||mode=="pcd-cache")points=ship::load_local_pcd(argv[2]);
  else if(mode=="offline"||mode=="ship-frame")points=ship::v15::read_xyz_cache(argv[2]);else throw std::runtime_error("INVALID_MODE");
  if(mode=="pcd-cache"){ship::v15::write_xyz_cache(points,argv[3]);return 0;}
  const auto loaded=std::chrono::steady_clock::now();const auto point_count=points.size();
  const ship::v15::StructuralRecognizer recognizer;ship::v15::StructuralModelCandidate result;ship::v15::RecognitionTimings stages;
  if(mode=="ship-frame"){ship::v15::ShipFrameCloud cloud;cloud.points=std::move(points);result=recognizer.recognize(cloud,&stages);}else result=recognizer.recognize_offline(points,&stages);
  const auto recognized=std::chrono::steady_clock::now();
  const std::filesystem::path output=argv[3];if(!output.parent_path().empty())std::filesystem::create_directories(output.parent_path());
  std::ofstream stream(output);ship::v15::write_model(stream,result);if(!stream)throw std::runtime_error("MODEL_WRITE_FAILED");
  stream.close();const auto written=std::chrono::steady_clock::now();
  const auto ms=[](auto a,auto b){return std::chrono::duration<double,std::milli>(b-a).count();};
  std::ofstream timing(output.string()+".timing.json");
  timing<<"{\"input_point_count\":"<<point_count<<",\"input_ms\":"<<ms(started,loaded)<<",\"recognize_ms\":"<<ms(loaded,recognized)
        <<",\"offline_frame_ms\":"<<stages.offline_frame_ms<<",\"deck_ms\":"<<stages.deck_ms<<",\"structure_ms\":"<<stages.structure_ms<<",\"topology_ms\":"<<stages.topology_ms
        <<",\"serialize_ms\":"<<ms(recognized,written)<<",\"total_ms\":"<<ms(started,written)<<"}\n";
  if(!timing)throw std::runtime_error("TIMING_WRITE_FAILED");
  std::cout<<"FRAME="<<result.frame_resolved<<" DECK="<<result.deck.valid<<" HATCHES="<<result.hatches.size()<<"\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
