#include <ship_perception/structure/offline_ship_frame.hpp>
#include <iostream>
#include <fstream>
using namespace ship::v15;
int main(int argc,char** argv) {try {
  Points p;
  if(argc==2) {
    std::ifstream f(argv[1],std::ios::binary);char magic[8];std::uint64_t n=0;f.read(magic,8);f.read(reinterpret_cast<char*>(&n),8);
    if(std::string(magic,7)!="SXYZV15"||n>12000000)throw std::runtime_error("CACHE_HEADER");
    for(std::uint64_t i=0;i<n;++i){float xyz[3];f.read(reinterpret_cast<char*>(xyz),12);p.emplace_back(xyz[0],xyz[1],xyz[2]);}
    if(!f)throw std::runtime_error("CACHE_TRUNCATED");
  }else{
    for(double x=-6;x<=6;x+=.1)for(double y=-4;y<=4;y+=.1){double z=(std::abs(x)<4 && std::abs(y)<2)?-3:0;p.emplace_back(float(x),float(y),float(z));}
  }
  const auto result=OfflineShipFrameProvider{}.resolve(p);
  std::cout<<"FRAME="<<result.valid<<" SCORE="<<result.deck.quality_score<<" SUPPORT="<<result.deck.support_count<<" REASON="<<result.reason<<"\n";
  if(!result.valid)return 1;
  std::cout<<"T_B_INPUT\n"<<result.cloud.T_B_input.matrix()<<"\n";
  if(argc==1) {
    if(std::abs(result.cloud.T_B_input.linear()(2,2)-1)>1e-5)throw std::runtime_error("UP");
    Transform rotated=Transform::Identity();rotated.linear()=(Eigen::AngleAxisd(.4,Eigen::Vector3d::UnitX())*Eigen::AngleAxisd(.2,Eigen::Vector3d::UnitY())).toRotationMatrix();
    for(auto& q:p)q=(rotated*q.cast<double>()).cast<float>();
    const auto raw=OfflineShipFrameProvider{}.resolve(p);if(!raw.valid || raw.cloud.input_alignment!=InputAlignment::RAW)throw std::runtime_error("RAW");
    if((raw.cloud.T_B_input.linear()*rotated.linear()*Eigen::Vector3d::UnitZ()-Eigen::Vector3d::UnitZ()).norm()>1e-5)throw std::runtime_error("DECK_NORMAL");
    std::cout<<"OFFLINE_DECK_NORMAL=PASS\n";
  }return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
