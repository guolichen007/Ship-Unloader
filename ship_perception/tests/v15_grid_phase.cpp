#include <ship_perception/structure/structural_recognizer.hpp>
#include <iostream>
using namespace ship::v15;
int main(){try{
  Points points;
  for(int x=-120;x<=120;++x)for(int y=-80;y<=80;++y){const double a=x*.05,b=y*.05;points.emplace_back(float(a),float(b),(a>-4&&a<4&&b>-2&&b<2)?-3.f:0.f);}
  for(double phase:{0.,.13,.27,.41}){
    Config config;config.geometry.grid_phase_x_m=phase;config.geometry.grid_phase_y_m=phase*.5;
    const auto model=StructuralRecognizer(config).recognize_offline(points);
    if(!model.frame_resolved||model.hatches.size()!=1||model.hatches[0].status!=HatchStatus::COMPLETE_OBSERVED)throw std::runtime_error("L0_PHASE_TOPOLOGY");
    const auto inverse=model.T_B_input.inverse();
    for(const auto& edge:model.hatches[0].boundaries){
      const auto a=inverse*edge.a,b=inverse*edge.b;
      for(double u:{0.,.25,.5,.75,1.}){const Eigen::Vector3d q=a+u*(b-a);
        const double distance=std::min(std::abs(std::abs(q.x())-4),std::abs(std::abs(q.y())-2));
        if(distance>.15)throw std::runtime_error("L0_GRID_USED_AS_FINAL_EDGE");
      }
    }
  }
  std::cout<<"L0_PHASE_INDEPENDENCE=PASS\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
