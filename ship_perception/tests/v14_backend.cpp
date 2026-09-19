#include <ship_perception/registration/backend.hpp>
#include <iostream>
#include <stdexcept>
using namespace ship;
using namespace ship::v14;
int main() {
  try {
    Points target;
    for(int x=-20;x<=20;++x) for(int y=-10;y<=10;++y) {
      target.emplace_back(float(x)*.14f,float(y)*.14f,0);
      if(y==-10 || x==20) for(int z=1;z<=8;++z) target.emplace_back(float(x)*.14f,float(y)*.14f,float(z)*.14f);
    }
    const Transform truth=pose({.08,-.04,.02},{.01,-.01,.02});
    Points source;for(const auto& p:target) source.push_back((truth.inverse()*p.cast<double>()).cast<float>());
    auto snapshot=std::make_shared<const TargetSnapshot>(target,1);
    SmallGicpBackend backend;
    for(Method method:{Method::GICP,Method::VGICP,Method::PCL_GICP}) {
      if(method==Method::PCL_GICP && !pcl_backend_available()) {std::cout<<"PCL_GICP=NOT_AVAILABLE\n";continue;}
      auto pcl=method==Method::PCL_GICP?make_pcl_backend(Config{}):nullptr;
      auto& engine=pcl?*pcl:static_cast<RegistrationBackend&>(backend);
      RegistrationRequest request{source,snapshot,Transform::Identity(),method};
      const auto result=engine.align(request);
      const double dt=(result.T_target_source.translation()-truth.translation()).norm();
      const double dr=rotation_degrees(result.T_target_source.linear().transpose()*truth.linear());
      std::cout<<method_name(method)<<" executed="<<result.backend_executed<<" converged="<<result.converged<<" valid="<<result.valid<<" translation="<<dt<<" rotation_deg="<<dr<<" reason="<<result.failure_reason<<std::endl;
      if(!result.backend_executed) throw std::runtime_error("真实后端未执行");
      if(!result.converged || !result.valid || result.mathematical_failure ||
         !healthy_transform(result.T_target_source,Config{}) || !std::isfinite(dt) || !std::isfinite(dr) || dt>.05 || dr>.5)
        throw std::runtime_error(std::string(method_name(method))+" 已知 SE3 有效性或精度门禁失败");
      request.initial_guess.matrix()(0,0)=2;
      auto invalid=engine.align(request);
      if(invalid.valid || invalid.backend_executed) throw std::runtime_error("非法初值被接受");
      Points empty; RegistrationRequest no_points{empty,snapshot,Transform::Identity(),method};
      if(engine.align(no_points).valid) throw std::runtime_error("空源点云被接受");
    }
    return 0;
  } catch(const std::exception& e) {std::cerr<<e.what()<<std::endl;return 1;}
}
