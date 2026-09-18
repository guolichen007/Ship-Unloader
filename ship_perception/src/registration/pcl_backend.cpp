#include <ship_perception/registration/backend.hpp>
#include <ship_perception/timing/latency.hpp>
#include <pcl/registration/gicp.h>

namespace ship { namespace v14 {
namespace {
class InstrumentedGicp : public pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ,pcl::PointXYZ> {
 public:
  std::size_t iterations() const {return static_cast<std::size_t>(this->nr_iterations_);}
};
class PclBackend final : public RegistrationBackend {
 public:
  explicit PclBackend(Config c):config_(std::move(c)) {}
  RegistrationResult align(const RegistrationRequest& q) override {
    const auto start=monotonic_ns(); RegistrationResult result;
    if(q.method!=Method::PCL_GICP) {result.failure_reason="UNSUPPORTED_METHOD";return result;}
    if(!healthy_transform(q.initial_guess,config_,&result.failure_reason)) return result;
    if(!q.target) {result.failure_reason="MISSING_TARGET";return result;}
    try {
      auto convert=[&](const Points& raw) {
        const auto points=preprocess(raw,config_);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        for(const auto& p:points) cloud->push_back(pcl::PointXYZ(p.x(),p.y(),p.z()));
        return cloud;
      };
      const auto source=convert(q.source), target=convert(q.target->points);
      if(source->size()<std::size_t(config_.registration.min_points) || target->size()<std::size_t(config_.registration.min_points)) {
        result.failure_reason="INSUFFICIENT_POINTS";return result;
      }
      InstrumentedGicp backend;
      backend.setInputSource(source);backend.setInputTarget(target);
      backend.setMaximumIterations(int(config_.registration.max_iterations));
      backend.setCorrespondenceRandomness(int(config_.registration.neighbors));
      backend.setMaxCorrespondenceDistance(config_.registration.correspondence_m);
      backend.setTransformationEpsilon(config_.registration.translation_epsilon_m*config_.registration.translation_epsilon_m);
      pcl::PointCloud<pcl::PointXYZ> aligned;
      result.backend_executed=true;
      backend.align(aligned,q.initial_guess.matrix().cast<float>());
      result.T_target_source.matrix()=backend.getFinalTransformation().cast<double>();
      result.converged=backend.hasConverged();
      result.iterations=backend.iterations();result.iterations_available=true;
      PointIndex index(q.target->points);
      result=validate_result(result,q,config_,index);
    } catch(const std::exception& e) {result.failure_reason=std::string("PCL_ERROR: ")+e.what();result.valid=false;}
    result.elapsed_ms=double(monotonic_ns()-start)*1e-6;
    return result;
  }
 private: Config config_;
};
}
std::unique_ptr<RegistrationBackend> make_pcl_backend(const Config& c) {return std::make_unique<PclBackend>(c);}
bool pcl_backend_available() {return true;}
}}
