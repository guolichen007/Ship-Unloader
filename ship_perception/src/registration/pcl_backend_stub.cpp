#include <ship_perception/registration/backend.hpp>
namespace ship { namespace v14 {
std::unique_ptr<RegistrationBackend> make_pcl_backend(const Config&) {
  throw std::runtime_error("PCL 对照未构建；正式 Ubuntu 通道必须启用 M0_WITH_PCL");
}
bool pcl_backend_available() {return false;}
}}
