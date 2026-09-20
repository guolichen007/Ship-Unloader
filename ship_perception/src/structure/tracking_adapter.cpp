#include <ship_perception/structure/tracking_adapter.hpp>
#include <ship_perception/structure/geometry_fit.hpp>

namespace ship { namespace v15 {
bool ShipFrameAccumulator::append(const v14::TrackingInput& input,const v14::TrackingResult& result,
                                 const v14::Config& tracking_config){
  failure_reason_.clear();
  auto reject=[&](const char* reason){failure_reason_=reason;return false;};
  if(!result.valid)return reject("INVALID_TRACKING_FRAME");
  if(last_frame_ && input.frame_id<=*last_frame_)return reject("NONMONOTONIC_FRAME_ID");
  if(!healthy(result.T_B_C)||!healthy(result.T_W_B))return reject("INVALID_TRACKING_TRANSFORM");
  v14::PreparedObservation observation;
  try{observation=v14::prepare_observation(input,tracking_config);}
  catch(const std::exception& error){return reject(error.what());}
  if(observation.points.empty())return reject("EMPTY_OBSERVATION");
  if(cloud_.points.size()+observation.points.size()>std::size_t(config_.geometry.max_input_points))return reject("ACCUMULATION_CAPACITY_EXCEEDED");
  Points current;current.reserve(observation.points.size());
  for(const auto& p:observation.points){const Eigen::Vector3d q=result.T_B_C*p.cast<double>();
    if(!q.allFinite()||q.cwiseAbs().maxCoeff()>config_.geometry.max_local_extent_m)return reject("INVALID_TRANSFORMED_POINT");
    current.push_back(q.cast<float>());
  }
  // Commit only after the entire frame passed validation; a failed append
  // cannot leave a partly updated cloud or change the accepted-frame count.
  cloud_.points.insert(cloud_.points.end(),current.begin(),current.end());
  ++cloud_.accepted_frames;last_frame_=input.frame_id;return true;
}
}} // namespace ship::v15
