#include <ship_perception/structure/tracking_adapter.hpp>
#include <iostream>
#include <limits>
using namespace ship;
void require(bool condition,const char* reason){if(!condition)throw std::runtime_error(reason);}
int main(){try{
  v14::TrackingInput input;input.frame_id=1;input.T_W_C=[](std::int64_t){return Transform::Identity();};
  v14::SensorObservation sensor;
  TimedPoint point;point.xyz={1,2,3};sensor.points.push_back(point);
  point.xyz={8,9,10};sensor.points.push_back(point); // transient returns have no removable GT flag
  sensor.T_C_L.translation()=Eigen::Vector3d(.1,.2,.3);input.sensors.push_back(sensor);
  v14::TrackingResult result;result.valid=true;result.T_B_C.translation()=Eigen::Vector3d(3,4,5);
  v15::ShipFrameAccumulator accumulator;
  require(accumulator.append(input,result),"VALID_FRAME_REJECTED");
  require(accumulator.cloud().points.size()==2&&accumulator.cloud().accepted_frames==1,"DYNAMIC_POINT_DROPPED");
  require((accumulator.cloud().points[0].cast<double>()-Eigen::Vector3d(4.1,6.2,8.3)).norm()<1e-5,"ESTIMATED_TRANSFORM_NOT_USED");
  require(!accumulator.cloud().offline&&accumulator.cloud().T_B_input.matrix().isIdentity(),"FIXED_GAUGE_CHANGED");
  require(!accumulator.append(input,result),"DUPLICATE_FRAME_ACCEPTED");
  input.frame_id=2;result.valid=false;require(!accumulator.append(input,result),"INVALID_FRAME_ACCEPTED");
  result.valid=true;result.T_B_C.linear()(0,0)=2;require(!accumulator.append(input,result),"NON_SE3_ACCEPTED");
  result.T_B_C=Transform::Identity();result.T_B_C.translation().x()=std::numeric_limits<double>::quiet_NaN();
  require(!accumulator.append(input,result),"NAN_ACCEPTED");
  require(accumulator.cloud().points.size()==2&&accumulator.cloud().accepted_frames==1,"REJECTION_MUTATED_CLOUD");
  result.T_B_C=Transform::Identity();input.T_W_C={};
  require(!accumulator.append(input,result)&&accumulator.cloud().points.size()==2,"INVALID_RUNTIME_INPUT_MUTATED_CLOUD");
  input.T_W_C=[](std::int64_t){return Transform::Identity();};
  v15::Config capacity;capacity.geometry.max_input_points=1;v15::ShipFrameAccumulator small(capacity);
  result.T_B_C=Transform::Identity();require(!small.append(input,result)&&small.cloud().points.empty(),"PARTIAL_CAPACITY_COMMIT");
  std::cout<<"V14_ESTIMATED_POSE_ADAPTER=PASS\n";return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
