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
    // A thin observed rim is still a 2D Deck support domain. Its boundary
    // cells must not vanish merely because their centres define an ROI hull.
    Points rim;for(int x=-44;x<=44;++x)for(int y=-24;y<=24;++y){
      const float px=x*.1f+.05f,py=y*.1f+.05f;
      rim.emplace_back(px,py,(std::abs(px)<4&&std::abs(py)<2)?-3.f:0.f);
    }
    const auto thin=OfflineShipFrameProvider{}.resolve(rim);
    if(!thin.valid||(thin.cloud.T_B_input.linear().row(2)-Eigen::RowVector3d(0,0,1)).norm()>1e-5)throw std::runtime_error("THIN_DECK_DOMAIN_CLIPPED");
    DeckPlaneCandidate a,b,duplicate;a.valid=b.valid=duplicate.valid=true;
    a.quality_score=.8;b.quality_score=.75;duplicate.quality_score=.79;
    a.support_region={{-5,-3,0},{5,-3,0},{5,3,0},{-5,3,0}};duplicate=a;duplicate.quality_score=.79;
    b=a;b.quality_score=.75;b.plane.offset=1;for(auto& v:b.support_region)v.z()=-1;
    if(select_deck_candidate({a,b,duplicate},Config{}).valid || select_deck_candidate({duplicate,b,a},Config{}).valid)throw std::runtime_error("DUPLICATE_HID_COMPETING_PLANE");
    if(!select_deck_candidate({a,duplicate},Config{}).valid)throw std::runtime_error("EQUIVALENT_PLANE_REJECTED");
    if(std::abs(result.cloud.T_B_input.linear()(2,2)-1)>1e-5)throw std::runtime_error("UP");
    Transform rotated=Transform::Identity();rotated.linear()=(Eigen::AngleAxisd(.4,Eigen::Vector3d::UnitX())*Eigen::AngleAxisd(.2,Eigen::Vector3d::UnitY())).toRotationMatrix();
    for(auto& q:p)q=(rotated*q.cast<double>()).cast<float>();
    const auto raw=OfflineShipFrameProvider{}.resolve(p);if(!raw.valid || raw.cloud.input_alignment!=InputAlignment::RAW)throw std::runtime_error("RAW");
    if((raw.cloud.T_B_input.linear()*rotated.linear()*Eigen::Vector3d::UnitZ()-Eigen::Vector3d::UnitZ()).norm()>1e-5)throw std::runtime_error("DECK_NORMAL");
    std::cout<<"OFFLINE_DECK_NORMAL=PASS\n";
    // OFFLINE_FRAME_COPLANAR_BACKGROUND_INVARIANCE:
    // A far-away coplanar wharf patch must not move the offline Ship Frame
    // origin or long axis — only the connected owned support may define them
    // (not a re-scan of every coplanar point). This regression pins P0-1.
    {Points deck_only;
      for(double x=-8;x<=8;x+=.25)for(double y=-6;y<=6;y+=.25)deck_only.emplace_back(float(x),float(y),(std::abs(x)<5&&std::abs(y)<3)?-4.f:0.f);
      const auto base=OfflineShipFrameProvider{}.resolve(deck_only);
      if(!base.valid)throw std::runtime_error("COPLANAR_BG_BASE_UNRESOLVED");
      Points with_wharf=deck_only;
      for(double x=40;x<=45;x+=.25)for(double y=40;y<=45;y+=.25)with_wharf.emplace_back(float(x),float(y),0.f);
      const auto shifted=OfflineShipFrameProvider{}.resolve(with_wharf);
      if(!shifted.valid)throw std::runtime_error("COPLANAR_BG_WHARF_UNRESOLVED");
      const double origin_drift=(base.cloud.T_B_input.translation()-shifted.cloud.T_B_input.translation()).norm();
      const double axis_drift=(base.cloud.T_B_input.linear().row(2)-shifted.cloud.T_B_input.linear().row(2)).norm();
      std::cout<<"COPLANAR_BG_ORIGIN_DRIFT="<<origin_drift<<" AXIS_DRIFT="<<axis_drift<<"\n";
      if(origin_drift>0.5)throw std::runtime_error("COPLANAR_BACKGROUND_MOVED_ORIGIN");
      if(axis_drift>1e-3)throw std::runtime_error("COPLANAR_BACKGROUND_ROTATED_AXIS");}
    // STRUCTURAL_DATUM_PRESERVES_SHIP_AXIS:
    // The structure datum XY must come from the Ship Frame X projected onto
    // the Deck plane, never from the point cloud. A tilted Deck only tilts Z;
    // X stays in the ship X-Z plane. This regression pins P0-2.
    {const Plane flat{Eigen::Vector3d::UnitZ(),0.0};
      const auto d0=ship_datum_frame(flat);
      if(!healthy(d0))throw std::runtime_error("DATUM_FLAT_UNHEALTHY");
      if((d0.linear().row(2)-Eigen::RowVector3d(0,0,1)).norm()>1e-9)throw std::runtime_error("DATUM_Z_NOT_DECK_NORMAL");
      if((d0.linear().row(0)-Eigen::RowVector3d(1,0,0)).norm()>1e-9)throw std::runtime_error("DATUM_X_NOT_SHIP_AXIS");
      const Eigen::Vector3d n=Eigen::Vector3d(std::sin(.1),0.,std::cos(.1)).normalized();
      const Plane tilted{n,0.5};
      const auto d1=ship_datum_frame(tilted);
      if(!healthy(d1))throw std::runtime_error("DATUM_TILTED_UNHEALTHY");
      const Eigen::Vector3d Z=d1.linear().row(2).transpose();
      const Eigen::Vector3d X=d1.linear().row(0).transpose();
      const Eigen::Vector3d Y=d1.linear().row(1).transpose();
      if((Z-n).norm()>1e-9)throw std::runtime_error("DATUM_TILTED_Z");
      if(std::abs(X.y())>1e-9)throw std::runtime_error("DATUM_X_LEFT_SHIP_XZ_PLANE");
      if(std::abs(X.dot(Z))>1e-9)throw std::runtime_error("DATUM_X_NOT_ORTHOGONAL_Z");
      if((Y-Z.cross(X)).norm()>1e-9)throw std::runtime_error("DATUM_Y_NOT_CROSS");
      if((d1.translation()-Eigen::Vector3d(0,0,0.5)).norm()>1e-9)throw std::runtime_error("DATUM_ORIGIN_NOT_SHIP_PROJECTION");
      std::cout<<"STRUCTURAL_DATUM_PRESERVES_SHIP_AXIS=PASS\n";}
  }return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
