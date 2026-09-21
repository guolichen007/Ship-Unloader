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
    {Config config;
      // Fixture 2D sampling is derived from the current candidate voxel so the
      // synthetic deck always satisfies the measured Deck-patch definition; a
      // hardcoded step could silently outlive a candidate_voxel_m change.
      const double fixture_step=config.geometry.candidate_voxel_m*0.5;
      std::cout<<"FIXTURE_STEP_M="<<fixture_step<<"\n";
      Points deck_only;
      for(double x=-8;x<=8;x+=fixture_step)for(double y=-6;y<=6;y+=fixture_step)deck_only.emplace_back(float(x),float(y),(std::abs(x)<5&&std::abs(y)<3)?-4.f:0.f);
      const auto base=OfflineShipFrameProvider{config}.resolve(deck_only);
      if(!base.valid)throw std::runtime_error("COPLANAR_BG_BASE_UNRESOLVED");
      Points with_wharf=deck_only;
      for(double x=40;x<=45;x+=fixture_step)for(double y=40;y<=45;y+=fixture_step)with_wharf.emplace_back(float(x),float(y),0.f);
      const auto shifted=OfflineShipFrameProvider{config}.resolve(with_wharf);
      if(!shifted.valid)throw std::runtime_error("COPLANAR_BG_WHARF_UNRESOLVED");
      // Physical Ship Frame origin in INPUT coordinates (B origin = centroid of
      // owned support). T_B_input.translation() is the input origin in B.
      const Eigen::Vector3d origin_base=base.cloud.T_B_input.inverse().translation();
      const Eigen::Vector3d origin_shifted=shifted.cloud.T_B_input.inverse().translation();
      const double owned_origin_drift=(origin_base-origin_shifted).norm();
      // Longitudinal axis: B.X expressed in input = row(0). Compare axes with an
      // absolute dot product so a 180° heading is treated as equivalent.
      const Eigen::Vector3d X_base=base.cloud.T_B_input.linear().row(0).transpose();
      const Eigen::Vector3d X_shifted=shifted.cloud.T_B_input.linear().row(0).transpose();
      const double owned_long_axis_cos=std::abs(X_base.dot(X_shifted));
      const double owned_z_drift=(base.cloud.T_B_input.linear().row(2)-shifted.cloud.T_B_input.linear().row(2)).norm();
      std::cout<<"OWNED_ORIGIN_DRIFT="<<owned_origin_drift<<" OWNED_LONG_AXIS_COS="<<owned_long_axis_cos<<" OWNED_Z_DRIFT="<<owned_z_drift<<"\n";
      if(owned_origin_drift>0.5)throw std::runtime_error("COPLANAR_BACKGROUND_MOVED_ORIGIN");
      if(owned_long_axis_cos<0.9999)throw std::runtime_error("COPLANAR_BACKGROUND_ROTATED_LONG_AXIS");
      // Legacy sensitivity negative control: reproduce the pre-fix "collect
      // every coplanar point then plane_frame" behaviour in test code only.
      // The fixture must visibly move the legacy frame, otherwise a PASS here
      // could just mean the wharf patch is too weak to expose the old bug.
      auto legacy_frame=[&](const Points& pts){
        const auto deck=detect_deck(pts,config);
        if(!deck.valid)throw std::runtime_error("LEGACY_DECK_UNRESOLVED");
        Points support;for(const auto& v:pts)if(std::abs(deck.plane.distance(v.cast<double>()))<=config.geometry.plane_inlier_m)support.push_back(v);
        return plane_frame(deck.plane,support);
      };
      const auto legacy_base=legacy_frame(deck_only),legacy_shifted=legacy_frame(with_wharf);
      const double legacy_origin_drift=(legacy_base.inverse().translation()-legacy_shifted.inverse().translation()).norm();
      const double legacy_long_axis_cos=std::abs(legacy_base.linear().row(0).dot(legacy_shifted.linear().row(0)));
      std::cout<<"LEGACY_COPLANAR_ORIGIN_DRIFT="<<legacy_origin_drift<<" LEGACY_LONG_AXIS_COS="<<legacy_long_axis_cos<<"\n";
      if(legacy_origin_drift<1.0)throw std::runtime_error("FIXTURE_LACKS_LEGACY_ATTACK_CAPABILITY");
      std::cout<<"OFFLINE_FRAME_COPLANAR_BACKGROUND_INVARIANCE=PASS\n";}
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
    // DATUM degenerate: Deck normal parallel to Ship X must fail closed. The
    // ship X projection collapses, ship_datum_frame yields NaN, and the
    // recognizer's healthy() check reports DATUM_AXIS_UNRESOLVED — never a
    // full-cloud PCA fallback.
    {const Plane parallel{Eigen::Vector3d::UnitX(),0.0};
      if(healthy(ship_datum_frame(parallel)))throw std::runtime_error("DATUM_PARALLEL_SHIP_X_NOT_REJECTED");
      const Plane antiparallel{-Eigen::Vector3d::UnitX(),0.0};
      if(healthy(ship_datum_frame(antiparallel)))throw std::runtime_error("DATUM_ANTIPARALLEL_SHIP_X_NOT_REJECTED");
      std::cout<<"DATUM_DEGENERACY=PASS\n";}
  }return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
