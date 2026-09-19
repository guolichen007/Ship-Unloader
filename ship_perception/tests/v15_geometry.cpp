#include <ship_perception/structure/linear_structure_detector.hpp>
#include <iostream>
using namespace ship::v15;
int main(){try{
  Points p;
  for(int ix=-120;ix<=120;++ix)for(int iy=-80;iy<=80;++iy){double x=ix*.05,y=iy*.05;bool low=((x>-4&&x<-1)||(x>1&&x<4))&&std::abs(y)<2;p.emplace_back(float(x),float(y),low?-3.f:0.f);}
  const auto structures=extract_structures(p,Config{});std::cout<<"OPENINGS="<<structures.openings.size()<<"\n";
  if(structures.openings.size()!=2)throw std::runtime_error("MULTI_OPENING");
  for(const auto& o:structures.openings){std::cout<<"EDGES="<<o.boundaries.size()<<"\n";for(const auto& e:o.boundaries)std::cout<<e.evidence_mechanism<<" "<<e.a.transpose()<<" -> "<<e.b.transpose()<<" support="<<e.observed_support_length<<"\n";if(o.boundaries.size()<4)throw std::runtime_error("VERTICAL_BLIND_BOUNDARY");}
  std::cout<<"TOP_DOWN_ZERO_VERTICAL_FACES=PASS\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
