#include <ship_perception/structure/geometry_fit.hpp>
#include <small_gicp/ann/kdtree.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>
#include <set>
#include <stdexcept>
#include <climits>

namespace ship { namespace v15 {
struct LocalIndex::Impl {
  std::shared_ptr<small_gicp::PointCloud> cloud;
  std::unique_ptr<small_gicp::KdTree<small_gicp::PointCloud>> tree;
  explicit Impl(const Points& p):cloud(std::make_shared<small_gicp::PointCloud>(p)) {
    if(!p.empty()) tree=std::make_unique<small_gicp::KdTree<small_gicp::PointCloud>>(cloud);
  }
};
LocalIndex::LocalIndex(const Points& p):impl_(std::make_unique<Impl>(p)){}
LocalIndex::~LocalIndex()=default;
std::vector<std::size_t> LocalIndex::neighbors(const Eigen::Vector3d& q,int k,double radius) const {
  if(!q.allFinite() || k<=0 || !std::isfinite(radius) || radius<=0 || !impl_->tree)return {};
  std::vector<std::size_t> ids(k);std::vector<double> d(k);
  const auto n=impl_->tree->knn_search(Eigen::Vector4d(q.x(),q.y(),q.z(),1),k,ids.data(),d.data());
  std::vector<std::size_t> out;
  for(std::size_t i=0;i<n;++i)if(d[i]<=radius*radius)out.push_back(ids[i]);
  return out;
}
double quantile(std::vector<double> v,double q) {
  if(v.empty())return 0;
  std::sort(v.begin(),v.end());return v[std::min(v.size()-1,std::size_t(std::ceil(q*v.size())-1))];
}
bool healthy(const Transform& t) {
  return t.matrix().allFinite() && (t.matrix().row(3)-Eigen::RowVector4d(0,0,0,1)).norm()<1e-9 &&
    std::abs(t.linear().determinant()-1)<1e-6 && (t.linear().transpose()*t.linear()-Eigen::Matrix3d::Identity()).norm()<1e-6;
}
Points voxelize(const Points& p,double size) {
  if(!std::isfinite(size)||size<=0)throw std::invalid_argument("INVALID_VOXEL_SIZE");
  using Key=std::array<std::int64_t,3>;
  struct Sum {Eigen::Vector3d sum=Eigen::Vector3d::Zero();std::size_t n=0;};
  std::map<Key,Sum> cells;
  for(const auto& point:p) {
    if(!point.allFinite() || point.cwiseAbs().maxCoeff()>1000)throw std::invalid_argument("INVALID_LOCAL_POINT");
    Key k;for(int a=0;a<3;++a)k[a]=std::int64_t(std::floor(double(point[a])/size));
    auto& s=cells[k];s.sum+=point.cast<double>();++s.n;
  }
  Points out;out.reserve(cells.size());for(const auto& kv:cells)out.push_back((kv.second.sum/double(kv.second.n)).cast<float>());
  return out;
}
std::vector<Normal> normals(const Points& p,const Config& c) {
  LocalIndex index(p);std::vector<Normal> out(p.size());
  for(std::size_t i=0;i<p.size();++i) {
    auto ids=index.neighbors(p[i].cast<double>(),int(c.geometry.normal_k),c.geometry.normal_radius_m);
    if(ids.size()<std::size_t(c.geometry.normal_min_points))continue;
    Eigen::Vector3d mean=Eigen::Vector3d::Zero();for(auto id:ids)mean+=p[id].cast<double>();mean/=double(ids.size());
    Eigen::Matrix3d cov=Eigen::Matrix3d::Zero();for(auto id:ids){const Eigen::Vector3d d=p[id].cast<double>()-mean;cov+=d*d.transpose();}
    cov/=double(ids.size());Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(cov);
    const auto e=eig.eigenvalues();if(eig.info()!=Eigen::Success || e[2]<1e-12 || e[1]<1e-12)continue;
    // A narrow but two-dimensional strip still has a stable plane normal.
    // Separate rank-two support from orthogonal residual, instead of confusing
    // anisotropic sampling with a non-planar surface.
    out[i].direction=eig.eigenvectors().col(0);out[i].planarity=1-std::max(0.,e[0])/e[1];
    out[i].spread=std::sqrt(std::max(0.,e[0])/e[1]);out[i].valid=out[i].planarity>=c.geometry.planarity_min && e[1]/e[2]>=c.geometry.normal_min_secondary_ratio;
  }return out;
}
PlaneFit fit_plane(const Points& p,const std::vector<std::size_t>& initial,const Eigen::Vector3d& up,const Config& c) {
  PlaneFit out;auto ids=initial;
  for(int pass=0;pass<4;++pass) {
    if(ids.size()<std::size_t(c.geometry.min_plane_points))return out;
    Eigen::Vector3d mean=Eigen::Vector3d::Zero();for(auto i:ids)mean+=p[i].cast<double>();mean/=double(ids.size());
    Eigen::Matrix3d cov=Eigen::Matrix3d::Zero();for(auto i:ids){const Eigen::Vector3d d=p[i].cast<double>()-mean;cov+=d*d.transpose();}
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(cov);if(eig.info()!=Eigen::Success || eig.eigenvalues()[1]<1e-9)return out;
    Eigen::Vector3d n=eig.eigenvectors().col(0);if(n.dot(up)<0)n=-n;
    out.plane={n,-n.dot(mean)};
    std::vector<std::size_t> trimmed;
    for(auto i:initial)if(std::abs(out.plane.distance(p[i].cast<double>()))<=c.geometry.plane_inlier_m)trimmed.push_back(i);
    if(trimmed==ids)break;ids=std::move(trimmed);
  }
  if(ids.size()<std::size_t(c.geometry.min_plane_points))return out;
  std::vector<double> residual;for(auto i:ids)residual.push_back(std::abs(out.plane.distance(p[i].cast<double>())));
  out.inliers=std::move(ids);out.p50=quantile(residual,.5);out.p95=quantile(residual,.95);out.valid=true;return out;
}
Transform plane_frame(const Plane& plane,const Points& p) {
  Eigen::Vector3d mean=Eigen::Vector3d::Zero();for(const auto& q:p)mean+=q.cast<double>();if(!p.empty())mean/=double(p.size());
  mean-=plane.distance(mean)*plane.normal;
  Eigen::Vector3d a=plane.normal.unitOrthogonal(),b=plane.normal.cross(a);
  Eigen::Matrix2d cov=Eigen::Matrix2d::Zero();for(const auto& q:p){Eigen::Vector3d d=q.cast<double>()-mean;Eigen::Vector2d u(a.dot(d),b.dot(d));cov+=u*u.transpose();}
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eig(cov);Eigen::Vector2d axis=eig.eigenvectors().col(1);
  Eigen::Vector3d x=a*axis.x()+b*axis.y();Eigen::Index major; x.cwiseAbs().maxCoeff(&major);if(x[major]<0)x=-x;
  Eigen::Matrix3d R;R.col(0)=x;R.col(1)=plane.normal.cross(x);R.col(2)=plane.normal;
  Transform t=Transform::Identity();t.linear()=R.transpose();t.translation()=-R.transpose()*mean;return t;
}
HeightGrid height_grid(const Points& p,const Config& c) {
  HeightGrid grid;const double s=c.geometry.coarse_voxel_m;
  for(std::size_t i=0;i<p.size();++i) {
    CellKey k{int(std::floor((p[i].x()-c.geometry.grid_phase_x_m)/s)),int(std::floor((p[i].y()-c.geometry.grid_phase_y_m)/s))};
    grid[k].ids.push_back(i);
  }
  for(auto& kv:grid){auto& cell=kv.second;std::vector<double> z;for(auto i:cell.ids)z.push_back(p[i].z());cell.lo=quantile(z,.1);cell.hi=quantile(z,.9);cell.median=quantile(z,.5);cell.center={s*(kv.first[0]+.5)+c.geometry.grid_phase_x_m,s*(kv.first[1]+.5)+c.geometry.grid_phase_y_m};}
  return grid;
}
std::vector<Opening> find_openings(const HeightGrid& grid,double z,const Config& c) {
  std::set<CellKey> lows;
  CellKey minimum{INT_MAX,INT_MAX},maximum{INT_MIN,INT_MIN};
  for(const auto& kv:grid)for(int axis=0;axis<2;++axis){minimum[axis]=std::min(minimum[axis],kv.first[axis]);maximum[axis]=std::max(maximum[axis],kv.first[axis]);}
  // A high cross-member does not erase actual returns beneath it. This is a
  // measured lower layer, never an inference of free space from an empty cell.
  for(const auto& kv:grid)if(kv.second.lo<z-c.roi.opening_drop_m && kv.second.lo>z-c.roi.max_opening_depth_m)lows.insert(kv.first);
  std::set<CellKey> visited;std::vector<Opening> out;const int dx[]={-1,1,0,0},dy[]={0,0,-1,1};
  for(const auto& start:lows) {
    if(visited.count(start))continue;
    Opening opening;std::queue<CellKey> todo;todo.push(start);visited.insert(start);std::size_t faces=0,supported=0;
    while(!todo.empty()) {
      auto k=todo.front();todo.pop();opening.cells.push_back(k);
      for(int axis=0;axis<2;++axis)if(k[axis]==minimum[axis] || k[axis]==maximum[axis])opening.touches_scan_boundary=true;
      for(int a=0;a<4;++a){CellKey next{k[0]+dx[a],k[1]+dy[a]};
        if(lows.count(next)){if(visited.insert(next).second)todo.push(next);continue;}
        if(!grid.count(next))opening.touches_scan_boundary=true;
        ++faces;bool support=false;
        for(int step=1;step<=int(std::ceil(c.roi.support_search_m/c.geometry.coarse_voxel_m));++step){
          auto it=grid.find({k[0]+dx[a]*step,k[1]+dy[a]*step});
          if(it!=grid.end() && std::abs(it->second.median-z)<c.roi.support_band_m){support=true;break;}
        }
        if(support)++supported;
        const auto& cell=grid.at(k);opening.boundary.push_back(cell.center+Eigen::Vector2d(dx[a],dy[a])*c.geometry.coarse_voxel_m*.5);
      }
    }
    opening.enclosure=faces?double(supported)/faces:0;
    const double area=opening.cells.size()*c.geometry.coarse_voxel_m*c.geometry.coarse_voxel_m;
    if(opening.cells.size()>=std::size_t(c.roi.min_opening_cells) && area>=c.roi.opening_min_area_m2 && area<=c.roi.opening_max_area_m2 && opening.enclosure>=c.roi.min_enclosure_ratio)out.push_back(std::move(opening));
  }return out;
}
double polygon_area(const std::vector<Eigen::Vector2d>& p) {double sum=0;for(std::size_t i=0;i<p.size();++i){auto& a=p[i];auto& b=p[(i+1)%p.size()];sum+=a.x()*b.y()-a.y()*b.x();}return sum*.5;}
std::vector<Eigen::Vector2d> convex_support_hull(std::vector<Eigen::Vector2d> p) {
  std::sort(p.begin(),p.end(),[](const auto& a,const auto& b){return a.x()<b.x()||(a.x()==b.x()&&a.y()<b.y());});
  p.erase(std::unique(p.begin(),p.end(),[](const auto& a,const auto& b){return a==b;}),p.end());
  if(p.size()<3)return {};
  auto cross=[](const auto& a,const auto& b,const auto& c){const Eigen::Vector2d u=b-a,v=c-a;return u.x()*v.y()-u.y()*v.x();};
  std::vector<Eigen::Vector2d> h;
  for(const auto& q:p){while(h.size()>1&&cross(h[h.size()-2],h.back(),q)<=0)h.pop_back();h.push_back(q);}
  const auto lower=h.size();
  for(auto i=p.rbegin()+1;i!=p.rend();++i){while(h.size()>lower&&cross(h[h.size()-2],h.back(),*i)<=0)h.pop_back();h.push_back(*i);}
  h.pop_back();return h;
}
bool inside(const Eigen::Vector2d& q,const std::vector<Eigen::Vector2d>& p) {
  bool result=false;if(p.empty())return false;
  for(std::size_t i=0,j=p.size()-1;i<p.size();j=i++)if((p[i].y()>q.y())!=(p[j].y()>q.y()) && q.x()<(p[j].x()-p[i].x())*(q.y()-p[i].y())/(p[j].y()-p[i].y())+p[i].x())result=!result;
  return result;
}
}} // namespace ship::v15
