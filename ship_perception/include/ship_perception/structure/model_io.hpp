#pragma once
#include <ship_perception/structure/types.hpp>
#include <iosfwd>
namespace ship { namespace v15 {
Points read_xyz_cache(const std::string& path,const Config& config=Config{});
void write_xyz_cache(const Points& p,const std::string& path);
void write_model(std::ostream& out,const StructuralModelCandidate& model);
}} // namespace ship::v15
