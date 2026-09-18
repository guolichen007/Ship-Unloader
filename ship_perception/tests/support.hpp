#pragma once
#include <ship_perception/core/types.hpp>
#include <ship_perception/config.hpp>
#include <cmath>
#include <ctime>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

namespace test {
inline void check(bool condition, const char* expression, int line) {
  if (!condition) throw std::runtime_error(std::string(expression) + " at line " + std::to_string(line));
}
inline void near(double actual, double expected, double tolerance, int line) {
  check(std::isfinite(actual) && std::isfinite(expected) &&
        std::abs(actual - expected) <= tolerance, "numeric tolerance", line);
}
template<class F> void throws(F f) {
  bool caught = false;
  try { f(); } catch (const std::exception&) { caught = true; }
  check(caught, "expected exception", __LINE__);
}
inline std::string json_string(const std::string& s) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c);
    else out << c;
  }
  out << '"';
  return out.str();
}
using Metrics = std::map<std::string, double>;
inline int run(const std::string& gate, const std::function<void(Metrics&)>& body) {
  Metrics metrics;
  std::string error;
  try { body(metrics); } catch (const std::exception& e) { error = e.what(); }
  const char* status = error.empty() ? "PASS_SYNTHETIC" : "FAIL";
  std::ofstream out("report.json");
  if (!out) { std::cerr << "cannot open report.json\n"; return 2; }
  out << std::setprecision(17) << "{\n\"gate\":" << json_string(gate)
      << ",\n\"status\":\"" << status << "\",\n\"git_sha\":\"" << ship::cfg::git_sha
      << "\",\n\"config_hash\":\"" << ship::cfg::config_hash
      << "\",\n\"dataset_id\":\"" << ship::cfg::dataset_id
      << "\",\n\"calibration_version\":\"" << ship::cfg::calibration_version
      << "\",\n\"mode\":\"EVALUATION_MODE\",\n\"timestamp\":" << std::time(nullptr)
      << ",\n\"scope\":\"M0 harness; no registration or site acceptance\",\n\"error\":"
      << json_string(error) << ",\n\"metrics\":{";
  bool first = true;
  for (const auto& m : metrics) {
    if (!first) out << ',';
    first = false;
    out << json_string(m.first) << ':' << m.second;
  }
  out << "}}\n";
  out.close();
  if (!out) return 2;
  std::cout << gate << '=' << status << " " << error << '\n';
  return error.empty() ? 0 : 1;
}
} // namespace test
#define CHECK(x) test::check(static_cast<bool>(x), #x, __LINE__)
#define NEAR(a,b,t) test::near((a),(b),(t),__LINE__)
