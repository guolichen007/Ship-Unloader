# M0 implementation

All transforms use `T_A_B = B -> A`, `p_A = T_A_B * p_B` and
`Eigen::Isometry3d`. Points are local float32; world transforms and origins are
double. Canonical ship axes are bow/port/deck-up, origin near hatch-row center.

Ubuntu 20.04 dependencies:

```sh
sudo apt-get install build-essential cmake git python3 libeigen3-dev libpcl-dev pkg-config
cmake -S ship_perception -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
# CMake 3.16-compatible invocation (avoid newer ctest --test-dir).
(cd build && ctest --output-on-failure)
```

Windows builds are developer checks only. Site data and vendor timestamp/PLC
adapters remain SITE_PENDING. small_gicp is not required in M0.
