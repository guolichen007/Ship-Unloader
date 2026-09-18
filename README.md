# Ship Unloader — M0

Target: Ubuntu 20.04 LTS, C++17, GCC 9, CMake 3.16, system Eigen/PCL.

New implementation lives in `ship_perception/`. Existing local `hold_detector/`
is read-only and is not imported into this new repository (including its datasets
and historical build outputs). Tests include a small synthetic fixture, so they
do not depend on private or unversioned assets.

Scope: frames, timestamps, synthetic replay, crane deskew, dual-LiDAR fusion,
G1–G7 infrastructure. Registration, EKF, observability estimation, hatch polygons,
PLC control and site acceptance are outside M0. G5 tests the harness only.

Implementation owner: Windows Codex. Independent validation owner: Ubuntu 20.04
ClaudeCLI. No baseline exists until the independent report accepts an exact SHA.

See [build and validation instructions](ship_perception/README.md) and the
unchanged user-supplied specifications in `docs/specs/`.
