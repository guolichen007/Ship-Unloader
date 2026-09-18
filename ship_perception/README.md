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

## Configuration and evidence

`config/m0.schema.json` defines version 1. `config/m0.json` is the sole default
parameter/threshold source. A build can select another complete config with
`-DM0_CONFIG=/path/to/config.json`. Configuration is frozen at configure time;
its exact bytes are hashed with SHA-256 and embedded in every gate report.
Reconfigure after changing checkout SHA. The official script always uses the
versioned default config and a new build directory.

`T_world_ship` is GT only. Frame clouds remain in local LiDAR coordinates;
deskew returns crane coordinates at `t_ref`. All recorded world poses and
`local_origin_world` are double. The fixture's ship axes are bow/port/deck-up.
Replay time starts at simulation epoch zero; vendor clock mapping remains pending.

## Replay export

```sh
python3 ship_perception/tools/replay.py --executable build/synthetic_replay \
  --input ship_perception/tests/fixtures/local_xyz.pcd --output validation/my-replay
# Omit --input to use the 378-point synthetic ship fixture.
```

Use a new output directory each time. Outputs are `points.csv` (per-point local
XYZ, intensity, lidar ID, frame stamp, reported offset, true time, source index),
`gt.csv` (double world pose), `calibration.csv` (true/reported extrinsics), and
`metadata.json` (SHA/config hash/input SHA-256/origin/mode/UTC timestamp).
Dynamic points have source index -1. Input PCD must already be LOCAL XYZ;
supplying absolute float32 UTM coordinates cannot recover precision already lost.
Choose its double origin with `local_origin_world_m` in the build configuration.
The default export deliberately enables disturbances. For ideal export, set
noise/drop/outlier/dynamic/timestamp/extrinsic perturbations to zero, decimation
to 1 and occlusion off in a separate versioned config.

The generator API supports independent crane and ship trajectories, smooth seeded
motion, drop/decimation, dynamic clusters, ROI occlusion, timing faults and sensor
extrinsics. Tests use isolated perturbations with numeric error assertions.
PCL is confined to the I/O adapter; internal points do not depend on PCL.

## Gate scope

| Gate | M0 evidence | Ubuntu independent target |
|---|---|---|
| G1 | compose/inverse/random transforms/local origin | PASS_LAB |
| G2 | checked signed/unsigned ns/us, overflow, order/IDs | PASS_SYNTHETIC |
| G3 | translation/rotation/combined crane deskew, timing fault | PASS_SYNTHETIC |
| G4 | dual extrinsics, noise/density/overlap/no-overlap | PASS_SYNTHETIC |
| G5 | 11 scenarios × 5 frames, GT and injected faults | PASS_LAB, HARNESS_ONLY |
| G6 | 5e5/6e6 origins, absolute vs local float precision | PASS_LAB |
| G7 | clock-domain checks, measured CSV, queue contracts | PASS_LAB_LOCAL |

Raw executables report PASS_SYNTHETIC/FAIL on every platform. The Ubuntu script
maps verified evidence to the target levels. ClaudeCLI must independently review
before acceptance. G5 does not estimate pose, covariance, observability or LOST;
its requirements are explicit future test oracles. No site PASS is generated.

EVALUATION_MODE is single-threaded, fixed seed/order, FIFO without frame drops.
Synthetic measurement point loss is deliberate data corruption, not queue loss.
REALTIME_MODE's queue is bounded and retains the latest frame on consumption;
M0 queue methods are single-threaded and require external synchronization if shared.
G7 records P50/P95/P99 for the local evaluation pipeline, separately tests realtime
queue behavior, and does not claim a site deadline or realtime throughput.

## Independent validation

Follow [exact-SHA handoff](../docs/VALIDATION_HANDOFF.md). On Ubuntu 20.04:

```sh
bash ship_perception/scripts/validate_ubuntu20.sh --sha "$(git rev-parse HEAD)" --base-sha NONE
```

For a Windows developer-only build without PCL, configure `-DM0_WITH_PCL=OFF`
and supply an installed Eigen3 package. PCD calls then fail explicitly; the
official Ubuntu validator forces PCL ON and also runs the PCD self-check.
There is no `_WIN32` fork in core math.
