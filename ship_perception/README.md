# 船体稳定跟踪工程

所有变换采用 `T_A_B: B → A`，变换与世界坐标为 double，局部点云为 float。small_gicp 固定源码随仓库提供，配置和构建不联网下载。核心接口不暴露 PCL 或 small_gicp 类型。

## Ubuntu20 构建

先安装系统依赖，再执行离线构建：

```bash
sudo apt-get install build-essential cmake git python3 libeigen3-dev libpcl-dev pkg-config
/usr/bin/cmake -S ship_perception -B build -DCMAKE_BUILD_TYPE=Release \
  -DEigen3_DIR=/usr/share/eigen3/cmake -DM0_WITH_PCL=ON
/usr/bin/cmake --build build --parallel 2
(cd build && /usr/bin/ctest --output-on-failure)
python3 ship_perception/tools/run_v14.py build/v14_evaluate full validation/v14-full-development
```

Eigen 配置目录因安装位置而异。正式脚本优先 `/usr/lib/cmake/eigen3`，其次 `/usr/share/eigen3/cmake`，记录实际版本、配置和头文件路径；要求系统 Eigen3.3.7。

正式验证请使用[精确 SHA 统一交接](../docs/V1.4版本_Ubuntu20验证交接说明.md)，它会新建构建目录并强制 PCL-ON。Quick 不能替代 Full。

## Windows 开发构建

使用支持 C++17 的编译器和 Eigen3.3.7。无 PCL 时只允许开发侧 `M0_WITH_PCL=OFF`，PCL 工厂显式报不可用。

```powershell
cmake -S ship_perception -B build-v14 -G 'Visual Studio 17 2022' -A x64 `
  -DEigen3_DIR='<Eigen3Config.cmake 所在目录>' -DM0_WITH_PCL=OFF
cmake --build build-v14 --config Release -j2
ctest --test-dir build-v14 -C Release --output-on-failure
python ship_perception/tools/run_v14.py build-v14/Release/v14_evaluate.exe full validation/v14-full-windows
```

Windows 结果只作为开发验证。PCL PCD 读写、PCL GICP 的编译运行和 Ubuntu20 系统依赖链必须由 Ubuntu 正式通道验证。

## 配置、门禁与产物

M0 的 `config/m0.json` 与原阈值保持不变。V1.4 使用 `config/v14.json` 和独立 schema，原始字节 SHA-256 嵌入程序；改变配置或 SHA 后重新配置构建。可用 `V14_CONFIG` 选择完整实验配置，但正式验证器使用仓库默认配置。

| 入口 | 内容 |
|---|---|
| G1/G2/G3/G4/G6/G7 | 原坐标、时间戳、吊机补偿、双雷达、float 精度与延时回归 |
| g5_harness | 原 11 类场景的生成、GT 与故障注入断言 |
| v14_backend | 真实 GICP/VGICP 已知 SE(3) 能力；PCL-ON 时追加对照 |
| v14_safety | 数学非法输出、错误合法位姿、候选、隔离、容量、revision 和支撑不足 |
| v14_validation_boundary | 缺证据、后端拼接、benchmark 不公平、非法接受及地图污染的验收反例 |
| G5 Quick | 50 帧、seed42、五个核心场景、两后端 benchmark 与独立闭环、GT 隔离和捕获诊断 |
| Full Acceptance | 200 帧、42/1337/2026、11 个必测及 6 个诊断场景、同一后端整体放行 |

主要产物为 `pose.csv`、`quality.csv`、`timing.csv`、`map.csv`；另有逐序列汇总、捕获范围、坐标自由度诊断、配置与环境元数据、哈希清单和统一 `report.json`。参考地图 PCD 与每十帧抽取的累积观测 PCD 分开保存；累积观测包含动态点，不能冒充语义静态地图。

## 旧回放导出

```bash
python3 ship_perception/tools/replay.py --executable build/synthetic_replay \
  --input ship_perception/tests/fixtures/local_xyz.pcd --output validation/my-replay
```

不提供 `--input` 则使用原 378 点夹具。导出包含局部点、报告时间、隐藏真值、外参和来源索引，属于离线评分数据，不能整体传入跟踪器。PCD 必须已经是局部坐标；绝对 float UTM 坐标已经丢失的精度不可恢复。
