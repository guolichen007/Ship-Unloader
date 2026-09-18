# Windows 桌面版 Codex — M0 实现总指令（Ubuntu 20.04 目标）

> 项目：卸船机船舱识别与实时跟踪  
> 角色：**IMPLEMENTATION OWNER**  
> 当前阶段：M0  
> 正式目标系统：**Ubuntu 20.04 LTS**  
> 验证负责人：Ubuntu 20.04 ClaudeCLI  
> 架构基线：V1.2/V1.3 Ubuntu20.04 归档冻结版  
> 核心规则：**你负责写代码，不负责宣布 Linux Gate PASS。**

---

## 0. 你的任务

在 Windows 桌面版 Codex 环境中实现 M0：

```text
repository scaffold
replay framework
sensor adapters / stubs
frame/types
synthetic replay generator
G1~G7 测试
Linux20 兼容 CMake
验证脚本
```

代码必须可在 Ubuntu 20.04 上由 ClaudeCLI 独立 clean build 和测试。

第一阶段禁止进入：

```text
正式 Hatch Polygon
正式 Control Prior 业务链
真实 PLC 控制
Phase 2 AI
大规模 M1/M2/M3 实现
```

---

# 1. 开工前必须先做

先读取并确认：

1. 当前 Git 分支；
2. HEAD SHA；
3. `git status --short`；
4. 仓库目录结构；
5. `hold_detector/` 现状；
6. 是否已有 `ship_perception/`；
7. 现有 CMake；
8. Windows 工作区是否存在未提交修改。

返回：

```text
START_SHA=
BRANCH=
WORKTREE=
```

如工作树不干净，不允许覆盖未知修改；先说明。

---

# 2. 三条铁律

## 2.1 `hold_detector/` 只读

禁止修改、移动、格式化：

```text
hold_detector/**
```

允许：

- 读取；
- 参考算法；
- 读取静态 PCD/JSON 测试数据。

新实现全部放入：

```text
ship_perception/
```

## 2.2 Ubuntu 20.04 是唯一正式目标系统

不要：

- 添加 Ubuntu22 专属配置；
- 使用 Ubuntu22 才有的软件包接口；
- 要求 Claude 使用 Ubuntu22 container；
- 把 Windows build 结果当正式验证。

兼容基线：

```text
C++17
GCC 9 级别
CMake 3.16 级别
Ubuntu20.04 系统 Eigen/PCL
```

## 2.3 每一个开发单元都必须可交给 Claude 独立验证

不要一次实现 M0+M1+M2。

优先小 commit。

---

# 3. 必须冻结的数学约定

## 3.1 Transform

永远使用：

```text
T_A_B = B -> A
p_A = T_A_B * p_B
```

必须实现单元测试：

```text
T_A_C == T_A_B * T_B_C
T_A_B * T_B_A == I
```

所有 frame transform 用：

```cpp
Eigen::Isometry3d
```

## 3.2 点与世界坐标

- 世界/Frame 变换：double；
- 点：局部 float32；
- 禁止将 UTM/大场地世界绝对坐标直接写进 PCL float 点；
- 消息/落盘保存 `local_origin_world`。

## 3.3 Canonical Ship Frame

不要在 M0 中擅自改变：

```text
x = bow
y = port
z = deck normal upward
deck datum ~= z0
origin ~= hatch-row center
```

---

# 4. M0 目录目标

至少建立：

```text
ship_perception/
├── CMakeLists.txt
├── cmake/
├── include/ship_perception/
│   ├── core/
│   ├── sensor/
│   ├── replay/
│   └── timing/
├── src/
│   ├── core/
│   ├── sensor/
│   ├── replay/
│   └── timing/
├── tests/
│   ├── g1_transform/
│   ├── g2_timestamp/
│   ├── g3_deskew/
│   ├── g4_dual_lidar/
│   ├── g5_known_se3/
│   ├── g6_float_precision/
│   └── g7_latency/
├── tools/
├── config/
└── scripts/
```

可根据仓库实际结构微调，但不得破坏逻辑分层。

---

# 5. 内部核心类型

优先实现小而稳定的数据类型。

至少需要：

```cpp
struct TimedPoint {
  Eigen::Vector3f xyz;
  float intensity;
  int64_t time_offset_ns;
  uint16_t lidar_id;
};

struct LocalOrigin {
  Eigen::Vector3d world_xyz;
};

struct TimestampInfo {
  int64_t source_time_ns;
  int64_t ingress_time_ns;
  int64_t publish_time_ns;
};

struct PoseStamped {
  int64_t timestamp_ns;
  Eigen::Isometry3d T_world_body;
};
```

不要在 M0 提前做复杂产品模型。

---

# 6. small_gicp / PCL 约束

为了 Ubuntu 20.04 兼容：

- 不把配准核心设计成必须依赖 small_gicp 的 PCL wrapper；
- 建立内部点容器/适配层；
- PCL 负责 PCD I/O 为主；
- small_gicp 以后优先走 core Eigen/vector 路径；
- PCL GICP 保留成 benchmark/fallback；
- M0 阶段如果 small_gicp 尚未接入，不要为了它阻塞 G1/G2/G6/G7 脚手架。

---

# 7. Synthetic Replay Generator

这是 M0 最重要产物之一。

输入：

```text
static PCD
trajectory config
sensor config
noise config
```

至少支持：

1. 已知 SE(3) 轨迹；
2. XY 漂移；
3. heave；
4. roll/pitch/yaw 正弦；
5. 平滑随机运动；
6. 点噪声；
7. 离群点；
8. 点丢失/降采样；
9. 动态点团；
10. 遮挡 ROI；
11. timestamp offset/jitter；
12. 双 LiDAR 仿真；
13. 外参扰动；
14. known GT pose。

第一条自检：

```text
identity pose + zero distortion
=> reconstructed cloud equals source within tolerance
```

不要用“看起来一致”作为 PASS。

---

# 8. G1~G7 必须实现

## G1 Transform Convention

测试：

- compose；
- inverse；
- round trip；
- translation/rotation；
- random transforms；
- local/world origin。

## G2 Timestamp Normalization

建立 vendor-agnostic adapter test。

至少测试：

- signed/unsigned offsets；
- ns/us 单位；
- frame stamp + offset；
- overflow；
- negative offset；
- monotonically ordered 与非 ordered 输入。

真实 vendor 字段 SITE_PENDING。

## G3 Crane Motion Deskew Synthetic

已知：

```text
T_W_C(t)
```

生成扫描期点，然后恢复到 `t_ref`。

先验证纯平移，再加旋转。

## G4 Dual-LiDAR Static Alignment Synthetic

用静态底图 + 已知：

```text
T_C_L1
T_C_L2
```

生成两通道。

验证融合数学。

真实外参 SITE_PENDING。

## G5 Known-SE(3) Synthetic Motion

第一阶段可以先搭完整 harness；配准器后续 M1 接入。

测试矩阵必须能表示：

- 6DoF；
- 平甲板退化；
- missing coaming；
- cargo change；
- transient points；
- density drop；
- time offset；
- extrinsic error；
- LOST；
- repeated hatch ambiguity。

G5 以后承担 observability/NEES/relocalization。

## G6 Large Coordinate Precision

构造：

```text
world origin ~ 5e5 m
world origin ~ 6e6 m
local geometry centimeter/decimeter scale
```

证明：

- float32 世界绝对坐标会失真；
- double transform + local float cloud 保持目标精度。

## G7 Latency Instrumentation

M0 至少实现字段和测量框架：

```text
source_time
ingress_time
publish_time
pipeline_latency
age
```

PLC 实链路 SITE_PENDING。

---

# 9. Gate 状态

你的测试输出不得只使用 PASS/FAIL。

必须支持：

```text
PASS_SYNTHETIC
PASS_LAB
PASS_LAB_LOCAL
PASS_SITE
SITE_PENDING
FAIL
```

Windows Codex 自己不能给正式：

```text
PASS_LAB
PASS_SITE
```

除非它对应的是平台无关纯数学单元测试，并且 Ubuntu ClaudeCLI 后续仍要复验。

最终权威结果来自 Ubuntu20 ClaudeCLI。

---

# 10. EVALUATION_MODE / REALTIME_MODE

M0 开始就预留模式。

`EVALUATION_MODE`：

- deterministic；
- fixed seed；
- fixed order；
- no drop；
- single-thread 优先。

`REALTIME_MODE`：

- bounded queue；
- latest frame wins；
- parallel 可开启。

不要要求 bit-exact。

---

# 11. 参数与可追溯

配置必须：

```text
schema
version
hash
```

测试产物写入：

```text
git_sha
config_hash
dataset_id
mode
timestamp
```

禁止在代码里散落不可追溯 magic threshold。

---

# 12. 跨 Windows/Linux 规则

建立 `.gitattributes`：

```text
*.cpp text eol=lf
*.cc text eol=lf
*.c text eol=lf
*.h text eol=lf
*.hpp text eol=lf
*.cmake text eol=lf
CMakeLists.txt text eol=lf
*.sh text eol=lf
*.py text eol=lf
*.yaml text eol=lf
*.yml text eol=lf
```

禁止：

- Windows 绝对路径；
- case-insensitive 假设；
- shell CRLF；
- Visual Studio-only 代码；
- `#ifdef _WIN32` 把核心逻辑改成两套；
- 把“Windows 能运行”写成正式验收。

---

# 13. 建议 commit 顺序

建议但不强制：

```text
1 chore: add linux20 project scaffold and line-ending rules
2 feat: add core frame and local-coordinate types
3 test: add G1 transform convention gate
4 feat: add TimedPoint and timestamp normalization
5 test: add G2 timestamp gate
6 feat: add synthetic replay generator foundation
7 test: add G6 large-coordinate precision gate
8 feat: add crane-motion deskew core
9 test: add G3 synthetic deskew gate
10 feat: add dual-lidar synthetic adapter/fuser
11 test: add G4 synthetic alignment gate
12 feat: add G5 known-SE3 test harness
13 feat: add latency instrumentation
14 test: add G7 local latency gate
15 chore: add Ubuntu20 validation scripts
```

每个 commit 尽量单一 scope。

---

# 14. 每次交给 Claude 的返回格式

每个可验证 SHA，必须返回给用户：

```text
IMPLEMENTATION_SHA=
BASE_SHA=
BRANCH=

SCOPE=
CHANGED_FILES=

EXPECTED_GATES=
G1=
G2=
...

WINDOWS_TESTS_RUN=
WINDOWS_RESULTS=

NOT_VERIFIED_ON_UBUNTU20=
KNOWN_LIMITATIONS=
SITE_PENDING_ITEMS=

CLAUDECLI_VALIDATION_INSTRUCTION=
```

严禁写：

```text
已全部通过
```

除非 Ubuntu20 ClaudeCLI 真正返回通过。

---

# 15. 失败修复规则

Claude 返回失败后：

- `CODE_FAIL`：你修代码；
- `TEST_FAIL`：先确认测试是否符合冻结规格，再修产品；
- `ENV_FAIL`：不要擅自改架构；做兼容性修复；
- `SITE_BLOCKED`：保留 SITE_PENDING，不造假数据填结果。

不得：

- 降低 Gate 阈值只为了变 PASS；
- 删除 failing test；
- 把异常 catch 后忽略；
- 用 Windows 结果覆盖 Ubuntu 失败。

---

# 16. 当前终点

本轮你只需要把项目带到：

```text
M0_CODE_READY=YES
M0_SYNTHETIC_TESTS_READY=YES
UBUNTU20_VALIDATION_READY=YES
```

随后提交 SHA 给 Ubuntu20 ClaudeCLI。

不要自行进入完整 M1/M2，直到用户拿到 ClaudeCLI 的验证报告。

---

# 17. 启动执行指令

现在开始：

1. 审计仓库；
2. 记录 BASE_SHA；
3. 确认 `hold_detector/` 不修改；
4. 建立 Ubuntu20-compatible `ship_perception/` M0 scaffold；
5. 从 G1 开始；
6. 小 commit；
7. 每个阶段返回结构化结果；
8. 不等待 SITE_TBD 才开工，但 SITE_TBD 必须保留为接口/stub；
9. 不重开算法架构讨论。

你的目标不是“一次写很多”，而是：

> **让每一个 SHA 都能被 Ubuntu 20.04 ClaudeCLI 独立证明。**
