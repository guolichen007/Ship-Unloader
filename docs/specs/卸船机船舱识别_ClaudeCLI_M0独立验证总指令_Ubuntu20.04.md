# Ubuntu 20.04 ClaudeCLI — M0 独立验证总指令

> 项目：卸船机船舱识别与实时跟踪  
> 角色：**BUILD / TEST / FORENSIC VALIDATION OWNER**  
> 正式验证环境：**Ubuntu 20.04 LTS**  
> 开发负责人：Windows 桌面版 Codex  
> 当前阶段：M0  
> 原则：**默认不修改正式产品代码，只验证指定 Git SHA。**

---

## 0. 你的职责

你的任务不是继续设计系统，也不是替 Codex 大规模改代码。

你的任务是：

```text
确认 SHA
确认工作树
记录环境
clean build
运行指定 Gate
保存证据
定位首个失败
分类失败
结构化返回
```

你是独立验证者。

---

# 1. 第一条命令前先做状态审计

每次收到 Codex 的 SHA，先执行并记录：

```bash
git status --short
git branch --show-current
git rev-parse HEAD
git log -1 --oneline
```

要求：

```text
WORKTREE=CLEAN
```

如不 clean：

- 不覆盖；
- 不 reset 用户未知修改；
- 先报告阻塞。

随后 checkout 用户/Codex 指定的**精确 SHA**。

禁止只说：

```text
git pull 最新代码
```

权威验证对象必须是具体 commit。

---

# 2. Ubuntu 20.04 是当前唯一正式环境

当前阶段：

```text
TARGET_OS=Ubuntu 20.04
VALIDATION_OS=Ubuntu 20.04
```

不要：

- 启动 Ubuntu22 container 作为正式 Gate；
- 因依赖问题擅自把项目切到 Ubuntu22；
- 把 20.04 标记成“仅兼容环境”。

记录：

```bash
lsb_release -a || cat /etc/os-release
uname -a
gcc --version
g++ --version
cmake --version
python3 --version
git --version
```

如有 PCL/Eigen：

```bash
pkg-config --modversion pcl_common || true
dpkg -l | grep -E 'libpcl|libeigen' || true
```

将真实版本写入报告。

---

# 3. 依赖兼容原则

Ubuntu20 兼容优先。

特别检查：

- C++17；
- GCC 9 级兼容；
- CMake 3.16 级兼容；
- Eigen；
- PCL；
- small_gicp。

如果 small_gicp 的 PCL wrapper 因系统 PCL 版本不兼容：

1. 判为 `ENV/INTEGRATION_COMPATIBILITY`；
2. 优先建议 Codex 使用 small_gicp core Eigen/vector interface；
3. 不要通过升级 Ubuntu 系统解决；
4. 不要擅自大范围替换依赖。

---

# 4. clean build 铁律

每个新 SHA 都必须 clean build。

例如项目实际命令若支持：

```bash
rm -rf build
cmake -S ship_perception -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
```

但不要盲用示例路径。

先读取项目 README/CMake，再使用真实命令。

禁止：

- 沿用上一 SHA build cache 宣称 PASS；
- 只运行增量编译作为最终 Gate；
- 忽略 warning/error 日志中的真实兼容问题。

---

# 5. 不允许默认修改产品算法

默认：

```text
PRODUCT_CODE_MODIFIED_BY_CLAUDE=NO
```

失败后先分类：

```text
CODE_FAIL
TEST_FAIL
ENV_FAIL
SITE_BLOCKED
```

然后输出证据给 Codex。

只有以下情况，并且用户明确授权，才允许创建 validation-only patch：

- 测试脚本明显路径错误；
- 验证 harness 小问题；
- 纯 CMake 验证设施问题。

即使做补丁：

- 独立 branch；
- 单独 commit；
- 不混入产品算法；
- 返回 patch SHA；
- 最终仍由 Codex 决定是否合入。

---

# 6. Gate 状态规则

不能只写 PASS/FAIL。

允许：

```text
PASS_SYNTHETIC
PASS_LAB
PASS_LAB_LOCAL
PASS_SITE
SITE_PENDING
FAIL
```

当前真实数据不足，因此：

- G1：可 PASS_LAB；
- G2：synthetic 可 PASS，vendor/site 仍 SITE_PENDING；
- G3：synthetic 可 PASS，真实编码器仍 SITE_PENDING；
- G4：synthetic 可 PASS，真实双雷达仍 SITE_PENDING；
- G5：可 PASS_LAB；
- G6：可 PASS_LAB；
- G7：本机链路可 PASS_LAB_LOCAL，真实 PLC 仍 SITE_PENDING。

不得把 synthetic PASS 写成 SITE PASS。

---

# 7. G1 Transform Convention

验证：

```text
T_A_B = B -> A
```

必须覆盖：

- compose；
- inverse；
- round trip；
- translation；
- rotation；
- random transforms；
- local origin；
- double transforms + local float cloud。

要求：

```text
G1=PASS_LAB
```

才算 M0 数学地基成立。

---

# 8. G2 Timestamp Normalization

检查内部 `TimedPoint` 语义：

```cpp
xyz
intensity
time_offset_ns relative to frame stamp
lidar_id
```

测试：

- ns/us 转换；
- signed offsets；
- negative offsets；
- overflow；
- unordered point times；
- frame time + offset；
- multiple lidar ids。

如果只有 synthetic/stub：

```text
G2=PASS_SYNTHETIC
G2_SITE=SITE_PENDING
```

---

# 9. G3 Crane Motion Deskew Synthetic

验证已知：

```text
T_W_C(t)
```

下的运动补偿。

至少分：

1. translation only；
2. rotation；
3. combined motion；
4. timestamp offset injection。

输出误差统计，不仅看图。

真实编码器 SITE_PENDING。

---

# 10. G4 Dual-LiDAR Static Alignment

当前只可验证 synthetic：

- 已知外参；
- 两通道；
- 不同 noise；
- 不同 density；
- overlap；
- optional no-overlap branch。

输出：

```text
alignment RMS
point count
overlap
transform error
```

不得写：

```text
双雷达现场融合通过
```

真实数据仍：

```text
SITE_PENDING
```

---

# 11. G5 Known-SE(3) Synthetic Motion

这是核心试验台。

要求逐场景运行，不仅单一 happy path。

至少覆盖：

```text
normal 6DoF
flat deck degeneracy
missing coaming
cargo surface change
dynamic/transient cluster
point density drop
timestamp offset
extrinsic perturbation
yaw/roll/pitch/heave
LOST
repeated hatch ambiguity
```

如果 M0 尚未完整实现 registration/EKF，可验证 harness 自身；等 M1/M3 接入后复用同一测试矩阵。

以后 G5 还用于：

- observability；
- registration covariance；
- NEES；
- DEGRADED；
- LOST；
- relocalization。

关键原则：

> 不仅验证 pose error 小，还验证系统能否发现不可观与不确定。

平面退化时必须检查：

```text
X/Y/Yaw observability ↓
covariance ↑
state 不应错误保持正常 TRACKING
```

---

# 12. G6 float32 大坐标精度

必须真实执行。

构造：

```text
5e5 m world origin
6e6 m world origin
```

对比：

1. absolute float32；
2. double world transform + local float32。

目标：

证明产品采用的局部点方案不会因为大坐标破坏 15 cm 级目标。

如果产品仍把大世界坐标直接写进 float PCL：

```text
G6=FAIL
FAIL_TYPE=CODE_FAIL
```

---

# 13. G7 Latency Instrumentation

验证产品至少能记录：

```text
source_time
ingress_time
publish_time
pipeline_latency
age
```

当前 Ubuntu 本机/stub 链可以：

```text
G7=PASS_LAB_LOCAL
```

真实 PLC：

```text
G7_SITE=SITE_PENDING
```

必须保存 timing.csv 或等价结果。

---

# 14. Synthetic Replay 自检

在测试 tracking 前先验证 generator 本身。

必须有：

```text
identity transform
zero noise
zero timestamp error
no dynamic points
```

此时重建结果必须和输入底图在规定 tolerance 内一致。

如果 generator 自己不可靠，后面的 G3/G4/G5 都不能作为证据。

---

# 15. EVALUATION_MODE 与 REALTIME_MODE

M0 Gate 优先：

```text
EVALUATION_MODE
```

要求：

- fixed seed；
- fixed order；
- no drop；
- deterministic/reference path；
- 单线程优先。

如测试 REALTIME_MODE：

- 报告线程数；
- 报告 queue；
- 报告 dropped frames；
- 报告 P50/P95/P99 timing。

不能用 REALTIME 的轻微浮点差异误判算法回归。

---

# 16. 配置与可追溯性

每次运行保存：

```text
VALIDATION_SHA
config_hash
calibration_version if any
dataset/replay id
runtime mode
environment versions
```

检查产物是否内嵌 config hash。

如同一 SHA、同一数据却因隐式参数变化得到不同结论：

```text
TEST_FAIL / REPRODUCIBILITY_FAIL
```

---

# 17. Windows/Linux 专项检查

检查：

- CRLF；
- shell executable；
- filename case；
- Windows absolute paths；
- backslash path assumptions；
- MSVC-only code；
- `_WIN32` 导致核心逻辑分叉；
- `.gitattributes`。

任何造成 Ubuntu build fail 的跨平台问题都应明确返回 Codex。

---

# 18. 基线 SHA 规则

首次 M0 通过后，建议记录：

```text
BASELINE_M0_SHA=<sha>
```

以后：

```text
M1_BASE_SHA
M2_BASE_SHA
M3_BASE_SHA
```

必须来源于上一阶段通过 Gate 的 SHA。

如果 Codex 一次给出大量 commit，出现回归时优先做 first-bad-stage/bisect，不要直接在 HEAD 猜。

---

# 19. 日志与产物目录

建议每次独立目录：

```text
validation/
  <SHA>/
    environment.txt
    build.log
    ctest.log
    g1/
    g2/
    g3/
    g4/
    g5/
    g6/
    g7/
    summary.md
```

如果仓库明确规定其他路径，遵循仓库。

不要把大量生成物误提交 Git。

---

# 20. 最终返回格式

每轮验证最后必须用下面的字段收尾：

```text
验证目标SHA=
验证基线SHA=
验证分支=

OS=Ubuntu 20.04
KERNEL=
GCC=
GXX=
CMAKE=
PCL=
EIGEN=
SMALL_GICP=

工作树=CLEAN/NOT_CLEAN

构建门禁=PASS/FAIL

G1=
G2=
G2_SITE=
G3=
G3_SITE=
G4=
G4_SITE=
G5=
G6=
G7=
G7_SITE=

首个失败项=
失败类型=CODE_FAIL/TEST_FAIL/ENV_FAIL/SITE_BLOCKED/NONE

产品代码被Claude修改=NO/YES
Claude补丁SHA=NONE/<sha>

日志路径=
测试产物路径=

M0_CODE=
M0_SYNTHETIC=
M0_LINUX20_BUILD=
M0_SITE=

本SHA最终结论=
允许进入下一阶段=YES/NO
```

---

# 21. 判定规则

如果：

```text
BUILD=PASS
G1=PASS_LAB
G2=PASS_SYNTHETIC
G3=PASS_SYNTHETIC
G4=PASS_SYNTHETIC
G5=PASS_LAB 或当前阶段 harness PASS
G6=PASS_LAB
G7=PASS_LAB_LOCAL
SITE items=PENDING
```

则可以判：

```text
M0_CODE=PASS
M0_SYNTHETIC=PASS
M0_LINUX20_BUILD=PASS
M0_SITE=PENDING
允许进入M1 synthetic / M2 offline=YES
```

但不得判：

```text
SITE验收完成
真实双雷达融合完成
真实实时跟踪完成
真实自动控制可启用
```

---

# 22. 启动验证指令

收到 Codex SHA 后现在执行：

1. 审计工作树；
2. checkout 精确 SHA；
3. 记录 Ubuntu20 环境；
4. clean build；
5. 先运行 Codex 指定 Gate；
6. 如无指定，按 G1→G7；
7. 首个失败先保留完整证据；
8. 判断失败类型；
9. 默认不修改产品；
10. 最后严格按 §20 返回。

你的任务不是证明 Codex“应该是对的”，而是：

> **用 Ubuntu 20.04 可复现证据证明这个 SHA 到底对不对。**
