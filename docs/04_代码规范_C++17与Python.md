# 代码规范：C++17 与 Python

## 1. 总则

优先级：

```text
正确性
> 可追溯性
> 安全失败
> 可维护性
> 性能
> 代码短
```

禁止为了“写得漂亮”隐藏物理语义或错误状态。

## 2. C++17

### 2.1 命名

建议：

```text
TypeName
function_name()
local_variable
member_variable_
kConstantName
EnumType
```

物理量必须带单位：

```text
distance_m
angle_rad
timeout_ms
voxel_leaf_m
```

禁止：

```text
threshold
value
tmp2
data1
pose
transform
```

如果其物理含义不明确。

### 2.2 坐标变换

统一：

```cpp
Eigen::Isometry3d T_W_B;
Eigen::Isometry3d T_B_L;
```

禁止：

```cpp
Eigen::Matrix4f pose;
auto T = ...;
```

用于核心船体姿态。

变换方向必须体现在变量名中。

### 2.3 Ownership

- 优先值语义；
- 独占所有权：`std::unique_ptr`；
- 共享所有权只有确实需要才用 `std::shared_ptr`；
- raw pointer 默认仅表示 non-owning；
- 禁止 `new/delete` 分散在业务代码。

### 2.4 const / RAII

- 能 `const` 就 `const`；
- 文件、锁、资源采用 RAII；
- 不允许隐式全局可变状态；
- 配置对象初始化后尽量不可变。

### 2.5 Error Handling

算法失败是正常业务状态。

优先：

```text
Status / Result
enum class FailureReason
std::optional
```

例如：

```cpp
enum class LocalDeckStatus {
  kOk,
  kInsufficientSupport,
  kCompetingPlane,
  kDegenerateGeometry
};
```

禁止：

```text
失败返回空矩形但继续当成功
异常后静默 fallback Global Deck
NaN 继续传播
```

异常不得跨核心模块边界作为日常控制流。

### 2.6 数值

- SE(3)：`double`；
- 点云：可 `float`；
- 比较浮点使用显式 tolerance；
- 所有 tolerance 具名、可配置、带单位；
- 禁止 magic number。

### 2.7 Logging

日志必须带：

```text
stage
scan/frame id
candidate/hatch id
reason
key metrics
```

禁止高频逐点日志。

### 2.8 Header

- 头文件自包含；
- include 最小化；
- public API 不暴露不必要的第三方类型；
- 不在头文件 `using namespace`。

## 3. Python 3.8

### 3.1 基本规则

- `pathlib.Path`；
- type hints；
- 小函数；
- 无 wildcard import；
- 无模块级可变全局状态；
- 所有 CLI 使用 `argparse`；
- 文件编码显式 UTF-8。

### 3.2 Determinism

训练/随机采样：

```text
seed
config
input SHA
output SHA
```

必须记录。

### 3.3 JSON

禁止：

```text
NaN
Infinity
-Infinity
```

Schema validator 必须 fail-closed。

### 3.4 文件写入

关键 manifest/report：

```text
write temp
fsync/close
atomic rename
```

避免中途写坏。

## 4. C++ / Python 一致性

如果两端读取同一 PCD：

必须验证：

```text
point_count
bbox
origin
resolution
heightmap dimensions
selected sample values
```

历史 16-byte PCD record bug 已证明：

> **不同解析器不能假设一致，必须建立 parser parity test。**

## 5. API 与模块边界

建议：

```text
io/
proposal/
geometry/
hypothesis/
tracking/
evaluation/
common/
```

核心算法不得依赖：

```text
legacy hold_detector source
validation artifact path
developer local path
sealed annotation
```

## 6. 配置

配置优先：

```text
immutable config struct
explicit defaults
config dump
config hash
```

每次真实评测记录配置 SHA。

## 7. 注释

注释解释：

```text
为什么
物理假设
失败边界
单位
证据来源
```

不要重复代码本身。

## 8. 格式工具建议

C++：

```text
clang-format
clang-tidy（逐步启用）
```

Python：

```text
black 或 ruff format
ruff
```

一旦选定版本必须 pin，避免不同机器产生全文件格式漂移。
