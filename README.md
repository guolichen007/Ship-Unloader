# Ship-Unloader — 卸船机船舱精准识别

面向散货卸船机双激光雷达场景的 **1～3 个船舱（Hatch）精准识别与整船刚体跟随** 系统。

当前唯一优先级：**先把船舱识别做准，再接整船跟踪；控制、PLC、抓取规划、Web 全部后置。**

## 一、目标

输入目标船区域内的原始点云，输出：

- 视野内 1～3 个独立 Hatch；
- 每个 Hatch 的 opening-side 结构边界；
- 每条边的 `OBSERVED / INFERRED / HISTORICAL / UNKNOWN` 证据来源；
- `VISIBLE / PARTIAL / UNRESOLVED` 状态；
- 固定 Ship Frame 下的 Hatch 几何；
- 船体位姿有效时，由统一 `T_W_B(t)` 更新全部 Hatch 世界位置。

**禁止把货堆轮廓、坡脚、码头边缘或固定高度等值线伪装成船舱钢结构边界。**

## 二、技术路线

```text
Raw XYZ / PCD / PLY
        │
        ▼
Strict Point Cloud Decode
        │
        ▼
Target Vessel Seed / Scene ROI
        │
        ├──────── Geometry Proposal
        │
        └──────── Optional Legacy CNN Proposal
                       │
                       ▼
                Candidate Union
                       │
                       ▼
             1 / 2 / 3 Hatch Solver
                       │
                       ▼
              Per-Hatch Local Deck
                       │
                       ▼
       opening-side 3D Profile Break
           + Visible 3D Face Evidence
                       │
                       ▼
                Boundary Fusion
                       │
                       ▼
       Hatch Polygon / PARTIAL / Failure
                       │
                       ▼
               Fixed Ship Frame B
```

**Production 主链不依赖 Global Deck / Global Structural ROI。**
旧 `hold_detector` 只作为只读 Legacy Proposal 参考，不作为产品真值或生产依赖。

## 三、当前阶段

| 阶段 | 目标 | 状态 |
|---|---|---|
| V1.4 | 整船 SE(3) 稳定跟踪基础 | 已冻结 |
| V1.5-R0 | 数据治理、场景契约、Proposal/Oracle 基线 | 当前 |
| V1.5-R1 | 可见 1～3 舱精准识别 | 待进入 |
| V1.5-R2 | 部分遮挡，不幻觉钢边 | 后续 |
| V1.5-T1 | 复用 V1.4 做整船刚体跟随 | R1 后 |
| V1.5-R3 | 完全覆盖：历史模型 / Prior-only | 最后 |

## 四、目录结构

```text
Ship Unloader/
├── README.md
├── CLAUDE.md
├── AGENTS.md
├── .gitignore
├── .gitattributes
├── docs/
│   ├── 00_项目总则与架构冻结.md
│   ├── 01_技术路线与理论可行性.md
│   ├── 02_数据标注与证据治理规范.md
│   ├── 03_工程化与Git协作规范.md
│   ├── 04_代码规范_C++17与Python.md
│   ├── 05_测试验证与版本发布规范.md
│   ├── 06_AI编码代理执行规范.md
│   └── 07_README与文档维护规则.md
├── ship_perception/              # 产品代码
├── third_party/                  # 固定第三方依赖
├── Ship-Unloader-Data/           # 永不推送
│   ├── raw/map/
│   ├── legacy/hold_detector/
│   ├── annotations/
│   └── manifests/
└── Ship-Unloader-Work/           # 永不推送
    ├── build/
    ├── validation/
    ├── logs/
    ├── reports/
    └── tmp/
```

## 五、环境

```text
Ubuntu       20.04
GCC          9.4
CMake        3.16.3
C++          C++17
PCL          1.10.0
Eigen        3.3.7
Python       3.8
small_gicp   fd29d8cf / v1.0.0
```

几何变换统一使用 `Eigen::Isometry3d`，记号统一为 `T_A_B`：**将 B 坐标表达转换到 A 坐标表达**。

## 六、核心原则

1. **识别优先**：先解决 Hatch 是否识别正确，再做控制。
2. **Fail-Closed**：证据不足时返回 `UNRESOLVED / AMBIGUOUS / UNKNOWN`，禁止强行补全。
3. **观测与推断分离**：任何 Prior / Historical / Inferred 都不得伪装成 Observed。
4. **单一 Ship Frame**：全部 Hatch 共享同一船体刚体位姿，不允许每个舱独立漂移。
5. **Development ≠ Commercial Validation**：当前 Development 数据只用于定位问题，不作为商业泛化证明。
6. **Raw Data Never in Git**：PCD / PLY / ZIP / build / validation / logs 永不推送。

## 七、当前数据门禁

主 Development：

```text
08-01    2 Hatch
08-08    1 Hatch
4-16     1 Hatch + FALSE_SPLIT negative
6-8      1 Hatch
7-16     1 Hatch
8-22     3 Hatch
----------------
合计      9 Hatch
```

特殊场景：

```text
8-17     PARTIAL_FOV
5-29     3-vessel stress
7-2      2-vessel stress
```

R1 必须逐实例报告召回、误合并、重复、跨船污染和逐边证据；禁止只报平均 IoU。

## 八、关键文档

- `docs/00_项目总则与架构冻结.md`
- `docs/01_技术路线与理论可行性.md`
- `docs/02_数据标注与证据治理规范.md`
- `docs/03_工程化与Git协作规范.md`
- `docs/04_代码规范_C++17与Python.md`
- `docs/05_测试验证与版本发布规范.md`
- `docs/06_AI编码代理执行规范.md`
- `docs/07_README与文档维护规则.md`

**任何产品算法修改前，必须先读取 Architecture Freeze 与对应阶段 Acceptance Contract。**
