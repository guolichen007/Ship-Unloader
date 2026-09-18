# M0 精确 SHA 独立验证交接

`BASELINE_M0_SHA=PENDING`。只允许 Ubuntu 20.04 ClaudeCLI 的独立报告确认后固化。

## 首次获取

```bash
git clone git@github.com:guolichen007/Ship-Unloader.git
cd Ship-Unloader
git status --short
git branch --show-current
git rev-parse HEAD
git log -1 --oneline
```

已有仓库先审计工作树；有未知修改时停止，不执行 reset 或覆盖。
将 Codex 返回的完整 SHA 赋给 `SHA`，不要使用分支名代替验证对象：

```bash
SHA='<Codex 返回的完整 40 位 SHA>'
git fetch origin
git checkout --detach "$SHA"
test "$(git rev-parse HEAD)" = "$SHA"
test -z "$(git status --porcelain)"
bash ship_perception/scripts/validate_ubuntu20.sh --sha "$SHA" --base-sha NONE --jobs 2
```

依赖命令见 `ship_perception/README.md`。脚本每次创建全新的
`validation/<SHA>/<UTC-run-id>/build`，不删除旧日志、不复用缓存。
记录 Ubuntu、内核、GCC/G++、CMake、Python、Git、Eigen/PCL 的实际版本；
正式通道强制 `M0_WITH_PCL=ON`。先运行 generator/PCD 自检，再逐个运行 G1→G7。
每个门禁保存 SHA/config_hash/dataset_id/mode/timestamp 与数值指标。
保存 `configure.log`、`build.log`、`ctest.log`、`report.json`、`summary.md`、
G5 `scenarios.csv`、G7 `timing.csv` 以及 PCD 回放文件。

## 给 ClaudeCLI 的指令

阅读 `docs/specs/` 中的 ClaudeCLI 独立验证总指令与冻结总纲，以及本文件。
只验证用户给定的精确 SHA；先审计工作树，再记录 Ubuntu 20.04 环境并 clean build。
运行上面的脚本，审查完整日志与测试实现，不把脚本建议结论当成独立审查。
默认不修改正式产品代码。按原验证文档 §20 返回结构化报告及日志位置。
如果失败，先保留证据，复核 CODE_FAIL / TEST_FAIL / ENV_FAIL / SITE_BLOCKED；
脚本分类只是初步定位。尤其区分代码断言失败和测试设计错误。

G5 本轮只验证 11 类场景的生成/GT/故障注入 harness。
不包含 registration、EKF、实际可观性/协方差/NEES、实际 LOST/重定位判断；
`future_requirement` 是未来估计器的验收要求，绝非本轮产品输出。
允许按 M0 harness 通过，报告须保留 `G5_SCOPE=HARNESS_ONLY`。
G2/G3/G4/G7 的 SITE 项持续为 SITE_PENDING。

## 基线与失败闭环

把最终报告返回本 Codex 任务。CODE_FAIL 由 Codex 修复并提交新 SHA，Ubuntu 对新 SHA
重新 clean build。TEST_FAIL 先对照冻结规格检查测试；不降低阈值或删除失败测试。
ENV_FAIL 仅做兼容修复，不升级目标系统。SITE_BLOCKED 保留 SITE_PENDING。

首个被 ClaudeCLI 独立确认符合 §21 的精确 SHA 才可记录为 `BASELINE_M0_SHA`。
后续由 Codex 在仓库保存报告摘要、证据引用，并创建指向被验证提交的基线标签。
标签必须指向被验证的原始 SHA，而不是随后新增报告的提交。当前脚本不创建标签、
不自动进入 M1/M2，不启用现场自动控制。
