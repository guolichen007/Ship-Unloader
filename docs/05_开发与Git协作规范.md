# 开发与 Git 协作规范

> 状态：FROZEN
> 适用版本：当前现场复核基线；V1.4 历史事实单列
> 最后更新：2026-10-15
> Owner：Ship-Unloader Engineering

## 分支、提交、PR 与 main

main 保存已收口工程基线；常规研发使用短期 `feature/* / fix/* / docs/* / codex/*`，一个任务一个分支。任务合入后删除开发分支，独立历史先归档。不得在 main 连续开发、force push main、重写正式 Tag 或丢失独立提交。未经用户明确授权不 merge/tag/删分支。本轮 2026-10-15 收口已获用户明确授权，仍先审计、备份与验证。

提交统一 `<type>(<scope>): <中文简述>`，type 为 feat/fix/refactor/test/docs/chore/perf/build/ci/revert。历史中英文类型不重写。正文包含背景、变更、验证、风险与 BASE_SHA；GitHub 面向人的描述中文，代码、命令、协议字段和分支名英文。一个逻辑一个 commit，不混合目录治理和算法调参、大规模格式化和功能改动、V1.4 frozen 和新识别开发。冻结文档整合单独 docs commit。

PR 至少有目标、BASE_SHA/HEAD_SHA、IN_SCOPE/OUT_OF_SCOPE、架构/数据影响、文件、验证/回归、限制、风险、回滚和证据。描述面向未看聊天的审查者，并与最终实现一致。架构变化先评审；Ubuntu 未验证明确写“待 Ubuntu20 独立验证”。SITE_PENDING 不包装成现场通过。main 工程收口不等于正式 release，产品放行仍满足[验收规范](04_测试与验收说明.md)。

建议 main 保护：PR、required checks、会话解决、禁止 force push/删除，可选线性历史、独立审批、过期审批失效。代理不擅改 ruleset；保护拒绝时保留分支并报告，不绕过。正式合入通常要求审查、Ubuntu20 clean build、适用阶段与旧 Gate、证据归档、无未解释失败。本轮是用户授权的当前工程基线快进，不把已知失败或缺失验证改为 PASS。

Issue 写中文现象、复现、期望/实际、证据、首个失败点、根因判断与责任建议。不要用模糊 update/misc/WIP/final 代替任务说明。

## 分支清理与历史保全

先 fetch/prune refs 和 tags，保存本地/远端分支 HEAD、Tag 对象、工作树及可达性清单。忽略的 Work 目录保留经对象核对的完整 Git 备份（含 reflog 与不可达对象），不运行破坏性 prune/gc。

对每条待删分支执行 `git merge-base --is-ancestor <head> <TARGET_HEAD>`：

- 已包含：记录 SAFE_DELETE_ANCESTOR，不额外 Tag。
- 独立/diverged：先在原 HEAD 建立带中文说明的 annotated archive Tag，推送并核验指向，再删 branch。检查本地未推送提交，不能只看 origin。
- 不可达历史也要保全，可用专门归档 Tag 与已核验备份；不能将快照当正式技术基线。

验证、main 快进与远端 SHA 核对完成后才清理被替代分支；删除前再次确认远端 HEAD，发生并发变化即停止该 ref 删除。main 仅 fast-forward，出现新独立提交时不 force push，保留现状报告。

正式 `baseline-v1.4-ubuntu20-20260919` 永久保留且指向原验证 SHA。研究 archive Tag 注明“历史研究支线，仅用于追溯；当前生产/现场基线不从该 Tag 开发”。正式 release 用 SemVer，说明 SHA、环境、数据/配置身份、验收、限制与回滚。Archive 不表示 release gate 通过。

历史文档可用 `git show 914f869:docs/<旧文件名>` 读取，不在 HEAD 新建 archive 目录。恢复研究时另开明确范围分支。

## 文档与资产

README 只写当前用途、环境、入口、能力/限制、阅读顺序和目录，不写过程法证或参数历史。docs 保留架构、算法、运行、数据、验收、协作与 V1.4 历史事实约 8 份，每条规则有唯一 canonical 来源。文档标状态、适用版本、日期和 Owner；先迁移有效结论，再删除过程文档并修复引用。

Git 只保存源码、配置/schema、测试、小型元数据、SHA 和当前文档。Raw PCD/PLY/ZIP、客户数据、build、validation、logs、临时包、私有模型、绝对开发机路径及秘密不新增 Git。已有受控合成夹具保留，不能把客户数据冒充夹具。数据与 Work/output 规则见[数据规范](03_数据与证据规范.md)。

## C++17 与 Python

核心 Python 保持 3.8 语法兼容；ROS2 Humble 正式运行使用 Ubuntu22/Python3.10 与 requirements-ros.txt 隔离依赖，不能把新数值依赖直接安装到 Ubuntu20 历史冻结环境。

正确性、追溯、安全失败优先于性能或代码短。物理量命名带单位，浮点比较用具名可配置 tolerance，不改既有阈值以过测试；位姿 double、Eigen::Isometry3d，方向 T_A_B，全部舱共用船体位姿。

C++ 明确 const/RAII、值语义和所有权，优先 unique_ptr；共享所有权有真实需求，raw pointer 仅非拥有。核心接口不暴露不必要第三方类型，头文件自包含，不在头文件 using namespace。不散布 new/delete，不引入隐式可变全局状态。失败用 Status/Result/optional，禁止空矩形继续当成功、非法矩阵/NaN 传播或异常静默回退 Global Deck。

Python 用 pathlib、类型提示、小函数、argparse，无 wildcard import，文件 UTF-8。JSON 禁 NaN/Infinity，schema fail-closed；关键写入 temp、flush/fsync/close、atomic rename。随机采样记录 seed/config/input/output SHA。两端 parser 比较点数、bbox、origin、resolution、栅格尺寸和样本值，不假设布局相同。

核心模块不依赖 legacy 源码、开发机路径、validation 产物或 sealed annotation。保持 IO/proposal/geometry/hypothesis/tracking/evaluation 职责分离，配置显式、尽量不可变并记录哈希。日志有 stage、scan/frame、candidate/hatch、reason 和关键指标，避免逐点刷屏。注释说明物理假设、证据来源和失败边界。格式工具固定版本，只处理本次文件，不重排 frozen 源。

## Codex / Claude 执行规范

开工读 README、AGENTS/CLAUDE 和当前架构、算法、运行、数据、验收及本规范。检查工作树，不覆盖用户改动；先报告 BRANCH、BASE_SHA、HEAD_SHA、WORKTREE、IN_SCOPE、OUT_OF_SCOPE、STOP_CONDITION。用户明确任务优先；常规实施选择不变成反复确认。

不自行改架构、split/holdout、验收门或 V1.4 frozen，不用 GT 补 runtime，不把 Cargo/Wharf 伪装成钢边，不跨阶段顺手实现控制、跟踪联动或大规模调参。只解决明确范围，到 STOP_CONDITION 停止。

Push 前检查 status、diff --check、cached stat、适用格式/测试/构建/frozen、原始数据和秘密。验证使用精确 SHA；提交后必要时刷新编译 SHA 再跑新鲜度相关 Gate。失败保存输入 ID、软件/配置 SHA、命令、stdout/stderr、环境、artifact 路径和 CONFIRMED/LIKELY/OPEN 根因，不删除 case、降门槛或把 unknown 改成功。

结束返回 PREV_SHA、NEW_SHA、CHANGED_FILES、COMMANDS、TEST_RESULT、REGRESSION_RESULT、WORKTREE_CLEAN、PUSH_STATUS、NEXT_BLOCKER；涉及冻结源另报 V1_4_FROZEN。失败和未执行必须明确，不省略、不改为 PASS。
