# 工程化与 Git 协作规范

> 状态：FROZEN
> 适用版本：V1.5-R / V1.5-T
> 最后更新：2026-09-22
> Owner：Ship-Unloader Engineering

> 目标：让任何开发者、Codex、ClaudeCLI 都遵循同一企业级流程，避免“能跑但不可追溯”。

## 1. 参考原则

本规范吸收以下公开工程实践：

- GitHub Protected Branches / Rulesets；
- GitHub Required Status Checks；
- GitHub CODEOWNERS；
- Conventional Commits 1.0；
- Semantic Versioning 2.0；
- Google C++ Style 的一致性原则。

项目按自身机器人/点云工程需求裁剪，不机械照搬。

## 2. Branch Model

```text
main
└── feature/v1-5r-*
└── fix/v1-5r-*
└── test/v1-5r-*
└── docs/*
└── chore/*
```

规则：

- `main` 永不直接开发；
- 一个任务一个 branch；
- 禁止长期万能分支；
- 禁止 force push 到 `main`；
- 禁止未经授权 merge / tag。

## 3. Main 保护建议

GitHub Ruleset / Branch Protection：

```text
Require pull request before merging       ON
Require status checks                     ON
Require conversation resolution           ON
Block force pushes                        ON
Block deletions                           ON
Require linear history                    RECOMMENDED
Dismiss stale approvals                   RECOMMENDED
Require latest push approval              RECOMMENDED
```

当团队成员增加后启用：

```text
CODEOWNERS approval
1+ independent approval
```

## 4. Commit 规范

格式：

```text
<type>(<scope>): <中文简述>
```

允许 type：

```text
feat
fix
refactor
test
docs
chore
perf
build
ci
revert
```

示例：

```text
feat(recognition): 增加1至3舱联合假设求解器
fix(parser): 修复16字节PCD记录解析错位
test(V1.5-R0): 增加多船与Partial-FOV契约回归
docs(架构): 冻结opening-side边界证据定义
chore(工程): 统一Data与Work目录治理
```

Commit Body 必须说明：

```text
背景
变更
验证
风险
```

禁止：

```text
update
fix bug
modify
temp
final
final2
try
```

## 5. 一个逻辑任务一个 Commit

不得混合：

```text
目录治理 + 产品算法
数据契约 + 阈值调参
大规模格式化 + 功能修改
V1.4 frozen + V1.5 feature
```

这样才能：

- bisect；
- rollback；
- code review；
- root-cause。

## 6. PR 模板字段

每个 PR 至少：

```text
目标
BASE_SHA
HEAD_SHA
IN_SCOPE
OUT_OF_SCOPE
Architecture Impact
Data Impact
Changed Files
Tests
Regression
Known Limitations
Risk
Rollback
Evidence
```

如修改架构：

```text
ADR / Architecture Review = REQUIRED
```

## 7. Required Checks 建议

```text
format
lint
unit
contract
synthetic
v14_frozen
build_ubuntu20
diff_check
```

后续增加：

```text
integration
real_data_development
performance
```

Development real-data 不宜直接在 public CI 上传原始 PCD。

## 8. CODEOWNERS

建议新增：

```text
.github/CODEOWNERS
```

初期仓库 owner 可以负责：

```text
/ship_perception/       @guolichen007
/docs/                  @guolichen007
/.github/               @guolichen007
/CLAUDE.md              @guolichen007
/AGENTS.md               @guolichen007
```

团队扩展后再拆：

```text
perception owner
validation owner
release owner
```

## 9. 版本号

采用 SemVer：

```text
MAJOR.MINOR.PATCH
```

但算法阶段可以使用：

```text
v1.5.0-r0
v1.5.0-r1
```

正式 release 后版本内容不得被修改；需要修复必须发布新版本。

## 10. Tag 规范

只有满足 release gate 才允许 tag。

示例：

```text
baseline-v1.4-ubuntu20-20260919
v1.5.0-r1
```

Tag 描述必须包含：

```text
commit SHA
environment
dataset manifest SHA
acceptance result
known limitations
```

## 11. 禁止提交

```text
PCD / PLY
map.zip
原始客户数据
large validation artifacts
build/
logs/
tmp/
private model assets
absolute local paths
secrets
```

提交前：

```bash
git status
git diff --check
git diff --cached --stat
git ls-files | grep -E '\.(pcd|ply|zip)$'
```

## 12. Task Start Contract

任何任务开始先返回：

```text
BRANCH=
BASE_SHA=
HEAD_SHA=
WORKTREE=
IN_SCOPE=
OUT_OF_SCOPE=
STOP_CONDITION=
```

## 13. Task End Contract

必须返回：

```text
PREV_SHA=
NEW_SHA=
CHANGED_FILES=
COMMANDS=
TEST_RESULT=
REGRESSION_RESULT=
WORKTREE_CLEAN=
PUSH_STATUS=
NEXT_BLOCKER=
```

失败不得省略。
