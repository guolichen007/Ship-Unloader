# AGENTS.md

面向 Codex / ClaudeCLI / 自动编码代理。

## 开始任务

必须先返回：

```text
BRANCH=
BASE_SHA=
HEAD_SHA=
WORKTREE=
IN_SCOPE=
OUT_OF_SCOPE=
STOP_CONDITION=
```

## 必读

- `README.md`
- `docs/00_项目总则与架构冻结.md`
- `docs/03_工程化与Git协作规范.md`
- `docs/04_代码规范_C++17与Python.md`
- `docs/05_测试验证与版本发布规范.md`

## 红线

- 不自行重设计架构；
- 不混合多个逻辑任务到一个 commit；
- 不把 GT 暴露给 runtime；
- 不把 Cargo/Wharf 伪装成 Hatch steel boundary；
- 不降低 acceptance gate；
- 不 merge / tag / force push；
- 不提交 raw data / build / validation / logs。

## 结束任务

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
