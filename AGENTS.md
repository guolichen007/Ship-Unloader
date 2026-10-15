# AGENTS.md

Codex / ClaudeCLI / 自动编码代理入口。执行细则见 [开发与 Git 协作规范](docs/05_开发与Git协作规范.md)，用户明确任务优先。

开工必读 README、CLAUDE 和 docs/00～05 当前规范；检查工作树并保护用户修改。先返回：

```text
BRANCH=
BASE_SHA=
HEAD_SHA=
WORKTREE=
IN_SCOPE=
OUT_OF_SCOPE=
STOP_CONDITION=
```

不自行改架构、算法阈值、验收门或 V1.4 冻结源；不把 GT 暴露给 runtime，不伪造钢边，不提交客户数据和运行产物。未经明确授权不 merge/tag/删分支；禁止 force push main 或丢失独立历史。

结束返回：

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
