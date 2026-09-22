# AGENTS.md

面向 Codex / ClaudeCLI 等编码代理的工程约束。开工前必读 `docs/工程执行规范.md` 与当前 Architecture Freeze 文档。

任务开始先返回：`BRANCH / BASE_SHA / HEAD_SHA / git status`，并写明本轮 `IN_SCOPE / OUT_OF_SCOPE / STOP CONDITION`。禁止一次 commit 混目录清理、架构修改与产品算法修改。任务结束返回 `PREV_SHA / NEW_SHA / CHANGED_FILES / COMMANDS / TEST_RESULT / REGRESSION_RESULT / WORKTREE_CLEAN / PUSH_STATUS / NEXT_BLOCKER`。

数据、构建、验证产物不得进 Git 根目录。
