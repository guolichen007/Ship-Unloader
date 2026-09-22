# CLAUDE.md

V1.5-R 识别优先项目。所有任务先读本文件与 `docs/工程执行规范.md`、当前 Architecture Freeze 文档。

- 原始数据、构建、验证产物一律放 repo 外（`SHIP_UNLOADER_DATA_ROOT` / `SHIP_UNLOADER_WORK_ROOT`）。
- `hold_detector/` 是只读 legacy 资产，产品代码禁止 include/link。
- 不自行修改架构、数据 split、门禁、阈值、holdout 状态、V1.4 冻结源码。
- 不 merge main、不 tag、不 force push、不访问 sealed holdout。
- 一项逻辑一个 commit，中文企业级提交信息。
