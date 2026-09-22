# CLAUDE.md

本项目为 **V1.5-R 船舱精准识别优先**。

开工前必须读取：

1. `README.md`
2. `docs/00_项目总则与架构冻结.md`
3. `docs/03_工程化与Git协作规范.md`
4. 当前阶段验收文档

核心规则：

- Production 主链不得重新依赖 Global Deck / Global Structural ROI。
- 原始数据放 `Ship-Unloader-Data/`，构建/验证放 `Ship-Unloader-Work/`，两者永不推送。
- `Ship-Unloader-Data/legacy/hold_detector/` 只读；产品代码禁止 include/link。
- 不自行改架构、数据 split、门禁、阈值、holdout 状态或 V1.4 frozen source。
- 不 merge main、不 tag、不 force push、不访问 sealed holdout annotation。
- 一项逻辑一个 commit；Git/README/验证说明使用中文，代码标识保持英文。
- 证据不足必须 fail-closed，不得为了 PASS 降低标准。
