# MHR-R2 本机接口与前端集成

## 启动

在独立科学环境中安装 `requirements-ros.txt`。Windows 命令提示符使用：

```bat
set "PYTHON_BIN=C:\path\to\review-venv\Scripts\python.exe"
scripts\run_manual_workbench.bat "D:\点云\输入.pcd" "D:\工作\复核"
```

Ubuntu 22.04 或 Ubuntu 20.04 的独立 Python 3.10–3.12 环境使用：

```bash
PYTHON_BIN=/path/to/review-venv/bin/python bash scripts/run_manual_workbench.sh '/data/输入.pcd' '/data/复核'
```

直接用 `python -m ship_perception.manual_review.workbench_server` 亦可。默认每次启动执行一次冻结自动流程，运行目录与 SQLite 修订目录分离。已存在可信运行结果时传 `--import-run 完整结果目录`。服务校验 03、05 与复核 compact 的身份及内容。合成试验使用 `python -m tools.mhr_synthetic demo.pcd`，以 `--skip-auto` 启动后点「新增船舱」。所有 HTTP 均为 127.0.0.1，本机启动参数决定帧来源。切到另一份 PCD 时由本机启动方开新服务。

打开终端打印的 `MHR_R2_URL`。链接中的令牌用于本机当前会话。原生按钮不需要手工输入坐标。选中舱后直接确认，或进入圈选拖拽。释放鼠标发一次重测，必要时设定粗框角度并手动重测。完整候选四角和主框一起更新，随后点击确认。作废和撤销在「其他操作」，确认整景独立于单舱。导出 JSON 和 CSV 保留来源、缺失 Z、待审对象和作废记录。

## SDK

同一服务的 `/example` 通过相同 ES 模块挂载实际组件，事件显示在示例标题栏。调用端加载同源静态脚本和样式，具备本机会话 token 后调用 `mount(root, { token, onReviewChange, onPreview, onCommit, onError })`。实例支持 `openFrame()`、`setTarget(id)`、`resize()`、`destroy()`。openFrame 刷新当前本机绑定帧，视图身份改变时丢弃上一帧工作状态。TS 声明位于 workbench.d.ts。前端只显示服务器候选，XYZ 与尺寸不在浏览器重新拟合。

## HTTP

每个 POST 含 view 返回的 session_id、frame_instance_id、view_id、manifest_sha256、wire_sha256、semantic_sha256、frame_id。携带 X-MHR-Token，写请求要求严格 Origin 和 JSON。状态不匹配以 409 拒绝，未知路径以 404 拒绝，过大请求以 413 拒绝。响应均有两个 false 安全字段。可机器读取的请求合同见 MHR-R2-openapi.json。

| 方法 | 路径 | 语义 |
| --- | --- | --- |
| GET | /api/v2/view | 帧身份、构图、自动流程状态、待审项目、修订及整景摘要 |
| GET | /api/v2/revisions | 历史事务日志 |
| POST | /api/v2/auto/confirm | 根据 bundle_digest、geometry_digest 和目标确认显示中的原封板复核框 |
| POST | /api/v2/auto/void | 记录自动目标误检作废 |
| POST | /api/v2/refine | CREATE 或 REFINE，提交像素与 XY 对照、request_id 和父修订 |
| POST | /api/v2/refine/cancel | 按 request_id 与 annotation_id 作废正在编辑的建议 |
| POST | /api/v2/refine/confirm | 仅确认当前视图和父修订上的匹配候选 ID、草稿与几何摘要 |
| POST | /api/v2/manual/commit | 显式人工粗框兜底，独立于重测，Z 全为空 |
| POST | /api/v2/undo | 生成新的回退修订 |
| POST | /api/v2/scene/confirm | 核对整景摘要，显式处理空场，所有自动目标独立判定后才可提交 |
| POST | /api/v2/export | 按整景摘要读取已记录坐标、待审和作废项目 |

R1 原入口和端点继续由原服务提供。R2 单独使用 r2-state 事务目录，旧 R1 数据可由 R1 查看，R2 不自动迁移旧签核。任一舱追加修订使旧整景签核失效。所有确认仅记录本机人工复核行为。可执行工具用明确的本机 PCD 与输出路径，运行时不读取验收历史坐标。
