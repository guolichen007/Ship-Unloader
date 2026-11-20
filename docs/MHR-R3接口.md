
# MHR-R3 实验工作台

Windows 使用 `scripts\run_manual_workbench.bat PCD 输出目录`，Ubuntu 使用
`bash scripts/run_manual_workbench.sh PCD 输出目录`。指定 `PYTHON_BIN` 可以复用
已有 Python 环境。安装依赖参见仓库的 `requirements-ros.txt` 和离线复核说明。
运行后在本机打开 `MHR_R3_URL` 指定的地址。默认执行本次原始 PCD 的冻结自动识别一次。
`--import-run DIR` 接受同帧并通过输出合同校验的原生运行目录，
`--skip-auto` 用于独立 CREATE 测量，`--protocol 2` 保留 R2 工作台。

在图上点击或从列表选择舱。重叠时从弹出的明确目标清单选择。正确的自动框直接确认。
重画会保留自动参考方向，也可先用两点确定斜舱的方向。松开鼠标提交局部 Raw 复算。
框、XYZ 和水平尺寸随当前预览同步变化。存在独立验证的候选竞争时，先明确选择一个编号，
再按确认。边证据栏中的剖面可提交一次侧边位置提示，提示仅约束重新搜索。
失败、取消和刷新保留上次确认。异常兜底可仅保存粗框，来源为 MANUAL_ONLY。
每个 Z 独立保存测量状态和质量。所有修订均为离线复核，钢边认证和设备控制均为 false。

## HTTP 与 SDK

沿用 v2 自动确认、作废、人工兜底、撤销、整景审核和导出。
新客户端显式挂载 `mount(host,{token,protocol:3,basePath:"/review/hatch"})`。
`basePath` 可为空或同源绝对路径前缀。启动进程传入同一个 `--base-path`。
`/review/hatch/example` 运行同一个组件。示例从本机 URL 片段获得令牌。
宿主经代理集成时保持同源，后端依然绑定回环并验证 Host、Origin 和令牌。
通过 `onReviewChange`、`onPreview`、`onCommit` 读取更新，调用 `destroy()` 后可重新挂载。

`POST /api/v3/refine` 继承 v2 圈选请求，新增可选 `edge_hint`。
提示绑定上一请求 ID、边名和地图 UV，必须在原边中段的法向窗口内。
`/api/v3/refine/profile` 返回至多 800 个当前合法训练观测的局部 offset/Z 样本。
`/api/v3/refine/select` 选择已验证服务器对象并生成新 `selection_id`。
`/api/v3/refine/confirm` 提交已绑定的暂存对象，要求同一帧、目标、自动包和父修订，且来源已核验。
`/api/v3/refine/cancel` 作废当前临时请求。过期、代次变化或搜索截断时禁止确认。
唯一结果可直接进入待确认，竞争候选必须显式选择。响应的 `geometry_digest` 覆盖完整服务器几何，
包括仅私有保存的 Raw 行证据，公开预览不自行重算该摘要。

## 发行与实验结果

从干净的项目工作树执行 `python -m tools.mhr_r3_package 输出目录` 生成运行包和前端 SDK。
运行包中的 `mhr-build.json` 绑定提交、依赖、协议和文件 SHA-256。解压后按该清单验证来源。
受版本管理的项目工作树可以直接运行。来源缺失或校验失败时提示 UNVERIFIED，
V3 Raw 正式确认被拒绝。公开包不携带现场 PCD 或私有证据。

实验 A 按共同方向组织有界联合搜索、独立留出 veto 和逐候选分隔验证。
合并数值别名需要共同 Raw 行、同层观测及整段端点约束。搜索未走完时只返回诊断。
实验 B 增加窄带内的壁面、顶面及交界证据，作为离线消融保留。
实验 A 和 B 的实景首画均为 0 个可确认候选。当前运行默认使用 A。
对于搜索范围大的实景和缺支撑边，仍需独立技术负责人复核，
实际操作员盲画、真实钢边精度和成功延迟均待验收。
