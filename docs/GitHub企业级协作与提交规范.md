# Ship-Unloader GitHub 企业级协作与提交规范

> 状态：生效  
> 适用范围：`guolichen007/Ship-Unloader` 全仓库  
> 目标：保证每次提交、推送、验证、PR、Issue、Tag 均可追溯、可审查、可回滚。  
> 语言规则：**GitHub 面向人的文案全部使用中文；代码、类名、变量、命令、协议字段、分支名等技术标识保留英文。**

## 1. 总体规则

1. `main` 只接收已通过 Ubuntu 20.04 独立验证的提交。
2. 开发使用功能分支，不在 `main` 上连续堆叠未验证实现。
3. 每次提交只解决一个明确问题，禁止混入无关修改。
4. 每个可验证提交必须对应精确 Git SHA。
5. 验证结果必须关联 Git SHA、配置哈希、数据/Replay ID、依赖版本与验证环境。
6. 禁止通过降低 Gate 阈值、删除失败测试、忽略异常来获得 PASS。
7. `docs/` 下所有文档文件名与正文使用中文；必要技术标识保留英文。
8. 历史英文提交不重写；本规范生效后全部按中文标准执行。

## 2. 分支命名

分支名作为技术标识使用 ASCII：

```text
feature/<阶段>-<内容>
fix/<阶段>-<内容>
test/<阶段>-<内容>
docs/<阶段>-<内容>
chore/<阶段>-<内容>
```

示例：

```text
chore/m0-closeout
feature/m1a-small-gicp-registration
test/m1a-registration-gates
fix/m1a-ubuntu20-build
```

## 3. Commit 标题规范

格式：

```text
<中文类型>(<阶段>): <中文动作 + 中文对象>
```

推荐类型：

```text
功能
修复
测试
文档
构建
重构
性能
维护
```

示例：

```text
构建(M0): 固定 Ubuntu20 系统 Eigen 解析路径
文档(M0): 固化 Ubuntu20 独立验证基线
功能(M1-A): 接入 small_gicp 配准后端
测试(M1-A): 增加已知 SE3 配准回归
```

禁止使用：`update`、`misc`、`WIP`、`final` 等模糊标题。

## 4. Commit 正文规范

关键提交必须写正文：

```text
背景：
- 为什么需要本提交。

变更：
- 修改了什么。
- 明确未修改什么。

验证：
- Windows 开发侧执行了什么。
- Ubuntu20 是否已验证；未验证必须明确写“待 Ubuntu20 独立验证”。

风险：
- 可能影响哪些模块。
- SITE_PENDING 项。

关联：
- BASE_SHA=<sha>
- 对应 Gate=<gate>
```

## 5. Push 前检查

```bash
git status --short
git diff --check
git log -1 --oneline
```

要求：

```text
工作树干净
无 CRLF/空白错误
提交标题符合中文规范
没有误改 hold_detector/
```

## 6. Pull Request 规范

PR 标题：

```text
<阶段>: <中文目标>
```

正文：

```markdown
## 背景
## 本次范围
## 明确不在本次范围
## 关键实现
## 验证结果
## Ubuntu20 独立验证
## 风险与待确认
## 回滚方式
## 关联信息
- BASE_SHA:
- IMPLEMENTATION_SHA:
- VALIDATION_SHA:
- 配置哈希:
- 相关 Gate:
```

未完成 Ubuntu20 验证时必须写：

```text
Ubuntu20 独立验证：待执行
```

## 7. Issue 规范

标题：

```text
[阶段][类型] 中文问题描述
```

正文：

```markdown
## 现象
## 复现步骤
## 期望结果
## 实际结果
## 证据
## 首个失败点
## 初步判断
## 建议责任人
```

## 8. Tag 规范

Tag 名用 ASCII：

```text
baseline-m0-ubuntu20-20260918
baseline-m1a-ubuntu20-YYYYMMDD
```

Tag 描述使用中文。基线 Tag 必须直接指向被验证 SHA。

## 9. docs 文档命名规范

推荐：

```text
docs/
├── 项目说明.md
├── 架构冻结版.md
├── GitHub协作与提交规范.md
├── M0阶段_Ubuntu20.04独立验证报告.md
├── M0阶段_验证交接说明.md
├── M1-A阶段_small_gicp配准实现与验证交接说明.md
└── 验证/
    └── <阶段>_<SHA前8位>_验证摘要.md
```

不得再新增纯英文文档名，如 `VALIDATION_HANDOFF.md`、`ARCHITECTURE.md`、`TEST_REPORT.md`。

## 10. 验证收尾字段

统一使用中文：

```text
验证目标SHA=
验证基线SHA=
工作树=
构建门禁=

旧门禁回归=
本阶段门禁=

首个失败项=
失败类型=
产品代码被Claude修改=

本阶段代码=
本阶段合成验证=
Ubuntu20验证=
现场验证=

最终结论=
允许进入下一阶段=
```

技术状态值可保留 `PASS_LAB / PASS_SYNTHETIC / PASS_SITE / SITE_PENDING / FAIL`。

## 11. 合并原则

只有满足以下条件才允许合并到 `main`：

```text
代码审查通过
Ubuntu20 clean build 通过
本阶段必需 Gate 通过
旧 Gate 无回归
验证报告已归档
无未解释失败
```

`SITE_PENDING` 不阻塞离线研发，但不得描述成现场通过。
