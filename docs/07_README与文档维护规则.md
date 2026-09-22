# README 与文档维护规则

> 状态：FROZEN
> 适用版本：V1.5-R / V1.5-T
> 最后更新：2026-09-22
> Owner：Ship-Unloader Engineering

## 1. README 只保留长期入口信息

README 回答：

1. 项目是什么；
2. 当前目标是什么；
3. 主技术路线是什么；
4. 怎么构建/运行；
5. 当前做到哪；
6. 去哪里看详细规范。

README 不放：

- 长篇失败法证；
- 临时实验记录；
- 全量参数历史；
- 每次 commit 日志；
- 过期计划。

## 2. 文档层级

```text
README
│
├─ Architecture Freeze
├─ Technical Feasibility
├─ Data Governance
├─ Engineering Standard
├─ Coding Standard
├─ Validation Standard
└─ Agent Standard
```

## 3. 状态标签

每份关键文档顶部必须写：

```text
状态: DRAFT / REVIEWED / FROZEN / DEPRECATED
适用版本:
最后更新:
Owner:
```

## 4. 禁止重复真值

同一个规则只允许一个 canonical 文档。

其它文档只能引用，不能复制后各自修改。

例如：

```text
坐标系规则 → Architecture Freeze
Git规则    → Engineering Standard
代码规则   → Coding Standard
验收规则   → Validation Standard
```

## 5. 变更要求

FROZEN 文档修改：

```text
单独 docs commit
明确变更原因
列出受影响模块
必要时新增 ADR
```

不得在 feature commit 里顺手修改 Architecture Freeze。
