# CompassCart 主机验证报告（macOS / Apple Silicon）

## 概览

| 项目 | 值 |
|---|---|
| 验证日期 | 2026-08-31 |
| 验证主机 | macOS 26.6.2（Build 25G83，Apple Silicon / arm64） |
| 验证版本 | 最终提交 `e213ed4`（PR #5 合并，候选 commit `900500b`，public 0.822490） |
| 结论 | ✅ 平台验证通过：200 会话官方评测分数与 Windows 参考**逐位一致** |

## 环境

| 项目 | 值 |
|---|---|
| 操作系统 | macOS 26.6.2（Darwin 25.6.0，arm64） |
| Python | 3.14.6（Anaconda，Clang 20.1.8） |
| numpy | 2.5.2 |
| onnxruntime | 1.29.0 |
| tokenizers | 0.23.1 |
| 数据 | `data/catalog.jsonl`（58 MB）、`data/public_set.jsonl`（200 会话） |
| 资产 | 5 个 dense 资产齐全，SHA256 校验通过 |

> 说明：本次验证使用仓库内已存在的 `.venv`（依赖已安装、可正常导入）。未做严格"冷安装"复现；如需从零冷安装验证，可另行执行。

## 验证结果

### 1. Smoke 测试（Agent 契约）

- `dense_available = True`（本地 ONNX dense 模型正常加载）。
- 一轮 `respond(...)` 返回 10 条推荐，首条 `B084JKY4S5`。
- `usage = {prompt_tokens: 0, completion_tokens: 0}` —— 零 token、零 API 成本。

### 2. 官方公开评测（200 会话）

| 指标 | Windows 参考 | macOS 实测 | 一致 |
|---|--:|--:|:--:|
| recommended_technical_score | 0.822490 | **0.822490** | ✅ |
| hit_rate_at_10 | 0.9650 | 0.965 | ✅ |
| mrr | 0.580968 | 0.580968 | ✅ |
| mttc | 2.715 | 2.715 | ✅ |
| efficiency | 0.8285 | 0.8285 | ✅ |
| fallback_count | 0 | 0 | ✅ |

场景分逐项一致：boundary `1.0000`、browsing `0.9750`、buying `0.9625`、intent_override `0.9333`。

### 3. 词法回退（禁用 Dense）

`COMPASSCART_DISABLE_DENSE=1` 下重跑，`recommended_technical_score` 仍为 **0.822490**，与 README 声称"禁用 dense 分数不变"一致。

### 4. 性能

| 项目 | macOS（arm64） | Windows 参考 |
|---|---:|---:|
| 200 会话评测墙钟时间 | 约 44 秒（含初始化） | init 约 19.6s + run 约 89.9s |

macOS Apple Silicon 上完整评测明显快于 Windows 参考。

## 结论

- 最终提交 `e213ed4` 在 macOS 上**通过平台验证**：dense 与 lexical 双路径可用，200 会话官方分数 **0.822490** 与 Windows 参考完全一致，零回退、零 token、零 API 成本。
- Linux 平台验证仍为 pending（未在本机执行）。
