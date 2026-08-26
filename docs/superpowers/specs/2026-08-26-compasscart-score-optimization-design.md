# CompassCart 泛化优先评分优化设计

## 1. 文档状态

- 日期：2026-08-26
- 基线版本：`compasscart-v2`（commit `c1d095b`，运行时代码候选 `4c41adf`）
- 已确认策略：优先提高 800 个私有会话上的泛化能力，不针对公开样本记忆答案
- 资产策略：先冻结现有 ONNX 模型和 50,000 条商品向量；只有代码优化形成稳定平台后才重新评估资产
- 设计状态：已由用户确认，可进入实施计划阶段

## 2. 背景与基线

Windows 主机上的未修改代码已于 2026-08-26 重新运行官方 200-session evaluator，精确复现发布基线：

| 指标 | 基线 |
|---|---:|
| TechnicalScore | 0.518309 |
| HitRate@10 | 0.625 |
| MRR | 0.321365 |
| MTTC | 5.530 |
| Efficiency | 0.547 |

开发折 1-4 的已发布结果为：

- TechnicalScore：`0.479726`、`0.568101`、`0.541363`、`0.487589`
- mean：`0.519195`
- std：`0.036878`
- selection score：`mean - 0.5 * std = 0.500756`

公开全集和原封存折结果已经被查看，因此折 5 不再被描述为无偏封存集。后续版本选择只使用开发折 1-4、无标签语义回归测试和扰动测试；800 个私有会话仍是最终真正的盲测。

## 3. 诊断证据

只读会话分析发现，主要损失来自约束语义不一致，而不是 dense 模型容量不足：

1. 粗粒度类别短语可被重复解析成 `category`、`style` 或 `brand`。例如 `Fashion Sneakers` 同时生成鞋类类别和 `style=sneaker`，`Boy Shorts` 可错误生成 `brand=boy`。
2. `no preference` 回答会持久化 `route_hint=browsing`，把 Buying 或 Intent Override 会话错误切换到 Browsing 权重。
3. 澄清回答中的开放文本可被转成精确硬约束，但属性索引只保存受限规范值。目标商品自己的描述因此可能无法满足由其描述生成的约束。
4. 过滤、属性召回和 fallback 使用不同的类别相等语义。即使过滤器接受目标，候选生成仍可能因为精确字符串不等而召不回目标。
5. Retriever 已计算 route-weighted RRF 分数，但 Ranker 主要重用 lexical/dense 分量，弱化了 attribute、profile 和融合顺序信号。

诊断原型在内存中放宽部分匹配后估算公开 TechnicalScore 可达到约 `0.67`。这项结果使用了公开全集失败信息，且不是生产代码运行结果，只证明“修复统一语义”值得优先实施，不能作为完成或提分声明。

## 4. 目标与非目标

### 4.1 目标

1. 同一用户约束在 parser、state、retrieval、fallback 和 ranker 中具有一致含义。
2. 粗类别文本不再无提示地污染 style 或 brand。
3. 可识别的规范澄清答案保持强约束；无法映射到规范词表的开放答案仍参与查询，但不制造不可能的硬过滤。
4. `no preference` 只拒绝当前问题，不改变已有购买/浏览意图。
5. 在不修改官方 evaluator、数据和 dense 资产的前提下提高开发折选择分数。
6. 每个优化阶段都能独立测试、独立评估和独立回退。

### 4.2 非目标

- 不根据 `sample_id`、目标 ASIN、公开 ground truth 或固定会话顺序建立规则。
- 不修改 evaluator、公开数据、catalog 内容或评分公式。
- 不使用 `ask_attribute="other"` 套取任意隐藏条件。
- 不为了 evaluator 自动轮转而在每一轮强制排除全部历史推荐；只有用户明确请求更多结果时才使用现有排除行为。
- 第一阶段不重新训练模型、不重新生成向量、不增加在线服务或 API。
- 不把公开全集的单次最高分作为唯一版本选择标准。

## 5. 推荐设计

### 5.1 Parser 消歧与路由保持

Parser 保留已有规范词表，但增加来源和语境边界：

- 首轮粗类别短语形成受保护 span。完全落在该 span 内的动态 style/brand alias 不再重复生成硬约束。
- style 只有在规范固定值或附近存在 `style`、`look`、`design` 等显式提示时才作为独立约束。
- brand 只有在规范固定值或附近存在 `brand`、`by`、`from`、`made by` 等显式提示时才作为独立约束。
- `no preference`、`no additional preference` 和等价回答返回当前属性拒绝状态，但 `route_hint=None`，保留 SessionState 的已有 route。
- 明确说出的购买/浏览意图仍有最高路由优先级。

上述规则依赖词项、span 和提示词，不依赖商品 ID 或公开样本 ID。

### 5.2 规范约束与开放查询证据

约束分成两种行为，不新增对外 API 字段：

1. **规范约束**：值能映射到 catalog 规范词表，继续使用 `eq`、`in`、`not_in`、预算范围等强语义。
2. **开放查询证据**：澄清回答无法映射到规范词表时，原文进入有界 query history，并可作为 soft preference；它不能直接成为 exact hard filter。

category 使用类别词项蕴含，而不是完整字符串相等：用户类别的规范词项必须包含于商品所有 category 值合并后的词项集合。规范化必须共享已有大小写、标点和单复数规则，并为 `hoodies -> hoodie` 等不规则情况增加有测试的规范形式。

开放全文匹配仅用于来源为 clarification 的允许属性，如 feature、style、brand、size 和 use_case。匹配条件是规范化短语或全部实义词存在于商品可搜索原文中；预算、否定条件以及首轮自由文本不走这一放宽路径。

### 5.3 Catalog 语义缓存与统一候选生成

CatalogIndex 在加载 50,000 条商品时一次性缓存：

- 每个商品的合并 category terms；
- 每个商品的 searchable terms；
- category term 到商品 ID 的倒排集合；
- 如性能测试证明必要，再为受控开放属性建立有限 token postings。

一个共享语义入口负责以下三处：

1. 判断候选是否满足 hard constraints；
2. 生成 attribute/category 候选；
3. 构造 category fallback。

候选生成不能继续使用与最终过滤不同的精确类别字符串规则。共享入口必须返回确定性、有序且只含 catalog-valid ID 的结果。

### 5.4 排序实验

排序改动在语义修复完成并建立新基线后单独进行，避免无法归因：

- 候选保留 route-weighted RRF/fused score、attribute contribution 和 profile contribution。
- Ranker 对各 source 分数先在当前候选集内规范化，再加入一个有上限的 fusion 分量。
- hard constraint coverage 仍优先；relaxed 候选仍排在 exact 候选之后。
- Browsing 的 MMR 只比较当前 `lambda=0.85` 与一个更偏相关性的有限候选值（包括 `1.0`）。不进行无界参数搜索。
- 每次只改变一个排序因素，并用相同 folds 和 seed 比较。

如果排序阶段没有提高 selection score，或提高均值但显著增加折间方差，该阶段全部撤销，保留语义修复版本。

### 5.5 Dense 资产决策

现有 ONNX 模型、tokenizer、量化向量和 SHA256 manifest 在本轮保持不变。只有满足以下全部条件时才另开设计：

1. 语义与排序阶段已经通过验收；
2. 剩余失败主要是 dense/lexical Top-150 均未召回，而不是过滤或重排；
3. 有可重复的 folds 1-4 资产消融基线；
4. 新资产仍满足离线、许可证、包大小、冷启动和内存要求。

## 6. 数据流

```text
message
  -> parser: protected category spans + explicit-cue aliases
  -> state: canonical constraints + bounded raw query evidence
  -> route: preserve route on no-preference
  -> catalog semantics: shared category/open-text matching
  -> lexical / dense / attribute / profile candidate lists
  -> route-weighted RRF
  -> constraint-aware ranker
  -> optional bounded MMR
  -> contract-safe Top 10 response
```

Override 开始新 goal 时继续清除旧 goal 的用户约束和旧问题状态。category 只有在新消息明确替换类别时才改变；不得因 alias 消歧规则而继承冲突的旧 style/brand。

## 7. 错误处理与性能

- 缓存只从只读 catalog 构建，不写回数据文件。
- 规范化或开放匹配失败时回到现有规范索引，不允许 Agent 抛出异常。
- dense 缺失、FTS 失败和空召回继续使用已有降级路径。
- 不在每次候选匹配时重新 flatten 完整商品 JSON。
- 完整 catalog 的单轮 P95 必须继续低于 1.5 秒；若新增缓存使冷启动内存增加超过 15%，必须记录并决定是否改用紧凑 postings。
- 推荐结果继续满足唯一、合法、最多 10 个 ID 的契约。

## 8. TDD 与回归覆盖

每项生产改动必须先有能正确失败的测试，再实现最小修复。至少覆盖：

1. Buying 会话回答 no-preference 后仍保持 Buying route。
2. Intent Override 会话回答 no-preference 后不丢失 override goal。
3. `Fashion Sneakers`、`Boy Shorts` 等类别短语不产生无提示 style/brand 硬约束。
4. 显式 `brand`、`by`、`style` 提示仍能产生对应约束。
5. 未识别的开放澄清文本不成为不可能的 hard constraint，但保留在 query evidence 中。
6. category token-union 对等价层级名称匹配，对无关类别不匹配。
7. hard filtering、attribute candidates 和 fallback 对同一 category constraint 返回一致语义。
8. 缓存结果与直接规范化结果一致，并且顺序确定。
9. fusion 分量能改变构造样例的最终顺序，但不能把 relaxed 候选提前到 exact 候选之前。
10. 官方 response contract、fallback、override 和 800-session 稳定性测试继续通过。

## 9. 实验与验收协议

### 9.1 阶段

| 阶段 | 内容 | 是否可独立回退 |
|---|---|---|
| S0 | 未修改基线 | 不适用 |
| S1 | parser 消歧、no-preference 路由修复 | 是 |
| S2 | 共享 category/open-query 语义及缓存 | 是 |
| S3 | fusion 排序实验 | 是 |
| S4 | 有界 MMR 实验 | 是 |

每个阶段均运行 folds 1-4。只有父阶段通过测试且当前阶段达到接受标准，才进入下一阶段。

### 9.2 选择指标

主指标：

```text
selection_score = mean(TechnicalScore on folds 1-4)
                  - 0.5 * std(TechnicalScore on folds 1-4)
```

最终代码相对 S0 必须满足：

- selection score 至少从 `0.500756` 提高到 `0.510756`；
- mean TechnicalScore 不低于 `0.529195`；
- folds 1-4 不出现超过 `0.02` 的单折 TechnicalScore 回退；
- Buying、Browsing、Intent Override 的开发折合并 HitRate 均不得下降超过 `0.025`；
- Boundary 样本少，若净损失一个命中即必须逐例解释并拒绝无明显总收益的阶段；
- MRR、MTTC、每场景指标、recover/regress 会话数都必须记录，不能只报告总分。

单一阶段若未满足上述稳定性要求则回退，不与下一阶段捆绑后再次尝试掩盖退化。

### 9.3 最终验证

候选确定后依次运行：

1. focused unit/integration tests；
2. 全部 pytest 与 lint；
3. folds 1-4 CV 和会话级 baseline diff；
4. 800 个无标签模拟稳定性测试；
5. 完整 200-session evaluator，作为最终公开结果报告；
6. dense 禁用和资产缺失 fallback；
7. 完整 catalog 延迟、内存和冷启动测量；
8. submission package 契约和 SHA256 校验。

完整公开分只在最终候选选定后报告，不用于新增针对性规则。报告必须同时列出 Windows 基线、新分数、绝对变化、四类场景变化和剩余可优化方向。

## 10. 风险与控制

| 风险 | 控制 |
|---|---|
| 放宽匹配提高召回却稀释 Top 10 | 只放宽 clarification/open 属性；分阶段检查 MRR 与 regressions |
| category token 匹配过宽 | 要求全部实义词蕴含；通用 taxonomy 词单独规范化 |
| parser 消歧漏掉真实 style/brand | 保留显式 cue 和规范固定值路径 |
| 新缓存增加冷启动内存 | 只缓存规范词项/紧凑 postings；设置 15% 增量审查线 |
| 排序权重过拟合公开集 | 仅少量有理论依据的候选；使用 selection score 与方差惩罚 |
| 平台浮点差异改变 Top-N 边界 | ID 作为稳定次级排序键；记录 Windows/macOS 会话 diff |
| 已查看 fold 5 导致错误的“盲测”声明 | 明确标注污染；不把 fold 5 用作版本选择门槛 |

## 11. 交付物

- 语义、路由、候选生成和可选排序代码修改；
- 对应的失败先行回归测试；
- 每阶段 folds 1-4 指标与接受/回退记录；
- 最终完整 evaluator 结果 JSON；
- baseline 与候选的会话级 recover/regress 报告；
- 剩余失败模式和下一轮优化建议；
- 不包含用户现有未跟踪文档或无关生成文件的独立提交。
