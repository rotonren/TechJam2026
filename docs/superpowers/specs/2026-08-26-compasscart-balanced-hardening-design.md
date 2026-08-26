# CompassCart 均衡提分与运行强化设计

## 1. 文档状态

- 日期：2026-08-26
- 当前版本：`b641ff97b0f4d7ae0c2fc7646e250492370231bd`
- 当前公开集 TechnicalScore：`0.660411`
- 用户目标：在保持离线、稳定、低成本和可复现的前提下继续提高私测泛化能力与公开分数
- 已批准方向：均衡泛化，依次处理官方环境稳健性、澄清语义、查询历史、最终排序、提问策略和自适应召回
- 书面规格状态：待用户复核后进入实施计划

## 2. 当前证据

当前版本在冻结的 50,000 商品目录和 200 条公开会话上取得：

| 指标 | 当前值 |
|---|---:|
| TechnicalScore | 0.660411 |
| HitRate@10 | 0.840000 |
| MRR | 0.376036 |
| MTTC | 4.620 |
| Efficiency | 0.638000 |

剩余 32 个未命中会话的只读重放显示：

- 25 个目标已进入最终候选，但位于 Top 10 之后；其中 15 个最好名次为 11-20。
- 7 个目标曾进入 route-weighted RRF Top 10，随后被最终 Ranker 降出 Top 10。
- 5 个终局失败来自澄清回答被错误转换成跨属性硬约束。
- 2 个终局失败未进入截断候选源；其中 1 个早期曾被召回，后来因查询历史漂移消失。
- 只有 1 个目标在全部轮次都未进入任何截断候选源，其首轮未截断 attribute 名次为 206。

因此下一阶段的主要瓶颈是排序校准和对话语义，而不是全局召回深度或模型容量。

当前运行证据为：完整折 P95 最高 `483.890 ms`，单 Agent 初始化约 `14.084 s`，单轮响应约 `241.339 ms`，工作集约 `467.9 MiB`，峰值约 `540.5 MiB`。官方没有公布初始化、内存或超时上限，因此资源风险必须与分数一起控制。

## 3. 目标与非目标

### 3.1 目标

1. 保证从任意当前工作目录导入根目录 `agent.py` 时都能找到并校验随包 Dense 资产。
2. 消除澄清回答的跨属性 alias 污染和不受支持的硬过滤。
3. 只让实质购物证据进入检索历史，避免控制话术挤掉真实需求。
4. 让最终 Ranker 保留 route、attribute 和跨来源共识证据，优先挽回 11-20 名目标。
5. 提问只选择可回答、可解析、可检索且具有正向预期收益的属性。
6. 只在有召回不足证据的 Buying 场景自适应增加 attribute 深度。
7. 在输出顺序不变的前提下减少重复排序、fallback、MMR 和属性提取工作。
8. 使用未包含公开目标的新代理集选择版本，降低继续查看公开失败导致的过拟合风险。

工程目标是将公开结果稳健推向约 `0.68-0.70`，但该范围不是验收承诺。真正的接受条件以代理集、回归门槛和资源门槛共同决定。

### 3.2 非目标

- 不修改 `evaluator/`、`data/public_set.jsonl`、`data/catalog.jsonl`、评分公式或 ground truth。
- 不在生产代码中加入 sample ID、目标 ASIN、公开答案、固定轮次答案或商品特判。
- 本阶段不重新训练 Dense 模型、不重新生成向量、不引入在线 API 或外部数据库。
- 不全局扩大 lexical/dense 候选深度，不直接关闭 Ranker，不接受仅靠公开全集最高分选出的参数。
- 不把延迟优化描述成确定性 TechnicalScore 增益；它只在避免超时、OOM 或 fallback 时保护分数。
- 不提交用户现有未跟踪 DOCX、原始公开评分 JSON 或无关生成文件。

## 4. 方案选择

### 4.1 保守语义修复

只修复资产路径、澄清 alias 和查询历史。风险最低，但没有充分利用 25 个已召回排序失败。

### 4.2 均衡泛化方案（采用）

先完成可证明的语义与运行修复，再进行有界排序、提问和 attribute 深度实验。每项独立提交、独立评估、独立回退。这条路线兼顾分数、私测稳定性和官方机器可运行性。

### 4.3 模型重型方案

加入 cross-encoder、替换 embedding 模型或训练复杂 LTR。潜在上限更高，但会增加包体、内存、CPU 延迟和过拟合风险。现有 32 个失败的来源分布不支持优先采用。

## 5. 推荐架构

### 5.1 运行根目录与资产解析

`agent.py` 继续作为唯一官方入口。运行时代码定义稳定的 submission root：

```text
submission root
  agent.py
  assets/
  src/compasscart/
```

运行时以 `src/compasscart` 所在文件的第二级父目录作为 submission root，并验证该目录同时包含根 `agent.py` 和 `assets/`。相对 Dense 资产路径统一相对于这个 root 解析；绝对路径保持不变。catalog 路径仍由 `Agent(catalog_path)` 显式控制，不改变官方 harness 的数据所有权。若布局验证失败，初始化必须给出可诊断的资产状态并安全降级，不能回退到当前工作目录猜测路径。

资产校验失败继续降级到 lexical，但 Trace/诊断必须区分：路径不存在、checksum 错误、依赖缺失和搜索期异常。新增契约测试必须从仓库外工作目录导入已解压提交包，并断言 Dense 可用，避免静默低分。

### 5.2 澄清上下文边界

Parser 将 `expected_attribute` 视为澄清回答的主要语义边界：

1. 值能映射到 pending attribute 的 catalog 支持词表时，生成该属性的结构化约束。
2. 短回答命中其他属性的动态 alias 时，默认不跨属性生成硬约束。
3. 只有消息含显式属性 cue、明确 goal/category 替换或标准 override 表达时，才允许跨属性解析。
4. 无法映射到 pending attribute 的开放文本保留为有界 soft query evidence，不制造 exact hard filter。
5. `no preference` 只拒绝 pending attribute，不改变现有 route 或 goal。

固定规范词和 catalog-derived 动态 alias 使用同一 gating 入口，防止某一类 alias 绕过语境检查。规则只依赖属性、cue、词表和消息结构，不依赖公开样本身份。

### 5.3 实质查询历史

解析结果增加内部控制信号，用于区分：

- 实质购物证据；
- no-preference 回答；
- 请求更多结果；
- evaluator/UI 控制话术；
- 真实 goal override。

只有实质证据进入 `query_history`。active constraints 始终独立加入 query text，因此过滤控制话术不会丢失结构化需求。真实 goal override 继续清空旧 goal 历史；普通 no-preference 不清空最后实质证据。

历史仍保持有界，不新增跨会话持久化。未知但包含可搜索商品词的开放文本不能仅因“没有解析出结构化约束”而被丢弃。

### 5.4 候选证据与最终排序

HybridRetriever 为每个候选保留以下内部证据：

- route-weighted RRF/fusion score；
- lexical、dense、attribute、profile 的来源贡献或名次；
- hard/soft constraint coverage；
- exact/relaxed 状态；
- deterministic ID tie-breaker。

Ranker 保留现有 hard conflict 和 exact-before-relaxed 规则。排序实验只增加有界特征，不整体绕过 Ranker：

1. **Attribute evidence**：将当前仅通过 fusion 间接保留的 attribute 来源作为显式规范化特征。候选权重只测试 `0.05` 和 `0.10`，并从 lexical/dense 权重中等量扣除，保证总权重不增加。
2. **Cross-source consensus**：候选至少同时出现在两个正权重非 profile 来源，且其中一个是 attribute 或 lexical 时，测试上限为 `0.025` 或 `0.05` 的规范化加成。
3. **Route-aware fusion retention**：保留 Retriever 已确定的路线差异，避免 Ranker 再次把所有路线近似压成相同 lexical/dense 配比。只比较当前 `0.10` 与 `0.15` 两个 fusion 上限。
4. **Top-boundary protection**：只有 pre-ranker RRF 名次不大于 10、exact、无 hard conflict 且满足上述共识条件的候选可获得最多 `0.025` 的边界特征；这不是强制保留名额，不能覆盖明显更高的结构化匹配。
5. **Adaptive diversity**：Browsing 只有在相邻候选相关性接近且结果高度重复时启用 MMR；不采用已被稳定性门槛拒绝的全局 `lambda=1.0`。

每次实验只改变一个因素。候选权重范围在实施计划中预先列出，不根据公开失败逐例扩大搜索空间。

### 5.5 提问策略

QuestionPolicy 的 utility 从当前固定回答概率扩展为：

```text
utility = candidate_reduction
          * answerability
          * parser_support
          * retrieval_support
          * remaining_turn_value
          - no_preference_risk
```

属性必须同时满足：候选集中至少两个有意义分区、值可由 Parser 接受、值可由 Catalog 匹配、该属性未被回答或拒绝。Buying 和 Override 优先能揭示 hard requirement 的属性；晚轮提高阈值，避免低收益提问。所有轮次仍立即返回推荐，提问不能阻止 Top 10 输出。

提问策略会改变后续对话轨迹，必须在排序稳定后单独评估，不能与排序权重同时调整。

### 5.6 自适应 attribute 深度

默认每来源上限保持 150。只有同时满足以下条件才增加 attribute 深度：

- route 为 Buying 或明确 hard constraints 存在；
- exact 候选不足或 attribute 结果在截断边界仍有有效匹配；
- 当前时间预算允许；
- 最大深度不超过预先设定的小范围上限。

lexical 和 dense 不随之全局扩大。该阶段直接证据只覆盖 1 个全轮 source-absent 失败，因此必须最后实施，且不得以显著延迟换取单个公开会话。

### 5.7 等价输出性能优化

下列改动必须通过输出顺序等价测试，不参与排序参数选择：

- Catalog 初始化时缓存一次全局 popularity 顺序，避免每轮排序 50,000 商品。
- fused exact 候选已经达到返回数量时，不构造完整 fallback 列表。
- 使用有界缓存复用 MMR diversity terms，避免每轮数万次重复 tokenization。
- QuestionPolicy 复用 `CatalogIndex.attributes`，不重新 flatten 每个候选商品。
- manifest 使用流式 SHA-256；Dense `.npy` 在跨平台验证通过后使用只读 mmap，避免不必要复制。
- 搜索期单次异常不永久关闭可选 backend；连续失败达到小阈值后才触发 circuit breaker，成功调用重置计数。
- `component_timeout_ms` 转为协作式总预算：进入可选阶段前检查剩余预算，超预算跳过 Dense/MMR 并保留 lexical/attribute 合法结果。不得用无法回收的后台线程伪造超时。

## 6. 数据与版本选择

公开 200 条会话已经被多次评测和失败归因，不能继续作为自由调参集。新版本选择使用两个互补的目录派生代理套件：代表性套件用于阶段选择，压力套件只做语义和场景回归。

1. 从 50,000 商品中排除公开集全部 target ASIN。
2. 代表性套件使用 seed `20260826` 抽取 2,000 个代理目标。抽样维度为 coarse category、price 缺失/目录四分位、`rating_number` 目录四分位和 searchable-field 完整度（0-2、3-4、5+）。算法先按 stable hash 为每个排除后仍有商品的 coarse category 保留一个目标，再用迭代分层补足样本，使后三个维度和 category 的边际配额尽量接近完整 catalog 分布；每一步选择“未满足配额最多、stable hash 最小”的商品。配额无法精确满足时，差额按固定维度顺序回流并记录。
3. 压力套件使用独立 seed `20260827` 抽取 800 个不同目标。对每个商品计算上述四个维度的逆频率权重之和，再按 stable hash 加权顺序抽样，专门提高稀疏 metadata、低流行度和少数类别覆盖；它不参与权重选择，只用于量化语义和场景回归。
4. 两个套件都保持官方 40/40/15/5 场景比例。目标与场景通过 seed 后的确定性分层 shuffle 关联，不按商品内容挑选场景。
5. 从参与者可见 metadata 按版本化规则构造 intent card，并加入固定清单中的措辞、no-preference 和 override 扰动。`user_profile` 从公开集中仅抽取安全 profile 对象，移除 sample/target 关联后按独立 seed `20260828` 确定性重排复用；`summary` 必须由其余 profile 字段重新生成，不复制原会话文本。
6. 代表性套件的 1,600 条用于四折开发，400 条作为冻结代理 audit。当前版本在 audit 上运行一次并只记录聚合基线；中间阶段不得运行 audit；全部候选固定后，最终候选只运行一次并只与聚合基线比较，不做逐会话调参。
7. 首次 baseline 前冻结生成器版本、所有 seed、strata 清单、profile 构造规则、catalog/public 输入 hash、目标 ID hash、生成输出 hash 和 config hash。任何生成规则变化都建立全新的代理版本和基线，不能与旧分数直接比较。
8. 所有生成物写入 ignored `var/`，不进入生产包或 Git。
9. 本轮强化的当前公开结果只在最终候选确定后运行一次，用作已知基准回归和结果报告。

代理生成器只允许读取公开 target ASIN 形成排除集合，并读取与 target 脱钩后的安全 `user_profile` 池；不得使用公开 intent card、场景答案、失败分类或会话结果生成代理数据，不得优先抽取已知失败商品，不得把公开或代理 target ID 写入生产配置。

## 7. 阶段与独立回退

| 阶段 | 内容 | 主要目的 | 可独立回退 |
|---|---|---|---|
| R0 | 任意 CWD 资产解析、错误分类 | 防止官方环境静默 lexical-only | 是 |
| P0 | 等价输出热点优化 | 降低超时和内存风险 | 是 |
| S1 | 澄清 alias gating | 修复 5 类硬冲突 | 是 |
| S2 | 实质查询历史 | 修复 query drift | 是 |
| S3 | 有界排序校准 | 挽回 Top 11-20 目标 | 是 |
| S4 | answerability-aware 提问 | 改善 MRR/MTTC 和 Override | 是 |
| S5 | 自适应 attribute 深度 | 处理真实召回不足 | 是 |

顺序约束：R0 和代理评测基线先完成；P0、S1、S2 可在输出/状态边界清晰时分别验证；S3 只能在语义稳定后开始；S4、S5 只能在 S3 候选固定后开始。实施拆为两份计划：第一份覆盖代理基线、R0 和 P0；第二份覆盖 S1-S5、最终验证与交付。第一份验收后才编写并执行第二份。

## 8. 错误处理

- Parser、Retriever、Ranker、QuestionPolicy 的异常继续被 Agent 边界转换为合法响应。
- 资产初始化错误必须留下安全、无绝对敏感路径的原因码；官方响应 schema 不增加字段。
- 所有 fallback 输出继续只含 catalog-valid、唯一、最多 10 个 ID。
- exact 候选存在时不得因性能预算直接返回 relaxed 候选。
- 代理数据生成失败必须终止实验，不得偷偷缩小或替换样本。
- 评分输出必须记录 commit、config hash、输入 hash、fallback count 和平台信息。

## 9. 测试设计

所有行为改动遵循失败先行测试。至少新增：

1. 从仓库外 CWD 导入解压包，Dense 资产仍可用。
2. 绝对自定义资产路径保持有效；相对路径以 submission root 解析。
3. 回答 style 问题时 `adjustable` 可按支持语义处理；回答 size 问题时 `boots` 不泄漏为 category hard filter。
4. 含显式 `category`、`size`、`style` cue 的真实跨属性 override 仍被识别。
5. no-preference、请求更多和控制话术不进入 query history；未知实质商品文本仍进入。
6. goal override 清除旧历史但保留 profile 约束。
7. attribute/consensus 特征能挽回构造的第 11 名候选，同时不改变 hard conflict 和 relaxed 顺序。
8. route-aware 排序在 Buying、Browsing、Override 三条路线使用预期证据。
9. MMR 只在满足重复度和分数边界条件时触发，并保持确定性 tie-break。
10. 提问只选择 parser/catalog 均支持的属性，晚轮低收益时返回 `None`。
11. 自适应深度只在规定条件触发，并遵守最大深度和时间预算。
12. popularity、lazy fallback、diversity cache 和 attribute reuse 前后候选及响应顺序逐项一致。
13. 单次 Dense/FTS 搜索异常不会永久关闭 backend；连续失败会安全降级。
14. 官方 contract、fallback、800-session 状态边界和完整 `410+` 测试继续通过。
15. 全量 Dense benchmark 在独立进程、仓库外 CWD 和冻结消息流下记录初始化、RSS/peak working set、P50/P95/max、fallback 和 Dense 可用状态。

## 10. 验收门槛

### 10.1 分数与泛化

每个 S 阶段必须相对其父阶段满足：

- 代理开发四折 selection score 至少提高 `0.003`，或修复明确正确性缺陷且 selection 不下降超过 `0.001`。
- 代理四折 mean TechnicalScore 不下降。
- 任一折 TechnicalScore 不下降超过 `0.015`。
- Buying、Browsing、Intent Override 合并 HitRate@10 均不下降超过 `0.02`。
- 开发折合并 Boundary 净损失 1 个命中即拒绝该阶段；阶段决策不得读取冻结 audit。
- 代理 audit 的当前版本聚合基线只记录一次，最终候选只运行一次；最终 TechnicalScore 不得低于基线，且无场景失效。

代理 audit 的“无场景失效”定义为：Buying、Browsing、Intent Override 的 HitRate@10 相对基线下降均不超过 `0.025`，Boundary 命中数不低于基线，任一场景 TechnicalScore 下降不超过 `0.03`，且四个场景均无 invalid response 或 fallback。

压力套件每个阶段都运行，但不用于选择具体权重。相对当前版本，其总体 TechnicalScore 不得下降超过 `0.01`，Buying、Browsing、Intent Override 的 HitRate@10 均不得下降超过 `0.025`，Boundary 命中数不得下降，所有响应必须合法且正常运行 fallback count 为 0。

最终公开回归要求：

- TechnicalScore 不低于 `0.655411`，即允许最多 `0.005` 的平台/泛化波动；低于该值直接拒绝。
- 目标值为 `0.68-0.70`，但不为追求目标值放宽上述门槛。
- 公开 recover/regress、MRR、MTTC 和四场景指标必须完整报告。

### 10.2 契约与资源

- 全部 pytest、Ruff、contract、fallback、离线安装和 extracted-package smoke 通过。
- evaluator/data/catalog/dense 资产 hash 与当前冻结值一致。
- 先新增统一 benchmark harness，再用未修改的 `b641ff9` 生成同口径 R0/P0 资源基线；报告中的 `483.890 ms`、`540.5 MiB` 和 `14.084 s` 只作历史参考，不直接与新 harness 混比。
- 每次 trial 使用全新子进程、完整 50,000 catalog、Dense enabled、仓库外 CWD、相同冻结 800-response 消息流；父进程不得预先构造 Agent。消息流由代表性开发集中的 200 个固定代理 session 在 `b641ff9` 上各捕获 4 轮 profile/message，后续 benchmark 无视候选 Agent 的提问变化并原样回放，同时校验 transcript hash。运行三次，不清空 OS 文件缓存，分别报告三次值和中位数，并记录 CPU、RAM、OS、Python、ONNX Runtime、进程 peak working set/RSS 口径及 Dense 状态。
- 初始化计时包含 Agent 构造、catalog 索引、manifest 校验和 Dense session 加载；响应计时不包含初始化。Windows 使用进程 `peak_wset` 与 `rss`，其他平台使用等价的最大 RSS 并在报告中标明单位转换。
- 所有阶段的 P95、初始化中位数和 peak 中位数不得比同 harness 的 `b641ff9` 基线高出超过 5%，单响应 max 必须低于 `1.5 s`。
- P0 只有在输出完全等价且 P95 至少改善 10% 或 peak/初始化至少一项改善 5% 时接受；其他两项不得回退超过 5%。资源优化目标仍为 Windows peak 低于 `500 MiB`、初始化低于 `12 s`。
- 正常代理/公开评分运行中的 fallback count、invalid response 和未捕获异常均为 0；故障注入测试必须继续证明 fallback 有效。
- Windows 结果必须确定。Apple Silicon 不是当前环境可声称完成的测试；它是外部发布前置条件：最终包和固定命令交付给指定 Mac 或 macOS CI，完成 package smoke、Dense 可用性、200-session 评分和 session diff 后，才允许报告“跨平台已验证”。没有可用 runner 时明确标记 `macOS verification pending`，不阻塞 Windows 代码实现，但阻塞跨平台完成声明。

## 11. 提交与交付

- 每阶段使用独立、可审查提交，不把多个未验证实验压成一个提交。
- 被拒绝实验不留在生产分支；结果与拒绝理由写入报告。
- 最终更新源码包、比赛提交包、SHA-256、评分 JSON、中文摘要和 GitHub `main`。
- 比赛提交包继续排除 evaluator、公开测试标签、完整 catalog、缓存、虚拟环境和内部实验数据。
- 源码整合包可以包含测试、工具和报告，但必须明确不是比赛上传包。

## 12. 风险与控制

| 风险 | 控制 |
|---|---|
| 继续针对公开 32 个失败过拟合 | 排除公开 target 的代理集选型；公开集只作最终一次回归 |
| 排序挽回 miss 却伤害现有 168 个 hit | session-level recover/regress、折/场景门槛、单因素实验 |
| alias gating 阻止真实 override | 显式 cue 与 goal/category override 独立通道 |
| 过滤控制话术时丢失未知商品需求 | 只过滤已识别控制意图；未知可搜索文本默认保留 |
| attribute 深度增加延迟 | 仅条件触发、有限上限、受总预算约束 |
| mmap 或缓存造成平台差异 | Windows/macOS 输出等价测试和 ID 次级排序 |
| 协作式预算降低 Dense 使用率 | 先消除确定性热点；预算只跳过可选阶段并记录原因 |
| 资源优化改变排序 | 输出等价测试；性能提交与评分提交分离 |
