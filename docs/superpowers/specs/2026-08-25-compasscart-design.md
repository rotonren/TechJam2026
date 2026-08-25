# CompassCart: TikTok TechJam 2026 Track 4 设计与团队规划书

## 1. 文档状态

- 日期：2026-08-25
- 赛题：Track 4 - Shopping Copilot: AI Conversational Search and Recommendations
- 项目代号：CompassCart
- 团队：5 人，多数成员为新手
- 开发主机：AMD Ryzen 5 7600、32 GB 内存、AMD RX 7600、Windows
- 已确认策略：评分与晋级优先；付费 API 可用于开发，但最终 Agent 必须离线独立运行
- 设计状态：已由团队负责人确认，可进入实施计划阶段

## 2. 执行摘要

CompassCart 是一个以转化效率为目标的多轮购物 Agent。它不把核心能力建立在在线 LLM 对话上，而是通过可复现的本地检索、版本化约束状态、场景路由和转化收益驱动的澄清策略，在最多 10 轮内尽早把隐藏目标商品排进 Top 10。

系统采用离线优先的混合架构：字段化 BM25、属性倒排索引和轻量语义向量负责候选召回；Constraint Ledger 负责约束积累、冲突与 Intent Override；Question Value Estimator 选择最能提高下一轮命中概率的属性；重排器同时优化 HitRate@10、MRR 和 MTTC。付费 API 只用于开发阶段生成措辞扰动、标注困难案例和分析失败，不是最终评分路径的依赖。

项目的唯一主创新点是“转化收益驱动的澄清策略”：系统不按固定顺序询问属性，而是根据当前候选分布、剩余轮数和预期 Top 10 命中增益选择下一问题。该能力直接对应官方三项技术指标，也容易通过消融实验和现场演示证明价值。

## 3. 官方约束与设计影响

| 官方条件 | 设计响应 |
|---|---|
| 50,000 个固定商品 | 共享只读目录索引；无需外部向量数据库 |
| 200 个公开会话、800 个私有会话 | 分层验证、封存审计折和措辞扰动，禁止直接记忆公开答案 |
| Buying、Browsing、Intent Override、Boundary 四类场景 | 路由、状态更新和指标均按场景独立记录 |
| 最多 10 轮，超出后零分 | 每轮都返回 Top 10；问题策略显式考虑剩余轮数 |
| 只评分前 10 个有效且唯一的 parent_asin | ResponseBuilder 负责去重、合法性检查和固定长度保护 |
| 最终评分可能关闭网络 | 最终 Agent 不调用外部 API；所有必要模型和配置本地化 |
| 最终环境可能限制 CPU、内存和超时 | CPU 优先、量化模型、候选集上限、分阶段超时和降级路径 |
| UI 不计分，使用 headless API | 不开发产品 UI；演示采用终端、轨迹和指标报告 |
| 禁止修改目录和注入虚假 ASIN | 目录只读；所有推荐必须来自官方 catalog ID 集合 |

## 4. 目标与非目标

### 4.1 产品目标

1. 在最多 10 轮内识别用户需求并尽早命中隐藏目标商品。
2. 对明确购买和开放浏览采用不同的候选生成策略。
3. 正确处理约束累积、用户改意和无偏好回答。
4. 在断网、缺失语义模型或部分索引失败时仍返回合法结果。
5. 让每次推荐和提问都能通过轨迹解释和离线指标复现。

### 4.2 内部晋级目标

官方未公布决赛分数线，因此下表是团队内部的 go/no-go 门槛，不代表官方保证。

| 指标 | 最低提交线 | 竞争目标 |
|---|---:|---:|
| TechnicalScore | 0.35 | 0.45 或更高 |
| HitRate@10 | 0.40 | 0.55 或更高 |
| MRR | 0.25 | 0.35 或更高 |
| MTTC | 7.0 或更低 | 5.5 或更低 |
| 有效完成率 | 100% | 100% |
| 单轮 P95 延迟 | 2 秒或更低 | 1 秒或更低 |
| 措辞扰动后分数下降 | 不超过 20% | 不超过 15% |
| 场景稳定性 | 四类均高于 starter | 无单类显著塌陷 |

### 4.3 非目标

- 不开发 UI、支付、真实交易或多用户并发平台。
- 不训练或全参数微调基础大模型。
- 不使用外部工业向量数据库。
- 不把在线 LLM、在线 embedding 或联网服务作为最终运行必需项。
- 不针对公开 evaluator 的固定句式、样本 ID 或公开答案建立记忆表。
- 不把 `ask_attribute="other"` 作为绕过正常提问策略的默认捷径。

## 5. 总体架构

```text
                         +----------------------+
catalog.jsonl ---------->| Catalog Compiler     |
                         | fields / FTS / vector|
                         +----------+-----------+
                                    |
user_profile                       shared read-only indexes
      |                             |
user_message --> Intent & Slot Parser
                    |
                    v
              Constraint Ledger
                    |
                    v
        Buying / Browsing / Override Router
                    |
          +---------+---------+
          |                   |
          v                   v
    Hybrid Retriever    Question Value Estimator
          |                   |
          +---------+---------+
                    v
          Constraint-aware Ranker
                    |
                    v
             Response Builder
             question + Top 10
                    |
                    v
            Trace / Metrics Sink
```

所有目录级数据均为只读共享对象；只有 SessionStore 保存会话状态。模块通过稳定的数据对象通信，不互相读取内部字段，从而允许五名成员并行开发与单独测试。

## 6. 核心数据结构与接口

### 6.1 Constraint

```python
@dataclass(frozen=True)
class Constraint:
    attribute: str
    value: str
    confidence: float
    is_hard: bool
    source: str
    created_turn: int
    intent_version: int
    status: Literal["active", "superseded", "rejected"]
```

`source` 只能是 `message`、`profile`、`clarification` 或 `inferred`。用户消息和澄清回答的权重高于 profile；profile 永远只能产生软约束。

### 6.2 SessionState

```python
@dataclass
class SessionState:
    session_id: str
    turn: int
    route: Literal["buying", "browsing"]
    intent_version: int
    constraints: list[Constraint]
    asked_attributes: list[str]
    no_preference_attributes: set[str]
    previous_recommendations: list[str]
    candidate_count: int
```

### 6.3 RetrievalPlan

```python
@dataclass(frozen=True)
class RetrievalPlan:
    route: str
    query_text: str
    hard_filters: dict[str, tuple[str, ...]]
    soft_preferences: dict[str, tuple[str, ...]]
    excluded_values: dict[str, tuple[str, ...]]
    ask_attribute: str | None
    candidate_limit: int = 500
```

### 6.4 模块接口

```python
state = dialog_engine.update(session_id, user_message, turn)
plan = question_policy.build_plan(state, user_profile)
candidates = catalog_index.search(plan)
ranked = ranker.rank(candidates, state, user_profile)
response = response_builder.build(ranked[:10], plan.ask_attribute)
```

任何模块失败时，调用方只能使用该模块声明的 fallback，不能访问其内部实现临时修补。

## 7. 模块设计

### 7.1 Catalog Compiler

职责：

- 规范化标题、分类、features、details、description、store 和 price。
- 提取 category、material、color、size、style、brand、budget、feature 和 use_case。
- 建立 parent_asin 合法集合和属性倒排表。
- 建立 SQLite FTS5 字段化索引。
- 加载预计算的量化语义向量。

字段化 BM25 初始权重为：title 6.0、categories 4.0、features 2.5、details 2.5、store 1.5、description 1.0。权重只能通过保留集实验调整。

语义编码器默认使用 `all-MiniLM-L6-v2` 的 ONNX int8 版本，输出 384 维向量。50,000 个商品向量以 int8 矩阵和缩放因子保存，目标总资产大小控制在 100 MB 内。若语义资产不可用，系统自动进入 lexical-only 模式。

### 7.2 Intent & Slot Parser

解析器采用三层决策：

1. 明确语义规则：价格、颜色、尺寸、材质、品牌和 override 表达。
2. 轻量词典与正则：将自由文本归一化到官方允许属性。
3. 本地线性分类器：处理开发期 API 生成的同义表达。

Buying 的初始信号包括明确硬条件、较高查询具体度和较小候选预估；Browsing 的初始信号包括开放探索表达、低具体度和大候选预估。若消息包含“改为”“忽略之前”“instead”“actually”等覆盖语义，或新约束与活跃约束发生高置信冲突，则进入 override 更新流程。

### 7.3 Constraint Ledger

- 每次 reset 创建 intent_version 1。
- 新约束与相同属性同值时提高置信度，不重复追加。
- 新约束与同属性旧值冲突时，显式用户输入胜出。
- 检测到 Intent Override 时 intent_version 加一；冲突的旧约束标记为 superseded。
- category 和未冲突的高置信硬约束可跨版本保留。
- profile 约束在任何用户冲突下立即降为 rejected。
- Boundary 场景返回“无偏好”后，该属性本会话不再询问。

Ledger 保留历史但检索器只读取 active 约束，这使现场演示能够清晰显示“旧要求为什么不再生效”。

### 7.4 Route Planner

路由不是一次性标签，而是每轮重新计算：

- Buying：属性召回 0.45、BM25 0.35、dense 0.20。
- Browsing：dense 0.45、BM25 0.30、category/profile 0.25。
- Override 后首轮：BM25 0.35、dense 0.35、属性召回 0.30，避免旧过滤造成零召回。

路由变化、权重和原因必须写入 trace，便于 D 负责人按场景分析。

### 7.5 Hybrid Retriever

并行候选源：

1. 字段化 BM25 Top 150。
2. 属性和 category 索引 Top 150。
3. dense cosine Top 150。
4. profile 轻先验 Top 50，仅用于 Browsing。

候选使用 weighted Reciprocal Rank Fusion 合并，`k=60`，去重后最多保留 500 个。硬过滤只对高置信显式约束执行；软约束和 profile 只参与加权。

若结果为空，按以下顺序逐级放宽：

1. 移除 profile 权重。
2. 将最低置信软约束改为纯排序特征。
3. 保留显式硬约束，扩大 lexical 和 dense 候选数。
4. 仍为空时返回同 category 高质量商品；category 也未知时返回全局高质量商品。

系统不得返回空 recommendations，除非官方目录本身为空。

### 7.6 Constraint-aware Ranker

初始归一化评分：

```text
score =
    0.30 * hard_constraint_coverage
  + 0.20 * lexical_score
  + 0.20 * dense_score
  + 0.10 * category_match
  + 0.10 * soft_preference_coverage
  + 0.05 * profile_affinity
  + 0.05 * quality_prior
  - 0.60 * explicit_hard_conflict
```

质量先验由 average_rating 和 log1p(rating_number) 归一化得到，只用于难以区分的候选。Browsing 路由在最终 Top 10 使用 `lambda=0.85` 的 MMR 轻度去重；Buying 不做多样性扩展，优先精确命中。

最终权重由 D 负责人进行受约束搜索：每个正权重只能在初始值上下 0.10 范围内变化，冲突惩罚保持不低于 0.40，防止在 200 个公开会话上形成不可解释的极端配置。

### 7.7 Question Value Estimator

这是项目的核心创新模块。对每个尚未询问且未被拒绝的属性 `a`，将当前候选按属性值分组。以当前排名分数归一化为候选概率，估计回答该属性后 Top 10 命中概率的期望变化：

```text
gain(a) = expected_top10_probability_after_answer(a)
          - current_top10_probability

utility(a) = gain(a)
             * attribute_coverage(a)
             * response_likelihood(a)
             * remaining_turn_factor
             - repeat_penalty
             - no_preference_penalty
```

规则：

- 候选数小于等于 10 时不再提问。
- 同一属性最多询问一次。
- turn 8 之后只在预期增益高于 0.15 时提问。
- Boundary 返回无偏好后，该属性 utility 永久为负。
- `other` 仅在所有标准属性 utility 低于 0.03、候选数大于 200 且从未使用时作为一次性 fallback。
- 无论是否提问，每轮都返回当前 Top 10。

### 7.8 Response Builder

Response Builder 不承担检索推理，只保证协议正确：

- `message` 为简短自然语言，不包含内部得分。
- `ask_attribute` 必须来自官方枚举或为 null。
- recommendations 按排序顺序去重并验证 parent_asin。
- 最多输出 10 项；不足 10 项时使用当前路由的安全 fallback 补齐。
- usage 在离线模式固定报告零 token。
- 任意异常都转换成合法响应，绝不让异常越过 `Agent.respond()`。

## 8. 单轮数据流

1. Evaluator 调用 `reset(session_id, user_profile)`，SessionStore 创建初始状态。
2. `respond()` 接收 user_message、turn 和 top_k，并校验 turn 在 1 到 10。
3. Parser 抽取意图和约束；Ledger 完成版本化合并。
4. Router 根据查询具体度、约束数量和候选规模选择 Buying 或 Browsing。
5. Question Value Estimator 基于上轮候选分布选择 ask_attribute。
6. Retriever 产生最多 500 个候选；Ranker 输出完整排序。
7. Response Builder 返回问题和 Top 10，并记录推荐 ID。
8. Trace Sink 记录路由、约束、候选数量、组件耗时、提问属性和 fallback，但不记录密钥或敏感数据。

内部单轮软预算为 800 ms：Parser 100 ms、检索 300 ms、重排 250 ms、响应和日志 150 ms。某阶段超时时立即使用上一阶段结果，不等待整轮超时。

## 9. 错误处理与降级

| 故障 | 降级行为 |
|---|---|
| ONNX 模型缺失或加载失败 | lexical + attribute 模式，写入一次警告 |
| 商品向量损坏或 checksum 不匹配 | 禁用 dense，不重新下载 |
| FTS5 不可用 | 使用内存 token 倒排和 TF-IDF 简化评分 |
| Parser 抛出异常 | 保留旧状态，将本轮作为 Browsing 自由文本查询 |
| 约束导致零候选 | 按既定四级放宽规则执行 |
| Ranker 超时 | 使用 RRF 顺序作为最终排序 |
| SessionState 丢失 | 使用当前消息和 profile 重建最小状态 |
| user_message 为空 | 不新增约束，返回安全推荐并询问 category |
| recommendation 非法或重复 | Response Builder 删除并用 fallback 补齐 |
| 日志写入失败 | 评分路径继续运行，禁用后续 trace 写入 |

所有 fallback 都必须有单元测试和集成测试。系统禁止静默联网、自动下载模型或在官方运行期间写入 catalog。

## 10. API 使用与成本策略

付费 API 只用于开发工具，不被 `submission/agent.py` 或其运行时依赖导入。

允许用途：

1. 为公开消息生成英文同义改写、顺序变化和含糊表达。
2. 对 parser 难例标注 intent、attribute 和 override。
3. 对 Top 20 候选做离线 pairwise 判断，帮助发现缺失的排序特征。
4. 总结失败轨迹并提出可验证假设。
5. 生成演示文案初稿，但最终内容由团队核验。

禁止用途：

- 生成或猜测私有会话答案。
- 把 API 输出直接作为公开样本的记忆映射。
- 在正式 Agent 中依赖在线调用。
- 把 API key 写入仓库、日志、视频或 Devpost。

默认开发 API 预算为 200 美元。每新增 50 美元预算必须在封存折之外证明至少 0.01 的 TechnicalScore 提升或明显提高扰动稳定性；总预算上限为 500 美元。无证据的批量生成立即停止。

## 11. 验证与实验设计

### 11.1 数据划分

- 使用固定随机种子按 scenario_type 和 difficulty_bucket 分成 5 个分层折。
- 折 1 到 4 用于轮换开发与权重选择。
- 折 5 在第 54 小时前封存，不向模块开发者提供逐样本失败信息。
- 第 52 小时必须仅根据折 1 到 4 选出唯一 release candidate。
- 第 54 小时只对该 candidate 运行一次折 5；不得根据折 5 切换权重、特征或候选版本。
- 若折 5 暴露协议错误、崩溃或数据损坏，只允许修复该确定性错误并完整重跑；不得针对评分结果调优。
- API 生成的 paraphrase 只能用于稳健性测试和 parser 训练，不得替代官方指标。

### 11.2 版本选择

```text
selection_score =
    mean(TechnicalScore across development folds 1-4)
    - 0.5 * std(TechnicalScore across development folds 1-4)
    - timeout_rate
    - invalid_response_rate
```

公开全集最高单次分数不能作为最终版本选择依据。折 5 的作用是审计：TechnicalScore 必须至少为 0.35，且四类场景不得出现接近零的塌陷；审计失败时保留结果并在限制说明中如实披露，而不是继续调参。

### 11.3 实验记录

每次完整实验必须记录：

- Git commit 和配置 hash。
- 总 TechnicalScore、HitRate@10、MRR、MTTC、Efficiency。
- 四类场景的分项指标。
- 各组件 P50/P95 延迟和启动耗时。
- fallback 次数、异常数和非法响应数。
- 与上一个稳定版本的唯一变量及假设。
- API 成本和生成数据版本，如果实验使用了开发 API。

## 12. 测试策略

### 12.1 单元测试

- 字段清洗和属性归一化。
- Intent、slot 和 override 解析。
- Constraint Ledger 的合并、覆盖、拒绝和版本行为。
- RRF、归一化、硬冲突惩罚和 MMR。
- Question Value 的覆盖率、重复惩罚和 turn 规则。
- ResponseBuilder 的枚举、去重、合法 ID 和固定上限。

### 12.2 契约测试

- `Agent.reset()` 和 `Agent.respond()` 完全符合官方 schema。
- turn 1 到 10 均返回合法对象。
- 任意异常输入不会抛出到 evaluator。
- recommendations 全部来自冻结 catalog。

### 12.3 场景集成测试

- Buying：硬约束保持且精确过滤。
- Browsing：候选足够多时主动提出高价值问题。
- Intent Override：旧约束被标记 superseded，新约束立即生效。
- Boundary：无偏好属性不重复询问。
- Miss：到 turn 10 前持续返回合法候选并逐步放宽。

### 12.4 变形与稳健性测试

- 同义改写不应改变主要 slot。
- 约束顺序变化不应改变最终 active 集合。
- 插入无关礼貌语不应显著改变排序。
- 大小写、标点和空白变化不应影响结果。
- 断网、API key 缺失和 DNS 失败不得影响正式 Agent。

### 12.5 性能与故障注入

- 800 个模拟会话连续运行，无内存持续增长。
- 冷启动不超过 60 秒，单轮 P95 不超过 2 秒。
- 分别禁用 dense、FTS、日志和缓存，验证声明的 fallback。
- 损坏向量 checksum、制造空候选和重复 ID，验证响应合法。

## 13. 五人分工

| 角色 | 主责 | 固定交付 | 交叉 Reviewer |
|---|---|---|---|
| A：队长与集成 | Agent 接口、配置、主分支、任务取舍 | 一键评测、稳定 main、发布包 | D |
| B：检索负责人 | Catalog Compiler、BM25、属性索引、dense、RRF | `CatalogIndex.search()` | C |
| C：对话负责人 | Parser、Ledger、Router、Question Value | `DialogEngine.update()`、`QuestionPolicy` | B |
| D：排序与实验 | Ranker、数据划分、指标、权重搜索、消融 | `Ranker.rank()`、实验报告 | A |
| E：可靠性与提交 | fallback、契约测试、性能、README、演示 | 测试报告、可复现提交包 | A |

协作约束：

- A 不独占大功能，首要职责是持续集成和缩小范围。
- 每项任务控制在 2 到 4 小时；超过 4 小时必须重新拆分。
- 每个 PR 只改变一个行为，附带测试或指标证据。
- 每 4 小时运行一次完整公开集评测并保存结果。
- 没有开发折改进证据的复杂能力不得合并。
- 第 60 小时后冻结架构，只允许修复、配置选择、文档和打包。

## 14. 72 小时执行计划

### 14.1 赛前准备

- 五人均能运行 starter 和官方 evaluator。
- 下载并验证 catalog checksum。
- 建立环境、Git 规则、任务看板和密钥管理。
- 复现官方 baseline：TechnicalScore 0.10671、HitRate@10 0.125、MRR 0.068034、MTTC 9.81。

### 14.2 H0-H8：可复现底座

- 完成统一 Agent 入口、指标记录、数据划分和最小测试。
- 验收：一条命令完成 200-session 评测；baseline 可重复；干净环境可启动。

### 14.3 H8-H24：混合召回 MVP

- 完成字段 BM25、属性索引、dense 和 RRF。
- 关卡：TechnicalScore 至少 0.22、HitRate@10 至少 0.25、P95 不超过 1.5 秒。
- 未达标时停止增加模型，优先修复字段清洗和查询构造。

### 14.4 H24-H42：状态和澄清策略

- 完成 Router、Ledger、Override、Boundary 和 Question Value。
- 关卡：TechnicalScore 至少 0.35、HitRate@10 至少 0.40、MRR 至少 0.25、MTTC 不高于 7。

### 14.5 H42-H54：排序和分数冲刺

- 执行受约束权重搜索和逐项消融。
- 目标：TechnicalScore 至少 0.45、HitRate@10 至少 0.55、MRR 至少 0.35、MTTC 不高于 5.5、开发折标准差不高于 0.04。

### 14.6 H54-H60：封存折和故障模拟

- 首次运行折 5，随后冻结架构。
- 执行措辞扰动、800-session、断网和故障注入。
- 验收：扰动分数下降不超过 15%，零崩溃、零非法响应。

### 14.7 H60-H72：冻结和提交

- H60-H64：干净环境重装和完整评测。
- H64-H68：README、Devpost、消融表、成本和限制说明。
- H68-H70：录制端到端演示，突出 Intent Override 和离线运行。
- H70-H72：检查提交包、依赖、密钥、checksum，并保留回滚缓冲。

## 15. Git 与工作流

- `main` 必须始终通过契约测试并能运行官方 evaluator。
- 分支使用 `codex/<ticket>-<summary>` 命名。
- 禁止直接向 main 推送未经审查的功能。
- 生成数据、catalog、模型缓存、API 响应和密钥不得进入普通源码提交。
- 允许提交经过许可的量化模型和商品向量，但必须包含来源、许可证和 SHA256。
- 每次稳定评分建立带指标的 Git tag，方便在最后阶段快速回滚。

## 16. 风险与缓解

| 风险 | 触发信号 | 缓解措施 |
|---|---|---|
| 公开集过拟合 | 总分升高但折间波动扩大 | 封存折、受约束权重、paraphrase 测试 |
| 过度依赖 evaluator 固定句式 | 改写后 parser 大幅失效 | API 扰动训练、规则与分类器双路径 |
| 在线 API 在决赛不可用 | 本地运行需要 key 或网络 | 正式 Agent 零网络依赖，API 仅在 tools 中 |
| AMD/Windows 模型兼容性 | ONNX 安装或运行失败 | CPU execution provider；lexical-only fallback |
| 模型资产过大 | 提交包超过内部 100 MB 目标 | int8 ONNX 和向量；必要时删除 dense 保底 |
| 新手并行导致接口冲突 | PR 修改多个模块或 main 长期不可运行 | 固定数据结构、短任务、交叉 reviewer |
| 复杂重排拖慢响应 | P95 超过 2 秒 | 候选上限、阶段预算、超时使用 RRF |
| Intent Override 保留旧条件 | override 场景明显低分 | 版本化 Ledger 和专门集成测试 |
| 提问过多导致 MTTC 偏高 | HitRate 尚可但 MTTC 大于 7 | 每轮推荐、turn-aware utility、后期少问 |
| 最后阶段无可提交版本 | H60 仍在改架构 | H60 强制冻结、稳定 tag 和发布清单 |

## 17. 评分维度与答辩策略

| 评审维度 | 提交证据 |
|---|---|
| Technical Execution 35% | TechnicalScore、场景分项、契约测试、断网与故障注入 |
| Innovation 20% | Question Value 公式、固定提问对照实验、状态版本化设计 |
| Impact 20% | 更少轮次降低用户认知负担；Buying 与 Browsing 的实际商业路径 |
| Feasibility 15% | CPU 运行、无外部数据库、零在线 token、资产和延迟报告 |
| Presentation 10% | 一条 Browsing 会话和一条 Intent Override 会话的完整轨迹 |

演示顺序：

1. 展示 baseline 指标和失败类型。
2. 输入模糊 Browsing 请求，展示候选数和系统选择澄清属性的原因。
3. 在 2 到 4 轮内展示目标进入 Top 10 和排名变化。
4. 展示 Intent Override，旧约束变为 superseded，新检索路径立即生效。
5. 关闭网络后再次运行，证明正式 Agent 不依赖 API。
6. 展示消融表：无 Question Value、无 dense、无 Ledger 和完整系统。

## 18. 目标目录结构

```text
techjam/
  agent.py
  requirements.txt
  README.md
  src/compasscart/
    catalog.py
    normalization.py
    parser.py
    state.py
    router.py
    retrieval.py
    ranker.py
    question_policy.py
    response.py
    tracing.py
    config.py
  assets/
    model/
    product_vectors/
    SHA256SUMS
  tools/
    build_assets.py
    run_cv.py
    analyze_failures.py
    generate_paraphrases.py
    package_submission.py
  tests/
    unit/
    integration/
    contract/
    performance/
  reports/
    experiments/
    final/
  docs/
    superpowers/specs/
```

`agent.py` 是唯一官方入口，只组装模块；业务逻辑必须位于 `src/compasscart`。开发 API 工具只能存在于 `tools`，并且不能被 `agent.py` 导入。

## 19. 最终完成定义

只有同时满足以下条件，项目才算完成：

1. 从干净环境按 README 一条命令完成安装和官方 evaluator 运行。
2. 完全断网、无 API key 时功能完整。
3. 800 个模拟会话零未捕获异常、零非法响应、零 catalog 外 ID。
4. 封存折 TechnicalScore 至少 0.35，四个开发折的 selection_score 选择过程可复现。
5. Buying、Browsing、Intent Override 和 Boundary 均有独立指标和测试。
6. 提交包不包含密钥、私有数据、organizer-only 文件或未声明服务。
7. README、Devpost 描述、成本披露、限制说明和演示视频全部完成。
8. 至少保留一个经过完整验证的稳定 Git tag，可在最终提交失败时回滚。

本规格将晋级概率置于功能数量之上。任何新增能力如果不能提高保留集指标、稳健性或评审证据，就不进入正式提交。
