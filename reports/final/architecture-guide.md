# CompassCart 是怎么跑起来的

给队友看的工作手册。`architecture.md` 是给评委看的一页纸架构图,这份是长版本 ——
当你要改点东西、跑个实验,或者在决赛现场被问到系统细节时,需要的是这一份。

## 一轮对话的形状

评测器调用 `Agent.respond(session_id, message, turn, 10)`,拿回一条回复、一个
`ask_attribute`,和十个 `parent_asin`。内部按顺序走七个阶段。每个阶段都做了包裹,
出错时降级,而不是让整轮失败。

```
message
   │
   ▼
1  SessionStore ─ 把这一轮解析进约束账本
   │              带版本、有上界、能处理 override
   ▼
2  PolicyMemory ─ 观察上一个问题有没有被回答
   │              (跨会话学习发生在这里)
   ▼
3  RoutePlanner ─ 判定 Buying 还是 Browsing,并给出融合权重
   │
   ▼
4  HybridRetriever ─ 词法 + 属性 + 画像 + 稠密,按加权 RRF 融合
   │                 约 500 个候选
   ▼
5  ConstraintRanker ─ 硬约束覆盖度、质量、Browsing 多样性
   │
   ▼
6  RerankStage ─ 按路由重排列表头部
   │             Browsing:短语邻接。Buying:默认不跑。
   ▼
7  QuestionPolicy + StrategySelector ─ 问什么,或者要不要问
   │
   ▼
   ResponseBuilder ─ 十个唯一且在目录中有效的 ID、token 用量、回复文本
```

## 每个阶段负责什么

| 阶段 | 文件 | 它唯一的职责 |
| --- | --- | --- |
| `SessionStore` | `state.py` | 维护一份带版本的账本,记录买家想要什么。override 是覆盖,不是追加。 |
| `MessageParser` | `parser.py` | 把消息变成带类型的约束:属性、值、软硬、操作符、来源。 |
| `PolicyMemory` | `evolution.py` | 估计每个属性有多大概率能被回答 —— 靠观测,不靠我们拍脑袋。 |
| `RoutePlanner` | `router.py` | 依据约束的具体程度判定 Buying / Browsing,并选定融合权重。 |
| `HybridRetriever` | `retrieval.py` | 从四路独立来源产出候选,按加权倒数排名融合。 |
| `ConstraintRanker` | `ranker.py` | 拿候选去对账本打分;精确匹配始终排在"已声明的放宽项"前面。 |
| `RerankStage` | `rerank.py` | 用词级阶段看不到的信号,重排列表头部。 |
| `StrategySelector` | `orchestration.py` | 决定提问策略弃权的那一轮怎么处理。 |
| `QuestionPolicy` | `question_policy.py` | 挑出预期转化收益最高的那个澄清问题。 |
| `ResponseBuilder` | `response.py` | 产出符合契约的回复:十个唯一有效 ID,如实的 token 计数。 |

## 本轮新增的三个阶段

### RerankStage —— 短语邻接重排

融合和约束打分都工作在词级别。一条以连续短语表述的需求 —— 比如
"water resistant rubber outsole" —— 对标题里恰好含有这个短语的商品,
并不会比对把这几个词散落在描述各处的商品得分更高。这个阶段补的就是这个缺失信号。

它只重排列表头部,从不改变列表成员,并且保持精确匹配的候选排在已声明放宽项之前。

**它只在 Browsing 上运行。** 同一个阶段实测下来:Browsing 命中率 `+0.038`,
Buying `-0.025`。Buying 轮次本身已经带着明确的硬约束,排序器信息充分,
这时候再重排就是在破坏已有的正确结果。

### PolicyMemory —— 会自我校正的问题先验

`QuestionPolicy` 给每个候选问题加权,依据是"买家有多大概率能回答它"。
那张表是手写的,而且从来没人验证过。现在 agent 把它当作先验,
再根据自己问出去的问题有没有被回答来更新后验。

第一个会话的行为和手写表完全一致。估计值按证据量逐级细化:
先是汇总的,然后按路由,再然后按买家分群,每一层只有在自己的桶里
攒够观测数之后才会生效。

它找出了我们填反的两个先验。`feature` 原本排在倒数第二档、值 `0.70`,
实际是最能问出东西的属性,`0.973`;`budget` 原本在最高档 `0.90`,
16 次尝试一次都没问出来。这后来被证明是关于目录本身的一个事实:
`parent_asin` 是父商品,不是尺码或颜色的 SKU,所以 91.3% 的商品根本没有尺码字段
—— 不管问法怎么变都答不出来。

学习过程从不接触 ground truth。agent 观察的是自己的问题有没有产生信息,
而不是自己的推荐对不对。

### StrategySelector —— 浪费掉的那一轮怎么用

当提问策略没有任何值得问的问题时,这一轮就白花了:买家会回一句
"那些选项还不太对,你问我某个具体属性吧",信息量为零。`open_probe`
改成问一个开放性问题;`exploit` 则在候选池已经小到"问一个问题不值一轮成本"时
干脆停止提问。

`open_probe` 在 536 轮里只触发十一次,却贡献了 `+0.018` 里的大部分。
**触发次数在这里不是衡量价值的好指标。**

## 配置

下面每一个默认值都是主办方实际会跑到的。每一层都有消融开关,
三个全部关掉会精确复现本轮开始前的分数 `0.761209`。

| 配置项 | 默认值 | 作用 |
| --- | --- | --- |
| `rerank_enabled` | `True` | 阶段 6 的总开关 |
| `rerank_backend` | `"phrase"` | Browsing 路由的后端:`phrase`、`cross_encoder`、`llm` |
| `rerank_window` | `50` | Browsing 路由上参与重排的候选数 |
| `rerank_weight` | `0.8` | 重排分数对排序器原始次序的影响力度 |
| `rerank_buying_backend` | `None` | Buying 路由的后端;`None` 表示沿用 Browsing 的 |
| `rerank_buying_window` | `None` | Buying 路由的窗口;`None` 表示沿用 `rerank_window` |
| `rerank_buying_weight` | `0.0` | **零:该阶段在 Buying 上不运行** |
| `rerank_buying_requires_override` | `False` | 把 Buying 后端限制为只在 override 轮次生效 |
| `rerank_prompt_style` | `"flat"` | 模型 prompt 形态:`flat`、`structured`、`adaptive` |
| `rerank_max_length` | `128` | 交叉编码器的序列长度 |
| `evolution_enabled` | `True` | 策略记忆的总开关 |
| `strategy_enabled` | `True` | 策略选择器的总开关 |

环境变量开关,做消融时不用改代码:

| 变量 | 效果 |
| --- | --- |
| `COMPASSCART_DISABLE_RERANK=1` | 关掉阶段 6 |
| `COMPASSCART_DISABLE_EVOLUTION=1` | 策略记忆退回手写先验 |
| `COMPASSCART_DISABLE_DENSE=1` | 只用词法召回;**分数完全一致,少占 156 MiB** |
| `COMPASSCART_LLM_BASE_URL` / `_API_KEY` / `_MODEL` | 可选模型后端的凭据 |

## 出错时降级到哪

可选路径里的任何东西都不会让一轮对话失败。

| 故障 | 结果 |
| --- | --- |
| 没有 LLM 凭据 | 后端构造时直接返回短语后端 |
| LLM 超时、拒答、回复无法解析,或回复不是一个合法排列 | 该轮不重排;连续失败三次后禁用该后端 |
| 交叉编码器资产缺失或损坏 | 退回短语后端 |
| 稠密资产缺失、损坏,或推理失败 | 退回纯 Python 词法召回 |
| FTS5 不可用 | 退回纯 Python 目录扫描 |
| 解析器、路由、召回、排序、重排、提问、策略、trace 任一抛异常 | 跳过该组件,并在 trace 的 `fallbacks` 里记名 |
| 重排窗口内没有可区分的候选 | 保留排序器原次序,而不是打乱成按 ID 的兜底排序 |

**一个已知缺口。** 构造期失败会退回短语后端;而运行期失败只是把该路由的重排关掉。
在当前发布的默认配置下这没有危害 —— LLM 只会被配置在 Buying 上,而 Buying
默认根本不重排 —— 但这两条路径降级到的位置并不相同,这是运气,不是设计。

## 资源实测

按进程测量,每种配置各跑一次独立运行。

| 配置 | 峰值 RSS | 仅 agent | 初始化 | TechnicalScore |
| --- | ---: | ---: | ---: | ---: |
| 默认 | 749.7 MiB | 409.6 MiB | 24418 ms | 0.822490 |
| `COMPASSCART_DISABLE_DENSE=1` | 593.2 MiB | 353.7 MiB | 19653 ms | 0.822490 |

上表在本轮早期用带评测框架的方式测得。在当前 commit `900500b` 上单独测 agent 进程
(直接构造 `Agent`,不带评测框架):初始化 `19218.6 ms`,峰值工作集 `440.7 MiB`,
单轮延迟中位数 `100.7 ms`。和上表"仅 agent"那一列一致(409.6 → 440.7,本轮新增了
三个阶段的代码)。

峰值里大约 240 MiB 是评测框架自己那份目录副本,任何提交都要付这个成本。
稠密召回被限定为"语义救援"用途,在公开集的 536 轮里一次都没触发过,
这就是关掉它分数完全一致的原因 —— 保留它是因为私有集可能会走到救援路径。

agent 不需要任何网络访问,默认路径上报的 token 数为零。

## 怎么跑

```powershell
$env:PYTHONPATH = "src;."
.\.venv\Scripts\python.exe -m tools.run_agent `
  --catalog data/catalog.jsonl --dataset data/public_set.jsonl `
  --output results.json --evidence-output results-evidence.json
```

| 任务 | 命令 |
| --- | --- |
| 完整测试 | `python -m pytest -q` |
| Lint | `python -m ruff check src tests tools` |
| 冻结输入校验 | `python -m tools.verify_frozen_inputs` |
| 交叉验证 | `python -m tools.run_cv --folds 1 2 3 4 5 --seed 2026` |
| 资源基准 | 两步:先 `--capture-transcript --proxy-root <dir> --output t.json`,再 `--transcript t.json --trials 3 --output bench.json` |

## 接下来读哪些

- `reports/final/rerank-results.md` —— 全部重排实验,包括被否决的四个和否决的理由
- `reports/final/evolution-results.md` —— 策略记忆与策略选择器、它们的消融,
  以及学到的先验揭示了什么
- `docs/attribute_schema.md` —— 分层属性 schema,以及目录发现为什么是可选项
- `reports/final/approach-evolution.md` —— 这套设计是按什么顺序演变出来的
- `reports/final/validation-evidence.md` —— 上面每条结论配一条能复现它的命令:
  冻结输入校验、交叉验证、密封折、学习层对照,以及消融阶梯
- `reports/final/design-record.html` —— 同样的脉络做成的自包含网页,含脉络图
