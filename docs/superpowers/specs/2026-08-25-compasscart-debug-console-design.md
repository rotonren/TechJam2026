# CompassCart 持久化调试控制台设计

## 1. 文档状态

- 日期：2026-08-25
- 状态：用户已选择 A（三栏实时工作台），并授权由实现方完成其余技术取舍
- 适用代码线：`codex/compasscart`
- 核心约束：复用当前 `Agent.reset()` / `Agent.respond()`，不修改解析、状态、检索、排序或提问算法
- 交付形式：官方评分包与团队调试控制台分离

## 2. 目标

为团队提供一个能长期保留调试记录的多轮购物 Agent 工作台。成员可以手动输入最多 10 轮消息，同时观察 Agent 的真实回复、Top 10 商品、会话状态和 trace，并将不准确的商品标记为失败案例。

该控制台必须满足三个层次的可用性：

1. 在任意安装了 Docker 的新电脑上可重复启动，而不是绑定当前开发机。
2. 容器和宿主机重启后，会话、备注和失败标记仍然存在。
3. 部署到持续运行的托管实例后，异地成员使用固定 HTTPS 地址访问，不依赖 Quick Tunnel 或当前 Codex 任务存活。

真正的公网持续运行需要托管账号、实例和持久磁盘。仓库负责提供完整的可部署软件和配置；开通托管账号、添加付款方式以及最终创建公网服务属于团队必须完成的外部操作，不能由源代码替代。

## 3. 非目标

- 不改变或复制 CompassCart 的核心算法。
- 不改变官方 evaluator、Agent contract 或评分输入输出。
- 不让调试服务成为官方离线评分的运行依赖。
- 不展示系统没有记录的最终 ranking score 或召回源贡献。
- 不建设公开注册、多租户权限、团队邀请或互联网规模并发。
- 不把 Cloudflare Quick Tunnel 当作生产部署。
- 不把组织方 catalog 提交到公开仓库或公开容器镜像。

## 4. 已评估方案

### 4.1 仅浏览器本地存储

优点是实现简单且无服务端数据库。缺点是换浏览器或换电脑后记录不会自动存在，无法满足团队共享和持续运行，因此只保留 JSON 导入导出作为备份能力。

### 4.2 单机调试服务 + Quick Tunnel

优点是能快速演示。缺点是本机、预览进程和 tunnel 必须一直运行，URL 会变化且没有持续性保证，因此只用于开发预览。

### 4.3 Docker 服务 + SQLite 持久卷 + 固定托管实例

这是选定方案。Docker 保证换电脑和换平台时的可移植性；SQLite 满足单实例团队工具的数据量与事务需求；持久卷保证重启和重新部署后的数据恢复；托管平台提供固定 HTTPS URL 和自动重启。

首选托管目标是 Render 的付费 Web Service + Persistent Disk，同时保留标准 Docker Compose 作为平台无关退路。Render 官方文档说明免费 Web Service 在 15 分钟无入站流量后会休眠，免费实例不支持 Persistent Disk；因此免费套餐不符合“持续在线且持久保存”的验收定义。

参考：

- <https://render.com/docs/free#spinning-down-on-idle>
- <https://render.com/docs/disks>

## 5. 双交付边界

### 5.1 官方评分包

`dist/compasscart-submission.zip` 继续由现有 allowlist 打包器生成，只包含 Agent 评分所需代码、依赖、资产和报告。它必须保持离线可运行，不导入调试服务，也不要求数据库、HTTP 端口或认证配置。

### 5.2 团队调试控制台

调试控制台存在于同一源码仓库，但作为独立开发与演示工具交付：

- 本地入口：Python 模块和 Docker Compose。
- 部署入口：Dockerfile、健康检查和 Render Blueprint 配置。
- 数据入口：组织方 catalog 通过本地只读挂载或私有持久磁盘提供。
- 调试数据：SQLite 数据库写入独立持久目录。
- 文档入口：README 中单独列出本地、Docker 和 Render 部署步骤。

Devpost、架构报告和演示视频可以呈现调试控制台，但技术评分仍只依赖官方 Agent。

## 6. 架构

```text
Browser
  |
  | HTTPS / JSON API
  v
Debug WSGI Server (one process, small HTTP thread pool)
  |                 |
  |                 +--> SQLite Debug Repository
  |                      per-operation connections / WAL
  |
  +--> bounded command queue
          |
          v
      Dedicated Agent Worker (exactly one thread)
          |
          +--> Read-only Debug Adapter
          |      state / exact last trace / product enrichment
          |
          +--> Existing root agent.Agent
                  reset() / respond()
                  |
                  +--> unchanged parser, ledger, router, retriever,
                       ranker, question policy, response builder

Read-only catalog + dense assets       Persistent debug volume
```

服务使用一个长期存活的 `Agent` 实例。一个专用 Agent worker 线程负责创建该实例，并执行所有 `reset()`、`respond()`、state、trace 和 catalog 读取。命令通过有界队列进入 worker，因此这些操作始终在创建 SQLite catalog 连接的同一线程串行执行。直接把共享 Agent 放进多线程 handler 会触发检索异常并可能被 fallback 隐藏，从而污染调试结论。

HTTP 线程可以在 Agent 初始化或执行请求时继续提供静态资源、liveness、readiness 和数据库读取，但任何进入 Agent 的操作都必须经过 worker 的串行执行边界。Debug Repository 每个操作创建短生命周期 SQLite 连接，不跨 HTTP 线程共享连接。首版只部署单进程、单 Agent worker 实例，这也符合 SQLite 和单挂载持久磁盘的约束。

## 7. 组件边界

### 7.1 Debug HTTP Server

职责：

- 提供静态前端和 JSON API。
- 读取 `HOST`、`PORT`、catalog、资产目录、数据库路径和访问口令配置。
- 限制请求体大小并返回统一 UTF-8 JSON 错误。
- 在 Agent 冷启动期间返回明确的 initializing 状态。
- 将 Agent 调用委派给串行执行器。
- 提供不泄露 catalog 内容的 liveness 与 readiness 检查。

它不得导入 parser、router、retriever 或 ranker 后单独调用内部方法。

HTTP 应用使用纯 WSGI 接口，避免引入完整 Web framework。本地入口使用标准库 WSGI server 与 `ThreadingMixIn`；Linux Docker 入口使用独立的 `requirements-debug.txt` 中的 Gunicorn，并固定 `--workers 1 --threads 4`。HTTP 线程只负责路由、认证和短数据库操作；唯一 Agent worker 负责所有 Agent 相关操作。官方 `requirements.txt` 不增加 Web 依赖。

### 7.2 Debug Adapter

新建会话时只执行一次：

```python
agent.reset(session_id, profile)
```

正常每轮只执行：

```python
response = agent.respond(session_id, message, turn, top_k=10)
```

服务重启后的 rehydration 会先执行一次 `reset()`，再按顺序重放 completed messages。正常多轮对话绝不能在每轮前 reset，否则会破坏累计约束和 Intent Override。

随后从同一实例只读组合：

- `response`：原样保存并返回。
- `agent.catalog.product(parent_asin)`：补充标题、价格、评分、店铺、分类、features 和 details。
- `agent.sessions.get(session_id)`：生成 SessionState 与完整 Constraint Ledger 快照。
- `agent.traces.records`：`respond()` 返回后立即读取最后一条记录，并校验其 session_id 和 turn 后保存。

商品展示顺序以 `response.recommendations` 为唯一真相源。`state.previous_recommendations` 不能替代响应，因为 ResponseBuilder 可能追加 fallback 商品。

当前 Agent 没有持久化最终排名分数或各召回源贡献。页面必须显示“当前版本未记录该数据”，不得根据位置、rating 或 trace 推测分数。

TraceSink 是全局有界队列，重放也会追加 trace。因此适配器不得事后按 `(session_id, turn)` 搜索历史 records；必须在同一个 Agent worker command 中捕获刚完成调用的最后一条 trace。rehydration 产生的 trace 只用于当次一致性判断，不写入历史 turn snapshot。

### 7.3 Debug Repository

使用 Python 标准库 `sqlite3`。数据库开启 foreign keys、WAL 和事务。数据表按下列职责拆分：

- `metadata`：数据库 schema version 和迁移记录。
- `sessions`：会话 ID、名称、profile JSON、Agent 版本、catalog checksum、配置 fingerprint、资产 manifest checksum、创建时间、更新时间和归档状态。
- `turns`：会话 ID、turn、请求 ID、状态、用户消息、Agent 原始响应、商品快照、state 快照、trace 快照、错误摘要和创建时间。
- `product_feedback`：turn、parent_asin、不准确原因、自由备注和更新时间。

`turns` 对 `(session_id, turn)` 和 `(session_id, request_id)` 分别建立唯一约束。`product_feedback` 对 `(session_id, turn, parent_asin)` 建立唯一约束。

持久层保存每轮不可变快照，使历史页面不会因 catalog 或代码更新而悄悄改变。用户修改只作用于会话名称、归档状态和人工反馈。

`turns.status` 只能为 `pending`、`completed` 或 `failed`。发送消息时，服务先确认该 session 没有 pending/failed turn，再用唯一请求 ID 在短事务中插入 pending turn，然后调用 Agent，最后在第二个事务中写入完整快照并改为 completed。请求 ID 在同一会话内唯一；客户端重试相同 ID 时返回已完成结果、当前 pending 状态或相同 failed turn 的可重试错误，不产生额外 turn。合法状态转换只有 `pending -> completed`、`pending -> failed` 和同一请求的 `failed -> pending`。

Agent 内存状态无法与 SQLite 形成同一原子事务。如果 Agent 调用成功后快照写入失败，服务必须把该 session 的内存状态标记为 dirty，在接受下一条消息前从最后一个 completed turn 重新 hydrate。服务不得在 Agent 已前进而数据库未记录的状态上继续对话。进程崩溃留下的 pending turn 在重启后也按 completed 历史重放，再重试该 pending 消息；重试结果成功后才标记 completed。

Agent 抛出异常时，即使没有响应也可能已经修改内部 session。该 turn 必须标记 failed，session 必须标记 dirty；重试前从 completed turns hydrate，再用相同 request ID 和原消息执行。failed turn 保留并占用预定的 turn number，但只有 completed turn 会进入历史重放与 10 轮计数。failed turn 不允许被下一条新消息越过。UI 只能重试相同消息，或结束该会话并克隆 completed 历史。这样 Agent 的内部 turn 和持久层 turn 不会因一次失败产生歧义。

### 7.4 Session Rehydrator

Agent 的会话状态只存在内存，因此服务重启后不能只从 SQLite 直接恢复内部对象。继续旧会话前，Rehydrator 必须：

1. 用存储的 profile 调用 `reset()`。
2. 按 turn 顺序重新调用历史用户消息。
3. 先验证 Agent 版本、catalog checksum、配置 fingerprint 和资产 manifest checksum，再验证重放 response 与 canonical SessionState 是否与快照一致。
4. 一致时允许继续下一轮。
5. 不一致时保留历史只读展示，并要求从该历史克隆一个“当前 Agent 版本的新会话”，不能静默混合两个版本的状态。

每个会话记录当前 Git commit 或发布版本，以及所有会影响确定性的输入指纹。容器构建使用 `COMPASSCART_VERSION` build argument 写入非敏感版本标签；本地 Git checkout 优先读取当前 commit。这样升级算法、catalog、配置或 dense assets 后，调试人员能区分“当时的结果”和“当前版本重放结果”。该机制只在调试层调用现有 Agent，不修改 Agent 的状态模型。

canonical SessionState 包含 turn、route、intent_version、完整 constraints（含 status/source/confidence/version）、asked_attributes、pending_attribute、query_history、no_preference_attributes、previous_recommendations 和 candidate_count；set 和无序 map 在比较前排序。它不包含 trace elapsed_ms 等非确定时间字段。仅比较推荐 ID 和 ask_attribute 不足以证明下一轮内部状态等价。

## 8. API 设计

所有 `/api/*` 路由除健康检查外都要求访问口令。

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/api/health/live` | 仅报告 WSGI 进程存活，始终返回最小 200 响应 |
| GET | `/api/health/ready` | 报告数据库与 Agent 状态；ready 为 200，其余状态为 503 |
| GET | `/api/sessions` | 列出未归档会话 |
| POST | `/api/sessions` | 使用 profile 创建并 reset 新会话 |
| GET | `/api/sessions/{id}` | 读取会话、所有轮次和人工反馈 |
| PATCH | `/api/sessions/{id}` | 修改名称或归档状态 |
| POST | `/api/sessions/{id}/messages` | 自动计算下一 turn 并调用当前 Agent |
| POST | `/api/sessions/{id}/clone` | 用当前版本重放 completed messages，创建独立新会话 |
| PUT | `/api/sessions/{id}/turns/{turn}/feedback/{asin}` | 保存或更新商品失败标记 |
| GET | `/api/sessions/{id}/export` | 导出可移植 JSON |
| POST | `/api/import` | 导入先前导出的会话快照 |

turn 完全由服务端根据已提交轮次计算，客户端不能跳号或覆盖历史。到 turn 10 后拒绝新消息，但仍允许查看、标注和导出。

导入只恢复历史快照和反馈，不盲目恢复 Agent 内存状态。若要继续对话，必须经过 rehydration 验证。

clone 接收可选 `through_turn`，创建新的 session ID，复制 profile 和来源关系，再从第一轮到指定 completed turn 逐条调用当前 Agent 并保存新的真实快照。它不复制旧 response、state 或商品反馈，也不会假装新版本产生了旧结果。默认克隆全部 completed turns；如果重放失败，新会话保留到最后一个 completed turn 并显示错误。

## 9. 页面设计

采用已批准的 A：三栏实时工作台。

### 9.1 顶部栏

- 产品名 `CompassCart Debug`。
- Agent 状态、当前会话和 turn 计数。
- 会话选择、新建会话、导入、导出和归档命令。
- 不把设置说明、快捷键或功能介绍作为长期可见页面文案。

### 9.2 左栏：多轮对话

- 新会话时编辑 `preference_tags`；发送第一轮后 profile 锁定。
- 展示用户消息、Agent message 和 ask_attribute。
- 文本框支持 Enter 发送、Shift+Enter 换行。
- 发送期间禁止重复提交。
- turn 10 后输入区变成新会话操作。

### 9.3 中栏：本轮 Top 10

- 严格按响应顺序显示 rank、ASIN、标题、价格、rating、rating count、store、category、features 和 details。
- 缺失价格或元数据时显示明确缺失态，不使用虚构占位值。
- 每项提供图标按钮标记“不准确”，展开后选择原因并填写备注。
- 原因枚举：违反明确约束、错误品类、预算不符、属性不符、重复或过于相似、其他。
- 页面不显示商品图片，因为当前 catalog 没有图片字段。

### 9.4 右栏：Agent 诊断

- 本轮 route、intent_version、candidate_count、elapsed_ms、ask_attribute 和 fallbacks。
- active、superseded 和 rejected 约束，包含 source、confidence、hard/soft、created_turn 和 version。
- asked_attributes、pending_attribute、no_preference_attributes 和 query_history。
- 可折叠原始 response、state 和 trace JSON。
- “分数与召回源贡献”固定显示未记录状态，而非空白或估计值。

### 9.5 响应式行为

- 桌面宽屏固定三栏，保持对话、商品和诊断同屏。
- 中等宽度将右栏变为可切换 inspector，不压缩商品标题。
- 手机端使用 `对话 / 推荐 / 诊断` 三个 tabs，输入区固定在对话 tab 底部。
- 固定工具栏、计数器和商品 rank 尺寸，动态内容不得导致布局跳动或文本重叠。

## 10. 认证与安全

- 本地默认绑定 `127.0.0.1`；容器与 Render 通过显式 `HOST=0.0.0.0` 监听。
- `COMPASSCART_DEBUG_TOKEN` 必须由环境变量提供，不能写入仓库、SQLite、日志、导出文件或前端资源。
- 浏览器登录页把口令保存在 `sessionStorage`，API 使用 `Authorization: Bearer` 传递。
- 服务端使用常量时间比较验证 token。
- 启动时拒绝缺失、常见示例值或短于 43 个字符的 token；文档使用 `python -c "import secrets; print(secrets.token_urlsafe(32))"` 生成具有 32 个随机字节的值。服务只能校验长度和示例黑名单，随机强度由生成流程保证。
- API JSON 请求体默认限制为 1 MiB；会话导入采用单独的受控上限。
- 静态文件只能来自固定 debug UI 目录，拒绝路径穿越。
- catalog 和 dense assets 只读挂载；SQLite 位于单独可写目录。
- 公网服务必须由平台终止 TLS。调试服务本身不实现证书管理。
- 未认证用户只能得到登录页面和不含内部路径、版本或 catalog 信息的健康状态。
- 所有 catalog、消息、备注和导入文本只通过 DOM `textContent` 渲染，禁止拼接到 `innerHTML`。
- 响应设置 `Content-Security-Policy: default-src 'self'; script-src 'self'; style-src 'self'; img-src 'none'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'`、`X-Content-Type-Options: nosniff`、`Referrer-Policy: no-referrer` 和 `Cache-Control: no-store`。
- 服务不返回 `Access-Control-Allow-Origin`，只接受 same-origin API 请求；状态修改请求还必须使用 JSON content type，降低跨站请求风险。

## 11. 持续运行与部署

### 11.1 本地 Docker

Docker Compose 统一使用仓库相对目录 `./var/debug` 作为默认持久目录，并将其映射到容器 `/var/data`。`var/` 加入 gitignore。这样数据库备份、迁移和换电脑都能通过一个明确目录完成，不依赖难以定位的匿名或 named volume。

Docker Compose 还提供：

- `restart: unless-stopped`，宿主机重启后自动恢复。
- 由 `.env` 指定的组织方 catalog 和 dense assets 只读 bind mount。
- SQLite、迁移记录和备份写入 `./var/debug`。
- 健康检查和启动日志。
- `.env.example` 仅列变量名和生成口令的命令，不包含真实 secret。

仓库提供备份命令，将 SQLite 一致性备份和 metadata 写入 `./var/debug/backups/`。迁移到新电脑时，复制仓库、`var/debug`、授权的 catalog 与 assets，重新生成本机 `.env` 后运行相同 Compose 命令。`restart: unless-stopped` 只有在 Docker daemon 随系统启动时才会自动恢复；本地部署仍只有当该电脑持续开机且网络可达时才是持续服务。

### 11.2 Render 固定部署

仓库提供 `render.yaml`，配置单个 Docker Web Service、`/api/health/live` 健康检查、`PORT`、secret 环境变量和 `/var/data` Persistent Disk。部署保持单实例，不启用横向扩容。`/api/health/ready` 由页面和运维人员判断 Agent 是否可用；setup_required 不会让平台把存活进程误判为崩溃并循环重启。

组织方 catalog 不进入公开 Git 或公开镜像。仓库提供本地 `tools/package_debug_runtime.py`，从当前已授权的 catalog、dense assets 和 SHA256 manifest 创建不进 Git 的 `dist/compasscart-debug-runtime.zip`。首次 Render 部署先进入 setup_required；团队使用 Render 文档支持的加密 `scp -s` 将该 zip 上传到 `/var/data/incoming/`，再在 Render Shell 运行 `python -m tools.install_debug_runtime /var/data/incoming/compasscart-debug-runtime.zip`。安装器限制解压路径和总大小，逐项验证 checksum，并原子安装到 `/var/data/runtime/catalog.jsonl` 与 `/var/data/runtime/assets/`。安装成功后重启服务，readiness 才转为 ready。

重新部署代码不会覆盖 `/var/data/runtime` 或 SQLite。更新 catalog/assets 时上传新版本包，安装到版本化临时目录，校验完成后原子切换 `current` 指针；失败时继续使用旧版本。缺少或损坏运行资产时服务保持 setup_required，不使用 fixture 或热门商品伪装成真实 Agent。

Render 付费实例提供持续运行能力；Persistent Disk 保存 SQLite 和私有运行资产，并提供平台快照。固定 URL、实时价格、账单、区域和账号成员由团队在 Render 控制台最终确认，设计不承诺一个可能变化的具体月费数字。

### 11.3 平台迁移

服务只依赖标准 Docker、环境变量和 POSIX 风格挂载路径。若 Render 不可用，可迁移到任何支持以下能力的平台：

- 长期运行的单实例容器。
- 固定 HTTPS 域名。
- 至少一个持久卷。
- secret 环境变量。
- 健康检查和自动重启。

迁移不改变前端、数据库 schema 或 Agent 调用边界。

## 12. 错误处理

| 故障 | 行为 |
|---|---|
| catalog 缺失或 checksum 失败 | 服务进入 setup_required，不接受会话消息 |
| Agent 冷启动 | 页面展示初始化状态，禁止发送 |
| Agent 调用异常 | 本轮返回结构化错误且不写入伪造成功快照 |
| Agent 内部 fallback | 正常保存响应，并在右栏突出显示 fallback 名称 |
| pending turn 写入失败 | 不调用 Agent，返回持久层错误 |
| Agent 成功后快照写入失败 | session 标记 dirty，返回未保存错误，下次操作前从 completed turns 重放 |
| 重启发现 pending turn | 从 completed turns 重放，然后用相同请求 ID 重试该消息 |
| turn 超过 10 | 返回 409，并引导创建或克隆新会话 |
| 重复提交 | 使用请求 ID 幂等返回原结果，不产生额外 turn |
| 重启后重放一致 | 恢复内存状态并允许继续 |
| 重启后重放不一致 | 历史只读，允许克隆为当前版本新会话 |
| 缺失商品元数据 | 保留 ASIN 和 rank，字段显示未提供 |
| 未授权请求 | 返回 401，不泄露服务内部信息 |
| liveness | 进程可服务 HTTP 时返回 200，不执行 catalog 或 Agent 查询 |
| readiness | ready 返回 200；initializing、setup_required 或 fatal 返回 503 和稳定状态码 |

## 13. 测试策略

### 13.1 单元测试

- Session、turn 和 feedback 的 SQLite CRUD 与事务回滚。
- schema version、向前迁移、pending/completed 状态机和 dirty session 恢复。
- JSON snapshot 序列化，包括 dataclass、set 和约束状态。
- 商品 enrich 以 response recommendations 为顺序真相源。
- token 常量时间认证、body size 和路径安全。
- CSP、安全响应头、same-origin 策略和所有不可信文本的 text-only DOM 渲染。
- turn 自动递增、上限和请求幂等。
- export/import schema 与恶意或损坏输入。

### 13.2 集成测试

- 使用现有 fixture catalog 启动真实 `Agent` 和 debug server。
- 创建会话、完成多轮对话、保存失败标记、导出并重新导入。
- 重启 server 后从 SQLite 读取历史并成功 rehydrate。
- Agent 调用异常或响应后数据库失败时标记 dirty，并在重试前恢复到最后 completed turn。
- clone endpoint 用当前版本生成新快照，不复制旧结果或反馈。
- 注入 Agent fallback 和异常，验证页面数据不混淆。
- 运行现有 unit、contract、integration 和 performance tests，证明核心评分行为未回归。
- 构建官方评分包并断言 debug server、SQLite、口令和部署配置没有进入 allowlist zip。

### 13.3 浏览器验收

- 在桌面和手机 viewport 验证登录、新会话、发送、切轮、标记、备注、导入和导出。
- 验证 loading、empty、setup_required、unauthorized、Agent error、turn 10 和 missing metadata 状态。
- 对批准的 A 方案设计与最终浏览器截图执行视觉对照。
- 检查文本不溢出、不重叠，商品顺序和诊断字段与 API 完全一致。

### 13.4 部署验收

- Docker Compose 停止并重新启动后，SQLite 会话和反馈仍存在。
- 容器重建后，从持久卷恢复并通过 rehydration 继续会话。
- 健康检查能区分 healthy、initializing 和 setup_required。
- 容器以只读 root filesystem 或最小可写范围运行，只有持久目录可写。
- Render 配置通过静态检查；真正创建服务需要团队账号和付费确认，不在自动测试中制造外部费用。
- runtime bundle 打包、zip-slip/zip-bomb 防护、checksum 验证、原子安装和版本回滚通过 fixture 测试。

## 14. 验收标准

1. 当前官方 Agent 的源文件和核心算法行为不因调试控制台而改变。
2. 成员能完成一段最多 10 轮的手工对话，每轮同时看到真实 Top 10 和对应诊断快照。
3. 成员能对商品标记不准确原因和备注，并在刷新、容器重启后继续查看。
4. 会话可以导出、导入，并能在 Agent 版本一致时重放恢复。
5. 页面明确区分真实可观察数据与当前未记录的 score/source contribution。
6. Docker Compose 在新电脑上使用相同命令启动。
7. 部署配置支持固定 URL、自动重启、secret 和持久磁盘；Quick Tunnel 只作为预览工具。
8. 官方评分包仍能在断网环境独立导入和响应，且不包含调试数据库、访问口令或调试服务。
9. README 和最终演示明确说明公网持续运行需要托管实例，绝不把免费休眠服务描述成 always-on。

## 15. 已知限制

- SQLite 和当前 Agent 决定了服务是单实例、串行 Agent 调用；这适合五人团队调试，不适合公共高并发产品。
- Render Persistent Disk 是付费能力。没有托管账号和付费实例时，仓库只能保证可部署与本地持久，不能凭空提供长期公网 URL。
- catalog 的授权与分发规则优先于部署便利；不得为一键部署把组织方数据公开上传。
- 升级 Agent 后历史响应可能无法完全重放，因此必须保留版本和不可变快照，不能承诺跨版本无差异继续。
- 商品 catalog 没有图片字段，首版调试页是文本商品信息工作台。
