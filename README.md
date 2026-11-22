# 流式语种分段服务（Streaming Language Segmenter）

把流式语种识别器逐帧输出的后验概率，整理成**稳定、可回放、可封存**的语种区间。
技术栈：FastAPI + NumPy，Docker Compose 启动。

- 会话创建时设定 **2～4 个语种**、帧长 `frame_ms`、切换所需连续帧数
  `switch_frames`、后验阈值 `posterior_threshold`（以及可选的切换惩罚
  `switch_penalty`）。
- 按全局序号 `start_seq` 提交帧块；服务用**带切换惩罚的 HMM 平滑状态**输出
  已确认区间。
- 只有新语种**连续达到阈值帧数**才切段；低置信帧沿用上一状态；单帧抖动
  不会切段。
- 完全重复的块返回**原结果**且不改变会话；迟到 / 越界 / 冲突块**一律拒绝**
  且不改变会话。
- 提交 `final: true`（或调用 `/seal`）后**封存**区间，之后不再接受新块。

---

## 1. 快速开始

```bash
docker compose up --build
```

服务监听 `http://localhost:8000`，交互式文档在 `/docs`。

本地开发/自测（无需 Docker）：

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q                     # 17 个单元/接口用例
uvicorn app.main:app --port 8000
python examples/demo.py http://127.0.0.1:8000
# 可选：真实 HTTP 端到端测试
SEGMENTER_BASE_URL=http://127.0.0.1:8000 python -m pytest tests/test_live.py -q
```

Docker 下跑测试（compose 的 `test` profile，会等 `segmenter` healthy 后执行）：

```bash
docker compose --profile test run --rm tests
```

`examples/demo.py` 只依赖标准库，演示了创建会话、正常分块、重复块、迟到块、
越界块、低置信尾帧、封存与封存后提交。

---

## 2. API

### 2.1 创建会话

`POST /sessions`

```json
{
  "languages": ["zh", "en", "ja"],
  "frame_ms": 20,
  "switch_frames": 3,
  "posterior_threshold": 0.55,
  "switch_penalty": 2.0
}
```

| 字段 | 约束 | 说明 |
|---|---|---|
| `languages` | 2～4 个、非空、去重（忽略大小写） | 顺序即每帧后验向量的列顺序 |
| `frame_ms` | 1～1000 | 单帧时长，用于换算毫秒区间 |
| `switch_frames` | 1～1000 | 建段/切段所需的**连续**帧数 |
| `posterior_threshold` | (0, 1] | 最大原始后验低于该值的帧视为低置信 |
| `switch_penalty` | [0, 50] | HMM 切换到其它语种的对数代价（纳特），越大越"粘" |

响应 `201`：

```json
{
  "session_id": "c5b6b0b8…",
  "languages": ["zh", "en", "ja"],
  "frame_ms": 20,
  "switch_frames": 3,
  "posterior_threshold": 0.55,
  "switch_penalty": 2.0
}
```

### 2.2 提交帧块

`POST /sessions/{session_id}/chunks`

```json
{
  "start_seq": 8,
  "posteriors": [[0.02, 0.96, 0.02], [0.02, 0.95, 0.03]],
  "final": false
}
```

- `start_seq`：本块第一帧的**全局序号**，从 0 开始，必须等于 `next_seq`。
- `posteriors`：形状 `(帧数, 语种数)`，每行非负、和 > 0（内部会归一化）。
- `final`：置 `true` 等价于在本块处理成功后立即封存会话。

响应 `200`：

```json
{
  "session_id": "…",
  "start_seq": 8,
  "end_seq": 14,
  "accepted": true,
  "duplicate": false,
  "sealed": false,
  "segments": [
    {"language": "zh", "start_frame": 0, "end_frame": 7,
     "start_ms": 0, "end_ms": 160, "duration_ms": 160}
  ],
  "open_segment": {"language": "en", "start_frame": 8, "end_frame": 14,
                   "start_ms": 160, "end_ms": 300, "duration_ms": 140},
  "current_language": "en",
  "pending_language": null,
  "pending_frames": 0,
  "next_seq": 15
}
```

- `segments`：**已关闭**区间（一旦关闭不会再变），闭区间 `start_frame..end_frame`。
- `open_segment`：已通过阈值确认、仍在延伸的当前区间（快照）。
- `pending_language` / `pending_frames`：尚未达到阈值的候选切换进度。
- `next_seq`：下一块的期望 `start_seq`。

### 2.3 其它端点

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/sessions/{id}` | 会话完整状态（参数、当前段、候选切换、`next_seq`） |
| `POST` | `/sessions/{id}/seal` | 封存；未闭合尾段按最后一帧封口，返回全部区间 |
| `GET` | `/sessions/{id}/segments` | 当前（或封存后的）区间列表 |
| `GET` | `/healthz` | 健康检查 |

### 2.4 错误

业务错误统一返回 `{"error": {"code": "...", "message": "..."}}`：

| HTTP | code | 场景 |
|---|---|---|
| 404 | — | 会话不存在 |
| 409 | `late_chunk` | `start_seq` < 期望序号（迟到/重放旧块但内容不同） |
| 409 | `out_of_order` | `start_seq` > 期望序号（越界、有缺口） |
| 409 | `chunk_conflict` | 同一 `start_seq` 已提交过，但内容或长度不同 |
| 409 | `session_sealed` | 封存后提交新块 |
| 413 | `chunk_too_large` | 单块超过 100 000 帧 |
| 422 | `dimension_mismatch` / `negative_probability` / `invalid_distribution` / `non_finite` / `invalid_seq` / `empty_chunk` | 参数或数据校验失败 |
| 422 | — | Pydantic 层结构校验（如语种数量不在 2～4） |

**关键不变式**：任何非 2xx 的请求都**不会**改变会话状态（帧序号、区间、
候选切换计数均不变），客户端可以用同一个 `start_seq` 修正后重试。

---

## 3. 实现说明

代码结构：

```
app/smoother.py   核心状态机（纯 NumPy，可独立测试）
app/main.py       FastAPI 路由、Pydantic 校验、错误映射、会话表
tests/test_api.py 17 个自测用例
examples/demo.py  端到端调用示例
```

### 3.1 平滑层：带切换惩罚的 HMM 前向滤波

把语种当作隐状态 `z_t ∈ {0..n-1}`，每帧后验 `p_t` 当作观测似然，在对数域
在线做前向滤波：

```
α_t(j) ∝ Σ_i α_{t-1}(i) · A(i,j) · p_t(j)
```

转移矩阵 `A` 给"保持"零对数代价、给"切换到其它语种" `-switch_penalty`，
再按行归一化：

```
log A(i,i) = 0,  log A(i,j) = -penalty (i≠j),  逐行 logsumexp 归一
```

于是 `switch_penalty=0` 时退化为逐帧独立判断；增大惩罚会让状态更不易被单帧
或短时观测拉走。滤波状态 `α` 是**增量**的：每来一帧只做一次 `O(n²)` 更新，
历史不需要重算，块如何切分都不影响结果（见自测
`test_chunk_splits_do_not_change_boundaries`）。

### 3.2 确认层：稳定分段

确认层在平滑状态之上再做一次连续帧门限：

1. **建段**：连续 `switch_frames` 帧被判为同一语种才建立首个区间，起点回退
   到第 0 帧，保证时间线从 0 开始连续。
2. **切段**：当前确认语种之外的候选语种需**连续** `switch_frames` 帧（每帧
   都非低置信）才切换。抖动期帧仍归旧段；真正切换时，旧段在候选起点前一帧
   关闭，新段从候选起点开始——帧不丢、区间首尾相接、边界由"连续达标"的
   位置决定。
3. **低置信**：每帧归一化原始后验的最大值 `< posterior_threshold` 时，该帧
   沿用上一确认语种（并入当前段），并**打断**候选切换的连续计数。
4. **封存**：`seal` 把仍在延伸的尾段按最后一帧关闭（该段此前已通过阈值
   确认），之后会话只读。

已关闭的 `segments` 是追加式列表、元素不可变；重放或后续块返回的都是同一
批对象的内容，因此"已确认区间"对客户端是单调、稳定的。

### 3.3 幂等、顺序与封存

- 每个被接受的块以 `start_seq` 为键缓存：结果快照 + 块长 + 内容指纹
  （形状 + 量化到 1e-9 的字节 SHA-256）。
- 命中缓存且长度、指纹一致 → 返回原结果并置 `duplicate: true`，**不重放**
  状态机，因此不会重复推进 `next_seq`，也不会改变候选切换计数。
- 命中缓存但内容/长度不同 → `409 chunk_conflict`。
- 未命中缓存时，`start_seq` 必须精确等于 `next_seq`：小于即 `late_chunk`，
  大于即 `out_of_order`，两者都不触碰状态机。
- 封存后：未缓存的块返回 `session_sealed`；**已缓存的块仍返回原结果**
  （便于客户端安全重试），响应中 `sealed: true`。

### 3.4 并发

会话表用一把进程内 `threading.Lock` 串行化所有读写，避免同一会话上的块竞争
导致帧顺序错乱。`uvicorn` 以单 worker 运行（`docker-compose.yml` 中显式
`--workers 1`），因为会话状态保存在进程内存。

---

## 4. 验证

```bash
python -m pytest -q
# 17 passed
```

另有 `tests/test_live.py`：设置 `SEGMENTER_BASE_URL` 后对真实运行中的服务发起
HTTP 请求，覆盖健康检查、建会话、正常块、重复块、迟到块、切换 + `final`
封存、封存后拒绝；未设置环境变量时自动跳过。

覆盖的场景：

| 用例 | 验证点 |
|---|---|
| `test_basic_build_switch_seal` | 建段、切段、毫秒换算、封存后拒绝新块 |
| `test_final_flag_seals_and_segments_endpoint` | `final: true` 结束标记 + `/segments` |
| `test_switch_needs_exactly_threshold_consecutive_frames` | 2 帧不切、第 3 帧才切；切点回退到候选首帧，旧段止于其前一帧 |
| `test_single_frame_jitter_does_not_switch` | en/zh 单帧抖动全部留在 zh 段 |
| `test_low_confidence_frames_stay_in_previous_state` | 低置信帧并入当前段、不切段 |
| `test_low_confidence_breaks_consecutive_run` | 低置信帧打断连续计数，不够阈值不切 |
| `test_duplicate_chunk_returns_original_result_and_keeps_state` | 重复块返回原结果、`next_seq` 不变、会话状态逐字段不变 |
| `test_duplicate_with_different_content_conflicts` | 同起点不同内容 → 409 且状态不变 |
| `test_duplicate_after_seal_still_returns_original` | 封存后重放已缓存块仍返回原结果 |
| `test_late_chunk_rejected_without_state_change` | 迟到块 → 409 `late_chunk` 且状态不变 |
| `test_out_of_order_chunk_rejected_without_state_change` | 越界块 → 409 `out_of_order` 且状态不变，随后正确序号可提交 |
| `test_frame_by_frame_equals_single_chunk` | 逐帧提交与单块提交的最终区间完全一致 |
| `test_chunk_splits_do_not_change_boundaries` | 任意切块方式不改变段边界 |
| `test_session_parameter_validation` | 语种数量 2～4、去重、阈值与帧数下界 |
| `test_chunk_validation` | 维度、负概率、零和、空块均 422 且不推进 `next_seq` |
| `test_unknown_session` | 404 |
| `test_direct_smoother_rejects_bad_seq` | 核心层序号校验 |

端到端（真实 HTTP 服务）：

```bash
docker compose up --build -d
python examples/demo.py http://127.0.0.1:8000
SEGMENTER_BASE_URL=http://127.0.0.1:8000 python -m pytest tests/test_live.py -q
docker compose down
```

演示输出中的关键片段：

- `chunk 0 (zh, jitter, zh)` → `open_segment` 仍是 `zh 0..7`，`segments` 为空
  （单帧 en 抖动未切段）。
- `duplicate chunk 0` → `duplicate: true`，`next_seq` 仍为 8，`open_segment`
  与首次完全一致。
- `chunk 1 (en x7)` → `segments` 出现 `zh 0..7`，`open_segment` 为 `en 8..14`，
  即切点正好是 en 连续达标的起点。
- `late chunk (start 6)` / `out-of-order chunk (start 20)` → 409，随后
  `next_seq` 仍为 15。
- `chunk 2 (ja x5 + low-confidence x2)` → 切到 ja，两帧低置信并入 ja 段。
- `seal` 后 `segments` 为 `zh 0..7`、`en 8..14`、`ja 15..21`，覆盖全部 22 帧
  且不重叠；封存后新块 409 `session_sealed`。

> 说明：本次交付环境未安装 Docker，`docker compose up` 未能在本机实际执行；
> Dockerfile / compose 按标准写法提供，服务本身已用 `uvicorn` 真实启动并跑通
> 上述端到端示例与全部单测。

---

## 5. 未完成事项 / 已知限制

1. **无持久化与多实例**：会话与幂等缓存都在进程内存，服务重启即丢失；单
   worker 部署。生产化需要把状态机快照/块结果放到 Redis 或数据库，并按
   `session_id` 做一致性路由或加分布式锁。
2. **无会话过期清理**：长时间运行会累积内存。应加 TTL / LRU 淘汰。
3. **只有前向滤波，没有维特比回溯**：当前段一旦关闭不会因后续帧修正。
   对"整段最优"有要求时，可在封存时对保留的观测做一次离线 Viterbi 重解码
   并重发最终区间（API 已把最终区间集中在 `/segments`，便于替换）。
4. **低置信判定用原始后验而非平滑后验**：阈值语义直观、可解释；若希望阈值
   作用在平滑后的置信度上，需要在 `_apply` 中改为传入过滤后验最大值，并
   补充对应自测。
5. **大块内存占用**：单块上限 100 000 帧，按 `float64` 约 100k×4×8B ≈ 3.2MB
   的请求体；流式超长音频应拆块提交（本就支持）。
6. **没有鉴权、限流与指标**：接口裸奔；接入生产需要 API Key/网关、限流、
   以及 Prometheus 指标（会话数、拒绝原因分布、处理延迟）。
7. **内容指纹量化到 1e-9**：极小概率把"数值上几乎相同"的块判为冲突；如需
   严格语义可改为按块哈希存原始字节。
8. **`seal` 不可撤销**，且封存后不保留重开接口；如需在封存后继续（例如用户
   取消结束），要新增明确的 `resume` 语义。
