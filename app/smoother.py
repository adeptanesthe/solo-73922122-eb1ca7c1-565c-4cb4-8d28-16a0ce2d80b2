"""带切换惩罚的流式语种分段状态机。

实现概要
--------
1. **平滑层（带切换惩罚）**：把语种视为隐状态、每帧后验视为观测，用 HMM
   前向滤波在对数域在线计算当前帧的过滤后验。转移矩阵给"保持当前语种"
   零对数代价、给"切换到其它语种" ``switch_penalty`` 纳特的代价。惩罚越大，
   状态越"粘"，单帧观测抖动越难把状态拉走。
2. **确认层（稳定分段）**：
   - 置信度门限作用在归一化后的原始后验上：最大后验低于
     ``posterior_threshold`` 的帧视为低置信，沿用上一确认状态、不推进切换
     计数（若尚未建立任何区间，则该帧暂不属于任何段）；
   - 首个语种需要连续 ``switch_frames`` 帧被判为同一语种才建立区间，首段
     起点回退到会话第 0 帧（其前的低置信/抖动帧并入首段，保证时间线连续）；
   - 段内切换同样需要新语种连续 ``switch_frames`` 帧才切段，抖动期帧仍归
     旧段（切点回退到候选起点，帧不丢失、区间首尾相接）。
3. **幂等与顺序**：按块起始序号缓存结果与内容指纹。完全相同的重放直接返回
   原结果且不改变会话；同起点但内容/长度不同返回 409；迟到（起点小于期望
   序号）或越界（起点大于期望序号）的块一律 409 且不改变会话。
4. **封存**：``seal`` 后区间固化，未闭合尾段按最后一帧封口；此后新块被拒绝
   （已缓存块的重放仍返回原结果）。
5. **批处理**：``forward_batch`` 把多个会话的逐帧前向递推按语种数分组后用
   NumPy 向量化执行；每个会话仍使用自己的转移矩阵和平滑状态。批处理只改变
   计算的组织方式，不改变单会话的状态机语义。
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

MAX_FRAMES_PER_CHUNK = 100_000
# 单批最多处理的会话数（每个会话一块）。API 层同样按此校验。
MAX_BATCH_SESSIONS = 16


class _ForwardCounter:
    """统计 NumPy 前向递推被调用的次数（逐会话逐帧 / 批量逐时间步）。

    仅用于自测与演示对比批量与逐路提交的计算次数，不参与业务逻辑。
    """

    def __init__(self) -> None:
        self._value = 0
        self._lock = threading.Lock()

    def inc(self) -> None:
        with self._lock:
            self._value += 1

    @property
    def value(self) -> int:
        with self._lock:
            return self._value

    def reset(self) -> None:
        with self._lock:
            self._value = 0


FORWARD_STEPS = _ForwardCounter()


class SessionError(Exception):
    """会话级业务错误，由 API 层映射为 4xx。"""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


@dataclass
class Segment:
    language: str
    start_frame: int
    end_frame: int  # 闭区间
    frame_ms: int = 0

    @property
    def start_ms(self) -> int:
        return self.start_frame * self.frame_ms

    @property
    def end_ms(self) -> int:
        return (self.end_frame + 1) * self.frame_ms

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.end_ms - self.start_ms,
        }


@dataclass
class ChunkResult:
    session_id: str
    start_seq: int
    end_seq: int
    duplicate: bool = False
    segments: List[Segment] = field(default_factory=list)  # 已关闭区间
    open_segment: Optional[Segment] = None  # 已确认、仍在延伸，返回时为快照
    current_language: Optional[str] = None
    pending_language: Optional[str] = None
    pending_frames: int = 0
    next_seq: int = 0

    def to_dict(self, sealed: bool = False) -> dict:
        return {
            "session_id": self.session_id,
            "start_seq": self.start_seq,
            "end_seq": self.end_seq,
            "accepted": True,
            "duplicate": self.duplicate,
            "sealed": sealed,
            "segments": [s.to_dict() for s in self.segments],
            "open_segment": (
                self.open_segment.to_dict() if self.open_segment is not None else None
            ),
            "current_language": self.current_language,
            "pending_language": self.pending_language,
            "pending_frames": self.pending_frames,
            "next_seq": self.next_seq,
        }


@dataclass
class _PreparedChunk:
    """已通过校验、等待执行（逐路或批量）的块。"""

    start_seq: int
    end_seq: int
    fingerprint: str
    posteriors: np.ndarray


def _fingerprint(posteriors: np.ndarray) -> str:
    """内容指纹：识别"同起点但内容不同"的冲突块。"""
    arr = np.ascontiguousarray(np.round(posteriors, 9), dtype=np.float64)
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("ascii"))
    h.update(arr.tobytes())
    return h.hexdigest()


def _logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    with np.errstate(divide="ignore"):
        out = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def _copy_segment(seg: Optional[Segment]) -> Optional[Segment]:
    if seg is None:
        return None
    return Segment(
        language=seg.language,
        start_frame=seg.start_frame,
        end_frame=seg.end_frame,
        frame_ms=seg.frame_ms,
    )


@dataclass
class BatchEntry:
    """批量入口的一项：一个会话、一块已结构校验的数据。"""

    smoother: "LanguageSmoother"
    start_seq: int
    posteriors: np.ndarray
    final: bool = False


@dataclass
class SubmitOutcome:
    """一次提交的回执快照：结果 + 提交时刻的封存状态。

    所有字段都在会话锁内取好，调用方不再回查会话，避免响应反映后续并发
    操作造成的状态漂移。
    """

    result: ChunkResult
    sealed: bool
    sealed_now: bool
    final_segments: List[Segment]


@dataclass
class BatchOutcome:
    """批量入口中单项的结果：要么是回执，要么是错误，二者必有其一。"""

    index: int
    session_id: str
    outcome: Optional[SubmitOutcome] = None
    error: Optional[SessionError] = None


class LanguageSmoother:
    """单个会话的流式平滑分段器。

    公开方法内部用 ``self.lock`` 串行化：逐路提交、批量提交与封存无论从哪个
    入口到达，同一会话上的操作都等价于按某个顺序逐一执行。
    """

    def __init__(
        self,
        session_id: str,
        languages: Sequence[str],
        frame_ms: int,
        switch_frames: int,
        posterior_threshold: float,
        switch_penalty: float = 2.0,
    ) -> None:
        self.session_id = session_id
        self.languages: List[str] = list(languages)
        self.n = len(self.languages)
        self.frame_ms = frame_ms
        self.switch_frames = switch_frames
        self.posterior_threshold = posterior_threshold
        self.switch_penalty = float(switch_penalty)

        # HMM 对数转移矩阵：保持 0 代价，切换到其它状态 -penalty，再行归一化。
        log_a = np.zeros((self.n, self.n), dtype=np.float64)
        off = ~np.eye(self.n, dtype=bool)
        log_a[off] = -self.switch_penalty
        log_a -= _logsumexp(log_a)[:, None]
        self._log_a = log_a

        # 前向滤波状态（对数域，初始均匀先验）。
        self._log_alpha = np.zeros(self.n, dtype=np.float64)

        # 确认层状态。
        self._segments: List[Segment] = []  # 已关闭区间
        self._open_segment: Optional[Segment] = None
        self._last_confirmed: Optional[int] = None
        self._pending_language: Optional[str] = None
        self._pending_run = 0
        self._frames_committed = 0

        # 幂等缓存与封存。
        self._chunk_cache: Dict[int, dict] = {}
        self._sealed = False
        self._final_segments: Optional[List[Segment]] = None

        # 同一会话上的所有入口共享此锁；批量提交只在真正执行前向递推时短暂
        # 持有它，因此批外会话仍可并发收块。
        self.lock = threading.RLock()

    # ------------------------------------------------------------------ 查询
    @property
    def sealed(self) -> bool:
        return self._sealed

    @property
    def next_seq(self) -> int:
        return self._frames_committed

    def final_segments(self) -> List[Segment]:
        if self._sealed:
            return list(self._final_segments or [])
        return list(self._segments)

    def state(self) -> dict:
        return {
            "session_id": self.session_id,
            "sealed": self._sealed,
            "languages": list(self.languages),
            "frame_ms": self.frame_ms,
            "switch_frames": self.switch_frames,
            "posterior_threshold": self.posterior_threshold,
            "switch_penalty": self.switch_penalty,
            "next_seq": self._frames_committed,
            "current_language": self._current_language(),
            "pending_language": self._pending_language,
            "pending_frames": self._pending_run,
            "segments": [s.to_dict() for s in self._segments],
            "open_segment": (
                self._open_segment.to_dict() if self._open_segment is not None else None
            ),
        }

    def _current_language(self) -> Optional[str]:
        if self._last_confirmed is None:
            return None
        return self.languages[self._last_confirmed]

    # ------------------------------------------------------------------ 输入
    def submit_chunk(self, start_seq: int, posteriors: np.ndarray) -> ChunkResult:
        return self.submit(start_seq, posteriors, final=False).result

    def submit(
        self, start_seq: int, posteriors: np.ndarray, final: bool = False
    ) -> SubmitOutcome:
        """逐路提交（可在同一临界区内附带 ``final`` 封存），返回回执快照。"""
        with self.lock:
            prepared, cached = self._prepare(start_seq, posteriors)
            if cached is not None:
                return self._finish(cached, final)
            result = self._execute(prepared)
            return self._finish(result, final)

    def _finish(self, result: ChunkResult, final: bool) -> SubmitOutcome:
        """在会话锁内完成可选封存并固化回执快照。"""
        sealed_now = False
        if final and not self._sealed:
            self._seal_locked()
            sealed_now = True
            # final 触发的封存会把尾段并入最终区间，回执反映封存后的完整列表。
            result = ChunkResult(
                session_id=result.session_id,
                start_seq=result.start_seq,
                end_seq=result.end_seq,
                duplicate=result.duplicate,
                segments=[_copy_segment(s) for s in self._final_segments or []],
                open_segment=None,
                current_language=result.current_language,
                pending_language=result.pending_language,
                pending_frames=result.pending_frames,
                next_seq=result.next_seq,
            )
        return SubmitOutcome(
            result=result,
            sealed=self._sealed,
            sealed_now=sealed_now,
            final_segments=list(self._final_segments or []),
        )

    def seal(self) -> List[Segment]:
        with self.lock:
            return self._seal_locked()

    def _seal_locked(self) -> List[Segment]:
        if self._sealed:
            return list(self._final_segments or [])
        self._sealed = True
        last = self._frames_committed - 1
        if self._open_segment is not None:
            # 尾段已通过连续帧阈值确认，按最后一帧封口。
            if last >= self._open_segment.start_frame:
                self._open_segment.end_frame = last
                self._segments.append(self._open_segment)
            self._open_segment = None
        self._final_segments = list(self._segments)
        return list(self._final_segments)

    # ---------------------------------------------------------------- 批量入口
    def _prepare(
        self, start_seq: int, posteriors: np.ndarray
    ) -> Tuple["_PreparedChunk", Optional[ChunkResult]]:
        """校验并判断块能否推进本会话；不改变任何状态。

        返回 ``(prepared, None)`` 表示可执行；返回 ``(prepared, cached)`` 表示
        命中幂等缓存，调用方应直接返回 ``cached``。错误以 ``SessionError``
        抛出，绝不触碰状态机。
        """
        arr = self._validate_chunk(start_seq, posteriors)
        num = int(arr.shape[0])
        end_seq = start_seq + num - 1
        fp = _fingerprint(arr)
        prepared = _PreparedChunk(
            start_seq=start_seq, end_seq=end_seq, fingerprint=fp, posteriors=arr
        )

        cached = self._chunk_cache.get(start_seq)
        if cached is not None:
            if cached["end_seq"] != end_seq or cached["fingerprint"] != fp:
                raise SessionError(
                    409,
                    "chunk_conflict",
                    "start_seq was already submitted with different content or length",
                )
            return prepared, self._rebuild_cached(cached)

        if self._sealed:
            raise SessionError(
                409, "session_sealed", "session is sealed; no new chunks accepted"
            )

        expected = self._frames_committed
        if start_seq < expected:
            raise SessionError(
                409,
                "late_chunk",
                f"chunk starts at {start_seq} but next expected seq is {expected}",
            )
        if start_seq > expected:
            raise SessionError(
                409,
                "out_of_order",
                f"chunk starts at {start_seq} but next expected seq is {expected}",
            )
        return prepared, None

    def _execute(self, prepared: "_PreparedChunk") -> ChunkResult:
        arr = prepared.posteriors
        for row in arr:
            self._step_frame(row)
        result = self._snapshot(prepared.start_seq, prepared.end_seq)
        self._chunk_cache[prepared.start_seq] = {
            "end_seq": prepared.end_seq,
            "fingerprint": prepared.fingerprint,
            "result": result,
        }
        return result

    def _step_frame(self, row: np.ndarray) -> None:
        """逐路提交的单帧推进：前向递推 + 确认层。"""
        frame_idx = self._frames_committed
        raw = row / float(row.sum())
        raw_confidence = float(raw.max())
        self._forward(raw)
        state = self._filtered_state()
        self._confirm(frame_idx, state, raw_confidence)
        self._frames_committed += 1

    def _snapshot(self, start_seq: int, end_seq: int) -> ChunkResult:
        return ChunkResult(
            session_id=self.session_id,
            start_seq=start_seq,
            end_seq=end_seq,
            duplicate=False,
            segments=[_copy_segment(s) for s in self._segments],
            open_segment=_copy_segment(self._open_segment),
            current_language=self._current_language(),
            pending_language=self._pending_language,
            pending_frames=self._pending_run,
            next_seq=self._frames_committed,
        )

    def _commit_vectorized(
        self, log_alpha: np.ndarray, prepared: "_PreparedChunk"
    ) -> ChunkResult:
        """批量执行完成后写回本会话的前向状态并记录幂等结果。"""
        self._log_alpha = log_alpha
        result = self._snapshot(prepared.start_seq, prepared.end_seq)
        self._chunk_cache[prepared.start_seq] = {
            "end_seq": prepared.end_seq,
            "fingerprint": prepared.fingerprint,
            "result": result,
        }
        return result

    # ---------------------------------------------------------------- 校验
    def _validate_chunk(self, start_seq: int, posteriors: np.ndarray) -> np.ndarray:
        if isinstance(start_seq, bool) or not isinstance(start_seq, (int, np.integer)):
            raise SessionError(422, "invalid_seq", "start_seq must be an integer")
        if start_seq < 0:
            raise SessionError(422, "invalid_seq", "start_seq must be >= 0")
        arr = np.asarray(posteriors, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise SessionError(422, "invalid_shape", "posteriors must be a 2-D array")
        if arr.shape[0] == 0:
            raise SessionError(422, "empty_chunk", "chunk must contain at least one frame")
        if arr.shape[0] > MAX_FRAMES_PER_CHUNK:
            raise SessionError(
                413, "chunk_too_large", f"chunk exceeds {MAX_FRAMES_PER_CHUNK} frames"
            )
        if arr.shape[1] != self.n:
            raise SessionError(
                422,
                "dimension_mismatch",
                f"expected {self.n} posteriors per frame, got {arr.shape[1]}",
            )
        if not np.all(np.isfinite(arr)):
            raise SessionError(422, "non_finite", "posteriors contain NaN or Inf")
        if np.any(arr < 0.0):
            raise SessionError(422, "negative_probability", "posteriors must be >= 0")
        sums = arr.sum(axis=1)
        if np.any(sums <= 0.0):
            raise SessionError(422, "invalid_distribution", "each frame must sum to > 0")
        return arr

    # ---------------------------------------------------------------- 内部实现
    def _rebuild_cached(self, cached: dict) -> ChunkResult:
        r: ChunkResult = cached["result"]
        return ChunkResult(
            session_id=r.session_id,
            start_seq=r.start_seq,
            end_seq=r.end_seq,
            duplicate=True,
            segments=list(r.segments),
            open_segment=_copy_segment(r.open_segment),
            current_language=r.current_language,
            pending_language=r.pending_language,
            pending_frames=r.pending_frames,
            next_seq=r.next_seq,
        )

    def _forward(self, row: np.ndarray) -> None:
        total = float(row.sum())
        obs = np.log(np.maximum(row / total, 1e-12))
        # alpha_t(j) ∝ sum_i alpha_{t-1}(i) * A(i,j) * p(obs_t | j)
        prev = self._log_alpha[:, None] + self._log_a
        log_alpha = _logsumexp(prev, axis=0) + obs
        log_alpha -= float(np.max(log_alpha))
        self._log_alpha = log_alpha
        FORWARD_STEPS.inc()

    def _filtered_state(self) -> int:
        """平滑层当前最优状态（对数域 argmax，等价于归一化后的 argmax）。"""
        return int(np.argmax(self._log_alpha))

    def _confirm(self, frame_idx: int, state: int, confidence: float) -> None:
        lang = self.languages[state]

        # 低置信：沿用上一确认状态，打断候选切换。
        if confidence < self.posterior_threshold:
            self._pending_language = None
            self._pending_run = 0
            if self._open_segment is not None:
                self._open_segment.end_frame = frame_idx
            return

        if self._last_confirmed is None:
            if self._pending_language == lang:
                self._pending_run += 1
            else:
                self._pending_language = lang
                self._pending_run = 1
            if self._pending_run >= self.switch_frames:
                self._last_confirmed = state
                # 首段回溯到会话开头：开头的低置信/抖动帧没有"上一确认状态"，
                # 归入首个已确认语种，保证区间覆盖整条时间线。
                self._open_segment = Segment(
                    language=lang,
                    start_frame=0,
                    end_frame=frame_idx,
                    frame_ms=self.frame_ms,
                )
                self._pending_language = None
                self._pending_run = 0
            return

        if state == self._last_confirmed:
            self._pending_language = None
            self._pending_run = 0
            if self._open_segment is not None:
                self._open_segment.end_frame = frame_idx
            return

        # 候选切换：连续计数。
        if self._pending_language == lang:
            self._pending_run += 1
        else:
            self._pending_language = lang
            self._pending_run = 1

        if self._pending_run < self.switch_frames:
            # 抖动期：仍归旧段。
            if self._open_segment is not None:
                self._open_segment.end_frame = frame_idx
            return

        # 达到连续帧阈值：切段，切点回退到候选起点，帧不丢、首尾相接。
        if self._open_segment is not None:
            closed_end = frame_idx - self._pending_run
            if closed_end >= self._open_segment.start_frame:
                self._open_segment.end_frame = closed_end
                self._segments.append(self._open_segment)
            self._open_segment = None
        self._last_confirmed = state
        self._open_segment = Segment(
            language=lang,
            start_frame=frame_idx - self._pending_run + 1,
            end_frame=frame_idx,
            frame_ms=self.frame_ms,
        )
        self._pending_language = None
        self._pending_run = 0


# ------------------------------------------------------------------ 批量执行
def submit_batch(entries: Sequence[BatchEntry]) -> List[BatchOutcome]:
    """批量提交多个会话各一块，返回与 ``entries`` 等长的逐项结果。

    语义与逐路调用 ``submit_chunk`` 完全一致：

    * 每项先独立校验、查幂等缓存、查序号；非法 / 冲突 / 封存 / 迟到 / 越界
      的项只产生该项自己的错误，不推进该会话，也不影响其它项；
    * 合法项按语种数分组，用 NumPy 一次性推进组内所有会话的 HMM 前向状态，
      每个会话使用自己的转移矩阵与平滑初值，确认层仍逐帧调用会话自己的规则；
    * 同一会话在一个批次里只能出现一次（由上层保证）；跨入口并发由每个会话
      自己的锁串行化，结果等价于某种串行顺序。
    """
    outcomes: List[Optional[BatchOutcome]] = [None] * len(entries)  # type: ignore[list-item]
    runnable: List[Tuple[int, "LanguageSmoother", _PreparedChunk, bool]] = []

    # 第一遍：在各自会话锁内完成校验与幂等判定。命中缓存、被拒绝的项不进入
    # 前向计算；这一步不改变任何会话状态。
    for idx, entry in enumerate(entries):
        s = entry.smoother
        try:
            with s.lock:
                prepared, cached = s._prepare(entry.start_seq, entry.posteriors)
                if cached is None:
                    runnable.append((idx, s, prepared, entry.final))
                else:
                    outcomes[idx] = BatchOutcome(
                        index=idx,
                        session_id=s.session_id,
                        outcome=s._finish(cached, entry.final),
                    )
        except SessionError as exc:
            outcomes[idx] = BatchOutcome(
                index=idx, session_id=s.session_id, error=exc
            )

    if runnable:
        _run_pending(runnable, outcomes)

    assert all(o is not None for o in outcomes)
    return outcomes  # type: ignore[return-value]


def _run_pending(
    runnable: List[Tuple[int, "LanguageSmoother", _PreparedChunk, bool]],
    outcomes: List[Optional[BatchOutcome]],
) -> None:
    """按语种数分组执行；拿不到锁的会话退化为逐项（阻塞）执行。"""
    groups: Dict[int, List[Tuple[int, "LanguageSmoother", _PreparedChunk, bool]]] = {}
    for item in runnable:
        groups.setdefault(item[1].n, []).append(item)

    for items in groups.values():
        # 按 session_id 排序加锁，避免多个批次互相等待形成死锁。用一个
        # 非阻塞探测判断整组是否都空闲；不空闲就不强行等待整组，改为逐项
        # 阻塞执行，保证批外会话不受影响、批内会话也能尽快推进。
        ordered = sorted(items, key=lambda it: it[1].session_id)
        acquired: List["LanguageSmoother"] = []
        for _, s, _, _ in ordered:
            if not s.lock.acquire(blocking=False):
                for got in acquired:
                    got.lock.release()
                acquired = []
                break
            acquired.append(s)
        if len(acquired) == len(ordered):
            try:
                _execute_group_locked(items, outcomes)
            finally:
                for s in reversed(acquired):
                    s.lock.release()
        else:
            for idx, s, prepared, final in items:
                with s.lock:
                    _execute_one_locked(idx, s, prepared, final, outcomes)


def _execute_one_locked(
    idx: int,
    s: "LanguageSmoother",
    prepared: _PreparedChunk,
    final: bool,
    outcomes: List[Optional[BatchOutcome]],
) -> None:
    """在已持有会话锁的前提下重新校验并逐路推进一项。"""
    try:
        fresh, cached = s._prepare(prepared.start_seq, prepared.posteriors)
        result = cached if cached is not None else s._execute(fresh)
        outcomes[idx] = BatchOutcome(
            index=idx,
            session_id=s.session_id,
            outcome=s._finish(result, final),
        )
    except SessionError as exc:
        outcomes[idx] = BatchOutcome(index=idx, session_id=s.session_id, error=exc)


def _execute_group_locked(
    items: List[Tuple[int, "LanguageSmoother", _PreparedChunk, bool]],
    outcomes: List[Optional[BatchOutcome]],
) -> None:
    """已持有一组会话锁：重新校验后按宽度向量化执行。"""
    ready: List[Tuple[int, "LanguageSmoother", _PreparedChunk, bool]] = []
    for idx, s, prepared, final in items:
        try:
            fresh, cached = s._prepare(prepared.start_seq, prepared.posteriors)
        except SessionError as exc:
            outcomes[idx] = BatchOutcome(
                index=idx, session_id=s.session_id, error=exc
            )
            continue
        if cached is not None:
            outcomes[idx] = BatchOutcome(
                index=idx,
                session_id=s.session_id,
                outcome=s._finish(cached, final),
            )
        else:
            ready.append((idx, s, fresh, final))

    by_width: Dict[int, List[Tuple[int, "LanguageSmoother", _PreparedChunk, bool]]] = {}
    for item in ready:
        by_width.setdefault(item[1].n, []).append(item)
    for w, group in by_width.items():
        _execute_vectorized(w, group, outcomes)


def _execute_vectorized(
    width: int,
    items: List[Tuple[int, "LanguageSmoother", _PreparedChunk, bool]],
    outcomes: List[Optional[BatchOutcome]],
) -> None:
    """一次完成组内所有会话的前向递推（按时间步对齐，最长的块决定步数）。

    每个时间步只做一次 NumPy ``logsumexp``，同时更新组内所有会话；每个会话
    使用自己的转移矩阵与进入本块前的 alpha，因此结果与逐路逐帧递推一致。
    各块的帧数可以不同：较短的会话在自己的最后一帧后冻结 alpha、不再推进。
    """
    m = len(items)
    speakers = [s for _, s, _, _ in items]
    chunks = [p.posteriors for _, _, p, _ in items]
    lengths = [int(c.shape[0]) for c in chunks]
    max_len = max(lengths)

    # 组内语种数一致，可堆成 (m, n) 与 (m, n, n) 的数组。
    log_a = np.stack([s._log_a for s in speakers])  # (m, n, n)
    alpha = np.stack([s._log_alpha for s in speakers])  # (m, n)

    raw = [c / c.sum(axis=1, keepdims=True) for c in chunks]
    log_obs = np.full((max_len, m, width), -np.inf, dtype=np.float64)
    confidence = np.zeros((max_len, m), dtype=np.float64)
    for slot, rows in enumerate(raw):
        log_obs[: lengths[slot], slot, :] = np.log(np.maximum(rows, 1e-12))
        confidence[: lengths[slot], slot] = rows.max(axis=1)

    for t in range(max_len):
        active_row = np.fromiter(
            (t < length for length in lengths), dtype=bool, count=m
        )
        active_idx = np.flatnonzero(active_row)
        # 向量化前向：alpha_t(j) ∝ sum_i alpha_{t-1}(i) * A(i,j) * obs_t(j)
        # 只计算本步仍有帧的会话，避免对 -inf 观测做减法产生 NaN 警告。
        prev = alpha[active_idx][:, :, None] + log_a[active_idx]  # (a, n, n)
        new_alpha = _logsumexp(prev, axis=1) + log_obs[t, active_idx]  # (a, n)
        new_alpha -= new_alpha.max(axis=1, keepdims=True)
        alpha[active_idx] = new_alpha
        FORWARD_STEPS.inc()

        # 确认层：各会话用自己的规则逐帧推进；已结束的会话跳过。
        for slot, s in enumerate(speakers):
            if not active_row[slot]:
                continue
            frame_idx = s._frames_committed
            state = int(np.argmax(alpha[slot]))
            s._confirm(frame_idx, state, float(confidence[t, slot]))
            s._frames_committed += 1

    # 收尾：写回 alpha、固化回执与幂等缓存。
    for slot, (idx, s, prepared, final) in enumerate(items):
        result = s._commit_vectorized(alpha[slot].copy(), prepared)
        outcomes[idx] = BatchOutcome(
            index=idx,
            session_id=s.session_id,
            outcome=s._finish(result, final),
        )
