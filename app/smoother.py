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
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

MAX_FRAMES_PER_CHUNK = 100_000


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


class LanguageSmoother:
    """单个会话的流式平滑分段器（非线程安全，由外层加锁）。"""

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
        arr = self._validate_chunk(start_seq, posteriors)
        num = int(arr.shape[0])
        end_seq = start_seq + num - 1
        fp = _fingerprint(arr)

        cached = self._chunk_cache.get(start_seq)
        if cached is not None:
            if cached["end_seq"] != end_seq or cached["fingerprint"] != fp:
                raise SessionError(
                    409,
                    "chunk_conflict",
                    "start_seq was already submitted with different content or length",
                )
            return self._rebuild_cached(cached)

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

        result = self._apply(start_seq, end_seq, arr)
        self._chunk_cache[start_seq] = {
            "end_seq": end_seq,
            "fingerprint": fp,
            "result": result,
        }
        return result

    def seal(self) -> List[Segment]:
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

    def _apply(self, start_seq: int, end_seq: int, posteriors: np.ndarray) -> ChunkResult:
        for row in posteriors:
            frame_idx = self._frames_committed
            raw = row / float(row.sum())
            raw_confidence = float(raw.max())
            self._forward(raw)
            state = self._filtered_state()
            self._confirm(frame_idx, state, raw_confidence)
            self._frames_committed += 1

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

    def _forward(self, row: np.ndarray) -> None:
        total = float(row.sum())
        obs = np.log(np.maximum(row / total, 1e-12))
        # alpha_t(j) ∝ sum_i alpha_{t-1}(i) * A(i,j) * p(obs_t | j)
        prev = self._log_alpha[:, None] + self._log_a
        log_alpha = _logsumexp(prev, axis=0) + obs
        log_alpha -= float(np.max(log_alpha))
        self._log_alpha = log_alpha

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
