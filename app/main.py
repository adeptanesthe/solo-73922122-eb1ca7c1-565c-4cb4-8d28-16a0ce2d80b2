"""FastAPI 服务：流式语种识别抖动 → 稳定分段。"""

from __future__ import annotations

import threading
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from .smoother import (
    FORWARD_STEPS,
    MAX_BATCH_SESSIONS,
    BatchEntry,
    BatchOutcome,
    LanguageSmoother,
    SubmitOutcome,
    SessionError,
    submit_batch,
)

app = FastAPI(
    title="Streaming Language Segmenter",
    version="2.0.0",
    description=(
        "把流式语种识别的逐帧后验整理成稳定语种区间。"
        "会话设定 2~4 个语种、帧长、切换连续帧数与后验阈值；"
        "服务用带切换惩罚的 HMM 平滑状态，只在满足连续帧阈值时切段。"
        "支持逐路提交与批量提交（每批最多 16 个不同会话，NumPy 合并前向递推）。"
    ),
)

_sessions: Dict[str, LanguageSmoother] = {}
_registry_lock = threading.Lock()


# --------------------------------------------------------------------- 模型
class SessionCreate(BaseModel):
    languages: List[str] = Field(
        ..., min_length=2, max_length=4, description="2~4 个语种标签，顺序即后验向量顺序"
    )
    frame_ms: int = Field(20, ge=1, le=1000, description="单帧时长（毫秒）")
    switch_frames: int = Field(
        5, ge=1, le=1000, description="切换（或首段建立）所需连续帧数"
    )
    posterior_threshold: float = Field(
        0.55, gt=0.0, le=1.0, description="低置信帧阈值：归一化原始后验最大值低于此值的帧沿用上一状态"
    )
    switch_penalty: float = Field(
        2.0, ge=0.0, le=50.0, description="HMM 切换到其它语种的对数惩罚（纳特）"
    )

    @field_validator("languages")
    @classmethod
    def _check_languages(cls, v: List[str]) -> List[str]:
        cleaned = [x.strip() for x in v]
        if any(not x for x in cleaned):
            raise ValueError("language tags must be non-empty")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("language tags must be unique")
        if len(set(x.lower() for x in cleaned)) != len(cleaned):
            raise ValueError("language tags must be unique (case-insensitive)")
        return cleaned


class ChunkSubmit(BaseModel):
    start_seq: int = Field(..., ge=0, description="本块第一帧的全局序号（从 0 开始）")
    posteriors: List[List[float]] = Field(
        ..., description="形状 (帧数, 语种数) 的后验概率"
    )
    final: bool = Field(False, description="置 true 等价于随后调用 /seal")

    @model_validator(mode="after")
    def _check_posteriors(self) -> "ChunkSubmit":
        if not self.posteriors:
            raise ValueError("posteriors must contain at least one frame")
        width = len(self.posteriors[0])
        if width == 0:
            raise ValueError("posteriors rows must be non-empty")
        if any(len(row) != width for row in self.posteriors):
            raise ValueError("all posterior rows must have the same length")
        return self


class SessionOut(BaseModel):
    session_id: str
    languages: List[str]
    frame_ms: int
    switch_frames: int
    posterior_threshold: float
    switch_penalty: float


class ChunkOut(BaseModel):
    session_id: str
    start_seq: int
    end_seq: int
    accepted: bool
    duplicate: bool
    sealed: bool
    segments: List[dict]
    open_segment: Optional[dict]
    current_language: Optional[str]
    pending_language: Optional[str]
    pending_frames: int
    next_seq: int


class SealOut(BaseModel):
    session_id: str
    sealed: bool
    segments: List[dict]
    next_seq: int


class BatchItemIn(BaseModel):
    """批量提交的一项。

    块级内容（``start_seq`` / ``posteriors``）故意保持宽松：单块的形状、类型、
    数值合法性由核心层逐项校验，坏块只让该项返回错误，不会让整批 422。
    """

    session_id: str
    start_seq: Any = Field(..., description="本块第一帧的全局序号")
    posteriors: Any = Field(..., description="形状 (帧数, 该会话语种数) 的后验概率")
    final: bool = Field(False, description="置 true 等价于随后对该会话调用 /seal")


class BatchSubmit(BaseModel):
    items: List[BatchItemIn] = Field(..., min_length=1, max_length=MAX_BATCH_SESSIONS)

    @field_validator("items")
    @classmethod
    def _check_items(cls, v: List[BatchItemIn]) -> List[BatchItemIn]:
        if len(v) > MAX_BATCH_SESSIONS:
            raise ValueError(f"at most {MAX_BATCH_SESSIONS} items per batch")
        ids = [item.session_id for item in v]
        if len(set(ids)) != len(ids):
            raise ValueError(
                "each session may appear at most once per batch; "
                "submit multiple chunks of one session one after another"
            )
        return v


class BatchErrorOut(BaseModel):
    code: str
    message: str
    status_code: int


class BatchItemOut(BaseModel):
    index: int
    session_id: str
    ok: bool
    receipt: Optional[ChunkOut] = None
    error: Optional[BatchErrorOut] = None


class BatchOut(BaseModel):
    accepted_count: int
    error_count: int
    results: List[BatchItemOut]


# --------------------------------------------------------------------- 错误
@app.exception_handler(SessionError)
async def _session_error_handler(_: Request, exc: SessionError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message}},
    )


def _get(session_id: str) -> LanguageSmoother:
    with _registry_lock:
        s = _sessions.get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="session not found")
    return s


def _lookup(session_id: str) -> Optional[LanguageSmoother]:
    with _registry_lock:
        return _sessions.get(session_id)


def _receipt_payload(out: SubmitOutcome) -> dict:
    """把回执快照转成响应体；所有字段都取自提交临界区内的快照。"""
    payload = out.result.to_dict(sealed=out.sealed)
    if out.sealed_now:
        # final 触发的封存会把尾段并入最终区间，响应反映封存后的完整列表。
        payload["segments"] = [seg.to_dict() for seg in out.final_segments]
        payload["open_segment"] = None
    return payload


def _error_payload(exc: SessionError) -> dict:
    return {"code": exc.code, "message": exc.message, "status_code": exc.status_code}


# --------------------------------------------------------------------- 路由
@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "sessions": len(_sessions)}


@app.get("/metrics")
def metrics() -> dict:
    """批量/逐路对比用的计数器；非生产监控指标。"""
    return {"forward_steps": FORWARD_STEPS.value}


@app.post("/metrics/reset")
def reset_metrics() -> dict:
    """把前向递推计数器归零，便于演示中做可重复的对比。"""
    FORWARD_STEPS.reset()
    return {"forward_steps": 0}


@app.post("/sessions", response_model=SessionOut, status_code=201)
def create_session(body: SessionCreate) -> SessionOut:
    sid = uuid.uuid4().hex
    smoother = LanguageSmoother(
        session_id=sid,
        languages=body.languages,
        frame_ms=body.frame_ms,
        switch_frames=body.switch_frames,
        posterior_threshold=body.posterior_threshold,
        switch_penalty=body.switch_penalty,
    )
    with _registry_lock:
        _sessions[sid] = smoother
    return SessionOut(
        session_id=sid,
        languages=body.languages,
        frame_ms=body.frame_ms,
        switch_frames=body.switch_frames,
        posterior_threshold=body.posterior_threshold,
        switch_penalty=body.switch_penalty,
    )


@app.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict:
    s = _get(session_id)
    with s.lock:
        return s.state()


@app.post("/sessions/{session_id}/chunks", response_model=ChunkOut)
def submit_chunk(session_id: str, body: ChunkSubmit) -> dict:
    s = _get(session_id)
    arr = np.asarray(body.posteriors, dtype=np.float64)
    with s.lock:
        outcome = s.submit(body.start_seq, arr, final=body.final)
        return _receipt_payload(outcome)


@app.post("/sessions/{session_id}/seal", response_model=SealOut)
def seal_session(session_id: str) -> SealOut:
    s = _get(session_id)
    with s.lock:
        segments = s.seal()
        return SealOut(
            session_id=session_id,
            sealed=True,
            segments=[seg.to_dict() for seg in segments],
            next_seq=s.next_seq,
        )


@app.get("/sessions/{session_id}/segments")
def get_segments(session_id: str) -> dict:
    s = _get(session_id)
    with s.lock:
        return {
            "session_id": session_id,
            "sealed": s.sealed,
            "segments": [seg.to_dict() for seg in s.final_segments()],
            "next_seq": s.next_seq,
        }


@app.post("/chunks/batch", response_model=BatchOut)
def submit_chunk_batch(body: BatchSubmit) -> BatchOut:
    """批量收块：每批最多 16 个不同会话，各交一个块。

    每项独立返回回执或错误；非法 / 冲突项不推进本会话，其余照常处理。
    单项的 HTTP 语义与逐路端点一致，错误体现在该项的 ``error`` 中。
    """
    entries: List[BatchEntry] = []
    entry_indices: List[int] = []
    results: List[Optional[BatchItemOut]] = [None] * len(body.items)

    for i, item in enumerate(body.items):
        s = _lookup(item.session_id)
        if s is None:
            results[i] = BatchItemOut(
                index=i,
                session_id=item.session_id,
                ok=False,
                error=BatchErrorOut(
                    code="session_not_found",
                    message=f"session {item.session_id} not found",
                    status_code=404,
                ),
            )
            continue
        try:
            arr = np.asarray(item.posteriors, dtype=np.float64)
        except (TypeError, ValueError):
            results[i] = BatchItemOut(
                index=i,
                session_id=item.session_id,
                ok=False,
                error=BatchErrorOut(
                    code="invalid_shape",
                    message="posteriors must be a 2-D array",
                    status_code=422,
                ),
            )
            continue
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        entry_indices.append(i)
        entries.append(
            BatchEntry(
                smoother=s,
                start_seq=item.start_seq,
                posteriors=arr,
                final=item.final,
            )
        )

    outcomes: List[BatchOutcome] = submit_batch(entries) if entries else []
    for i, outcome in zip(entry_indices, outcomes):
        item = body.items[i]
        if outcome.error is not None:
            results[i] = BatchItemOut(
                index=i,
                session_id=item.session_id,
                ok=False,
                error=BatchErrorOut(**_error_payload(outcome.error)),
            )
        else:
            assert outcome.outcome is not None
            results[i] = BatchItemOut(
                index=i,
                session_id=item.session_id,
                ok=True,
                receipt=ChunkOut(**_receipt_payload(outcome.outcome)),
            )

    final_results = [r for r in results if r is not None]
    return BatchOut(
        accepted_count=sum(1 for r in final_results if r.ok),
        error_count=sum(1 for r in final_results if not r.ok),
        results=final_results,
    )
