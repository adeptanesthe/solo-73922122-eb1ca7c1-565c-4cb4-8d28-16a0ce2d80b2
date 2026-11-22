"""FastAPI 服务：流式语种识别抖动 → 稳定分段。"""

from __future__ import annotations

import threading
import uuid
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from .smoother import ChunkResult, LanguageSmoother, SessionError

app = FastAPI(
    title="Streaming Language Segmenter",
    version="1.0.0",
    description=(
        "把流式语种识别的逐帧后验整理成稳定语种区间。"
        "会话设定 2~4 个语种、帧长、切换连续帧数与后验阈值；"
        "服务用带切换惩罚的 HMM 平滑状态，只在满足连续帧阈值时切段。"
    ),
)

_sessions: Dict[str, LanguageSmoother] = {}
_lock = threading.Lock()


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


# --------------------------------------------------------------------- 错误
@app.exception_handler(SessionError)
async def _session_error_handler(_: Request, exc: SessionError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message}},
    )


def _get(session_id: str) -> LanguageSmoother:
    with _lock:
        s = _sessions.get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="session not found")
    return s


# --------------------------------------------------------------------- 路由
@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "sessions": len(_sessions)}


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
    with _lock:
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
    with _lock:
        return s.state()


@app.post("/sessions/{session_id}/chunks", response_model=ChunkOut)
def submit_chunk(session_id: str, body: ChunkSubmit) -> dict:
    s = _get(session_id)
    arr = np.asarray(body.posteriors, dtype=np.float64)
    with _lock:
        result: ChunkResult = s.submit_chunk(body.start_seq, arr)
        sealed_now = False
        if body.final and not s.sealed:
            s.seal()
            sealed_now = True
        payload = result.to_dict(sealed=s.sealed)
        if sealed_now:
            # final 触发的封存会把尾段并入最终区间，响应反映封存后的完整列表。
            payload["segments"] = [seg.to_dict() for seg in s.final_segments()]
            payload["open_segment"] = None
    return payload


@app.post("/sessions/{session_id}/seal", response_model=SealOut)
def seal_session(session_id: str) -> SealOut:
    s = _get(session_id)
    with _lock:
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
    with _lock:
        return {
            "session_id": session_id,
            "sealed": s.sealed,
            "segments": [seg.to_dict() for seg in s.final_segments()],
            "next_seq": s.next_seq,
        }
