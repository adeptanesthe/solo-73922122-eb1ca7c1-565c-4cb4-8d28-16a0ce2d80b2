"""可选的真实 HTTP 端到端测试。

默认跳过；设置 SEGMENTER_BASE_URL 后运行（例如 docker compose 的 tests 服务）：
    SEGMENTER_BASE_URL=http://127.0.0.1:8000 python -m pytest tests/test_live.py -q
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import pytest

BASE = os.environ.get("SEGMENTER_BASE_URL")

pytestmark = pytest.mark.skipif(
    not BASE, reason="set SEGMENTER_BASE_URL to run live HTTP tests"
)


def _post(path: str, payload: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def _get(path: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(BASE + path, timeout=15) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def _frame(i: int, n: int = 2, hi: float = 0.95) -> list[float]:
    row = [round((1.0 - hi) / (n - 1), 6)] * n
    row[i] = hi
    return row


def test_live_end_to_end():
    status, _ = _get("/healthz")
    assert status == 200

    status, session = _post(
        "/sessions",
        {
            "languages": ["zh", "en"],
            "frame_ms": 20,
            "switch_frames": 3,
            "posterior_threshold": 0.55,
        },
    )
    assert status == 201, session
    sid = session["session_id"]

    chunk = {"start_seq": 0, "posteriors": [ _frame(0) for _ in range(5)]}
    status, first = _post(f"/sessions/{sid}/chunks", chunk)
    assert status == 200, first
    assert first["next_seq"] == 5

    # 重复块返回原结果且不推进。
    status, dup = _post(f"/sessions/{sid}/chunks", chunk)
    assert status == 200 and dup["duplicate"] is True
    assert dup["next_seq"] == 5

    # 迟到块被拒且不改变会话。
    status, late = _post(
        f"/sessions/{sid}/chunks",
        {"start_seq": 2, "posteriors": [_frame(1)]},
    )
    assert status == 409 and late["error"]["code"] == "late_chunk"
    status, state = _get(f"/sessions/{sid}")
    assert state["next_seq"] == 5 and state["current_language"] == "zh"

    # 切到 en：3 帧达阈值。
    status, switched = _post(
        f"/sessions/{sid}/chunks",
        {"start_seq": 5, "posteriors": [_frame(1) for _ in range(3)], "final": True},
    )
    assert status == 200, switched
    assert switched["sealed"] is True
    assert switched["current_language"] == "en"
    assert [(s["language"], s["start_frame"], s["end_frame"]) for s in switched["segments"]] == [
        ("zh", 0, 4),
        ("en", 5, 7),
    ]

    # 封存后新块 409。
    status, after = _post(
        f"/sessions/{sid}/chunks",
        {"start_seq": 8, "posteriors": [_frame(0)]},
    )
    assert status == 409 and after["error"]["code"] == "session_sealed"

    # 封存后的最终区间完整覆盖全部帧。
    status, final = _get(f"/sessions/{sid}/segments")
    assert status == 200
    assert [(s["language"], s["start_frame"], s["end_frame"]) for s in final["segments"]] == [
        ("zh", 0, 4),
        ("en", 5, 7),
    ]


def test_live_batch_endpoint():
    """真实 HTTP 下的批量入口：同批正常 / 冲突 / 未知会话，逐项回执或错误。"""
    session_ids = []
    for _ in range(3):
        status, session = _post(
            "/sessions", {"languages": ["zh", "en"], "switch_frames": 3}
        )
        assert status == 201
        session_ids.append(session["session_id"])

    ok_chunk = [_frame(0) for _ in range(4)] + [_frame(1) for _ in range(4)]
    # session 0 先提交一块，供批量里制造内容冲突。
    status, _ = _post(
        f"/sessions/{session_ids[0]}/chunks",
        {"start_seq": 0, "posteriors": [_frame(0) for _ in range(4)]},
    )
    assert status == 200

    status, batch = _post(
        "/chunks/batch",
        {
            "items": [
                {"session_id": session_ids[0], "start_seq": 0, "posteriors": ok_chunk},
                {"session_id": session_ids[1], "start_seq": 0, "posteriors": ok_chunk},
                {"session_id": "missing", "start_seq": 0, "posteriors": ok_chunk},
                {"session_id": session_ids[2], "start_seq": 0, "posteriors": ok_chunk},
            ]
        },
    )
    assert status == 200, batch
    assert batch["accepted_count"] == 2
    assert batch["error_count"] == 2
    results = {r["index"]: r for r in batch["results"]}
    assert results[0]["error"]["code"] == "chunk_conflict"
    assert results[2]["error"]["code"] == "session_not_found"
    assert results[1]["receipt"]["next_seq"] == 8
    assert results[3]["receipt"]["next_seq"] == 8

    # 冲突 / 未知项都没有推进对应会话。
    status, st0 = _get(f"/sessions/{session_ids[0]}")
    assert status == 200 and st0["next_seq"] == 4

    # 批量 + final 封存。
    status, sealed = _post(
        "/chunks/batch",
        {
            "items": [
                {
                    "session_id": session_ids[1],
                    "start_seq": 8,
                    "posteriors": [_frame(0) for _ in range(3)],
                    "final": True,
                }
            ]
        },
    )
    assert status == 200, sealed
    receipt = sealed["results"][0]["receipt"]
    assert receipt["sealed"] is True
    assert receipt["open_segment"] is None
    assert [(s["language"], s["start_frame"], s["end_frame"]) for s in receipt["segments"]] == [
        ("zh", 0, 3),
        ("en", 4, 7),
        ("zh", 8, 10),
    ]
