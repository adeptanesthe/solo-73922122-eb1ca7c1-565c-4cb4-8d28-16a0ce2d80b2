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
