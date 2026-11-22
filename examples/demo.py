"""端到端调用示例：创建会话 → 分块提交 → 重复块 → 迟到块 → 封存。

先启动服务（二选一）：
    docker compose up --build
    # 或本地：
    uvicorn app.main:app --port 8000

再运行：
    python examples/demo.py [base_url]

只依赖标准库，便于在容器外直接执行。
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
LANGUAGES = ["zh", "en", "ja"]


def post(path: str, payload: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(payload).encode() if payload is not None else b""
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def get(path: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(BASE + path, timeout=10) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def frame(lang_idx: int, hi: float = 0.95) -> list[float]:
    n = len(LANGUAGES)
    row = [round((1.0 - hi) / (n - 1), 6)] * n
    row[lang_idx] = hi
    return row


def show(label: str, status: int, body: dict) -> None:
    summary = {
        k: body.get(k)
        for k in (
            "session_id",
            "languages",
            "frame_ms",
            "switch_frames",
            "posterior_threshold",
            "switch_penalty",
            "accepted",
            "duplicate",
            "sealed",
            "current_language",
            "pending_language",
            "pending_frames",
            "next_seq",
            "segments",
            "open_segment",
            "error",
        )
        if k in body
    }
    print(f"\n[{label}] HTTP {status}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    status, session = post(
        "/sessions",
        {
            "languages": LANGUAGES,
            "frame_ms": 20,
            "switch_frames": 3,
            "posterior_threshold": 0.55,
            "switch_penalty": 2.0,
        },
    )
    show("create session", status, session)
    sid = session["session_id"]

    # 块 0：5 帧 zh，随后 1 帧 en 抖动，再 2 帧 zh —— 抖动不应切段。
    chunk0 = {
        "start_seq": 0,
        "posteriors": [frame(0) for _ in range(5)]
        + [frame(1)]
        + [frame(0) for _ in range(2)],
    }
    status, body = post(f"/sessions/{sid}/chunks", chunk0)
    show("chunk 0 (zh, jitter, zh)", status, body)

    # 重复提交块 0：应返回原结果（duplicate=true），会话不变。
    status, dup = post(f"/sessions/{sid}/chunks", chunk0)
    show("duplicate chunk 0", status, dup)

    # 块 1：3 帧 en（达到切换阈值）+ 4 帧 en。
    status, body = post(
        f"/sessions/{sid}/chunks",
        {"start_seq": 8, "posteriors": [frame(1) for _ in range(7)]},
    )
    show("chunk 1 (en x7)", status, body)

    # 迟到块：起始序号 6 < 期望序号 15，应被拒且不改变会话。
    status, late = post(
        f"/sessions/{sid}/chunks",
        {"start_seq": 6, "posteriors": [frame(0) for _ in range(2)]},
    )
    show("late chunk (start 6)", status, late)

    # 越界块：起始序号 20 > 期望序号 15，同样被拒。
    status, gap = post(
        f"/sessions/{sid}/chunks",
        {"start_seq": 20, "posteriors": [frame(0)]},
    )
    show("out-of-order chunk (start 20)", status, gap)

    # 块 2：5 帧 ja，后 2 帧低置信（0.6/0.2/0.2 最大后验 0.6 达标；这里用 0.45 低置信）。
    low = [0.45, 0.30, 0.25]
    status, body = post(
        f"/sessions/{sid}/chunks",
        {
            "start_seq": 15,
            "posteriors": [frame(2) for _ in range(5)] + [low, low],
        },
    )
    show("chunk 2 (ja x5 + low-confidence x2)", status, body)

    # 封存。
    status, sealed = post(f"/sessions/{sid}/seal")
    show("seal", status, sealed)

    # 封存后提交新块：应 409。
    status, after = post(
        f"/sessions/{sid}/chunks",
        {"start_seq": 22, "posteriors": [frame(0)]},
    )
    show("submit after seal", status, after)

    status, final = get(f"/sessions/{sid}/segments")
    print("\n[final segments]")
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
