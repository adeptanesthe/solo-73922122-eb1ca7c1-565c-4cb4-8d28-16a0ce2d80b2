"""端到端调用示例：逐路 / 批量提交、幂等、错误隔离、封存与计算量对比。

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


def frame(lang_idx: int, hi: float = 0.95, n: int | None = None) -> list[float]:
    n = n or len(LANGUAGES)
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


def create_session(**overrides) -> str:
    body = {
        "languages": LANGUAGES,
        "frame_ms": 20,
        "switch_frames": 3,
        "posterior_threshold": 0.55,
        "switch_penalty": 2.0,
    }
    body.update(overrides)
    status, session = post("/sessions", body)
    show("create session", status, session)
    assert status == 201, session
    return session["session_id"]


def metrics() -> int:
    status, body = get("/metrics")
    assert status == 200
    return body["forward_steps"]


def reset_metrics() -> None:
    status, _ = post("/metrics/reset")
    assert status == 200


def batch(items: list[dict]) -> dict:
    status, body = post("/chunks/batch", {"items": items})
    show(f"batch submit ({len(items)} sessions)", status, body)
    assert status == 200, body
    return body


def main() -> None:
    sid = create_session()

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

    # 块 2：5 帧 ja，后 2 帧低置信（0.45 最大后验 < 0.55）。
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

    demo_batch_equivalence()
    demo_cost_comparison()


def demo_batch_equivalence() -> None:
    """批量入口：16 路混合块，含非法块 / 冲突块 / 未知会话，逐项回执或错误。"""
    print("\n" + "=" * 70)
    print("批量收块：16 个不同会话，同一批提交；坏块只影响自己")
    print("=" * 70)

    batch_sids = [create_session() for _ in range(16)]
    # 其中 4 路先提交一块，供后面制造冲突/迟到。
    warm = [frame(0) for _ in range(3)]
    for sid in batch_sids[:4]:
        post(f"/sessions/{sid}/chunks", {"start_seq": 0, "posteriors": warm})

    chunk_z = [frame(0) for _ in range(4)] + [frame(1) for _ in range(4)]
    chunk_e = [frame(1) for _ in range(2)] + [frame(0) for _ in range(6)]
    chunk_j = [frame(2) for _ in range(5)]

    items = [
        # 0：冲突块（起点 0 已提交过不同内容）。
        {"session_id": batch_sids[0], "start_seq": 0, "posteriors": chunk_z},
        # 1：迟到块。
        {"session_id": batch_sids[1], "start_seq": 1, "posteriors": chunk_z},
        # 2：正常块。
        {"session_id": batch_sids[2], "start_seq": 3, "posteriors": chunk_z},
        # 3：未知会话。
        {"session_id": "does-not-exist", "start_seq": 0, "posteriors": chunk_z},
        # 4：维度错误（会话是 3 语种，这里只给 2 列）。
        {"session_id": batch_sids[3], "start_seq": 3, "posteriors": [[0.6, 0.4]]},
        # 5..12：正常但长度、内容各异。
        *[
            {
                "session_id": sid,
                "start_seq": 0,
                "posteriors": [chunk_z, chunk_e, chunk_j][i % 3],
            }
            for i, sid in enumerate(batch_sids[4:12])
        ],
        # 13..15：正常，最后一个带 final 封存。
        {"session_id": batch_sids[12], "start_seq": 0, "posteriors": chunk_z},
        {"session_id": batch_sids[13], "start_seq": 0, "posteriors": chunk_e},
        {
            "session_id": batch_sids[15],
            "start_seq": 0,
            "posteriors": chunk_j,
            "final": True,
        },
    ]
    body = batch(items)
    print(
        f"\n汇总：accepted={body['accepted_count']} error={body['error_count']}"
    )
    for r in body["results"]:
        if r["ok"]:
            rec = r["receipt"]
            segs = [(s["language"], s["start_frame"], s["end_frame"]) for s in rec["segments"]]
            print(
                f"  [{r['index']:2d}] ok   {r['session_id'][:8]}… "
                f"next_seq={rec['next_seq']} current={rec['current_language']} "
                f"sealed={rec['sealed']} segments={segs}"
            )
        else:
            e = r["error"]
            print(
                f"  [{r['index']:2d}] err  {r['session_id'][:8]}… "
                f"{e['status_code']} {e['code']}: {e['message']}"
            )

    # 坏块没有推进对应会话：0/1/3/4 号的 next_seq 保持原样。
    print("\n坏块不推进会话（可查状态）：")
    for r in body["results"]:
        if not r["ok"] and r["session_id"] in batch_sids[:4]:
            status, st = get(f"/sessions/{r['session_id']}")
            print(
                f"  {r['session_id'][:8]}… next_seq={st['next_seq']} "
                f"sealed={st['sealed']}"
            )

    # 封存项的回执已包含固化后的完整区间。
    sealed_item = body["results"][-1]["receipt"]
    print(
        "\nfinal 项回执：sealed=",
        sealed_item["sealed"],
        "open_segment=",
        sealed_item["open_segment"],
        "segments=",
        [(s["language"], s["start_frame"], s["end_frame"]) for s in sealed_item["segments"]],
    )


def demo_cost_comparison() -> None:
    """同一组等长、等语种数的块：批量 vs 逐路的前向递推次数。"""
    print("\n" + "=" * 70)
    print("计算量对比：8 个会话 × 每块 12 帧 × 2 语种（块完全同形）")
    print("=" * 70)

    n_sessions = 8
    base = (
        [frame(0, n=2) for _ in range(4)]
        + [frame(1, n=2) for _ in range(4)]
        + [frame(0, n=2) for _ in range(4)]
    )
    assert len(base) == 12

    seq_sids = [create_session(languages=["zh", "en"]) for _ in range(n_sessions)]
    bat_sids = [create_session(languages=["zh", "en"]) for _ in range(n_sessions)]

    reset_metrics()
    for sid in seq_sids:
        status, _ = post(f"/sessions/{sid}/chunks", {"start_seq": 0, "posteriors": base})
        assert status == 200
    seq_steps = metrics()

    reset_metrics()
    body = post(
        "/chunks/batch",
        {"items": [{"session_id": sid, "start_seq": 0, "posteriors": base} for sid in bat_sids]},
    )
    assert body[0] == 200, body
    bat_steps = metrics()

    print(f"\n逐路提交：{n_sessions} 次请求，NumPy 前向递推 {seq_steps} 次")
    print(f"批量提交：1 次请求，NumPy 前向递推 {bat_steps} 次")
    print(f"倍数：{seq_steps / bat_steps:.1f}×")

    status, seq_segs = get(f"/sessions/{seq_sids[0]}/segments")
    status2, bat_segs = get(f"/sessions/{bat_sids[0]}/segments")
    print(
        "\n两种方式最终区间一致：",
        seq_segs["segments"] == bat_segs["segments"],
        [(s["language"], s["start_frame"], s["end_frame"]) for s in seq_segs["segments"]],
    )

    # 封存批量侧的会话，展示批量 + final 的用法。
    status, sealed = post(
        "/chunks/batch",
        {
            "items": [
                {
                    "session_id": bat_sids[0],
                    "start_seq": 12,
                    "posteriors": [frame(1, n=2) for _ in range(3)],
                    "final": True,
                }
            ]
        },
    )
    show("batch append + final", status, sealed)


if __name__ == "__main__":
    main()
