"""自测：批量收块入口的行为、与逐路提交的等价性、并发与计算量。"""

from __future__ import annotations

import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app, _sessions
from app.smoother import FORWARD_STEPS, BatchEntry, LanguageSmoother, submit_batch


def strong(i: int, n: int = 2, hi: float = 0.95) -> list[float]:
    v = [round((1.0 - hi) / (n - 1), 6)] * n
    v[i] = hi
    return v


def rows(seq: str, n: int = 2) -> list[list[float]]:
    """按字符串构造帧序列：'z' -> 语种 0，'e' -> 语种 1，'j' -> 语种 2。"""
    idx = {"z": 0, "e": 1, "j": 2}
    return [strong(idx[ch], n) for ch in seq]


@pytest.fixture(autouse=True)
def _clear_sessions():
    _sessions.clear()
    FORWARD_STEPS.reset()
    yield
    _sessions.clear()


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def make_session(client: TestClient, **overrides) -> str:
    body = {
        "languages": ["zh", "en"],
        "frame_ms": 20,
        "switch_frames": 3,
        "posterior_threshold": 0.55,
        "switch_penalty": 2.0,
    }
    body.update(overrides)
    r = client.post("/sessions", json=body)
    assert r.status_code == 201, r.text
    return r.json()["session_id"]


def batch(client: TestClient, items: list[dict]) -> dict:
    r = client.post("/chunks/batch", json={"items": items})
    assert r.status_code == 200, r.text
    return r.json()


# ------------------------------------------------------------ 批量基本行为
def test_batch_accepts_up_to_16_sessions(client: TestClient):
    sids = [make_session(client) for _ in range(16)]
    payload = [
        {"session_id": sid, "start_seq": 0, "posteriors": rows("zzzzz")} for sid in sids
    ]
    body = batch(client, payload)
    assert body["accepted_count"] == 16
    assert body["error_count"] == 0
    assert [r["index"] for r in body["results"]] == list(range(16))
    for r in body["results"]:
        assert r["ok"] is True
        assert r["receipt"]["accepted"] is True
        assert r["receipt"]["next_seq"] == 5
        assert r["receipt"]["current_language"] == "zh"


def test_batch_rejects_17_sessions(client: TestClient):
    sid = make_session(client)
    payload = [
        {"session_id": sid, "start_seq": 0, "posteriors": rows("zz")} for _ in range(17)
    ]
    r = client.post("/chunks/batch", json={"items": payload})
    assert r.status_code == 422


def test_batch_rejects_duplicate_session_in_one_batch(client: TestClient):
    sid = make_session(client)
    payload = [
        {"session_id": sid, "start_seq": 0, "posteriors": rows("zz")},
        {"session_id": sid, "start_seq": 2, "posteriors": rows("zz")},
    ]
    r = client.post("/chunks/batch", json={"items": payload})
    assert r.status_code == 422


def test_batch_mixed_lengths_and_language_counts(client: TestClient):
    a = make_session(client, languages=["zh", "en"], switch_frames=3)
    b = make_session(client, languages=["zh", "en", "ja"], switch_frames=2)
    c = make_session(client, languages=["a", "b", "c", "d"], switch_frames=4)
    body = batch(
        client,
        [
            {
                "session_id": a,
                "start_seq": 0,
                "posteriors": rows("zzzzeee"),
            },
            {
                "session_id": b,
                "start_seq": 0,
                "posteriors": rows("zzjje", n=3),
            },
            {
                "session_id": c,
                "start_seq": 0,
                "posteriors": rows("zzzz", n=4),
            },
        ],
    )
    assert body["accepted_count"] == 3
    by_id = {r["session_id"]: r["receipt"] for r in body["results"]}
    assert by_id[a]["next_seq"] == 7
    assert by_id[b]["next_seq"] == 5
    assert by_id[c]["next_seq"] == 4
    assert by_id[a]["current_language"] == "en"
    assert by_id[b]["current_language"] == "ja"
    assert by_id[c]["current_language"] == "a"


def test_batch_unknown_session_only_fails_that_item(client: TestClient):
    a = make_session(client)
    body = batch(
        client,
        [
            {"session_id": "nope", "start_seq": 0, "posteriors": rows("zz")},
            {"session_id": a, "start_seq": 0, "posteriors": rows("zz")},
        ],
    )
    assert body["accepted_count"] == 1
    assert body["error_count"] == 1
    assert body["results"][0]["error"]["code"] == "session_not_found"
    assert body["results"][0]["error"]["status_code"] == 404
    assert body["results"][1]["ok"] is True
    assert client.get(f"/sessions/{a}").json()["next_seq"] == 2


# ------------------------------------------------- 非法块 / 冲突块不推进
def test_batch_invalid_chunk_does_not_advance_session(client: TestClient):
    a = make_session(client)
    b = make_session(client)
    # a 提交维度错误（3 列，会话只有 2 个语种）；b 正常。
    body = batch(
        client,
        [
            {"session_id": a, "start_seq": 0, "posteriors": [[0.5, 0.3, 0.2]]},
            {"session_id": b, "start_seq": 0, "posteriors": rows("zzz")},
        ],
    )
    assert body["results"][0]["error"]["code"] == "dimension_mismatch"
    assert body["results"][1]["ok"] is True
    assert client.get(f"/sessions/{a}").json()["next_seq"] == 0
    assert client.get(f"/sessions/{b}").json()["next_seq"] == 3

    # 修正后同一 start_seq 可重试。
    ok = batch(
        client,
        [{"session_id": a, "start_seq": 0, "posteriors": rows("zzz")}],
    )
    assert ok["results"][0]["ok"] is True


def test_batch_conflicting_chunk_does_not_advance_session(client: TestClient):
    a = make_session(client)
    first = batch(client, [{"session_id": a, "start_seq": 0, "posteriors": rows("zzz")}])
    assert first["results"][0]["ok"] is True
    before = client.get(f"/sessions/{a}").json()

    # 同起点不同内容 -> chunk_conflict；同批另一个新会话照常推进。
    b = make_session(client)
    body = batch(
        client,
        [
            {"session_id": a, "start_seq": 0, "posteriors": rows("eee")},
            {"session_id": b, "start_seq": 0, "posteriors": rows("zzz")},
        ],
    )
    assert body["results"][0]["error"]["code"] == "chunk_conflict"
    assert body["results"][1]["ok"] is True
    assert client.get(f"/sessions/{a}").json() == before
    assert client.get(f"/sessions/{b}").json()["next_seq"] == 3


def test_batch_late_and_out_of_order_rejected_per_item(client: TestClient):
    a = make_session(client)
    b = make_session(client)
    batch(client, [{"session_id": a, "start_seq": 0, "posteriors": rows("zz")}])

    body = batch(
        client,
        [
            {"session_id": a, "start_seq": 1, "posteriors": rows("z")},
            {"session_id": b, "start_seq": 5, "posteriors": rows("z")},
        ],
    )
    assert body["results"][0]["error"]["code"] == "late_chunk"
    assert body["results"][1]["error"]["code"] == "out_of_order"
    assert client.get(f"/sessions/{a}").json()["next_seq"] == 2
    assert client.get(f"/sessions/{b}").json()["next_seq"] == 0


# ------------------------------------------------------- 与逐路提交等价
def _run_sequential(client: TestClient, sid: str, chunks: list[list[list[float]]]) -> None:
    start = 0
    for chunk in chunks:
        r = client.post(
            f"/sessions/{sid}/chunks", json={"start_seq": start, "posteriors": chunk}
        )
        assert r.status_code == 200, r.text
        start += len(chunk)


def test_batch_matches_sequential_incremental_state(client: TestClient):
    """逐路 vs 批量：每轮回执与可查状态逐字段一致（含多次往返）。"""
    sid_seq = make_session(client, switch_frames=3)
    sid_batch = make_session(client, switch_frames=3)
    rounds = [
        rows("zzzzz"),
        rows("eezz"),
        rows("e"),
        rows("zzzz"),
    ]

    receipts_seq = []
    start = 0
    for chunk in rounds:
        r = client.post(
            f"/sessions/{sid_seq}/chunks", json={"start_seq": start, "posteriors": chunk}
        )
        receipts_seq.append(r.json())
        start += len(chunk)

    receipts_batch = []
    start = 0
    for chunk in rounds:
        body = batch(
            client,
            [{"session_id": sid_batch, "start_seq": start, "posteriors": chunk}],
        )
        receipts_batch.append(body["results"][0]["receipt"])
        start += len(chunk)

    for seq, bat in zip(receipts_seq, receipts_batch):
        assert bat["start_seq"] == seq["start_seq"]
        assert bat["end_seq"] == seq["end_seq"]
        assert bat["duplicate"] == seq["duplicate"]
        assert bat["sealed"] == seq["sealed"]
        assert bat["segments"] == seq["segments"]
        assert bat["open_segment"] == seq["open_segment"]
        assert bat["current_language"] == seq["current_language"]
        assert bat["pending_language"] == seq["pending_language"]
        assert bat["pending_frames"] == seq["pending_frames"]
        assert bat["next_seq"] == seq["next_seq"]

    st_seq = client.get(f"/sessions/{sid_seq}").json()
    st_batch = client.get(f"/sessions/{sid_batch}").json()
    st_seq.pop("session_id")
    st_batch.pop("session_id")
    assert st_seq == st_batch


def test_batch_final_flag_seals_same_as_sequential(client: TestClient):
    a = make_session(client)
    b = make_session(client)
    client.post(
        f"/sessions/{a}/chunks",
        json={"start_seq": 0, "posteriors": rows("zzzz"), "final": True},
    )
    body = batch(
        client,
        [{"session_id": b, "start_seq": 0, "posteriors": rows("zzzz"), "final": True}],
    )
    rec = body["results"][0]["receipt"]
    assert rec["sealed"] is True
    assert rec["open_segment"] is None
    assert [(x["language"], x["start_frame"], x["end_frame"]) for x in rec["segments"]] == [
        ("zh", 0, 3)
    ]
    assert client.get(f"/sessions/{b}/segments").json()["sealed"] is True
    assert client.get(f"/sessions/{b}").json() == client.get(f"/sessions/{a}").json() | {
        "session_id": b
    }


def test_batch_after_seal_returns_per_item_error(client: TestClient):
    a = make_session(client)
    client.post(f"/sessions/{a}/seal")
    body = batch(client, [{"session_id": a, "start_seq": 0, "posteriors": rows("zz")}])
    assert body["results"][0]["error"]["code"] == "session_sealed"
    assert body["results"][0]["error"]["status_code"] == 409
    assert client.get(f"/sessions/{a}").json()["next_seq"] == 0


def test_batch_duplicate_replay_returns_original(client: TestClient):
    a = make_session(client)
    payload = [{"session_id": a, "start_seq": 0, "posteriors": rows("zzz")}]
    first = batch(client, payload)["results"][0]["receipt"]
    dup = batch(client, payload)["results"][0]["receipt"]
    assert dup["duplicate"] is True
    assert dup["next_seq"] == first["next_seq"]
    assert dup["segments"] == first["segments"]
    assert dup["open_segment"] == first["open_segment"]


# ---------------------------------------------------------- 计算量对比
def test_batch_calls_forward_fewer_times_than_sequential(client: TestClient):
    """同一组等长、等语种数的块：批量前向递推次数明显少于逐路。"""
    n_sessions = 8
    n_frames = 12
    chunk = rows("zzzzeeezzzz" + "z", n=2)
    assert len(chunk) == n_frames

    seq_ids = [make_session(client) for _ in range(n_sessions)]
    bat_ids = [make_session(client) for _ in range(n_sessions)]

    FORWARD_STEPS.reset()
    for sid in seq_ids:
        r = client.post(
            f"/sessions/{sid}/chunks", json={"start_seq": 0, "posteriors": chunk}
        )
        assert r.status_code == 200
    sequential_steps = FORWARD_STEPS.value

    FORWARD_STEPS.reset()
    body = batch(
        client,
        [{"session_id": sid, "start_seq": 0, "posteriors": chunk} for sid in bat_ids],
    )
    batch_steps = FORWARD_STEPS.value
    assert body["accepted_count"] == n_sessions

    assert sequential_steps == n_sessions * n_frames
    assert batch_steps == n_frames
    assert batch_steps < sequential_steps

    # 两种方式的最终区间一致。
    final_seq = [
        client.get(f"/sessions/{sid}/segments").json()["segments"] for sid in seq_ids
    ]
    final_bat = [
        client.get(f"/sessions/{sid}/segments").json()["segments"] for sid in bat_ids
    ]
    assert final_seq == final_bat
    assert final_seq[0], "expected at least one confirmed segment"


def test_forward_counter_matches_work(client: TestClient):
    sid = make_session(client)
    FORWARD_STEPS.reset()
    batch(client, [{"session_id": sid, "start_seq": 0, "posteriors": rows("zzzzz")}])
    assert FORWARD_STEPS.value == 5
    assert client.get("/metrics").json() == {"forward_steps": 5}


# ---------------------------------------------------------------- 并发
def _make_direct(languages=("zh", "en"), switch_frames=3) -> LanguageSmoother:
    return LanguageSmoother(
        "s-" + str(id(object())),
        list(languages),
        frame_ms=20,
        switch_frames=switch_frames,
        posterior_threshold=0.55,
        switch_penalty=2.0,
    )


def test_same_session_sequential_chunks_through_batch_are_ordered():
    """同一会话连续两块只能按序号逐块提交；乱序项得到错误且不推进。"""
    s = _make_direct()
    out1 = submit_batch([BatchEntry(s, 0, np.array(rows("zzzz")))])
    assert out1[0].error is None
    assert out1[0].outcome.result.next_seq == 4

    out2 = submit_batch(
        [
            BatchEntry(s, 8, np.array(rows("zz"))),
            BatchEntry(s, 4, np.array(rows("zzzz"))),
        ]
    )
    # 批内同一会话出现两次超出 API 约束；核心层按顺序处理：越界项报错，
    # 正确项推进。
    codes = [o.error.code if o.error else "ok" for o in out2]
    assert codes == ["out_of_order", "ok"]
    assert s.next_seq == 8


def test_concurrent_mixed_entries_equivalent_to_serial_order():
    """批量与逐路并发投递同一会话：最终状态等价于某种串行顺序。"""
    s = _make_direct()
    chunks = [np.array(rows("zzzzzz")), np.array(rows("zzzzzz"))]
    errors: list[Exception] = []

    def via_batch() -> None:
        try:
            submit_batch([BatchEntry(s, 0, chunks[0])])
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    def via_single() -> None:
        try:
            # 与批量竞争：若批量的块先落地，这一块应报 out_of_order，反之
            # 应成功；无论哪种，最终状态都必须是按某个顺序串行处理的结果。
            s.submit(6, chunks[1])
        except Exception as exc:
            if not hasattr(exc, "code"):
                errors.append(exc)

    threads = [threading.Thread(target=via_batch), threading.Thread(target=via_single)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    # 无论谁先，12 帧都只能有一条合法路径：块 0 在 0..5，块 1 在 6..11。
    assert s.next_seq == 12
    st = s.state()
    assert st["pending_frames"] in (0, 1, 2)


def test_batch_does_not_block_unrelated_session(monkeypatch):
    """批量执行期间，批外会话仍能收块。

    批量线程在前向递推里被挂住（直到 ``release`` 才继续）；若批外会话的
    提交被整批的锁挡住，它就会一直等到批量结束。这里用"先放行批外会话、
    再放行批量"的次序断言：批外会话在批量仍挂起时就能完成。
    """
    participants = [_make_direct() for _ in range(4)]
    outsider = _make_direct()
    outsider_chunk = np.array(rows("zz"))
    release = threading.Event()
    batch_inside = threading.Event()
    outsider_done = threading.Event()

    real_logsumexp = __import__("app.smoother", fromlist=["_logsumexp"])._logsumexp
    calls = {"n": 0}

    def blocked_logsumexp(x, axis=-1):
        calls["n"] += 1
        if calls["n"] == 1:
            batch_inside.set()
            release.wait(5.0)
        return real_logsumexp(x, axis=axis)

    monkeypatch.setattr("app.smoother._logsumexp", blocked_logsumexp)

    def run_batch() -> None:
        submit_batch([BatchEntry(s, 0, np.array(rows("zzzz"))) for s in participants])

    def run_outsider() -> None:
        outsider.submit(0, outsider_chunk)
        outsider_done.set()

    batch_thread = threading.Thread(target=run_batch)
    batch_thread.start()
    assert batch_inside.wait(2.0), "batch never entered forward recursion"

    # 批量仍阻塞在 forward 内：批外会话必须能立刻完成。
    threading.Thread(target=run_outsider).start()
    assert outsider_done.wait(2.0), "unrelated session was blocked by batch inference"
    assert batch_thread.is_alive(), "batch should still be blocked"

    release.set()
    batch_thread.join(5.0)
    assert not batch_thread.is_alive()
    assert outsider.next_seq == 2
    assert all(s.next_seq == 4 for s in participants)
