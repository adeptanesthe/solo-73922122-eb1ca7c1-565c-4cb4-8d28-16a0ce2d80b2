"""自测：核心状态机 + API 行为（重复、迟到、越界、切换边界、低置信、封存）。"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app, _sessions
from app.smoother import LanguageSmoother, SessionError


def strong(i: int, n: int, hi: float = 0.97) -> list[float]:
    v = [round((1.0 - hi) / (n - 1), 6)] * n
    v[i] = hi
    return v


@pytest.fixture(autouse=True)
def _clear_sessions():
    _sessions.clear()
    yield
    _sessions.clear()


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def make_session(client: TestClient, **overrides) -> dict:
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
    return r.json()


# ------------------------------------------------------------------ 基本分段
def test_basic_build_switch_seal(client: TestClient):
    s = make_session(client)
    sid = s["session_id"]
    rows = [strong(0, 2) for _ in range(5)] + [strong(1, 2) for _ in range(5)]
    r = client.post(f"/sessions/{sid}/chunks", json={"start_seq": 0, "posteriors": rows})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["accepted"] is True
    assert body["duplicate"] is False
    assert body["current_language"] == "en"
    assert body["next_seq"] == 10
    # 已关闭区间：zh 0..4；en 段仍在延伸。
    assert body["segments"] == [
        {
            "language": "zh",
            "start_frame": 0,
            "end_frame": 4,
            "start_ms": 0,
            "end_ms": 100,
            "duration_ms": 100,
        }
    ]
    assert body["open_segment"]["language"] == "en"
    assert body["open_segment"]["start_frame"] == 5
    assert body["open_segment"]["end_frame"] == 9

    seal = client.post(f"/sessions/{sid}/seal")
    assert seal.status_code == 200
    segs = seal.json()["segments"]
    assert [(x["language"], x["start_frame"], x["end_frame"]) for x in segs] == [
        ("zh", 0, 4),
        ("en", 5, 9),
    ]
    # 封存后新块被拒。
    late = client.post(
        f"/sessions/{sid}/chunks", json={"start_seq": 10, "posteriors": [strong(0, 2)]}
    )
    assert late.status_code == 409
    assert late.json()["error"]["code"] == "session_sealed"


def test_final_flag_seals_and_segments_endpoint(client: TestClient):
    s = make_session(client)
    sid = s["session_id"]
    r = client.post(
        f"/sessions/{sid}/chunks",
        json={
            "start_seq": 0,
            "posteriors": [strong(0, 2) for _ in range(4)],
            "final": True,
        },
    )
    assert r.status_code == 200
    assert r.json()["sealed"] is True
    assert r.json()["open_segment"] is None
    assert [(x["language"], x["start_frame"], x["end_frame"]) for x in r.json()["segments"]] == [
        ("zh", 0, 3)
    ]
    segs = client.get(f"/sessions/{sid}/segments").json()
    assert segs["sealed"] is True
    assert [(x["language"], x["start_frame"], x["end_frame"]) for x in segs["segments"]] == [
        ("zh", 0, 3)
    ]


# ------------------------------------------------------------- 切换边界
def test_switch_needs_exactly_threshold_consecutive_frames(client: TestClient):
    s = make_session(client, switch_frames=3)
    sid = s["session_id"]
    rows = [strong(0, 2) for _ in range(4)] + [strong(1, 2) for _ in range(2)]
    r = client.post(f"/sessions/{sid}/chunks", json={"start_seq": 0, "posteriors": rows})
    body = r.json()
    # 只有 2 帧 en：未达阈值，不切段，仍归 zh。
    assert body["current_language"] == "zh"
    assert body["pending_language"] == "en"
    assert body["pending_frames"] == 2
    assert body["segments"] == []
    assert body["open_segment"]["language"] == "zh"
    assert body["open_segment"]["end_frame"] == 5

    # 第 3 帧 en 到达：切段，切点回退到 en 首帧（4）。
    r2 = client.post(
        f"/sessions/{sid}/chunks", json={"start_seq": 6, "posteriors": [strong(1, 2)]}
    )
    body2 = r2.json()
    assert body2["current_language"] == "en"
    assert body2["segments"][-1] == {
        "language": "zh",
        "start_frame": 0,
        "end_frame": 3,
        "start_ms": 0,
        "end_ms": 80,
        "duration_ms": 80,
    }
    assert body2["open_segment"]["language"] == "en"
    assert body2["open_segment"]["start_frame"] == 4
    assert body2["open_segment"]["end_frame"] == 6


def test_single_frame_jitter_does_not_switch(client: TestClient):
    s = make_session(client, switch_frames=3)
    sid = s["session_id"]
    rows = (
        [strong(0, 2) for _ in range(4)]
        + [strong(1, 2)]
        + [strong(0, 2) for _ in range(2)]
        + [strong(1, 2)]
        + [strong(0, 2) for _ in range(2)]
    )
    body = client.post(
        f"/sessions/{sid}/chunks", json={"start_seq": 0, "posteriors": rows}
    ).json()
    assert body["current_language"] == "zh"
    assert body["segments"] == []
    assert body["open_segment"]["language"] == "zh"
    assert body["open_segment"]["end_frame"] == len(rows) - 1


def test_low_confidence_frames_stay_in_previous_state(client: TestClient):
    s = make_session(client, switch_frames=3, posterior_threshold=0.8)
    sid = s["session_id"]
    # 0.6/0.4 的最大后验 0.6 < 0.8：低置信。
    low = [0.6, 0.4]
    rows = [strong(0, 2) for _ in range(4)] + [low, low, low, low]
    body = client.post(
        f"/sessions/{sid}/chunks", json={"start_seq": 0, "posteriors": rows}
    ).json()
    assert body["current_language"] == "zh"
    assert body["pending_language"] is None
    assert body["segments"] == []  # 尚未发生切换
    assert body["open_segment"]["end_frame"] == 7  # 低置信帧并入当前段


def test_low_confidence_breaks_consecutive_run(client: TestClient):
    s = make_session(client, switch_frames=3, posterior_threshold=0.8)
    sid = s["session_id"]
    low = [0.3, 0.7]  # max 0.7 < 0.8 -> 低置信，且原始 argmax 是 en
    rows = [strong(0, 2) for _ in range(4)] + [strong(1, 2), strong(1, 2), low, strong(1, 2)]
    body = client.post(
        f"/sessions/{sid}/chunks", json={"start_seq": 0, "posteriors": rows}
    ).json()
    # 两帧 en + 低置信打断 + 一帧 en：未连续达 3 帧。
    assert body["current_language"] == "zh"
    assert body["pending_frames"] == 1


# ------------------------------------------------------------------ 重复块
def test_duplicate_chunk_returns_original_result_and_keeps_state(client: TestClient):
    s = make_session(client)
    sid = s["session_id"]
    chunk = {"start_seq": 0, "posteriors": [strong(0, 2) for _ in range(5)]}
    first = client.post(f"/sessions/{sid}/chunks", json=chunk).json()
    state_after_first = client.get(f"/sessions/{sid}").json()

    dup = client.post(f"/sessions/{sid}/chunks", json=chunk)
    assert dup.status_code == 200
    d = dup.json()
    assert d["duplicate"] is True
    assert d["next_seq"] == first["next_seq"]
    assert d["segments"] == first["segments"]
    assert d["open_segment"] == first["open_segment"]
    # 会话状态未被改动。
    assert client.get(f"/sessions/{sid}").json() == state_after_first

    # 后续块仍从正确的序号继续。
    nxt = client.post(
        f"/sessions/{sid}/chunks",
        json={"start_seq": 5, "posteriors": [strong(1, 2)]},
    )
    assert nxt.status_code == 200
    assert nxt.json()["next_seq"] == 6


def test_duplicate_with_different_content_conflicts(client: TestClient):
    s = make_session(client)
    sid = s["session_id"]
    client.post(
        f"/sessions/{sid}/chunks",
        json={"start_seq": 0, "posteriors": [strong(0, 2) for _ in range(3)]},
    )
    before = client.get(f"/sessions/{sid}").json()
    bad = client.post(
        f"/sessions/{sid}/chunks",
        json={"start_seq": 0, "posteriors": [strong(1, 2) for _ in range(3)]},
    )
    assert bad.status_code == 409
    assert bad.json()["error"]["code"] == "chunk_conflict"
    assert client.get(f"/sessions/{sid}").json() == before


def test_duplicate_after_seal_still_returns_original(client: TestClient):
    s = make_session(client)
    sid = s["session_id"]
    chunk = {"start_seq": 0, "posteriors": [strong(0, 2) for _ in range(4)]}
    first = client.post(f"/sessions/{sid}/chunks", json=chunk).json()
    client.post(f"/sessions/{sid}/seal")
    dup = client.post(f"/sessions/{sid}/chunks", json=chunk)
    assert dup.status_code == 200
    assert dup.json()["duplicate"] is True
    assert dup.json()["sealed"] is True
    assert dup.json()["segments"] == first["segments"]


# ------------------------------------------------------------- 迟到 / 越界
def test_late_chunk_rejected_without_state_change(client: TestClient):
    s = make_session(client)
    sid = s["session_id"]
    client.post(
        f"/sessions/{sid}/chunks",
        json={"start_seq": 0, "posteriors": [strong(0, 2) for _ in range(4)]},
    )
    before = client.get(f"/sessions/{sid}").json()
    late = client.post(
        f"/sessions/{sid}/chunks",
        json={"start_seq": 2, "posteriors": [strong(1, 2) for _ in range(2)]},
    )
    assert late.status_code == 409
    assert late.json()["error"]["code"] == "late_chunk"
    assert client.get(f"/sessions/{sid}").json() == before


def test_out_of_order_chunk_rejected_without_state_change(client: TestClient):
    s = make_session(client)
    sid = s["session_id"]
    before = client.get(f"/sessions/{sid}").json()
    gap = client.post(
        f"/sessions/{sid}/chunks",
        json={"start_seq": 3, "posteriors": [strong(0, 2)]},
    )
    assert gap.status_code == 409
    assert gap.json()["error"]["code"] == "out_of_order"
    assert client.get(f"/sessions/{sid}").json() == before

    # 缺口被拒后，正确序号仍可提交。
    ok = client.post(
        f"/sessions/{sid}/chunks",
        json={"start_seq": 0, "posteriors": [strong(0, 2)]},
    )
    assert ok.status_code == 200


# ------------------------------------------------------------- 增量一致性
def test_frame_by_frame_equals_single_chunk():
    rows = [
        strong(0, 3),
        strong(0, 3),
        strong(1, 3),
        strong(0, 3),
        strong(1, 3),
        strong(1, 3),
        strong(1, 3),
        strong(2, 3),
        strong(2, 3),
        strong(2, 3),
        strong(2, 3),
    ]
    params = dict(frame_ms=20, switch_frames=3, posterior_threshold=0.55, switch_penalty=2.0)
    one = LanguageSmoother("a", ["zh", "en", "ja"], **params)
    one.submit_chunk(0, np.array(rows))
    one_segs = [s.to_dict() for s in one.seal()]

    many = LanguageSmoother("b", ["zh", "en", "ja"], **params)
    for i, row in enumerate(rows):
        many.submit_chunk(i, np.array([row]))
    many_segs = [s.to_dict() for s in many.seal()]
    assert one_segs == many_segs


def test_chunk_splits_do_not_change_boundaries():
    rows = [strong(0, 2) for _ in range(4)] + [strong(1, 2) for _ in range(6)]
    params = dict(frame_ms=20, switch_frames=3, posterior_threshold=0.55, switch_penalty=2.0)
    one = LanguageSmoother("a", ["zh", "en"], **params)
    one.submit_chunk(0, np.array(rows))
    one_segs = [s.to_dict() for s in one.seal()]

    many = LanguageSmoother("b", ["zh", "en"], **params)
    many.submit_chunk(0, np.array(rows[:2]))
    many.submit_chunk(2, np.array(rows[2:5]))
    many.submit_chunk(5, np.array(rows[5:]))
    many_segs = [s.to_dict() for s in many.seal()]
    assert one_segs == many_segs


# ------------------------------------------------------------- 校验与错误
def test_session_parameter_validation(client: TestClient):
    one = client.post("/sessions", json={"languages": ["zh"]})
    assert one.status_code == 422
    five = client.post("/sessions", json={"languages": ["a", "b", "c", "d", "e"]})
    assert five.status_code == 422
    dup = client.post("/sessions", json={"languages": ["zh", "zh"]})
    assert dup.status_code == 422
    bad_thr = client.post("/sessions", json={"languages": ["zh", "en"], "posterior_threshold": 0})
    assert bad_thr.status_code == 422
    bad_frames = client.post("/sessions", json={"languages": ["zh", "en"], "switch_frames": 0})
    assert bad_frames.status_code == 422


def test_chunk_validation(client: TestClient):
    s = make_session(client)
    sid = s["session_id"]
    wrong_width = client.post(
        f"/sessions/{sid}/chunks", json={"start_seq": 0, "posteriors": [[0.5, 0.3, 0.2]]}
    )
    assert wrong_width.status_code == 422
    assert wrong_width.json()["error"]["code"] == "dimension_mismatch"

    neg = client.post(
        f"/sessions/{sid}/chunks", json={"start_seq": 0, "posteriors": [[1.2, -0.2]]}
    )
    assert neg.status_code == 422
    assert neg.json()["error"]["code"] == "negative_probability"

    zero = client.post(
        f"/sessions/{sid}/chunks", json={"start_seq": 0, "posteriors": [[0.0, 0.0]]}
    )
    assert zero.status_code == 422
    assert zero.json()["error"]["code"] == "invalid_distribution"

    empty = client.post(f"/sessions/{sid}/chunks", json={"start_seq": 0, "posteriors": []})
    assert empty.status_code == 422

    # 失败的请求没有推进会话。
    assert client.get(f"/sessions/{sid}").json()["next_seq"] == 0


def test_unknown_session(client: TestClient):
    r = client.post("/sessions/nope/chunks", json={"start_seq": 0, "posteriors": [[0.9, 0.1]]})
    assert r.status_code == 404
    assert client.get("/sessions/nope").status_code == 404


def test_direct_smoother_rejects_bad_seq():
    sm = LanguageSmoother("x", ["zh", "en"], 20, 3, 0.55)
    with pytest.raises(SessionError) as e:
        sm.submit_chunk(-1, np.array([[0.9, 0.1]]))
    assert e.value.code == "invalid_seq"
    with pytest.raises(SessionError) as e2:
        sm.submit_chunk(1, np.array([[0.9, 0.1]]))
    assert e2.value.code == "out_of_order"
