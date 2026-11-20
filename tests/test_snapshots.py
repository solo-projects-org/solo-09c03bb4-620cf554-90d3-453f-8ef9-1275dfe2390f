"""Snapshot-pinning, swap/query interleaving and restart reclamation tests."""
from __future__ import annotations

import threading

import duckdb
import pytest

from tests.factories import upload, write_csv, write_parquet


def _table_names(db_path) -> set[str]:
    con = duckdb.connect(str(db_path))
    try:
        return {
            row[0]
            for row in con.execute(
                "SELECT table_name FROM information_schema.tables"
            ).fetchall()
        }
    finally:
        con.close()


def _new_round1_files(tmp_path, tag):
    csv_path = tmp_path / f"r1_{tag}.csv"
    pq_path = tmp_path / f"r1_{tag}.parquet"
    write_csv(csv_path, [
        ("a1", "A", 1, 3.0), ("a2", "A", 0, 7.0), ("b1", "B", 1, 4.0),
    ])
    write_parquet(pq_path, [
        ("a1", "rep1", 4.0), ("a2", "rep1", 6.0), ("b1", "rep1", 4.0),
        ("a1", "rep2", 2.0), ("a2", "rep2", 8.0), ("b1", "rep2", 4.0),
    ])
    return csv_path, pq_path


def _new_round2_files(tmp_path, tag):
    csv_path = tmp_path / f"r2_{tag}.csv"
    pq_path = tmp_path / f"r2_{tag}.parquet"
    write_csv(csv_path, [
        ("c1", "A", 1, 6.0), ("c2", "A", 1, 4.0), ("d1", "B", 0, 5.0),
    ])
    write_parquet(pq_path, [
        ("c1", "rep1", 5.0), ("c2", "rep1", 5.0), ("d1", "rep1", 5.0),
        ("c1", "rep2", 7.0), ("c2", "rep2", 3.0), ("d1", "rep2", 5.0),
    ])
    return csv_path, pq_path


def test_snapshot_pair_pins_both_rounds_through_two_swaps(app_env, paths, tmp_path):
    """Both old generations survive swaps of either round until the query ends."""
    from app import analysis, db

    client, data_dir = app_env
    r1_csv, r1_pq, r2_csv, r2_pq = paths
    assert upload(client, 1, r1_csv, r1_pq).status_code == 200
    assert upload(client, 2, r2_csv, r2_pq).status_code == 200

    gen1_before = db.current_generation(1)
    gen2_before = db.current_generation(2)
    expected = analysis.compute("A")

    entered = threading.Event()
    release = threading.Event()
    real_main_proportion = analysis._main_proportion

    def blocking_main_proportion(con, gen, district):
        if district == "A":
            entered.set()
            release.wait(5)
        return real_main_proportion(con, gen, district)

    analysis._main_proportion = blocking_main_proportion
    try:
        result_holder = {}

        def run_query():
            result_holder["payload"] = analysis.compute("A")

        query = threading.Thread(target=run_query)
        query.start()
        assert entered.wait(5), "query never entered the pinned read section"

        # Swap both rounds while the query holds its pair snapshot.
        r1_new_csv, r1_new_pq = _new_round1_files(tmp_path, "n1")
        r2_new_csv, r2_new_pq = _new_round2_files(tmp_path, "n2")
        assert upload(client, 1, r1_new_csv, r1_new_pq).status_code == 200
        assert upload(client, 2, r2_new_csv, r2_new_pq).status_code == 200

        gen1_after = db.current_generation(1)
        gen2_after = db.current_generation(2)
        assert (gen1_after, gen2_after) != (gen1_before, gen2_before)

        names = _table_names(data_dir / "bus_survey.duckdb")
        # The in-flight query still reads both retired generations.
        for gen in (gen1_before, gen2_before, gen1_after, gen2_after):
            assert f"s_gen_{gen}" in names
            assert f"r_gen_{gen}" in names
    finally:
        release.set()
        analysis._main_proportion = real_main_proportion
        query.join(5)

    assert not query.is_alive()
    # The answer is computed from the start-of-query pair, unaffected by swaps.
    assert result_holder["payload"]["round1"]["generation"] == gen1_before
    assert result_holder["payload"]["round2"]["generation"] == gen2_before
    assert result_holder["payload"]["point_estimate"] == pytest.approx(
        expected["point_estimate"]
    )
    assert result_holder["payload"]["ci_low"] == pytest.approx(expected["ci_low"])
    assert result_holder["payload"]["ci_high"] == pytest.approx(expected["ci_high"])

    # After the query ends, the retired generations are reclaimed; the two
    # current generations remain.
    names = _table_names(data_dir / "bus_survey.duckdb")
    for gen in (gen1_before, gen2_before):
        assert f"s_gen_{gen}" not in names
        assert f"r_gen_{gen}" not in names
    for gen in (gen1_after, gen2_after):
        assert f"s_gen_{gen}" in names
        assert f"r_gen_{gen}" in names


def test_snapshot_pair_always_a_coexistent_pair(app_env, paths, tmp_path):
    """Under repeated interleaved swaps every pinned pair was once current."""
    from app import db

    client, _ = app_env
    r1_csv, r1_pq, r2_csv, r2_pq = paths
    upload(client, 1, r1_csv, r1_pq)
    upload(client, 2, r2_csv, r2_pq)

    # Record every (round1, round2) generation pair that is ever published.
    history: set[tuple[int, int]] = set()
    history.add((db.current_generation(1), db.current_generation(2)))
    real_commit = db.commit_generation

    def recording_commit(round_no, *args, **kwargs):
        new_gen = real_commit(round_no, *args, **kwargs)
        history.add((db.current_generation(1), db.current_generation(2)))
        return new_gen

    db.commit_generation = recording_commit
    try:
        seen_pairs: set[tuple[int, int]] = set()

        def pin_loop(stop: threading.Event):
            while not stop.is_set():
                pair = db.acquire_snapshot_pair()
                seen_pairs.add(pair)
                db.release_snapshot_pair(pair)

        stop = threading.Event()
        pinner = threading.Thread(target=pin_loop, args=(stop,))
        pinner.start()

        r1_new = _new_round1_files(tmp_path, "p1")
        r2_new = _new_round2_files(tmp_path, "p2")
        for _ in range(10):
            assert upload(client, 1, *r1_new).status_code == 200
            assert upload(client, 2, *r2_new).status_code == 200

        stop.set()
        pinner.join(5)
    finally:
        db.commit_generation = real_commit

    assert seen_pairs, "pinner never acquired a pair"
    # No None slots once both rounds are loaded, and every observed pair is
    # one that was actually published (no mixed, never-coexistent generations).
    for pair in seen_pairs:
        assert pair[0] is not None and pair[1] is not None
        assert pair in history, f"pinned pair {pair} was never simultaneously current"


def test_restart_reclaims_pinned_retired_generations(app_env, paths, tmp_path):
    """A process killed mid-query leaves pins behind; restart must reclaim them."""
    from app import db

    client, data_dir = app_env
    r1_csv, r1_pq, r2_csv, r2_pq = paths
    upload(client, 1, r1_csv, r1_pq)
    upload(client, 2, r2_csv, r2_pq)
    gen1_first = db.current_generation(1)
    gen2_first = db.current_generation(2)

    # A long-running query pins the first generations.
    pinned = db.acquire_snapshot_pair()
    assert pinned == (gen1_first, gen2_first)

    # Both rounds are swapped while the dead query still holds its pins.
    r1_new_csv, r1_new_pq = _new_round1_files(tmp_path, "r1x")
    r2_new_csv, r2_new_pq = _new_round2_files(tmp_path, "r2x")
    assert upload(client, 1, r1_new_csv, r1_new_pq).status_code == 200
    assert upload(client, 2, r2_new_csv, r2_new_pq).status_code == 200
    gen1_now = db.current_generation(1)
    gen2_now = db.current_generation(2)

    names_before = _table_names(data_dir / "bus_survey.duckdb")
    for gen in (gen1_first, gen2_first, gen1_now, gen2_now):
        assert f"s_gen_{gen}" in names_before

    # Simulate SIGKILL: the process vanishes without releasing the snapshot.
    # A fresh process opens the same database file and runs startup recovery.
    db._writer.close()
    db._writer = None
    db.init_db()

    names_after = _table_names(data_dir / "bus_survey.duckdb")
    # Retired tables kept alive only by the dead process's pins are gone.
    for gen in (gen1_first, gen2_first):
        assert f"s_gen_{gen}" not in names_after
        assert f"r_gen_{gen}" not in names_after
    # The two current generations survived and stay queryable.
    for gen in (gen1_now, gen2_now):
        assert f"s_gen_{gen}" in names_after
        assert f"r_gen_{gen}" in names_after
    assert db.current_generation(1) == gen1_now
    assert db.current_generation(2) == gen2_now

    # Refcounts start clean: a subsequent swap must reclaim immediately again.
    r1_next_csv, r1_next_pq = _new_round1_files(tmp_path, "r1y")
    assert upload(client, 1, r1_next_csv, r1_next_pq).status_code == 200
    names_final = _table_names(data_dir / "bus_survey.duckdb")
    assert f"s_gen_{gen1_now}" not in names_final
    assert f"r_gen_{gen1_now}" not in names_final

    # The live data still answers queries through the HTTP entry point.
    payload = client.get("/api/result", params={"district": "A"}).json()
    assert payload["available"] is True
    assert payload["round1"]["generation"] == db.current_generation(1)
    assert payload["round2"]["generation"] == gen2_now


def test_restart_keeps_data_when_no_swap_ever_happened(app_env, paths):
    """Current generations survive a restart even if a query was in flight."""
    from app import db

    client, data_dir = app_env
    r1_csv, r1_pq, r2_csv, r2_pq = paths
    upload(client, 1, r1_csv, r1_pq)
    upload(client, 2, r2_csv, r2_pq)
    gen1, gen2 = db.current_generation(1), db.current_generation(2)

    db.acquire_snapshot_pair()  # pins never released, simulating a hard kill
    db._writer.close()
    db._writer = None
    db.init_db()

    names = _table_names(data_dir / "bus_survey.duckdb")
    assert {f"s_gen_{gen1}", f"r_gen_{gen1}", f"s_gen_{gen2}", f"r_gen_{gen2}"} <= names
    payload = client.get("/api/result", params={"district": "A"}).json()
    assert payload["available"] is True
    assert payload["round1"]["generation"] == gen1
    assert payload["round2"]["generation"] == gen2


def test_test_hooks_disabled_by_default(app_env):
    """The query-hold control surface must not exist in a normal deployment."""
    from app import main

    assert main.TEST_HOOKS_ENABLED is False
    client, _ = app_env
    assert client.post("/api/test/hold/enable").status_code == 404
    assert client.get("/api/test/hold/entered").status_code == 404
    assert client.post("/api/test/hold/release").status_code == 404
    # The debug catalog stays available for operational inspection.
    assert client.get("/api/debug/generations").status_code == 200


def test_failed_swap_during_held_query_keeps_pinned_data(app_env, paths, tmp_path):
    """An invalid upload mid-query neither publishes nor disturbs the snapshot."""
    from app import db

    client, data_dir = app_env
    r1_csv, r1_pq, r2_csv, r2_pq = paths
    upload(client, 1, r1_csv, r1_pq)
    upload(client, 2, r2_csv, r2_pq)
    gen1, gen2 = db.current_generation(1), db.current_generation(2)

    held = threading.Event()
    release = threading.Event()

    def hook():
        held.set()
        release.wait(5)

    db.snapshot_held_hook = hook
    try:
        payload: dict = {}

        def query():
            from app import analysis
            payload["result"] = analysis.compute("A")

        t = threading.Thread(target=query)
        t.start()
        assert held.wait(5)

        # Invalid round-2 replacement: replicate coverage hole.
        bad_pq = tmp_path / "bad_inflight.parquet"
        write_parquet(bad_pq, [
            ("t1", "rep1", 5.0), ("t2", "rep1", 5.0), ("t3", "rep1", 5.0),
            ("t1", "rep2", 6.0),  # t2/t3 missing for rep2
        ])
        resp = upload(client, 2, r2_csv, bad_pq)
        assert resp.status_code == 422

        names = _table_names(data_dir / "bus_survey.duckdb")
        assert f"s_gen_{gen2}" in names and f"r_gen_{gen2}" in names
        # No staging tables survive the rolled-back upload.
        assert not {n for n in names if n.startswith("stage_")}
        assert db.current_generation(2) == gen2
    finally:
        db.snapshot_held_hook = None
        release.set()
        t.join(5)

    assert not t.is_alive()
    assert payload["result"]["round1"]["generation"] == gen1
    assert payload["result"]["round2"]["generation"] == gen2
    assert payload["result"]["available"] is True
