"""End-to-end lifecycle test: swap/query interleaving and hard-kill restart.

Exercises the real uvicorn process (single worker, same entry point as the
Docker image) against an on-disk DuckDB database:

1. two rounds are uploaded and a slow query is started;
2. while the query is in flight both rounds are swapped, twice, and the
   service is SIGKILL-ed (no graceful shutdown, no snapshot release);
3. the service is restarted on the same data directory: retired tables must
   have been reclaimed automatically while the two current generations stay
   queryable; a further swap after restart must reclaim immediately again.

Run:
  python scripts/lifecycle_test.py --rows 2000000 --out /tmp/lifecycle
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import duckdb  # noqa: E402

from tests.factories import make_large  # noqa: E402

PORT = 8131
BASE = f"http://127.0.0.1:{PORT}"


def post_files(round_no: int, csv_path: Path, pq_path: Path) -> tuple[int, dict]:
    boundary = "----lifeboundary" + os.urandom(8).hex()
    body = b""
    for field, path in (("sample_csv", csv_path), ("replicate_parquet", pq_path)):
        body += f"--{boundary}\r\n".encode()
        body += (
            f'Content-Disposition: form-data; name="{field}"; '
            f'filename="{path.name}"\r\n\r\n'
        ).encode()
        body += path.read_bytes() + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{BASE}/api/rounds/{round_no}",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


def get_json(path: str, timeout: float = 600) -> dict:
    with urllib.request.urlopen(BASE + path, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def post_json(path: str) -> dict:
    req = urllib.request.Request(BASE + path, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def wait_hold_entered(timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if get_json("/api/test/hold/entered", timeout=10)["entered"]:
            return
        time.sleep(0.3)
    raise RuntimeError("held query never reported entry")


def wait_ready(timeout: float = 90.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            get_json("/api/status", timeout=5)
            return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("server did not become ready")


def start_server(data_dir: Path, log_path: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env.update({
        "DATA_DIR": str(data_dir),
        "UPLOAD_TMP_DIR": str(data_dir / "tmp"),
        "DUCKDB_MEMORY_LIMIT": "180MB",
        "DUCKDB_THREADS": "2",
        "ENABLE_TEST_HOOKS": "1",
        "PYTHONPATH": str(ROOT),
    })
    log = open(log_path, "wb")
    return subprocess.Popen(
        [
            sys.executable, "-I", "-m", "uvicorn",
            "app.main:app", "--host", "127.0.0.1", "--port", str(PORT),
            "--workers", "1",
        ],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        cwd=str(ROOT),
    )


def table_names_offline(db_path: Path) -> set[str]:
    """Direct catalog read; only valid while no server holds the file lock."""
    con = duckdb.connect(str(db_path), read_only=False)
    try:
        return {
            r[0]
            for r in con.execute(
                "SELECT table_name FROM information_schema.tables"
            ).fetchall()
        }
    finally:
        con.close()


def debug_tables() -> set[str]:
    return set(get_json("/api/debug/generations")["tables"])


def gen_tables(gen: int) -> set[str]:
    return {f"s_gen_{gen}", f"r_gen_{gen}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2_000_000)
    ap.add_argument("--out", default="/tmp/bus_lifecycle")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data_dir = out / "data"
    data_dir.mkdir(exist_ok=True)
    db_path = data_dir / "bus_survey.duckdb"

    n_reps = min(500, args.rows)
    n_samples = max(args.rows // n_reps, 1)
    print(f"[plan] {n_samples} samples x {n_reps} reps per generation")

    # Two independently valid generations per round (different seeds ->
    # different content, same schema and rep-id space).
    print("[gen] building datasets ...")
    r1a = make_large(out / "r1a", n_samples, n_reps, seed=7)
    r2a = make_large(out / "r2a", n_samples, n_reps, seed=23)
    r1b = make_large(out / "r1b", n_samples, n_reps, seed=71)
    r2b = make_large(out / "r2b", n_samples, n_reps, seed=113)

    proc = start_server(data_dir, out / "server1.log")
    try:
        wait_ready()
        print(f"[srv] first instance pid={proc.pid} ready")

        code, p1a = post_files(1, *r1a)
        assert code == 200, p1a
        code, p2a = post_files(2, *r2a)
        assert code == 200, p2a
        g1a, g2a = p1a["generation"], p2a["generation"]
        print(f"[up] initial generations: round1={g1a} round2={g2a}")

        status = get_json("/api/status")
        district = status["districts"][0]
        baseline = get_json(
            "/api/result?" + urllib.parse.urlencode({"district": district})
        )
        assert baseline["available"]
        print(
            f"[q] baseline {district}: point={baseline['point_estimate']:.6f} "
            f"ci=[{baseline['ci_low']:.6f},{baseline['ci_high']:.6f}]"
        )

        # Hold every new query right after it pins its snapshot pair, so the
        # swaps below are guaranteed to happen while the query is in flight.
        post_json("/api/test/hold/enable")

        query_result: dict = {}

        def held_query():
            try:
                query_result["payload"] = get_json(
                    "/api/result?" + urllib.parse.urlencode({"district": district}),
                    timeout=600,
                )
            except Exception as exc:  # the process will be killed mid-request
                query_result["error"] = repr(exc)

        import threading

        qt = threading.Thread(target=held_query, daemon=True)
        qt.start()
        wait_hold_entered()
        print("[hold] query in flight with its snapshot pair pinned")

        code, p1b = post_files(1, *r1b)
        assert code == 200, p1b
        code, p2b = post_files(2, *r2b)
        assert code == 200, p2b
        g1b, g2b = p1b["generation"], p2b["generation"]
        print(f"[swap] while query runs: round1={g1b} round2={g2b}")

        names = debug_tables()
        assert gen_tables(g1a) <= names and gen_tables(g2a) <= names, (
            "retired generations must stay readable for the in-flight query"
        )
        assert gen_tables(g1b) <= names and gen_tables(g2b) <= names
        print("[pin] all four generations present during in-flight query")

        # Let the blocked query finish: it must report the start-of-query pair.
        post_json("/api/test/hold/release")
        qt.join(60)
        assert not qt.is_alive()
        assert "payload" in query_result, query_result
        done = query_result["payload"]
        assert (done["round1"]["generation"], done["round2"]["generation"]) == (
            g1a,
            g2a,
        ), "in-flight query mixed generations that were never simultaneously current"
        assert abs(done["point_estimate"] - baseline["point_estimate"]) < 1e-12
        assert abs(done["ci_low"] - baseline["ci_low"]) < 1e-12
        assert abs(done["ci_high"] - baseline["ci_high"]) < 1e-12
        print("[query] released query returned its start-of-query pair unchanged")

        # A fresh query now sees the new pair.
        after_swap = get_json(
            "/api/result?" + urllib.parse.urlencode({"district": district})
        )
        assert after_swap["round1"]["generation"] == g1b
        assert after_swap["round2"]["generation"] == g2b

        # Start another held query, then hard-kill the process mid-request:
        # no lifespan shutdown, snapshot pins are never released.
        post_json("/api/test/hold/enable")
        def doomed_query():
            # Expected: the server is SIGKILL-ed before answering.
            try:
                get_json(
                    "/api/result?"
                    + urllib.parse.urlencode({"district": district})
                )
            except Exception:
                pass

        qt2 = threading.Thread(target=doomed_query, daemon=True)
        qt2.start()
        wait_hold_entered()
        held_debug = get_json("/api/debug/generations")
        held_refs = {g["generation"]: g["refs"] for g in held_debug["generations"]}
        assert held_refs[g1b] >= 1 and held_refs[g2b] >= 1, held_debug
        proc.send_signal(signal.SIGKILL)
        proc.wait(20)
        print(f"[kill] pid={proc.pid} terminated with SIGKILL mid-query")
    finally:
        if proc.poll() is None:
            proc.kill()

    # While no process holds the database file, confirm the hard-killed
    # process left the generations pinned by its in-flight query on disk.
    names_killed = table_names_offline(db_path)
    assert gen_tables(g1b) <= names_killed and gen_tables(g2b) <= names_killed, (
        "pinned generations must remain on disk until restart reclaims them"
    )
    print("[killed] pinned generations still on disk (awaiting restart reclamation)")

    # Restart on the same data directory (same Docker volume semantics).
    proc = start_server(data_dir, out / "server2.log")
    try:
        wait_ready()
        print(f"[srv] restarted instance pid={proc.pid} ready")

        names = debug_tables()
        # Retired generations held only by the dead process are reclaimed.
        assert not (gen_tables(g1a) & names), "retired round1 tables survive restart"
        assert not (gen_tables(g2a) & names), "retired round2 tables survive restart"
        # The two current generations are intact.
        assert gen_tables(g1b) <= names and gen_tables(g2b) <= names
        debug = get_json("/api/debug/generations")
        assert all(g["refs"] == 0 for g in debug["generations"]), debug
        assert {g["generation"] for g in debug["generations"] if g["current"]} == {
            g1b,
            g2b,
        }
        print("[reclaim] retired generations dropped on startup; current kept")

        status = get_json("/api/status")
        assert status["rounds"]["1"]["generation"] == g1b
        assert status["rounds"]["2"]["generation"] == g2b
        post_restart = get_json(
            "/api/result?" + urllib.parse.urlencode({"district": district})
        )
        assert post_restart["available"]
        assert post_restart["round1"]["generation"] == g1b
        assert post_restart["round2"]["generation"] == g2b
        print(
            f"[q] post-restart {district}: point={post_restart['point_estimate']:.6f}"
        )

        # Stale pins must not block reclamation of future retirements.
        code, p1c = post_files(1, *r1a)  # re-upload the first dataset
        assert code == 200, p1c
        g1c = p1c["generation"]
        names = debug_tables()
        assert not (gen_tables(g1b) & names), (
            "swap after restart failed to reclaim the retired generation"
        )
        assert gen_tables(g1c) <= names
        print(f"[swap] post-restart swap reclaimed {g1b} -> published {g1c}")

        # Failed swap still leaves the current data queryable.
        bad_pq = out / "bad.parquet"
        dc = duckdb.connect()
        try:
            dc.execute(
                "CREATE TEMP TABLE b AS "
                "SELECT * FROM read_parquet(?) LIMIT 50",
                [str(r2b[1])],
            )
            dc.execute("COPY b TO ? (FORMAT PARQUET)", [str(bad_pq)])
        finally:
            dc.close()
        code, payload = post_files(2, r2b[0], bad_pq)
        assert code == 422, payload
        still = get_json(
            "/api/result?" + urllib.parse.urlencode({"district": district})
        )
        assert still["round2"]["generation"] == g2b
        print("[swap-fail] invalid upload rejected; current generation unchanged")

        print("[OK] lifecycle checks passed")
        return 0
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(20)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
