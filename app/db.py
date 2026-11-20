"""DuckDB storage layer.

Data model
----------
Each accepted upload creates a new *generation* for one round (1 or 2).
The round's tables are named ``s_gen_<g>`` (samples) and ``r_gen_<g>``
(replicate weights), so generations never overwrite each other in place.

``meta_current`` maps round -> current generation; ``meta_generations``
records provenance; ``meta_refcount`` pins generations while a query is
running so an upload cannot drop tables a query still reads.

A query always pins *both* rounds through :func:`acquire_snapshot_pair`
inside one critical section, so the two generations are guaranteed to have
been simultaneously current.  On every startup the refcount table is reset:
a process killed mid-query can never release its pins, and stale rows would
otherwise keep every later-retired generation alive forever.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import duckdb

def _data_dir() -> Path:
    return Path(os.environ.get("DATA_DIR", "/data"))


def _db_path() -> Path:
    return _data_dir() / "bus_survey.duckdb"


def _tmp_dir() -> Path:
    return _data_dir() / "tmp"


# Module-level handles are refreshed on every init_db(); helpers resolve
# paths through the functions above so test fixtures can switch DATA_DIR.
DB_PATH: Path = _db_path()
TMP_DIR: Path = _tmp_dir()
MEMORY_LIMIT = os.environ.get("DUCKDB_MEMORY_LIMIT", "180MB")
THREADS = int(os.environ.get("DUCKDB_THREADS", "2"))

_writer: duckdb.DuckDBPyConnection | None = None
_lock = threading.RLock()

# Optional test hook: invoked immediately after a pair snapshot is pinned
# (the lock has been released, the pins remain). Used to hold a query in a
# known "snapshot acquired, tables still in use" state. Inert in production.
snapshot_held_hook = None


def init_db() -> None:
    """Open the writer connection and prepare metadata tables."""
    global _writer, DB_PATH, TMP_DIR
    data_dir = _data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = _tmp_dir()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    DB_PATH = _db_path()
    TMP_DIR = tmp_dir
    if _writer is not None:
        try:
            _writer.close()
        except Exception:
            pass
    _writer = duckdb.connect(str(DB_PATH), read_only=False)
    _writer.execute(f"SET memory_limit = '{MEMORY_LIMIT}'")
    _writer.execute(f"SET threads = {THREADS}")
    _writer.execute(f"SET temp_directory = '{TMP_DIR}'")
    _writer.execute(
        """
        CREATE TABLE IF NOT EXISTS meta_generations (
            generation BIGINT PRIMARY KEY,
            round      INTEGER NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
            n_samples  BIGINT NOT NULL,
            n_replicates BIGINT NOT NULL,
            n_rep_rows  BIGINT NOT NULL
        )
        """
    )
    _writer.execute(
        """
        CREATE TABLE IF NOT EXISTS meta_current (
            round INTEGER PRIMARY KEY,
            generation BIGINT NOT NULL
        )
        """
    )
    _writer.execute(
        """
        CREATE TABLE IF NOT EXISTS meta_refcount (
            generation BIGINT PRIMARY KEY,
            refs INTEGER NOT NULL
        )
        """
    )
    # Leftover staging tables from a process killed mid-upload.
    rows = _writer.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_name LIKE 'stage\\_s\\_%' ESCAPE '\\' "
        "   OR table_name LIKE 'stage\\_r\\_%' ESCAPE '\\'"
    ).fetchall()
    for (name,) in rows:
        _writer.execute(f'DROP TABLE IF EXISTS "{name}"')
    # A process killed mid-query never ran its releases: reset every pin so
    # references left by the previous process cannot block reclamation.
    _writer.execute("DELETE FROM meta_refcount")
    # Every known generation starts with zero references after the reset.
    _writer.execute(
        "INSERT INTO meta_refcount (generation, refs) "
        "SELECT generation, 0 FROM meta_generations"
    )
    # Reclaim generations already retired before the restart. The two rounds'
    # current generations stay published and queryable; only non-current
    # generations (kept alive solely by the dead process's pins) are dropped.
    retired = _writer.execute(
        "SELECT generation FROM meta_generations g "
        "WHERE NOT EXISTS (SELECT 1 FROM meta_current c WHERE c.generation = g.generation)"
    ).fetchall()
    for (generation,) in retired:
        _drop_generation(generation)


def _check() -> duckdb.DuckDBPyConnection:
    if _writer is None:
        raise RuntimeError("database not initialised")
    return _writer


def read_connection() -> duckdb.DuckDBPyConnection:
    """Short-lived connection for one query.

    DuckDB refuses mixed read_only/read_write connections to the same file
    in one process, so readers use the same configuration as the writer and
    rely on the application layer (writers hold ``writer_lock`` and only
    mutate via :func:`commit_generation`) for isolation.
    """
    con = duckdb.connect(str(_db_path()), read_only=False)
    con.execute(f"SET memory_limit = '{MEMORY_LIMIT}'")
    con.execute(f"SET threads = {THREADS}")
    # temp_directory is database-wide: setting it again on a later connection
    # fails once the writer has spilled ("Cannot switch temporary directory
    # after the current one has been used"), so only init_db sets it.
    return con


def writer_lock():
    return _lock


def samples_table(generation: int) -> str:
    return f"s_gen_{generation}"


def replicate_table(generation: int) -> str:
    return f"r_gen_{generation}"


def current_generation(round_no: int) -> int | None:
    with _lock:
        row = _check().execute(
            "SELECT generation FROM meta_current WHERE round = ?", [round_no]
        ).fetchone()
        return int(row[0]) if row else None


def generation_info(generation: int) -> tuple | None:
    with _lock:
        return _check().execute(
            "SELECT round, created_at, n_samples, n_replicates, n_rep_rows "
            "FROM meta_generations WHERE generation = ?",
            [generation],
        ).fetchone()


def acquire_snapshot(round_no: int) -> int | None:
    """Pin the round's current generation; returns it or None if no data."""
    with _lock:
        con = _check()
        gen = current_generation(round_no)
        if gen is None:
            return None
        con.execute(
            "INSERT INTO meta_refcount VALUES (?, 1) "
            "ON CONFLICT (generation) DO UPDATE SET refs = meta_refcount.refs + 1",
            [gen],
        )
        return gen


def acquire_snapshot_pair() -> tuple[int | None, int | None]:
    """Atomically pin the two rounds' current generations for one query.

    Both reads and both refcount bumps happen in a single critical section,
    so the returned pair was simultaneously current at query start: a swap
    of either round during the query cannot mix generations that never
    coexisted. A ``None`` slot means that round has no published data yet.
    """
    with _lock:
        con = _check()
        pair = (
            current_generation(1),
            current_generation(2),
        )
        for gen in pair:
            if gen is None:
                continue
            con.execute(
                "INSERT INTO meta_refcount VALUES (?, 1) "
                "ON CONFLICT (generation) DO UPDATE SET refs = meta_refcount.refs + 1",
                [gen],
            )
    if snapshot_held_hook is not None:
        snapshot_held_hook()
    return pair


def release_snapshot(generation: int) -> None:
    with _lock:
        row = _check().execute(
            "SELECT refs FROM meta_refcount WHERE generation = ?", [generation]
        ).fetchone()
        if row is None:
            # Startup reclamation may already have removed a dead row.
            return
        refs = max(0, int(row[0]) - 1)
        _check().execute(
            "UPDATE meta_refcount SET refs = ? WHERE generation = ?",
            [refs, generation],
        )


def release_snapshot_pair(generations: tuple[int | None, int | None]) -> None:
    """Release a pair of pins and drop any generation that becomes orphaned.

    A generation is only reclaimed when it carries no references and is no
    longer a round's current generation; the current data is never dropped.
    """
    with _lock:
        for generation in generations:
            if generation is None:
                continue
            release_snapshot(generation)
            drop_if_unreferenced(generation)


def commit_generation(
    round_no: int,
    stage_samples: str,
    stage_replicates: str,
    n_samples: int,
    n_replicates: int,
    n_rep_rows: int,
) -> int:
    """Atomically promote staging tables and retire the round's old generation.

    Must run while holding ``writer_lock()`` and inside no open transaction
    (the staging work has already been committed/rolled back by the caller).
    """
    con = _check()
    new_gen = con.execute(
        "SELECT COALESCE(max(generation), 0) + 1 FROM meta_generations"
    ).fetchone()[0]
    with _lock:
        con.execute("BEGIN TRANSACTION")
        try:
            con.execute(
                f'ALTER TABLE "{stage_samples}" RENAME TO "{samples_table(new_gen)}"'
            )
            con.execute(
                f'ALTER TABLE "{stage_replicates}" RENAME TO "{replicate_table(new_gen)}"'
            )
            con.execute(
                "INSERT INTO meta_generations "
                "(generation, round, n_samples, n_replicates, n_rep_rows) "
                "VALUES (?, ?, ?, ?, ?)",
                [new_gen, round_no, n_samples, n_replicates, n_rep_rows],
            )
            old_row = con.execute(
                "SELECT generation FROM meta_current WHERE round = ?", [round_no]
            ).fetchone()
            con.execute(
                "INSERT INTO meta_current VALUES (?, ?) "
                "ON CONFLICT (round) DO UPDATE SET generation = excluded.generation",
                [round_no, new_gen],
            )
            con.execute(
                "INSERT INTO meta_refcount VALUES (?, 0) "
                "ON CONFLICT DO NOTHING",
                [new_gen],
            )
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        if old_row is not None:
            old_gen = old_row[0]
            refs = con.execute(
                "SELECT refs FROM meta_refcount WHERE generation = ?", [old_gen]
            ).fetchone()[0]
            if refs == 0:
                _drop_generation(old_gen)
            # If refs > 0 the generation is dropped lazily by release_snapshot
            # callers via drop_if_unreferenced().
        return new_gen


def drop_if_unreferenced(generation: int) -> None:
    with _lock:
        con = _check()
        row = con.execute(
            "SELECT refs FROM meta_refcount WHERE generation = ?", [generation]
        ).fetchone()
        if row is not None and int(row[0]) <= 0:
            # Only drop generations no longer current.
            current = con.execute(
                "SELECT 1 FROM meta_current WHERE generation = ?", [generation]
            ).fetchone()
            if current is None:
                _drop_generation(generation)


def _drop_generation(generation: int) -> None:
    con = _check()
    # Table names follow a fixed convention; drop them by exact name.
    con.execute(f'DROP TABLE IF EXISTS "{samples_table(generation)}"')
    con.execute(f'DROP TABLE IF EXISTS "{replicate_table(generation)}"')
    con.execute("DELETE FROM meta_refcount WHERE generation = ?", [generation])
    con.execute("DELETE FROM meta_generations WHERE generation = ?", [generation])
