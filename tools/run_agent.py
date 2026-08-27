from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

from agent import Agent
from evaluator.local_evaluator import MAX_TURNS, catalog_index, evaluate, load_jsonl


def _reject_existing(destination: Path) -> None:
    if os.path.lexists(destination):
        raise FileExistsError(f"output already exists: {destination}")


def _same_file(path: Path, expected: os.stat_result) -> bool:
    try:
        current = os.stat(path, follow_symlinks=False)
    except OSError:
        return False
    return (current.st_dev, current.st_ino) == (expected.st_dev, expected.st_ino)


def _unlink_if_same_file(path: Path, expected: os.stat_result) -> None:
    if _same_file(path, expected):
        path.unlink(missing_ok=True)


def _registered_tempfile(destination: Path) -> tuple[int, Path, os.stat_result]:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        identity = os.fstat(descriptor)
    except BaseException:
        try:
            os.close(descriptor)
        finally:
            temporary.unlink(missing_ok=True)
        raise
    return descriptor, temporary, identity


def _prove_hardlink_publication(parent: Path) -> None:
    try:
        with tempfile.TemporaryDirectory(
            prefix=".run-agent-probe.", dir=parent
        ) as probe_dir:
            probe_root = Path(probe_dir)
            descriptor, source_name = tempfile.mkstemp(prefix="source.", dir=probe_root)
            source = Path(source_name)
            try:
                try:
                    handle = os.fdopen(descriptor, "wb")
                except BaseException:
                    os.close(descriptor)
                    raise
                with handle:
                    handle.write(b"exclusive hardlink publication probe\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                linked = probe_root / "destination.link"
                os.link(source, linked)
                if not os.path.samefile(source, linked):
                    raise RuntimeError(
                        "hardlink publication preflight verification failed"
                    )
                occupied = probe_root / "occupied"
                sentinel = b"must not be overwritten\n"
                occupied.write_bytes(sentinel)
                try:
                    os.link(source, occupied)
                except FileExistsError:
                    pass
                else:
                    raise RuntimeError("exclusive hardlink publication is unavailable")
                if occupied.read_bytes() != sentinel:
                    raise RuntimeError(
                        "exclusive hardlink publication overwrote a file"
                    )
            finally:
                source.unlink(missing_ok=True)
    except RuntimeError:
        raise
    except OSError as error:
        raise RuntimeError("exclusive publication requires hardlink support") from error


def _preflight_outputs(output: str, evidence_output: str) -> tuple[Path, Path]:
    requested = (Path(output), Path(evidence_output))
    for destination in requested:
        _reject_existing(destination)

    raw, evidence = (destination.resolve() for destination in requested)
    if raw == evidence:
        raise ValueError("raw and evidence outputs must resolve to distinct paths")
    for destination in (raw, evidence):
        _reject_existing(destination)

    parents = list(dict.fromkeys((raw.parent, evidence.parent)))
    for parent in parents:
        parent.mkdir(parents=True, exist_ok=True)
    for parent in parents:
        _prove_hardlink_publication(parent)
    for destination in (raw, evidence):
        _reject_existing(destination)
    return raw, evidence


def _attempted_turns(result: object) -> list[int]:
    if not isinstance(result, dict):
        raise ValueError("official evaluation result must be an object")  # noqa: TRY004
    sessions = result.get("sessions")
    if not isinstance(sessions, list):
        raise ValueError("official evaluation sessions must be a list")  # noqa: TRY004
    sample_count = result.get("sample_count")
    if (
        not isinstance(sample_count, int)
        or isinstance(sample_count, bool)
        or sample_count < 0
        or sample_count != len(sessions)
    ):
        raise ValueError("official evaluation session count is invalid")

    attempted_turns: list[int] = []
    for session in sessions:
        if not isinstance(session, dict) or not isinstance(session.get("hit"), bool):
            raise ValueError("official evaluation session is invalid")  # noqa: TRY004
        first_hit_turn = session.get("first_hit_turn")
        if session["hit"]:
            if (
                not isinstance(first_hit_turn, int)
                or isinstance(first_hit_turn, bool)
                or not 1 <= first_hit_turn <= MAX_TURNS
            ):
                raise ValueError("official evaluation session turn is invalid")
            attempted_turns.append(first_hit_turn)
        else:
            if first_hit_turn is not None:
                raise ValueError("official evaluation session turn is invalid")
            attempted_turns.append(MAX_TURNS)
    return attempted_turns


def _trace_evidence(agent: object, result: object) -> dict[str, object]:
    attempted_turns = _attempted_turns(result)
    records = getattr(getattr(agent, "traces", None), "records", None)
    if not isinstance(records, list):
        raise ValueError("trace records must be a list")  # noqa: TRY004
    attempted_response_count = sum(attempted_turns)
    if len(records) != attempted_response_count:
        raise ValueError("trace count does not match attempted responses")

    fallback_count = 0
    fallback_counts: Counter[str] = Counter()
    record_index = 0
    seen_sessions: set[str] = set()
    for turn_count in attempted_turns:
        first_record = records[record_index]
        if not isinstance(first_record, dict):
            raise ValueError("trace records are malformed")  # noqa: TRY004
        session_id = first_record.get("session_id")
        if (
            not isinstance(session_id, str)
            or not session_id
            or session_id in seen_sessions
        ):
            raise ValueError("trace records are stale, malformed, or out of order")
        seen_sessions.add(session_id)
        for expected_turn in range(1, turn_count + 1):
            record = records[record_index]
            trace_turn = record.get("turn") if isinstance(record, dict) else None
            if (
                not isinstance(record, dict)
                or record.get("session_id") != session_id
                or not isinstance(trace_turn, int)
                or isinstance(trace_turn, bool)
                or trace_turn != expected_turn
            ):
                raise ValueError("trace records are stale, malformed, or out of order")
            fallbacks = record.get("fallbacks")
            if not isinstance(fallbacks, list):
                raise ValueError("trace fallbacks must be a list")  # noqa: TRY004
            if fallbacks:
                fallback_count += 1
                fallback_counts.update(str(item) for item in fallbacks)
            record_index += 1
    if fallback_count:
        raise ValueError("top-level trace fallbacks must be empty")

    return {
        "attempted_response_count": attempted_response_count,
        "trace_count": len(records),
        "trace_call_count_consistent": True,
        "fallback_count": fallback_count,
        "fallback_counts": dict(sorted(fallback_counts.items())),
    }


def _serialize(payload: object) -> bytes:
    return (json.dumps(payload, indent=2) + "\n").encode("utf-8")


def _publish(entries: list[tuple[Path, bytes]]) -> None:
    staged: list[tuple[Path, Path, os.stat_result]] = []
    for destination, _ in entries:
        _reject_existing(destination)
    try:
        for destination, payload in entries:
            descriptor, temporary, identity = _registered_tempfile(destination)
            staged.append((destination, temporary, identity))
            try:
                handle = os.fdopen(descriptor, "wb")
            except BaseException:
                os.close(descriptor)
                raise
            with handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())

        for destination, temporary, identity in staged:
            _reject_existing(destination)
            try:
                os.link(temporary, destination)
            except FileExistsError:
                raise
            except OSError as error:
                raise RuntimeError(
                    "exclusive publication requires hardlink support"
                ) from error
            if not _same_file(destination, identity):
                raise RuntimeError("exclusive publication verification failed")
    finally:
        for _, temporary, identity in staged:
            _unlink_if_same_file(temporary, identity)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evidence-output", required=True)
    args = parser.parse_args()
    output, evidence_output = _preflight_outputs(args.output, args.evidence_output)
    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    agent = Agent(args.catalog)
    result = evaluate(agent, samples, catalog_ids, categories, products)
    evidence = _trace_evidence(agent, result)
    raw_payload = _serialize(result)
    evidence_payload = _serialize(evidence)
    _publish([(evidence_output, evidence_payload), (output, raw_payload)])
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "sessions"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
