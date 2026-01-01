# tests/test_paper_cases.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import pytest

# Make "src" importable when running pytest from repo root or tests/
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.engine import CatelingoEngine, Tri
from src.loader import load_scenario_json, load_validator_yaml


def _load_engine() -> CatelingoEngine:
    # Prefer repo data/validator.yaml; fall back to repo root validator.yaml
    cand = [
        REPO_ROOT / "data" / "validator.yaml",
        REPO_ROOT / "validator.yaml",
    ]
    for path in cand:
        if path.exists():
            spec = load_validator_yaml(path)
            return CatelingoEngine(spec)
    raise FileNotFoundError("validator.yaml not found (expected data/validator.yaml or validator.yaml)")


@pytest.fixture(scope="session")
def engine() -> CatelingoEngine:
    return _load_engine()


def _iter_scenario_json_paths() -> list[Path]:
    scenarios_dir = REPO_ROOT / "scenarios"
    if not scenarios_dir.exists():
        raise FileNotFoundError(
            f"scenarios directory not found: {scenarios_dir}\n"
            f"Create it and put JSON scenarios under it (e.g., scenarios/paper_examples/*.json)."
        )

    paths = sorted(p for p in scenarios_dir.rglob("*.json") if p.is_file())
    if not paths:
        raise FileNotFoundError(f"No .json files found under: {scenarios_dir}")
    return paths


def _assert_expected(res, expected: dict, *, json_path: Path) -> None:
    # --- TRI ---
    exp_tri = expected.get("tri")
    if not exp_tri:
        raise AssertionError(f"{json_path}: missing expected.tri")
    try:
        exp_tri_enum = Tri(exp_tri)
    except Exception:
        raise AssertionError(f"{json_path}: invalid expected.tri={exp_tri!r} (must be one of SAT/UNSAT/UNKNOWN)")

    assert res.tri == exp_tri_enum, (
        f"{json_path}: tri mismatch (expected={exp_tri_enum.value}, got={res.tri.value})"
    )

    applied_ids = {r.constraint_id for r in res.applied}

    # --- constraints that must appear ---
    must_in: Iterable[str] = expected.get("must_include_constraints", []) or []
    for cid in must_in:
        assert cid in applied_ids, (
            f"{json_path}: expected constraint_id not applied: {cid}. applied={sorted(applied_ids)}"
        )

    # --- constraints that must NOT appear ---
    must_not: Iterable[str] = expected.get("must_not_include_constraints", []) or []
    for cid in must_not:
        assert cid not in applied_ids, (
            f"{json_path}: unexpected constraint_id applied: {cid}. applied={sorted(applied_ids)}"
        )

@pytest.mark.parametrize("json_path", _iter_scenario_json_paths(), ids=lambda p: str(Path(p).relative_to(REPO_ROOT)))
def test_scenario_json(engine: CatelingoEngine, json_path: Path) -> None:
    inp = load_scenario_json(json_path)
    res = engine.verify(inp)

    expected = inp.get("expected")
    if expected is None:
        raise AssertionError(f"{json_path}: missing top-level 'expected' block")

    _assert_expected(res, expected, json_path=json_path)

