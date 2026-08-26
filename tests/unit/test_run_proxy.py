from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.proxy_dataset import sha256_file


def _sample(scenario: str = "buying") -> dict:
    behavior = {}
    if scenario == "intent_override":
        behavior = {
            "override": {
                "turn": 3,
                "old_value": "soft blue",
                "new_value": "cotton",
                "message": "Actually, ignore my earlier preference. What I need is: cotton.",
            }
        }
    return {
        "sample_id": "proxy_sample_1",
        "scenario_type": scenario,
        "user_profile": {"summary": "safe"},
        "ground_truth": {"parent_asin": "A"},
        "intent_card": {
            "hard_constraints": ["cotton"],
            "soft_preferences": ["soft blue"],
        },
        "behavior": behavior,
    }


def test_proxy_dialogue_variant_is_deterministic_and_discloses_buying_hard_constraint():
    from tools.run_proxy import ProxyDialogue

    sample = _sample()
    first_disclosed: set[str] = set()
    second_disclosed: set[str] = set()
    first = ProxyDialogue(sample, 1).initial_message("shirts", first_disclosed)
    second = ProxyDialogue(sample, 1).initial_message("shirts", second_disclosed)

    assert "cotton" in first
    assert first == second
    assert first_disclosed == second_disclosed == {"cotton"}


def test_proxy_dialogue_variants_change_wording_not_disclosed_state():
    from tools.run_proxy import ProxyDialogue

    sample = _sample()
    messages: list[str] = []
    disclosed: list[set[str]] = []
    for variant in range(4):
        state: set[str] = set()
        messages.append(ProxyDialogue(sample, variant).initial_message("shirts", state))
        disclosed.append(state)

    assert len(set(messages)) == 4
    assert disclosed == [{"cotton"}] * 4


def test_proxy_dialogue_boundary_only_uses_no_preference_once():
    from tools.run_proxy import ProxyDialogue

    dialogue = ProxyDialogue(_sample("boundary"), 0)
    disclosed: set[str] = set()
    first, boundary_used = dialogue.customer_reply("material", disclosed, False)
    second, later_used = dialogue.customer_reply("color", disclosed, boundary_used)

    assert "don't have a preference" in first
    assert boundary_used is True
    assert "soft blue" in second
    assert later_used is True
    assert disclosed == {"soft blue"}


def test_proxy_dialogue_classifies_and_discloses_like_official_reply():
    from tools.run_proxy import ProxyDialogue

    dialogue = ProxyDialogue(_sample(), 2)
    disclosed: set[str] = set()
    material, used = dialogue.customer_reply("material", disclosed, False)
    other, used = dialogue.customer_reply("unknown", disclosed, used)

    assert "cotton" in material
    assert disclosed == {"cotton", "soft blue"}
    assert "soft blue" in other
    assert used is False


def test_override_initial_message_keeps_new_value_hidden():
    from tools.run_proxy import ProxyDialogue

    disclosed: set[str] = set()
    message = ProxyDialogue(_sample("intent_override"), 3).initial_message("shirts", disclosed)

    assert "shirts" in message
    assert "soft blue" in message
    assert "cotton" not in message
    assert disclosed == set()


def test_write_audit_report_is_aggregate_only_and_exclusive(tmp_path: Path):
    from tools.run_proxy import write_audit_report

    destination = tmp_path / "audit" / "baseline.json"
    result = {"hit_rate_at_10": 1.0, "sessions": [{"sample_id": "secret"}]}
    write_audit_report(destination, result, {"suite": "representative"})

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload == {"suite": "representative", "aggregate": {"hit_rate_at_10": 1.0}}
    with pytest.raises(FileExistsError):
        write_audit_report(destination, result, {"suite": "representative"})


def test_load_proxy_suite_validates_manifest_hash_and_count(tmp_path: Path):
    from tools.run_proxy import load_proxy_suite

    rows = [{"sample_id": "one"}, {"sample_id": "two"}]
    dataset = tmp_path / "representative.jsonl"
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest = {
        "counts": {"representative": 2, "stress": 0},
        "output_hashes": {"representative.jsonl": sha256_file(dataset), "stress.jsonl": "unused"},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    loaded_manifest, loaded_rows = load_proxy_suite(tmp_path, "representative")
    assert loaded_manifest == manifest
    assert loaded_rows == rows
    dataset.write_text(dataset.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash|count"):
        load_proxy_suite(tmp_path, "representative")


def test_select_proxy_rows_enforces_dev_audit_and_stress_guards():
    from tools.run_proxy import select_proxy_rows

    representative = [{"sample_id": f"r{fold}", "proxy_fold": fold} for fold in range(1, 6)]
    stress = [{"sample_id": "s1"}, {"sample_id": "s2"}]

    assert [fold for fold, _ in select_proxy_rows(representative, "representative", [1, 4], None)] == [1, 4]
    assert select_proxy_rows(representative, "representative", None, "baseline") == [(5, [representative[-1]])]
    assert select_proxy_rows(stress, "stress", [1], None) == [(None, stress)]
    with pytest.raises(ValueError):
        select_proxy_rows(representative, "representative", [5], None)
    with pytest.raises(ValueError):
        select_proxy_rows(stress, "stress", None, "baseline")
    with pytest.raises(ValueError):
        select_proxy_rows([], "stress", None, None)


def test_evaluate_proxy_matches_official_metrics_and_counts_invalid_responses():
    from tools.run_proxy import evaluate_proxy

    class ScriptedAgent:
        def __init__(self):
            self.calls = 0
            self.session_id = ""

        def reset(self, session_id: str, user_profile: dict) -> None:
            self.session_id = session_id

        def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> object:
            self.calls += 1
            if self.calls == 1:
                return "not a response"
            return {
                "message": "ok",
                "ask_attribute": None,
                "recommendations": ["A"],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            }

    sample = _sample()
    agent = ScriptedAgent()
    result = evaluate_proxy(agent, [sample], {"A"}, {"A": ["Clothing", "Shirts"]}, {"A": {"parent_asin": "A"}})

    expected_id = "proxy_eval_" + hashlib.sha256(b"proxy_sample_1").hexdigest()[:20]
    assert agent.session_id == expected_id
    assert result["invalid_response_count"] == 1
    assert result["reported_token_usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    assert result["hit_rate_at_10"] == result["mrr"] == 1.0
    assert result["mttc"] == 2.0
    assert result["efficiency"] == 0.9
    assert result["recommended_technical_score"] == 0.98
    assert result["sessions"] == [{"sample_id": "proxy_sample_1", "scenario_type": "buying", "hit": True, "first_hit_turn": 2, "best_rank": 1, "reciprocal_rank": 1.0}]


def test_evaluate_proxy_counts_exception_once_and_preserves_later_usage():
    from tools.run_proxy import evaluate_proxy

    class ExceptionThenHitAgent:
        def reset(self, session_id: str, user_profile: dict) -> None:
            pass

        def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
            if turn == 1:
                raise RuntimeError("temporary failure")
            return {
                "message": "ok",
                "ask_attribute": None,
                "recommendations": ["A"],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }

    result = evaluate_proxy(
        ExceptionThenHitAgent(),
        [_sample()],
        {"A"},
        {"A": ["Clothing", "Shirts"]},
        {"A": {"parent_asin": "A"}},
    )

    assert result["invalid_response_count"] == 1
    assert result["reported_token_usage"] == {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}


def test_frozen_evaluator_hash_is_unchanged():
    assert sha256_file(Path("evaluator/local_evaluator.py")) == "84ea899707452de249ca62abee77c4b40ab7a3139b5cc798ac30c9f521f91b30"
