"""The two config files, and the difference between them.

The manifest and the runtime config look alike and are not. A reader pointed at
the manifest gets an empty mapping back and every declared value quietly
reverts to a built-in default — the agent runs, the tests pass, and nothing
that was configured is in effect. These tests pin the split.
"""

import pathlib

import pytest
import yaml

from framework.schemas.invocation_context import InvocationContext, TrustLevel
from src.graph.graph import GovernmentProcedureQAAgent, runtime_config

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_MANIFEST = _REPO_ROOT / "config" / "agent.yaml"
_RUNTIME = _REPO_ROOT / "config" / "config.yaml"


def _load(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# ── the manifest carries identity only ───────────────────────


def test_manifest_keys_are_all_at_root():
    manifest = _load(_MANIFEST)
    assert "agent" not in manifest, "the registry reads root level; a nested block is invisible"
    for key in (
        "id",
        "name",
        "namespace",
        "version",
        "enabled",
        "category",
        "generation_mode",
        "industry",
        "base_type",
        "class",
        "required_trust_level",
        "requires",
    ):
        assert key in manifest, f"manifest is missing {key}"


def test_manifest_class_path_actually_imports():
    """The entry point is a real dotted path, not a description of one."""
    import importlib

    module_path, _, class_name = _load(_MANIFEST)["class"].rpartition(".")
    assert getattr(importlib.import_module(module_path), class_name) is GovernmentProcedureQAAgent


def test_manifest_declares_no_secrets_because_none_are_required():
    """Declaring an unprovisioned secret makes the agent fail at compile time.

    Nothing in this template calls for a secret, so the list is empty — and it
    has to stay empty for as long as that is true.
    """
    assert _load(_MANIFEST)["requires"]["secrets"] == []
    src = _REPO_ROOT / "src"
    for path in src.rglob("*.py"):
        assert "secrets.require(" not in path.read_text(encoding="utf-8"), path


def test_manifest_carries_no_runtime_parameters():
    """Runtime tuning in the manifest is tuning nothing — nobody reads it there."""
    manifest = _load(_MANIFEST)
    for key in ("retrieval", "egov_corpus_path", "municipal_kb_path", "enable_mynumber_block", "eligibility_mode"):
        assert key not in manifest


# ── the runtime config carries the live values ───────────────


def test_runtime_config_declares_every_value_the_code_reads():
    config = _load(_RUNTIME)
    assert config["retrieval"]["top_k"]
    assert config["retrieval"]["score_threshold"] is not None
    assert config["egov_corpus_path"]
    assert config["municipal_kb_path"]
    assert config["enable_mynumber_block"] is True
    assert config["eligibility_mode"] == "indeterminate"
    assert config["security"]["s3_gate_enabled"] is True


def test_runtime_config_declares_nothing_the_code_ignores():
    """A shipped value nothing reads is worse than no value at all.

    It reads as configuration to whoever finds it, and changing it does nothing.
    """
    declared = set(_load(_RUNTIME))
    src_text = "\n".join(p.read_text(encoding="utf-8") for p in (_REPO_ROOT / "src").rglob("*.py"))
    for key in declared:
        assert key in src_text, f"config declares {key!r} but no code reads it"


def test_loader_reads_the_runtime_file(monkeypatch, tmp_path):
    """runtime_config() resolves against the repo root, not the working directory."""
    monkeypatch.chdir(tmp_path)
    assert runtime_config()["egov_corpus_path"]


# ── a declared value visibly reaches the pipeline ────────────


def _run(config, question="転入届 必要書類 提出窓口"):
    agent = GovernmentProcedureQAAgent(config=config)
    agent.compile()
    return agent.invoke(question, ctx=InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL))


def test_declared_threshold_reaches_the_retrieval_nodes():
    """Raise the floor and retrieval visibly stops matching.

    Asserting that the value is *present* in the config proves nothing; the
    failure this guards against is a value that is declared, loaded, and then
    never handed to the node that would act on it.
    """
    permissive = _run({**runtime_config(), "retrieval": {"top_k": 5, "score_threshold": 0.2}})
    assert permissive["egov_hits"] not in (None, "[]")

    strict = _run({**runtime_config(), "retrieval": {"top_k": 5, "score_threshold": 0.99}})
    assert strict["egov_hits"] == "[]"


def test_declared_corpus_path_reaches_the_retrieval_nodes():
    """Point the corpus somewhere empty and the answer changes accordingly."""
    import json

    result = _run(
        {
            **runtime_config(),
            "egov_corpus_path": "config/does-not-exist.jsonl",
            "municipal_kb_path": "config/does-not-exist.jsonl",
        }
    )
    assert json.loads(result["answer"])["status"] == "low_confidence"


def test_disabling_the_output_gate_refuses_construction():
    with pytest.raises(ValueError):
        GovernmentProcedureQAAgent(config={**runtime_config(), "security": {"s3_gate_enabled": False}})
