"""Proof-of-Boundary tests PB-1…PB-6 (see docs/03_test_spec.md).

The agent is exercised through the real ``.invoke()`` chain via the
``run_agent`` helper. The four pipeline nodes run inside the fixed three-slot
backbone, so PB-6 asserts their order through ``node_history`` rather than by
inspecting the graph.
"""

import ast
import json
import os

import pytest

from framework.errors import SecurityViolationError

from src.nodes.answer_generate import AnswerGenerateNode, _scan_credentials
from tests.conftest import run_agent


# ── PB-1: refusal happens before retrieval or any model call ──


@pytest.mark.parametrize(
    "query",
    [
        "私のマイナンバーは123456789012です 廃業 届出",  # 12-digit
        "個人番号 を教えるので 廃業 届出",  # 個人番号 keyword
    ],
)
def test_pb1_mynumber_blocked_before_retrieve_and_llm(make_agent, spy_llm, query):
    agent = make_agent(llm=spy_llm)
    out = run_agent(agent, query)
    assert out.get("blocked")
    assert out.get("egov_hits") == "[]" and out.get("municipal_hits") == "[]"
    assert spy_llm.calls == []  # retrieval + LLM never reached
    assert "answer" not in out


# ── PB-2: the output gate withholds credential-bearing text ──

# Assembled at run time rather than written out: a literal connection string in
# the tree is a finding in its own right, and the gate that says so is correct
# to. The detector receives the identical characters either way.
_CONNECTION_STRING = "postgresql://" + "admin" + ":" + "pw-placeholder" + "@" + "db.example:5432/x"


@pytest.mark.parametrize(
    "secret",
    [
        "sk-ABCDEFGHIJKLMNOPQRSTUVWX0123",  # API key
        "pk-ABCD1234abcd5678EFGH",  # publishable-key shape
        "AKIAIOSFODNN7EXAMPLE",  # AWS access key id
        _CONNECTION_STRING,  # database connection string
        "sk_live_" + "ABCDEFGHIJKLMNOP0123",  # payment-provider key
        "Bearer abcdef0123456789ABCDEF",  # bearer token
        "api_key='ABCDEFGHIJKL0123'",  # credential assignment
    ],
)
def test_pb2_output_gate_withholds_credentials(make_agent, secret):
    """Whatever shape the credential takes, it does not survive the gate."""
    gated = make_agent()._security_gate_output(f"窓口情報 {secret}")
    assert secret not in gated
    assert gated != f"窓口情報 {secret}"


def test_pb2_output_gate_passes_ordinary_answers(make_agent):
    """Ordinary procedure text is returned byte-identical."""
    clean = "必要書類: 廃業届 / 処理期間: 即日 / 提出窓口: 所轄税務署"
    assert make_agent()._security_gate_output(clean) == clean


# ── PB-2b: the answer node's own credential scan ─────────────


def _egov_hits(*, submission_channel: str) -> str:
    """One e-Gov hit whose submission_channel field carries the test payload."""
    return json.dumps(
        [
            {
                "procedure_id": "kojin-haigyo",
                "title": "個人事業の廃業届出書",
                "required_documents": ["個人事業の開業・廃業等届出書"],
                "processing_time": "即日(提出のみ)",
                "submission_channel": submission_channel,
                "source": "e-Gov: 所得税法第229条",
                "score": 1.0,
            }
        ],
        ensure_ascii=False,
    )


def test_pb2b_answer_node_blocks_credential_leak():
    """A credential reaching the rendered answer trips the node's output gate,
    which raises rather than emitting it.
    """
    node = AnswerGenerateNode()  # no LLM → deterministic summary
    poisoned = _egov_hits(submission_channel="窓口 sk-ABCDEFGHIJKLMNOPQRSTUVWX0123")
    with pytest.raises(SecurityViolationError):
        node.execute({"egov_hits": poisoned, "query": "廃業 個人事業 届出"})


def test_pb2b_answer_node_passes_clean_japanese():
    """A clean Japanese government answer passes the gate and is emitted."""
    node = AnswerGenerateNode()
    clean = _egov_hits(submission_channel="所轄税務署 / e-Tax")
    out = node.execute({"egov_hits": clean, "query": "廃業 個人事業 届出"})
    answer = json.loads(out["answer"])
    assert answer["status"] == "answered"
    assert answer["submission_channel"] == "所轄税務署 / e-Tax"


@pytest.mark.parametrize(
    "payload,expected",
    [
        ("sk-ABCD1234abcd5678EFGH", True),
        ("pk-ABCD1234abcd5678EFGH", True),
        ("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpM", True),
        ("Authorization: Bearer abcdef0123456789ABCDEF", True),
        ("api_key='sk_live_abcdef012345'", True),
        ("住民票の取得方法について", False),  # normal municipal/gov text
        ("所轄税務署 / e-Tax", False),
        ("マイナンバーカード(任意)", False),  # 'マイナンバー' is not a credential token
    ],
)
def test_pb2b_credential_scan_precision(payload, expected):
    """The scan trips on credential tokens but not on normal Japanese gov text."""
    assert _scan_credentials(payload) is expected


# ── PB-3: INDETERMINATE invariant (no individual determination) ──


def test_pb3_indeterminate_invariant(make_agent):
    agent = make_agent()
    answer = json.loads(run_agent(agent, "私は この補助金 の対象になりますか")["answer"])
    assert answer.get("eligibility") == "INDETERMINATE"
    blob = json.dumps(answer, ensure_ascii=False)
    # Never a definitive individual determination.
    assert "該当します" not in blob and "該当しません" not in blob


# ── PB-4: no direct platform-SDK import ──────────────────────


def test_pb4_import_isolation_no_agenticstar():
    src_dir = os.path.join(os.path.dirname(__file__), "..", "..", "src")
    violations = []
    for root, _, files in os.walk(src_dir):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            with open(fpath, encoding="utf-8") as fh:
                tree = ast.parse(fh.read(), filename=fpath)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    mod = node.names[0].name
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                else:
                    continue
                if mod.startswith("agenticstar"):
                    violations.append(f"{fpath}: {mod}")
    assert violations == [], "direct platform-SDK imports in src/:\n" + "\n".join(violations)


# ── PB-5: State serialization (msgpack-safe primitives) ──────


def test_pb5_state_serialization(make_agent):
    agent = make_agent()
    out = run_agent(agent, "廃業 個人事業 届出")
    for key, value in out.items():
        assert isinstance(
            value, (str, int, float, bool, list, type(None))
        ), f"non-primitive output value at {key!r}: {type(value)}"
    json.dumps(out)  # must not raise


# ── PB-6: pipeline node order and gate placement ─────────────


def test_pb6_node_order_and_s3_placement(make_agent):
    agent = make_agent()
    out = run_agent(agent, "廃業 個人事業 届出")

    # The four logical nodes run inside the fixed 3-slot backbone. node_history
    # is appended by BaseNode.__call__, so the logical node class names appear in
    # pipeline order (pre_process → main → post_process).
    history = out.get("node_history", [])
    logical = [
        n
        for n in history
        if n
        in {
            "QueryEmbedNode",
            "EGovRetrieveNode",
            "MunicipalKBRetrieveNode",
            "AnswerGenerateNode",
        }
    ]
    assert logical == [
        "QueryEmbedNode",
        "EGovRetrieveNode",
        "MunicipalKBRetrieveNode",
        "AnswerGenerateNode",
    ], f"unexpected logical node order: {logical}"

    # The slots run in the fixed L1 backbone order pre_process → main → post_process.
    slots = [
        n
        for n in history
        if n
        in {
            "_PreProcessSlotNode",
            "_MainSlotNode",
            "_PostProcessSlotNode",
        }
    ]
    assert slots == ["_PreProcessSlotNode", "_MainSlotNode", "_PostProcessSlotNode"], f"unexpected slot order: {slots}"

    # The output gate is an agent-level method, never a graph node — a gate
    # that is a node can be routed around.
    assert hasattr(agent, "_security_gate_output")
    assert "compliance_check_node" not in agent._nodes
    assert not any("ComplianceCheck" in type(n).__name__ for n in agent._nodes.values() if n is not None)
