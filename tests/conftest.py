"""Shared test fixtures.

``framework.*`` comes from the installed ``agenticstar-agentcore`` package.
These fixtures supply a deterministic corpus, an agent factory and an
``invoke()`` helper, so tests never read the shipped ``config/*.jsonl`` files
and never reach a live service.
"""

import os
import sys

import pytest

from framework.schemas.invocation_context import InvocationContext, TrustLevel

# Repo root (src.*) on the path so local `pytest tests/` works without install.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ── fixture corpora (deterministic; share procedure_id for the override path) ──

EGOV_CORPUS = [
    {
        "procedure_id": "kojin-haigyo",
        "title": "個人事業の廃業届出書",
        "keywords": ["廃業", "個人事業", "届出"],
        "required_documents": ["個人事業の開業・廃業等届出書"],
        "processing_time": "即日(提出のみ)",
        "submission_channel": "所轄税務署 / e-Tax",
        "source": "e-Gov: 所得税法第229条",
    },
    {
        "procedure_id": "tennyu-todoke",
        "title": "転入届",
        "keywords": ["転入", "引越", "住民登録"],
        "required_documents": ["転出証明書", "本人確認書類"],
        "processing_time": "即日",
        "submission_channel": "市区町村役場",
        "source": "e-Gov: 住民基本台帳法第22条",
    },
    {
        "procedure_id": "kensetsu-kyoka",
        "title": "建設業許可申請",
        "keywords": ["建設業", "許可", "申請"],
        "required_documents": ["建設業許可申請書", "財務諸表"],
        "processing_time": "知事許可 約30日",
        "submission_channel": "都道府県庁",
        "source": "e-Gov: 建設業法第3条",
    },
]

MUNICIPAL_CORPUS = [
    {
        "procedure_id": "tennyu-todoke",  # supplements the e-Gov 転入届
        "title": "転入届(渋谷区)",
        "municipality": "渋谷区",
        "keywords": ["転入", "引越", "渋谷", "住民登録"],
        "required_documents": ["転出証明書", "本人確認書類", "渋谷区:窓口の事前予約を推奨"],
        "processing_time": "即日(混雑時は要予約)",
        "submission_channel": "渋谷区役所 戸籍住民課",
        "source": "渋谷区: 住民異動の手続き案内",
    },
]

_BASE_CONFIG = {
    "retrieval": {"top_k": 5, "score_threshold": 0.2},
    "egov_corpus_path": None,
    "municipal_kb_path": None,
    "enable_mynumber_block": True,
    "eligibility_mode": "indeterminate",
    "security": {"s3_gate_enabled": True},
}


class SpyLLM:
    """LLM client that records calls so tests can assert it was/ wasn't used."""

    def __init__(self):
        self.calls = []

    def complete(self, messages: list) -> dict:
        self.calls.append(messages)
        return {"content": "（要約は省略）"}


@pytest.fixture
def spy_llm():
    return SpyLLM()


@pytest.fixture
def make_agent(spy_llm):
    """Factory: build an agent with injected corpora.

    Corpora go in through the constructor, because the nodes are built inside
    the slot orchestrators and are not reachable from outside afterwards.
    ``make_agent()`` gives both corpora; ``make_agent(municipal=[])`` gives the
    base corpus alone.
    """
    from src.graph import GovernmentProcedureQAAgent

    def _factory(egov=None, municipal=None, llm=None, config=None):
        return GovernmentProcedureQAAgent(
            config=dict(config or _BASE_CONFIG),
            llm_client=llm if llm is not None else spy_llm,
            egov_corpus=EGOV_CORPUS if egov is None else egov,
            municipal_corpus=MUNICIPAL_CORPUS if municipal is None else municipal,
        )

    return _factory


def run_agent(agent, query, municipality=None):
    """Invoke an agent through the real ``.invoke()`` chain and return its output.

    The question travels as ``user_input``; an optional municipality travels in
    ``input_context``, mirroring how a gateway calls the agent. The trust level
    is the one the entry node requires.
    """
    ctx = InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL)
    input_context = {"municipality": municipality} if municipality else None
    return agent.invoke(query, ctx=ctx, input_context=input_context)
