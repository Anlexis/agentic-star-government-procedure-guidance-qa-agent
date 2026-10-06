"""Graph composition for the government-procedure question-answering agent.

``AgentBaseGraph`` provides a fixed three-slot backbone — ``pre_process`` /
``main`` / ``post_process``, wrapped by ``initialize`` / ``finalize``.
``compile()`` refuses to build a graph with any slot unregistered, and the
route to ``post_process`` is taken only on a successful status. The pipeline

    QueryEmbed → eGovRetrieve → MunicipalKBRetrieve → AnswerGenerate

is therefore laid over the three slots by small orchestrator nodes that run
their sub-nodes in order, merge each partial result into the accumulated
state, honour an upstream refusal or error rather than overwriting it, and set
the status the backbone routes on:

  pre_process  : QueryEmbed                       (identifier guard, term set)
  main         : eGovRetrieve → MunicipalKBRetrieve
  post_process : AnswerGenerate, then the agent-level output gate

The output gate lives on the agent, not in a node of its own: a gate that is
itself a graph node can be routed around, and the point of the gate is that
nothing reaches the caller without passing it.

Runtime parameters come from ``config/config.yaml``; when that file is absent
the defaults below apply so the agent still constructs. Tests inject corpora
through the ``egov_corpus`` / ``municipal_corpus`` constructor arguments,
since the nodes are built inside the slot orchestrators.
"""

from __future__ import annotations

import json
import re
import os
from pathlib import Path
from typing import Any, ClassVar, Mapping, Optional

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from framework.security.credential_detector import detect_credentials
from framework.utils.config_loader import load_agent_config
from shared.utils.audit_logger import emit_trace_event

from ..nodes.answer_generate import AnswerGenerateNode
from ..nodes.egov_retrieve import EGovRetrieveNode
from ..nodes.municipal_kb_retrieve import MunicipalKBRetrieveNode
from ..nodes.query_embed import QueryEmbedNode, validate_municipality
from ..schemas.state import GovernmentProcedureQAState

# The repo root is two levels above src/graph/; config/config.yaml sits under it.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Fallback defaults mirror config/config.yaml, used only when that file is absent.
_DEFAULTS: dict[str, Any] = {
    "retrieval": {"top_k": 5, "score_threshold": 0.2},
    "llm": {"model": "gpt-4o", "max_tokens": 1024, "temperature": 0.2},
    "egov_corpus_path": "config/egov_procedures.jsonl",
    "municipal_kb_path": "config/municipal_supplement.jsonl",
    "enable_mynumber_block": True,
    "eligibility_mode": "indeterminate",
    "security": {"s3_gate_enabled": True},
}

# Output-gate credential patterns. These SUPPLEMENT framework detection rather
# than restating it: `detect_credentials` owns the shared shapes (API keys,
# JWTs, AWS key ids, bearer tokens, connection strings) and the patterns below
# add the two forms it does not carry. Re-listing the framework's own shapes
# here would let the two sets drift, and a value the framework blocks but this
# gate misses is not merely undetected — it makes the framework raise further
# out, past the point where this template can explain what happened.
_EXTRA_CREDENTIAL_PATTERNS = (
    re.compile(r"pk-[A-Za-z0-9]{16,}"),
    re.compile(r"""(?i)(api[_-]?key|secret|password|token)\s*=\s*['\"][^'\"]{8,}['\"]"""),
)

_S3_BLOCK_NOTICE = "出力に機密情報パターンを検出したため回答を保留しました。" "所管の窓口に直接お問い合わせください。"


def _contains_credential(text: str) -> bool:
    """True when the text carries a credential-shaped token.

    Framework detection first, then the two supplementary shapes — so this
    gate blocks everything the framework blocks, and a little more.
    """
    return bool(detect_credentials(text)) or any(p.search(text) for p in _EXTRA_CREDENTIAL_PATTERNS)


def _apply_s3(output: str) -> str:
    """Withhold an answer that carries credential-shaped content.

    On a hit the offending content is replaced by a notice directing the user
    to a service counter; otherwise the output passes through unchanged.
    """
    text = str(output)
    if _contains_credential(text):
        return _S3_BLOCK_NOTICE
    return text


def runtime_config() -> dict[str, Any]:
    """Load ``config/config.yaml`` — the live runtime parameters.

    The registry loads this file and hands it to the graph constructor; the
    standalone HTTP entry point does the same, so the declared retrieval
    tuning, corpus paths and safety switches are live in both deployments.

    Reading ``config/agent.yaml`` here instead would return nothing usable: the
    manifest carries identity and compile-time requirements only, so a reader
    pointed at it degrades silently to the built-in defaults while every
    declared value looks configured.
    """
    loaded = load_agent_config(_REPO_ROOT)
    return dict(loaded) if isinstance(loaded, dict) else {}


def _resolve_corpus_path(path: Any) -> Optional[str]:
    """Make a configured corpus path absolute against the repo root.

    The configured paths are relative (`config/...`), which resolves against the
    process working directory. Started from anywhere but the repo root, every
    lookup then reads a file that is not there — and a missing corpus is
    deliberately non-fatal, so the agent answers "nothing found" to every
    question instead of failing. Anchoring the path here means the only way to
    get an empty corpus is to actually have one.
    """
    if not path:
        return None
    text = str(path)
    return text if os.path.isabs(text) else str(_REPO_ROOT / text)


def _is_error(result: Any) -> bool:
    status = result.get("status") if isinstance(result, dict) else None
    return status in (AgentStatus.ERROR, AgentStatus.ERROR.value)


class _SequentialSlotNode(FunctionNode):
    """Run an ordered list of sub-nodes inline, merging partial dicts.

    Mirrors the released ``RAGMainNode`` / a peer template ``{**state, **r}`` merge:
      - if the INCOMING state already carries a refusal or an error (the
        identifier hard block, for instance), this slot is a pass-through
        no-op. The backbone always runs initialize→pre_process→main→route,
        so a downstream slot must never overwrite an upstream refusal;
      - otherwise each sub-node is called so its own trust gate runs, partial
        results are merged, and the slot returns the accumulated dict with a
        successful status — required, because the route to ``post_process``
        is taken only on success.
    Returns the accumulated partial dict (not the whole state).
    """

    # The slot nodes are the graph's outer boundary: every request arrives
    # through them, and the entry node they wrap only runs for a caller the
    # entry point has already authenticated. Declared explicitly rather than
    # inherited, so the level is visible where the node is defined.
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    _sub_nodes: tuple[Any, ...] = ()
    _slot_name: ClassVar[str] = "slot"

    def execute(self, state: dict[str, Any], config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        # Honour an upstream refusal or error. A refusal travels in state
        # rather than as a terminal status, so the backbone still reaches the
        # answer slot — which short-circuits on it and generates nothing.
        # Errors stay terminal.
        if _is_error(state):
            self._trace(state, "upstream_error", 0)
            return {"status": state.get("status")}
        if state.get("blocked"):
            self._trace(state, "skipped_after_refusal", 0)
            return {"status": AgentStatus.SUCCESS.value}

        merged = dict(state)
        accumulated: dict[str, Any] = {}
        sub_history: list[str] = []
        for node in self._sub_nodes:
            result = node(merged)  # runs the node's trust gate, then execute()
            # Each sub-node appends its own class name to node_history;
            # collect them so the pipeline order survives the overwrite-merge.
            sub_history.extend(result.get("node_history", []))
            merged = {**merged, **result}
            accumulated = {**accumulated, **result}
            if _is_error(result):
                accumulated["node_history"] = sub_history
                self._trace(state, "sub_node_error", len(sub_history))
                return {**accumulated, "status": result.get("status")}
        accumulated["node_history"] = sub_history
        accumulated["status"] = AgentStatus.SUCCESS.value
        self._trace(state, "completed", len(sub_history))
        return accumulated

    def _trace(self, state: Mapping[str, Any], outcome: str, sub_node_count: int) -> None:
        """Record how this stage of the pipeline finished.

        Counts and an outcome label only — never question text, retrieved
        records or the rendered answer.
        """
        emit_trace_event(
            "pipeline_stage",
            {"stage": self._slot_name, "outcome": outcome, "sub_nodes_run": sub_node_count},
            state,
        )


class _PreProcessSlotNode(_SequentialSlotNode):
    """First stage: carry the tenant selector through, then normalize the question."""

    _slot_name: ClassVar[str] = "pre_process"

    def __init__(self, query_embed: QueryEmbedNode) -> None:
        self._sub_nodes = (query_embed,)

    def execute(self, state: dict[str, Any], config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        # Lift the tenant selector out of input_context: invoke() packs caller
        # data there, and the supplement lookup reads state["municipality"].
        # An explicit state["municipality"] wins. The value is validated here,
        # at the boundary that owns the caller contract, so nothing downstream
        # has to wonder whether it was checked.
        raw = state.get("municipality")
        if raw is None:
            raw = (state.get("input_context") or {}).get("municipality")
        municipality = validate_municipality(raw)
        result = super().execute(state, config)
        if municipality and not _is_error(result):
            result.setdefault("municipality", municipality)
        return result


class _MainSlotNode(_SequentialSlotNode):
    """Middle stage: e-Gov corpus lookup, then the municipal supplement lookup."""

    _slot_name: ClassVar[str] = "main"

    def __init__(self, egov: EGovRetrieveNode, municipal: MunicipalKBRetrieveNode) -> None:
        self._sub_nodes = (egov, municipal)

    def execute(self, state: dict[str, Any], config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        # After a refusal the lookups are skipped, but empty hit sets are still
        # surfaced so "nothing was retrieved" is explicit rather than inferred.
        # No model is reached downstream either way.
        if not _is_error(state) and state.get("blocked"):
            return {
                "egov_hits": "[]",
                "municipal_hits": "[]",
                "status": AgentStatus.SUCCESS.value,
            }
        return super().execute(state, config)


class _PostProcessSlotNode(_SequentialSlotNode):
    """Final stage: compose the answer, then apply the agent-level output gate."""

    _slot_name: ClassVar[str] = "post_process"

    def __init__(self, answer: AnswerGenerateNode) -> None:
        self._sub_nodes = (answer,)

    def execute(self, state: dict[str, Any], config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        result = super().execute(state, config)
        # Gate the produced answer. A refused question leaves `answer` unset,
        # so there is nothing to gate.
        answer = result.get("answer")
        if isinstance(answer, str) and answer:
            result["answer"] = _apply_s3(answer)
        return result


class GovernmentProcedureQAAgent(AgentBaseGraph):
    """Question answering over administrative-procedure records.

    A retrieval-then-generate pipeline: score the question against the
    corpora, then compose an answer strictly from what was retrieved.
    """

    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        llm_client: Optional[Any] = None,
        egov_corpus: Optional[list[dict[str, Any]]] = None,
        municipal_corpus: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        super().__init__()
        cfg = config if isinstance(config, dict) else runtime_config()
        self._config = {**_DEFAULTS, **cfg} if cfg else dict(_DEFAULTS)
        self._llm_client = llm_client
        self._egov_corpus = egov_corpus
        self._municipal_corpus = municipal_corpus

        # Refuse to construct at all if the output gate has been switched off:
        # an agent that can answer without gating its output is not this agent.
        security = self._config.get("security") or {}
        if isinstance(security, dict) and security.get("s3_gate_enabled", True) is False:
            raise ValueError(
                "The output gate is mandatory (security.s3_gate_enabled must be true) "
                "— refusing to construct GovernmentProcedureQAAgent"
            )

    @property
    def name(self) -> str:
        return "GovernmentProcedureQAAgent"

    @property
    def state_schema(self) -> type:
        return GovernmentProcedureQAState

    # ── Builder hook: the three-slot backbone ───────────────

    def register_nodes(self) -> None:
        super().register_nodes()  # adds the initialize and finalize slots
        cfg = self._config
        retrieval = cfg.get("retrieval") or {}
        top_k = int(retrieval.get("top_k", 5))
        threshold = float(retrieval.get("score_threshold", 0.2))

        query_embed = QueryEmbedNode(enable_mynumber_block=bool(cfg.get("enable_mynumber_block", True)))
        egov = EGovRetrieveNode(
            corpus_path=_resolve_corpus_path(cfg.get("egov_corpus_path")),
            top_k=top_k,
            score_threshold=threshold,
            corpus=self._egov_corpus,
        )
        municipal = MunicipalKBRetrieveNode(
            corpus_path=_resolve_corpus_path(cfg.get("municipal_kb_path")),
            top_k=top_k,
            score_threshold=threshold,
            corpus=self._municipal_corpus,
        )
        answer = AnswerGenerateNode(
            llm_client=self._llm_client,
            eligibility_mode=cfg.get("eligibility_mode", "indeterminate"),
        )

        self._nodes["pre_process"] = _PreProcessSlotNode(query_embed)
        self._nodes["main"] = _MainSlotNode(egov, municipal)
        self._nodes["post_process"] = _PostProcessSlotNode(answer)

    # ── Output gate (agent level) ─────────────────────

    def _security_gate_output(self, output: Any, config: Optional[Any] = None) -> str:
        """Mandatory output gate — do not disable.

        Applies the credential filter to the rendered output. The eligibility
        invariant is enforced where the answer is composed; this is the
        agent-level backstop over whatever that produced.
        """
        return _apply_s3(str(output))

    # ── Output surfacing ─────────────────────────────

    #: Fields that carry retrieved or generated content out to the caller.
    #: Everything named here is withheld when the run does not succeed.
    _CONTENT_FIELDS = ("query_terms", "egov_hits", "municipal_hits", "answer")

    #: Generic fallbacks when no more specific message is available. Never
    #: silence: the runner rejects a missing/empty "output" outright (Marketplace
    #: chat contract), so every path through get_output() must leave one behind.
    _NO_ANSWER_NOTICE = "回答を生成できませんでした。所轄の窓口にご確認ください。"
    _SYSTEM_ERROR_NOTICE = "処理中にエラーが発生しました。時間をおいて再度お試しください。"

    @staticmethod
    def _chat_text(answer_json: Any) -> str:
        """Pull the one field inside the structured ``answer`` meant for chat.

        ``state["answer"]`` is machine-structured JSON (status, citations,
        required_documents, ...) for callers that want structure; ``summary``
        (or ``notice`` for the low-confidence / indeterminate shapes) is the
        only part of it written to be read directly. The Marketplace runner's
        output contract wants a plain string here, not the JSON envelope.
        """
        if not isinstance(answer_json, str):
            return ""
        try:
            payload = json.loads(answer_json)
        except ValueError:
            return answer_json
        if isinstance(payload, dict):
            text = payload.get("summary") or payload.get("notice")
            if isinstance(text, str) and text.strip():
                return text
        return answer_json

    def get_output(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Assemble the caller-facing result.

        A run that did not succeed carries NO content. That matters more than
        it looks: the output gate withholds the rendered answer by refusing,
        which leaves the retrieved records themselves sitting in state — and
        those records are what the answer was built from, so surfacing them
        hands over exactly the content the gate just refused to release. The
        error reply therefore carries identifiers and status only.

        ``output`` is the plain-text slot the Marketplace runner delivers to
        chat; it is derived from the domain fields above rather than being a
        field of its own, but it must be populated on every path — including
        a refusal or a system error — or the runner rejects the invocation
        even when ``status`` is SUCCESS.
        """
        status = state.get("status")
        succeeded = status in (AgentStatus.SUCCESS, AgentStatus.SUCCESS.value)

        out: dict[str, Any] = {
            "blocked": state.get("blocked"),
            "status": status,
            "trace_id": state.get("trace_id"),
            "correlation_id": state.get("correlation_id"),
            "node_history": state.get("node_history", []),
        }
        if not succeeded:
            for field in self._CONTENT_FIELDS:
                out[field] = None
            # Framework nodes append f"[node] {e}\n{traceback.format_exc()}" to
            # error_log (framework/nodes/base_node.py) -- a full stack trace with
            # source paths, never meant to reach the caller. A fixed, contentless
            # notice is deliberate here, matching this method's own "identifiers
            # and status only" rule for a run that did not succeed.
            out["output"] = self._SYSTEM_ERROR_NOTICE
            return out

        out["query_terms"] = state.get("query_terms")
        out["egov_hits"] = state.get("egov_hits")
        out["municipal_hits"] = state.get("municipal_hits")
        # `answer` appears only when one was produced — a refused question is
        # answered by the refusal itself, not by an empty answer object.
        answer = state.get("answer")
        if answer is not None:
            out["answer"] = answer
            out["output"] = self._chat_text(answer)
        else:
            out["output"] = state.get("blocked") or self._NO_ANSWER_NOTICE
        return out
