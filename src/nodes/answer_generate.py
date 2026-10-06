"""AnswerGenerate — composes the final answer from what was retrieved.

  - Builds the three structured fields — required documents, processing time,
    submission channel — plus citations, and writes them to ``state["answer"]``
    as JSON.
  - **Merge and override.** A supplement record extends or replaces the base
    record for the same procedure; otherwise the highest-scoring record wins,
    with the supplement preferred on a tie. This node is the only one that
    sees both hit sets, so the merge belongs here.
  - **Eligibility is never decided.** When the question asks whether *this
    person* qualifies, the answer carries the general procedure and an explicit
    INDETERMINATE marker. No model is consulted for such a question — a
    determination about an individual is not something the record set can
    support, and generating one anyway would be the worst failure this agent
    could have.
  - **No retrieval, no answer.** An empty hit set produces an explicit "ask at
    the counter" reply, not a composed one, and no model call.

The structured fields are extracted from the retrieved records. An injected
client is used only for the natural-language summary; the fields themselves are
never model-generated.

Before returning, the node scans the answer it is about to emit for
credential-shaped content and raises rather than emitting it. The agent-level
gate is the backstop behind that.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Mapping, Optional

from framework.errors import SecurityViolationError
from framework.nodes import BaseNode
from framework.security.credential_detector import detect_credentials

# Audit events use the module-level function, not a node method.
from shared.utils.audit_logger import emit_trace_event

_logger = logging.getLogger(__name__)

# Supplementary credential shapes. `detect_credentials` owns the shared set
# (API keys, JWTs, AWS key ids, bearer tokens, connection strings); these add
# the forms it does not carry. Restating the framework's own shapes here would
# let the two sets drift apart, and a shape this scan misses but the framework
# catches fails further out, where this node can no longer explain it.
_EXTRA_CREDENTIAL_PATTERNS = (
    re.compile(r"pk-[A-Za-z0-9]{16,}"),
    re.compile(
        r"(?i)\b(?:api[_-]?key|secret|password|passwd|access[_-]?token|token)\b\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{12,}"
    ),
)


def _scan_credentials(text: str) -> bool:
    """True when the rendered answer carries a credential-shaped token.

    Framework detection first, then the supplementary shapes, so this scan
    blocks everything the framework blocks and a little more.
    """
    return bool(detect_credentials(text)) or any(p.search(text) for p in _EXTRA_CREDENTIAL_PATTERNS)


# Markers of an *individual* eligibility question ("am I eligible / do I
# qualify"), as opposed to "what is this procedure". Kept conservative so plain
# procedure questions are not mis-flagged.
_ELIGIBILITY_PATTERNS = [
    re.compile(r"該当(しますか|する(の)?(か|でしょうか)|します)"),
    re.compile(r"対象(に?なりますか|ですか|になる(の)?(か|でしょうか))"),
    re.compile(r"(受給|申請|利用)(でき(ますか|る(の)?か)|資格)"),
    re.compile(r"資格(が)?あり(ますか|ますでしょうか)"),
    re.compile(r"要件を満たし(ますか|ているか)"),
    re.compile(r"(am i|are we)\s+eligible", re.IGNORECASE),
    re.compile(r"do i (qualify|need)", re.IGNORECASE),
    re.compile(r"can i (apply|claim|get)", re.IGNORECASE),
]

_WINDOW_NOTICE = "該当する手続きが特定できませんでした。お手数ですが、所轄の窓口に直接お問い合わせください。"
_INDETERMINATE_NOTICE = (
    "個別の適格性（該当・非該当）はこのサービスでは判定できません。"
    "一般的な手続き内容をご案内します。具体的な適格性は所轄の窓口にご確認ください。"
)

# Fields a municipal record may supplement / override on the e-Gov base record.
_OVERRIDABLE_FIELDS = ("title", "required_documents", "processing_time", "submission_channel")


class AnswerGenerateNode(BaseNode):
    """Synthesize the final structured answer from retrieved records."""

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        eligibility_mode: str = "indeterminate",
    ) -> None:
        # LLM is injected (DI) for the summary narrative; None → deterministic.
        self.llm_client = llm_client
        self.eligibility_mode = eligibility_mode

    def execute(self, state: Mapping[str, Any], config: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        # A refused question retrieved nothing and gets no composed answer;
        # the refusal itself is what the caller sees.
        if state.get("blocked"):
            return {}

        egov = self._parse_hits(state.get("egov_hits"))
        municipal = self._parse_hits(state.get("municipal_hits"))
        query_text = self._query_text(state)
        is_eligibility = self.eligibility_mode == "indeterminate" and self._is_eligibility_query(query_text)

        # Low / zero confidence — no records to ground an answer.
        if not egov and not municipal:
            answer: dict[str, Any] = {
                "status": "low_confidence",
                "required_documents": [],
                "processing_time": None,
                "submission_channel": None,
                "summary": _WINDOW_NOTICE,
                "notice": _WINDOW_NOTICE,
                "citations": [],
            }
            if is_eligibility:
                answer["eligibility"] = "INDETERMINATE"
            self._trace(config, query_text, 0, is_eligibility, "low_confidence")
            return self._emit({"answer": json.dumps(answer, ensure_ascii=False)}, state)

        primary, citations = self._merge(egov, municipal)

        if is_eligibility:
            # Eligibility: general procedure info + INDETERMINATE; never an
            # individual determination, and the LLM is not consulted here.
            answer = {
                "status": "indeterminate",
                "eligibility": "INDETERMINATE",
                "required_documents": primary.get("required_documents", []),
                "processing_time": primary.get("processing_time"),
                "submission_channel": primary.get("submission_channel"),
                "summary": _INDETERMINATE_NOTICE,
                "notice": _INDETERMINATE_NOTICE,
                "citations": citations,
            }
            self._trace(config, query_text, len(citations), True, "indeterminate")
            return self._emit({"answer": json.dumps(answer, ensure_ascii=False)}, state)

        # Normal grounded answer.
        answer = {
            "status": "answered",
            "required_documents": primary.get("required_documents", []),
            "processing_time": primary.get("processing_time"),
            "submission_channel": primary.get("submission_channel"),
            "summary": self._summary(primary),
            "citations": citations,
        }
        self._trace(config, query_text, len(citations), False, "answered")
        return self._emit({"answer": json.dumps(answer, ensure_ascii=False)}, state)

    # ── output gate (node level) ─────────────────────────────

    def _security_gate_input(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Input gate — pass-through.

        The framework declares this abstract, so a concrete node has to supply
        it. This node consumes only already-retrieved records; the refusal
        screens ran upstream, before anything was looked up.
        """
        return state

    def _security_gate_output(self, result: dict[str, Any]) -> dict[str, Any]:
        """Output gate — runs the node's credential scan.

        The framework declares this abstract. It delegates to the scan below,
        which reads the rendered answer and raises on credential-shaped
        content, so nothing leaked is ever emitted. A clean answer passes
        through and the node's partial result flows on unchanged.
        """
        self._extra_security_gate_output(result)
        return result

    def _extra_security_gate_output(self, state: Mapping[str, Any]) -> None:
        """Scan the answer this node is about to emit.

        Reads the rendered answer and raises if it carries credential-shaped
        content, so the offending text is never emitted. A clean answer passes
        silently. Invoked explicitly from ``execute()`` before the answer is
        returned.
        """
        answer = state.get("answer")
        if answer is None:
            return
        if _scan_credentials(str(answer)):
            raise SecurityViolationError(
                "AnswerGenerateNode",
                "no credential-shaped content in the rendered answer",
                "credential-pattern match",
            )

    def _emit(self, result: dict[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
        """Gate the produced answer, record the event, and return the result."""
        # Scan the answer we are about to emit; raises on a leak.
        self._extra_security_gate_output({**state, **result})
        answer = result.get("answer")
        if isinstance(answer, str):
            # Record that an answer was produced — its length only, never its
            # text and never anything the caller supplied.
            emit_trace_event("answer_generated", {"answer_len": len(answer)}, state)
        return result

    # ── merge ────────────────────────────────────────────────

    @classmethod
    def _merge(
        cls, egov: list[dict[str, Any]], municipal: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Pick the primary record (municipal overrides e-Gov for the same
        procedure; else highest score, municipal preferred on a tie) and build
        the citation list from every contributing record.
        """
        tagged = [("egov", r) for r in egov] + [("municipal", r) for r in municipal]
        # Highest score first; municipal wins ties (override priority).
        tagged.sort(
            key=lambda item: (float(item[1].get("score", 0.0)), item[0] == "municipal"),
            reverse=True,
        )
        primary_origin, primary = tagged[0]
        primary = dict(primary)

        # Municipal supplement/override on the same procedure as the e-Gov base.
        if primary_origin == "egov":
            pid = primary.get("procedure_id")
            override = next(
                (r for o, r in tagged if o == "municipal" and r.get("procedure_id") == pid),
                None,
            )
            if override:
                for field in _OVERRIDABLE_FIELDS:
                    if override.get(field):
                        primary[field] = override[field]

        citations = [
            {
                "procedure_id": r.get("procedure_id"),
                "title": r.get("title"),
                "source": r.get("source"),
                "origin": origin,
                "score": r.get("score"),
            }
            for origin, r in tagged
        ]
        return primary, citations

    # ── summary ──────────────────────────────────────────────

    def _summary(self, primary: Mapping[str, Any]) -> str:
        """Natural-language summary. Uses the injected LLM if present (grounded
        in the structured fields); otherwise composed deterministically. The
        structured fields themselves are never taken from the LLM.
        """
        docs = primary.get("required_documents") or []
        docs_str = "、".join(d for d in docs if isinstance(d, str)) or "（窓口にご確認ください）"
        ptime = primary.get("processing_time") or "（窓口にご確認ください）"
        channel = primary.get("submission_channel") or "（窓口にご確認ください）"
        deterministic = (
            f"必要書類: {docs_str} / 処理期間: {ptime} / 提出窓口: {channel}。" "正確な運用は提出窓口にご確認ください。"
        )
        if self.llm_client is None:
            return deterministic
        prompt = (
            "以下の行政手続き情報を、必要書類・処理期間・提出窓口がわかるように日本語で簡潔に要約してください。"
            "個別の適格性（該当・非該当）の判断は行わないでください。\n"
            f"必要書類: {docs_str}\n処理期間: {ptime}\n提出窓口: {channel}\n"
            f"手続き名: {primary.get('title', '')}"
        )
        try:
            # shared.services.llm clients (AzureOpenAIClient et al.) expose
            # .complete(messages) -> dict with a "content" key, not a plain
            # string return and not a LangChain .invoke()/AIMessage surface.
            # See shared/services/llm/base_llm.py's canonical response shape.
            response = self.llm_client.complete([{"role": "user", "content": prompt}])
        except Exception:
            return deterministic
        content = response.get("content") if isinstance(response, dict) else None
        return content if isinstance(content, str) and content.strip() else deterministic

    # ── helpers ──────────────────────────────────────────────

    @staticmethod
    def _parse_hits(hits_json: Any) -> list[dict[str, Any]]:
        if not isinstance(hits_json, str) or not hits_json.strip():
            return []
        try:
            payload = json.loads(hits_json)
        except json.JSONDecodeError:
            return []
        if not isinstance(payload, list):
            return []
        return [r for r in payload if isinstance(r, dict)]

    @staticmethod
    def _query_text(state: Mapping[str, Any]) -> str:
        query = state.get("query") or state.get("user_input")
        if isinstance(query, str) and query.strip():
            return query
        # Fall back to the cleaned original captured in query_terms.
        qt = state.get("query_terms")
        if isinstance(qt, str):
            try:
                payload = json.loads(qt)
            except json.JSONDecodeError:
                return ""
            if isinstance(payload, dict):
                original = payload.get("original")
                if isinstance(original, str):
                    return original
        return ""

    @staticmethod
    def _is_eligibility_query(text: str) -> bool:
        return any(p.search(text) for p in _ELIGIBILITY_PATTERNS)

    def _trace(
        self,
        config: Optional[Mapping[str, Any]],
        query_text: str,
        procedure_count: int,
        is_eligibility: bool,
        status: str,
    ) -> None:
        """Record the shape of the answer: question type, how many procedures
        contributed, and whether it was an indeterminate one. Never the
        question text and never anything personal.
        """
        _logger.info(
            "s4_audit",
            extra={
                "audit": {
                    "node": "AnswerGenerate",
                    "query_type": "eligibility" if is_eligibility else "procedure",
                    "procedure_count": procedure_count,
                    "indeterminate_count": 1 if status == "indeterminate" else 0,
                }
            },
        )
