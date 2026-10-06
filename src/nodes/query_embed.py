"""QueryEmbed — first node of the pipeline.

Three responsibilities, in this order:

  - **Refuse before you look anything up.** A question carrying an individual
    identification number (マイナンバー / 個人番号), or a chat-template control
    token, is refused here — before retrieval and before any model call. The
    refusal is written to ``state["blocked"]`` and the rest of the pipeline
    short-circuits on it.
  - Strip markup and trim the question.
  - Produce ``query_terms``: the normalized term set the retrieval nodes score
    against.

The identifier block is unconditional while ``enable_mynumber_block`` is true,
which is the default. Turning it off is a testing affordance, not a
configuration option for a running deployment.
"""

from __future__ import annotations

import json
import re
from typing import Any, ClassVar, Mapping, Optional

from framework.nodes import BaseNode
from framework.schemas.trust_level import TrustLevel

# Audit events use the module-level function, not a node method.
from shared.utils.audit_logger import emit_trace_event

_HTML = re.compile(r"<[^>]+>")

# Upper bound on the question. Administrative-procedure questions are short;
# anything past this is not a question the corpus can answer.
_MAX_QUERY_CHARS = 2000

# Chat-template control tokens, screened as a CLASS rather than as a list of
# instruction phrases. These are the delimiters a model uses to tell system
# text from user text, so a payload carrying one is trying to be read as a
# different speaker — regardless of what it then says. Phrase-matching screens
# miss this entirely: the words after the token can be perfectly ordinary.
_CONTROL_TOKEN_PATTERNS = (
    re.compile(r"<\|[^|]{0,64}\|>"),  # <|im_start|>, <|endoftext|>, ...
    re.compile(r"\[/?INST\]", re.IGNORECASE),  # [INST] / [/INST]
    re.compile(r"<</?SYS>>", re.IGNORECASE),  # <<SYS>> / <</SYS>>
    re.compile(r"<\|?im_(start|end)\|?>", re.IGNORECASE),
)


def _has_control_token(text: str) -> bool:
    """True when the text carries a chat-template control token."""
    return any(p.search(text) for p in _CONTROL_TOKEN_PATTERNS)


def _sanitize_query(query: str) -> str:
    """Strip markup tags and trim surrounding whitespace.

    Sanitizing is not refusing. Stripping `<...>` spans silently deletes a
    control token and forwards whatever it was wrapping, which turns a
    recognizable attack into ordinary-looking text. So the screen runs on the
    raw string AND on the result of this function — once to catch tokens
    before they are removed, once to catch directives that only become
    contiguous after removal.
    """
    return _HTML.sub("", query).strip()


# 12-digit My Number pattern (individual or corporate number).
# Also catch explicit keywords in case of partial or masked numbers.
_MYNUMBER_PATTERNS = [
    re.compile(r"\b\d{12}\b"),  # 12-digit sequence
    re.compile(r"個人番号", re.IGNORECASE),
    re.compile(r"マイナンバー", re.IGNORECASE),
    re.compile(r"my\s*number", re.IGNORECASE),
]

_BLOCK_MESSAGE = (
    "このクエリはマイナンバー（個人番号）に関する情報を含む可能性があるため、"
    "安全のためお答えできません。マイナンバーはこのサービスに入力しないでください。"
    "手続き内容についての一般的なご質問はお気軽にどうぞ。"
)


_INJECTION_MESSAGE = (
    "このご質問には対話テンプレートの制御記号が含まれているため、処理できません。"
    "記号を取り除いたうえで、手続きの内容をそのままご記入ください。"
)

# The tenant selector is caller-controlled. Real municipality names are short
# and made of letters, digits, spaces and hyphens — Japanese or Latin. The
# class deliberately excludes every markup and control character, so a value
# that reaches the corpus filter cannot also be a payload.
_MUNICIPALITY_RE = re.compile(r"^[\w\u3000-\u303f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uff00-\uffef \-]{1,64}$")


class QueryEmbedError(ValueError):
    """Raised when the caller's input does not meet the node's contract."""


def validate_municipality(value: Any) -> Optional[str]:
    """Return the tenant selector, or raise if the caller sent something else.

    Absent is fine — the supplement corpus is optional and an unscoped lookup
    is the documented default. Present means it must be a bounded string over
    the inert class above. The rejected value is never repeated back; the
    caller is told which field to fix.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise QueryEmbedError("municipality must be a string")
    candidate = value.strip()
    if not candidate:
        return None
    if not _MUNICIPALITY_RE.match(candidate):
        raise QueryEmbedError("municipality must be 1-64 characters of letters, digits, spaces or hyphens")
    return candidate


class QueryEmbedNode(BaseNode):
    """Entry node: refusal screens, then question normalization.

    Declares the caller trust level this pipeline requires. The framework
    refuses to run ``execute()`` for a caller below it, so this declaration is
    the pipeline's entry gate rather than a hint.
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def __init__(self, enable_mynumber_block: bool = True) -> None:
        self.enable_mynumber_block = bool(enable_mynumber_block)

    def execute(self, state: Mapping[str, Any], config: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        # invoke() puts the caller query in user_input; tests/legacy may pass query.
        query = state.get("query") or state.get("user_input")
        if not isinstance(query, str) or not query.strip():
            raise QueryEmbedError("query is required and must be a non-empty string")
        if len(query) > _MAX_QUERY_CHARS:
            raise QueryEmbedError(f"query must be at most {_MAX_QUERY_CHARS} characters")

        # Sanitize first, then screen BOTH forms: the raw string still carries
        # any control token, and the stripped string is where a directive split
        # across markup becomes contiguous. Screening only one of the two is
        # how a sanitizer turns into an attack surface.
        clean = _sanitize_query(query)
        if _has_control_token(query) or _has_control_token(clean):
            emit_trace_event(
                "query_refused",
                {"reason": "control_token"},
                state,
            )
            return {
                "blocked": _INJECTION_MESSAGE,
                "query_terms": None,
            }

        # Hard block: refuse before any retrieval or generation runs.
        if self.enable_mynumber_block and self._contains_mynumber(query):
            # Record the refusal decision.
            emit_trace_event(
                "query_refused",
                {"reason": "identification_number"},
                state,
            )
            return {
                "blocked": _BLOCK_MESSAGE,
                "query_terms": None,
            }

        # Produce term set for in-memory retrieval scoring.
        terms = self._extract_terms(clean)
        # Record that a question was normalized — the term count only.
        emit_trace_event(
            "query_embed",
            {"term_count": len(terms)},
            state,
        )
        return {"query_terms": json.dumps({"original": clean, "terms": terms}, ensure_ascii=False)}

    # ── gate overrides ───────────────────────────────────────
    # The framework declares both gates abstract, so a concrete node has to
    # supply them. This node's refusal screens run inline in execute(), before
    # anything is looked up, so there is nothing left for a separate gate to
    # do and both are pass-throughs.

    def _security_gate_input(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        return state

    def _security_gate_output(self, result: dict[str, Any]) -> dict[str, Any]:
        return result

    # ── helpers ──────────────────────────────────────────────

    @staticmethod
    def _contains_mynumber(text: str) -> bool:
        return any(p.search(text) for p in _MYNUMBER_PATTERNS)

    @staticmethod
    def _extract_terms(text: str) -> list[str]:
        """Extract a normalized term set for overlap-based retrieval scoring.

        Splits on whitespace + common CJK punctuation; lower-cases Latin tokens;
        removes single-character Latin stop tokens; keeps CJK tokens as-is.
        The retrieve nodes call this same function so term-scoring is consistent.
        """
        # Split on whitespace and common delimiters (CJK / ASCII punctuation).
        raw = re.split(r"[\s　、。，．「」（）]+", text)
        terms: list[str] = []
        seen: set[str] = set()
        for tok in raw:
            tok = tok.strip("「」（）().,、。　 \t")
            if not tok:
                continue
            # Lower-case Latin; keep CJK / hiragana / katakana as-is.
            normalized = tok.lower() if tok.isascii() else tok
            # Skip single-char ASCII tokens (articles, particles etc.).
            if normalized.isascii() and len(normalized) <= 1:
                continue
            if normalized not in seen:
                seen.add(normalized)
                terms.append(normalized)
        return terms
