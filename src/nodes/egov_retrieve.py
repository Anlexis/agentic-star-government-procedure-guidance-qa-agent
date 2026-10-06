"""eGovRetrieve — scores the question against the base procedure corpus.

  - Scores the term set QueryEmbed produced against the published-procedure
    corpus and returns the top-k records.
  - Scoring is in-memory term overlap over a JSONL file. There is no vector
    store and no live API call, which makes retrieval deterministic and the
    whole pipeline testable without a network.
  - ``retrieval.score_threshold`` drops weak matches. An empty hit set is a
    normal outcome: downstream, it produces an explicit "ask at the counter"
    reply rather than an answer nothing supports.

The corpus is injectable through the constructor so tests can supply a small
fixture; otherwise it is read lazily from ``egov_corpus_path``. A missing or
unreadable corpus yields an empty hit set rather than an exception — an agent
that cannot retrieve should still be able to say so.

Each record is a JSON object::

    {"procedure_id": ..., "title": ..., "keywords": [...],
     "required_documents": [...], "processing_time": ...,
     "submission_channel": ..., "source": ...}
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Mapping, Optional

from framework.nodes import BaseNode

# Audit events use the module-level function, not a node method.
from shared.utils.audit_logger import emit_trace_event

# Defaults mirror the `retrieval` block of config/config.yaml.
_DEFAULT_TOP_K = 5
_DEFAULT_SCORE_THRESHOLD = 0.2

# Record fields whose text contributes to the searchable term set, in priority
# order. Lists (keywords / required_documents) are flattened token-wise.
_SEARCHABLE_FIELDS = ("title", "keywords", "required_documents")


class EGovRetrieveNode(BaseNode):
    """Retrieve top-k e-Gov procedure records by query term overlap."""

    def __init__(
        self,
        corpus_path: Optional[str] = None,
        top_k: int = _DEFAULT_TOP_K,
        score_threshold: float = _DEFAULT_SCORE_THRESHOLD,
        corpus: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        self.corpus_path = corpus_path
        self.top_k = int(top_k)
        self.score_threshold = float(score_threshold)
        # Injected corpus wins over lazy file load (DI / test path).
        self._corpus = corpus

    def execute(self, state: Mapping[str, Any], config: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        # Nothing is retrieved after a refusal, or for a state with no terms.
        if state.get("blocked"):
            return {"egov_hits": json.dumps([], ensure_ascii=False)}

        terms = self._query_terms(state.get("query_terms"))
        if not terms:
            return {"egov_hits": json.dumps([], ensure_ascii=False)}

        query_set = set(terms)
        scored: list[tuple[float, dict[str, Any]]] = []
        for record in self._load_corpus():
            score = self._score(query_set, record)
            if score >= self.score_threshold:
                hit = dict(record)
                hit["score"] = round(score, 4)
                scored.append((score, hit))

        # Highest score first; take top-k.
        scored.sort(key=lambda pair: pair[0], reverse=True)
        hits = [hit for _, hit in scored[: self.top_k]]
        # Record the lookup — counts only, never the records themselves.
        emit_trace_event(
            "egov_retrieve",
            {"hits_returned": len(hits), "top_k": self.top_k},
            state,
        )
        return {"egov_hits": json.dumps(hits, ensure_ascii=False)}

    # ── gate overrides ───────────────────────────────────────
    # The framework declares both gates abstract, so a concrete node has to
    # supply them. This node holds no content gate of its own: refusal happens
    # upstream in QueryEmbed, and output gating happens where the answer is
    # composed and again at the agent level. Both are pass-throughs.

    def _security_gate_input(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        return state

    def _security_gate_output(self, result: dict[str, Any]) -> dict[str, Any]:
        return result

    # ── corpus loading ───────────────────────────────────────

    def _load_corpus(self) -> list[dict[str, Any]]:
        """Return the corpus records.

        An injected ``corpus`` takes precedence. Otherwise the JSONL at
        ``corpus_path`` is read lazily and cached. A missing or unreadable
        file, and any malformed line, are tolerated: retrieval degrades to an
        empty result rather than taking the process down.
        """
        if self._corpus is not None:
            return self._corpus
        if not self.corpus_path:
            self._corpus = []
            return self._corpus

        records: list[dict[str, Any]] = []
        try:
            with open(self.corpus_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # skip malformed line, keep going
                    if isinstance(obj, dict):
                        records.append(obj)
        except OSError:
            records = []  # missing/unreadable corpus → empty (non-fatal)

        self._corpus = records
        return self._corpus

    # ── scoring ──────────────────────────────────────────────

    def _score(self, query_set: set[str], record: Mapping[str, Any]) -> float:
        """Term-overlap score in [0, 1]: |query ∩ record| / |query|."""
        record_terms = self._record_terms(record)
        if not record_terms:
            return 0.0
        overlap = query_set & record_terms
        return len(overlap) / len(query_set)

    @classmethod
    def _record_terms(cls, record: Mapping[str, Any]) -> set[str]:
        """Build a normalized term set from a record's searchable fields."""
        tokens: list[str] = []
        for field in _SEARCHABLE_FIELDS:
            value = record.get(field)
            if isinstance(value, str):
                tokens.extend(cls._normalize_terms(value))
            elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
                for item in value:
                    if isinstance(item, str):
                        tokens.extend(cls._normalize_terms(item))
        return set(tokens)

    @staticmethod
    def _query_terms(query_terms_json: Any) -> list[str]:
        """Extract the normalized term list QueryEmbed stored in query_terms."""
        if not isinstance(query_terms_json, str) or not query_terms_json.strip():
            return []
        try:
            payload = json.loads(query_terms_json)
        except json.JSONDecodeError:
            return []
        if not isinstance(payload, dict):
            return []
        terms = payload.get("terms")
        if isinstance(terms, list):
            return [t for t in terms if isinstance(t, str)]
        return []

    @staticmethod
    def _normalize_terms(text: str) -> list[str]:
        """Normalize text into a term list.

        Mirrors ``QueryEmbedNode._extract_terms`` so the question side and the
        corpus side are normalized identically — otherwise the overlap score
        measures the difference between two tokenizers rather than relevance.
        Split on whitespace and common CJK/ASCII punctuation, lower-case ASCII
        tokens, keep CJK as-is, drop single-character ASCII tokens, deduplicate.
        """
        raw = re.split(r"[\s　、。，．「」（）]+", text)
        terms: list[str] = []
        seen: set[str] = set()
        for tok in raw:
            tok = tok.strip("「」（）().,、。　 \t")
            if not tok:
                continue
            normalized = tok.lower() if tok.isascii() else tok
            if normalized.isascii() and len(normalized) <= 1:
                continue
            if normalized not in seen:
                seen.add(normalized)
                terms.append(normalized)
        return terms
