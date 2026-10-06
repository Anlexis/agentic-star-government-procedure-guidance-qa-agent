"""MunicipalKBRetrieve — scores the question against the local supplement.

  - Scores the same term set against the tenant supplement corpus and writes
    the top-k records to ``state["municipal_hits"]``.
  - The supplement is optional. A missing or empty corpus is a supported
    configuration and yields an empty hit set — the answer is then composed
    from the base corpus alone.
  - When ``state["municipality"]`` is set, the lookup is scoped to it: a record
    tagged for a different municipality is excluded, while an untagged record
    applies everywhere. This is what keeps one tenant's guidance out of
    another tenant's answers.

Scoring reuses ``EGovRetrieveNode``'s term functions so both lookups normalize
question and corpus the same way. Only the corpus, the tenant filter and the
output field differ.

The two hit sets stay separate in state. Merging them — and letting a local
record override the base one for the same procedure — happens in
AnswerGenerate, which is the only node that sees both.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional

from framework.nodes import BaseNode

# Audit events use the module-level function, not a node method.
from shared.utils.audit_logger import emit_trace_event

# Single source of truth for term-overlap scoring.
from .egov_retrieve import EGovRetrieveNode

_DEFAULT_TOP_K = 5
_DEFAULT_SCORE_THRESHOLD = 0.2


class MunicipalKBRetrieveNode(BaseNode):
    """Retrieve top-k municipal-KB records by query term overlap (optional KB)."""

    def __init__(
        self,
        corpus_path: Optional[str] = None,
        top_k: int = _DEFAULT_TOP_K,
        score_threshold: float = _DEFAULT_SCORE_THRESHOLD,
        corpus: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        # corpus_path is the municipal KB JSONL (config `municipal_kb_path`).
        self.corpus_path = corpus_path
        self.top_k = int(top_k)
        self.score_threshold = float(score_threshold)
        self._corpus = corpus

    def execute(self, state: Mapping[str, Any], config: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        # Nothing is retrieved after a refusal.
        if state.get("blocked"):
            return {"municipal_hits": json.dumps([], ensure_ascii=False)}

        terms = EGovRetrieveNode._query_terms(state.get("query_terms"))
        if not terms:
            return {"municipal_hits": json.dumps([], ensure_ascii=False)}

        municipality = state.get("municipality")
        query_set = set(terms)
        scored: list[tuple[float, dict[str, Any]]] = []
        for record in self._load_corpus():
            # Tenant scoping: with a municipality selected, drop records tagged
            # for a different one. Untagged records apply everywhere.
            if municipality:
                rec_muni = record.get("municipality")
                if rec_muni and rec_muni != municipality:
                    continue
            # Same record-term builder as the base lookup, so both sides are
            # normalized identically; the overlap ratio is the same formula.
            record_terms = EGovRetrieveNode._record_terms(record)
            if not record_terms:
                continue
            score = len(query_set & record_terms) / len(query_set)
            if score >= self.score_threshold:
                hit = dict(record)
                hit["score"] = round(score, 4)
                scored.append((score, hit))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        hits = [hit for _, hit in scored[: self.top_k]]
        # Record the lookup — counts and scope only, never the records.
        # Whether a tenant scope was applied, not which one: the selector is
        # caller-supplied text, and an audit record is not a place to put
        # caller-supplied text.
        emit_trace_event(
            "municipal_kb_retrieve",
            {"hits_returned": len(hits), "tenant_scoped": bool(municipality)},
            state,
        )
        return {"municipal_hits": json.dumps(hits, ensure_ascii=False)}

    # ── gate overrides ───────────────────────────────────────
    # The framework declares both gates abstract, so a concrete node has to
    # supply them. This node holds no content gate of its own: refusal happens
    # upstream, the tenant filter runs inline in execute(), and output gating
    # happens where the answer is composed and again at the agent level.

    def _security_gate_input(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        return state

    def _security_gate_output(self, result: dict[str, Any]) -> dict[str, Any]:
        return result

    # ── corpus loading ───────────────────────────────────────

    def _load_corpus(self) -> list[dict[str, Any]]:
        """Return the municipal KB records.

        An injected ``corpus`` takes precedence. Otherwise the JSONL at
        ``corpus_path`` is read lazily and cached. The supplement is optional,
        so a missing or unreadable file and any malformed line yield an empty
        record set and the answer is composed from the base corpus alone.
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
            records = []  # missing/unreadable KB → empty (optional, non-fatal)

        self._corpus = records
        return self._corpus
