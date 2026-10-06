"""State schema for the government-procedure question-answering agent.

- Flat TypedDict — primitives and JSON strings only, so the state stays
  serializable across process boundaries.
- Every structured payload (retrieval hits, the answer) travels as a JSON
  string rather than a nested object.
- Extends the framework's own state type; the local class is named for this
  agent so it never shadows it.
- All fields are Optional, which is what "not required" means here. The
  question itself arrives as ``user_input``; ``query`` is accepted as well.
"""

from typing import Optional

from framework.schemas.agent_state import AgentState


class GovernmentProcedureQAState(AgentState):
    """State for the government-procedure FAQ RAG pipeline.

    Flow: QueryEmbed → eGovRetrieve → MunicipalKBRetrieve → AnswerGenerate
    """

    # ── Input ──
    query: Optional[str]
    """Natural-language administrative-procedure question."""

    municipality: Optional[str]
    """Optional tenant municipality selector for the municipal KB."""

    # ── Computed ──
    query_terms: Optional[str]
    """JSON: normalized term set / embedding representation from QueryEmbed
    (used by the retrieve nodes for term-overlap scoring)."""

    egov_hits: Optional[str]
    """JSON array: top-k e-Gov corpus records from eGovRetrieve
    (procedure_id, title, required_documents, processing_time,
    submission_channel, source, score)."""

    municipal_hits: Optional[str]
    """JSON array: top-k municipal KB records from MunicipalKBRetrieve.
    Empty array when no municipal KB is configured (e-Gov-only answer)."""

    answer: Optional[str]
    """JSON: final structured answer — required_documents / processing_time /
    submission_channel + citations, or an INDETERMINATE / low-confidence
    response. Produced by AnswerGenerate and passed through the output gate."""

    blocked: Optional[str]
    """Set when QueryEmbed refuses the question: a safe, user-facing reason.
    When set, retrieval and the model call are both skipped."""
