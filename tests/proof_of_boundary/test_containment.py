"""PB-2c — what the caller receives when the output gate refuses.

The gate withholds the rendered answer by refusing to emit it. That leaves the
retrieved records themselves sitting in state, and those records are what the
answer was built from — so surfacing them hands over exactly the content the
gate just refused to release. These tests fix that: a run that does not succeed
carries identifiers and status out, and nothing else.

Each case is driven end to end through the real invoke chain, because the leak
lives in what the caller finally receives, not in any single node's return.
"""

import json

import pytest

from tests.conftest import EGOV_CORPUS, run_agent

# Assembled at run time: a literal connection string in the tree is a finding in
# its own right. The detector receives the identical characters either way.
_CONNECTION_STRING = "postgresql://" + "admin" + ":" + "pw-placeholder" + "@" + "db.example:5432/x"

# Two groups, and the difference matters.
#
# The framework's own detector recognises the first group, so the run fails in
# the `main` slot — the records never even reach state, and containment is the
# framework's doing rather than this template's.
#
# It does NOT recognise the second group. Those runs retrieve successfully, put
# the poisoned records into state, and only fail later when this template's own
# scan reads the composed answer. That is the path where the records are sitting
# in state at the moment the run fails, and it is the path these tests exist for.
_FRAMEWORK_DETECTED = [
    "sk-ABCDEFGHIJKLMNOPQRSTUVWX0123",
    "AKIAIOSFODNN7EXAMPLE",
    _CONNECTION_STRING,
    "sk_live_" + "ABCDEFGHIJKLMNOP0123",
]
_TEMPLATE_ONLY = [
    "pk-ABCD1234abcd5678EFGH",
    "api_key='ABCDEFGHIJKL0123'",
]


def _poisoned_corpus(secret: str) -> list:
    """The base corpus with a credential planted in a rendered field."""
    poisoned = dict(EGOV_CORPUS[0])
    poisoned["submission_channel"] = f"窓口 {secret}"
    return [poisoned]


@pytest.mark.parametrize("secret", _TEMPLATE_ONLY + _FRAMEWORK_DETECTED)
def test_refused_run_carries_no_credential(make_agent, secret):
    out = run_agent(make_agent(egov=_poisoned_corpus(secret), municipal=[]), "廃業 個人事業 届出")
    assert secret not in json.dumps(out, ensure_ascii=False, default=str)


@pytest.mark.parametrize("secret", _TEMPLATE_ONLY)
def test_refused_run_carries_no_retrieved_records(make_agent, secret):
    """The records the refused answer was built from are withheld too.

    This is the case the framework does not catch: retrieval succeeded, so the
    records are in state when the gate refuses. Surfacing them anyway would
    release the very content the refusal was about.
    """
    out = run_agent(make_agent(egov=_poisoned_corpus(secret), municipal=[]), "廃業 個人事業 届出")
    assert out["status"] != "success"
    assert out["egov_hits"] is None
    assert out["municipal_hits"] is None
    assert out["answer"] is None


@pytest.mark.parametrize("secret", _TEMPLATE_ONLY + _FRAMEWORK_DETECTED)
def test_error_envelope_carries_no_traceback_or_paths(make_agent, secret):
    """No stack frames, no source paths, no exception class names."""
    blob = json.dumps(
        run_agent(make_agent(egov=_poisoned_corpus(secret), municipal=[]), "廃業 個人事業 届出"),
        ensure_ascii=False,
        default=str,
    )
    for tell in ("Traceback", 'File "', '.py"', "/src/", "SecurityViolationError", "site-packages"):
        assert tell not in blob, f"error envelope leaked {tell!r}"


def test_refused_run_still_returns_correlation_identifiers(make_agent):
    """Withholding content is not the same as withholding everything.

    An operator still has to be able to find this run in the logs, so the
    identifiers survive even when every content field is cleared.
    """
    out = run_agent(make_agent(egov=_poisoned_corpus(_TEMPLATE_ONLY[0]), municipal=[]), "廃業 個人事業 届出")
    assert out["trace_id"]
    assert out["correlation_id"]
    assert out["node_history"]


def test_successful_run_still_carries_its_content(make_agent):
    """The control: clearing applies to failures only, not to every run.

    Without this, a get_output that returned nothing at all would pass every
    assertion above.
    """
    out = run_agent(make_agent(), "廃業 個人事業 届出")
    assert out["status"] == "success"
    assert out["egov_hits"] and out["egov_hits"] != "[]"
    answer = json.loads(out["answer"])
    assert answer["status"] == "answered"
    assert answer["required_documents"]
