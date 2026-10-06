"""End-to-end boundary tests through the real ASGI /invoke entry point.

The whole stack the way an external caller reaches it: HTTP adapter, Bearer
trust promotion, runtime config loading, the compiled graph, and the output
gate.

  - an authenticated question produces a real answer computed from the shipped
    corpus, with citations — not a fixed baseline;
  - the tenant selector reaches the supplement lookup and visibly changes the
    answer;
  - a missing or wrong Bearer token gives 401 with a generic body;
  - a selector outside the contract is refused and never echoed;
  - oversized structured parameters are refused at the adapter (413);
  - a credential-shaped structured value is refused at the adapter (400) naming
    the field, because the framework's own gate would otherwise fail the FIRST
    node of the graph with nothing the caller could act on.
"""

import importlib
import json
import os
import pathlib

import pytest

_TOKEN = "invoke-endpoint-test-token"
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def client():
    """A test client over the real app, with Bearer auth switched on."""
    from fastapi.testclient import TestClient

    os.environ["INVOKE_AUTH_TOKEN"] = _TOKEN
    # Import after the variable is set: the adapter reads it per request, but
    # the module also builds and compiles the agent at import time.
    server = importlib.import_module("src.api.server")
    importlib.reload(server)
    with TestClient(server.app) as test_client:
        yield test_client
    os.environ.pop("INVOKE_AUTH_TOKEN", None)


def _auth():
    return {"Authorization": f"Bearer {_TOKEN}"}


def _post(client, **body):
    return client.post("/invoke", json=body, headers=_auth())


# ── the public path does real work ───────────────────────────


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_deploy_payload_produces_a_real_answer(client):
    """The sign-off payload must actually exercise the pipeline.

    A payload that returns `low_confidence` would let a deployment smoke test
    pass while retrieval was completely broken.
    """
    payload = json.loads((_REPO_ROOT / "deploy" / "invoke_payload.json").read_text(encoding="utf-8"))
    body = _post(client, input=payload["input"], session_id=payload["session_id"]).json()

    assert body["status"] == "success"
    answer = json.loads(body["answer"])
    assert answer["status"] == "answered"
    assert answer["required_documents"]
    assert answer["submission_channel"]
    assert answer["citations"]


def test_tenant_selector_reaches_the_supplement_lookup(client):
    """The selector crosses the graph boundary, not just the request body."""
    question = "転入 引越 住民登録 手続き"
    scoped = json.loads(_post(client, input=question, input_context={"municipality": "渋谷区"}).json()["answer"])
    assert "municipal" in {c["origin"] for c in scoped["citations"]}
    assert "渋谷区" in scoped["submission_channel"]


def test_another_tenants_guidance_is_excluded(client):
    """Tenant isolation: scoping to one municipality drops another's records.

    This is the direction worth pinning. An unscoped lookup legitimately sees
    every supplement record, so comparing scoped against unscoped proves
    nothing — the two can agree while the filter does nothing at all.
    """
    question = "転入 引越 住民登録 手続き"
    other = json.loads(_post(client, input=question, input_context={"municipality": "世田谷区"}).json()["answer"])
    assert "municipal" not in {c["origin"] for c in other["citations"]}
    assert "渋谷区" not in json.dumps(other, ensure_ascii=False)


# ── auth boundary ────────────────────────────────────────────


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer wrong-token"},
        {"Authorization": _TOKEN},  # missing the scheme
        {"Authorization": "Bearer "},
    ],
)
def test_bad_or_missing_token_is_401_with_a_generic_body(client, headers):
    response = client.post("/invoke", json={"input": "転入届 必要書類"}, headers=headers)
    assert response.status_code == 401
    detail = response.json()["detail"]
    assert detail == "Token is invalid or expired."
    assert _TOKEN not in detail


# ── structured parameters are bounded and screened ───────────


def test_oversized_input_context_is_refused_at_the_adapter(client):
    response = _post(client, input="転入届 必要書類", input_context={"municipality": "x" * 300_000})
    assert response.status_code == 413


@pytest.mark.parametrize(
    "credential",
    [
        "sk-ABCDEFGHIJKLMNOPQRSTUVWX0123",
        "eyJ" + "a" * 12 + "." + "b" * 12 + "." + "c" * 12,
        "AKIAIOSFODNN7EXAMPLE",
        "Bearer abcdef0123456789ABCDEF",
        "postgresql://" + "u" + ":" + "pw-placeholder" + "@" + "db.example:5432/x",
    ],
)
def test_credential_in_input_context_is_refused_readably(client, credential):
    """Refused at the adapter, naming the field, never echoing the value.

    Without this the request still fails — but inside the framework's own gate,
    at the first node of the graph, with an opaque error the caller cannot act
    on. The refusal set here is the framework's own detector, so what this
    rejects and what that gate blocks are one set by construction.
    """
    response = _post(client, input="転入届 必要書類", input_context={"note": credential})
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "input_context.note" in detail
    assert credential not in detail


def test_credential_in_an_undeclared_field_is_still_refused(client):
    """Undeclared keys are not stripped — they reach the first node and detonate.

    "Our contract only declares inert identifiers" is not immunity: a validator
    ignores keys it does not know, and ignoring is not removing.
    """
    response = _post(
        client,
        input="転入届 必要書類",
        input_context={"municipality": "渋谷区", "attachment": "sk-ABCDEFGHIJKLMNOPQRSTUVWX0123"},
    )
    assert response.status_code == 400
    assert "input_context.attachment" in response.json()["detail"]


def test_hostile_field_name_is_reported_by_position_not_echoed(client):
    """Field names are caller data too."""
    hostile = "<script>" + "x" * 80
    response = _post(client, input="転入届 必要書類", input_context={hostile: "sk-ABCDEFGHIJKLMNOPQRSTUVWX0123"})
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "input_context field #1" in detail
    assert hostile not in detail


def test_ordinary_domain_text_on_the_same_channel_still_passes(client):
    """The screen must not fire on real values — the other failure direction."""
    body = _post(client, input="転入 引越 住民登録 手続き", input_context={"municipality": "渋谷区"}).json()
    assert body["status"] == "success"


# ── the caller contract is enforced through the endpoint ─────


@pytest.mark.parametrize("bad", ["区" * 100, {"nested": "x"}, 12345, "<script>alert(1)</script>"])
def test_selector_outside_the_contract_is_refused_and_never_echoed(client, bad):
    body = _post(client, input="転入届 必要書類", input_context={"municipality": bad}).json()
    assert body["status"] != "success"
    assert json.dumps(bad, ensure_ascii=False) not in json.dumps(body, ensure_ascii=False)


def test_identifier_question_is_refused_through_the_endpoint(client):
    body = _post(client, input="マイナンバー 123456789012 転入届").json()
    assert body["blocked"]
    # A refused question produces no answer at all, rather than an empty one.
    assert "answer" not in body
    assert body["egov_hits"] == "[]"
