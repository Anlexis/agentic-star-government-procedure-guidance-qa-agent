"""AGENTIC STAR Marketplace entrypoint — one-shot Pod process.

Referenced by this repo's Dockerfile as the image `CMD`. Compiles the agent,
provisions its secrets, then hands off to shared.bootstrap.marketplace_app
for the Marketplace lifecycle (identity, input, events, terminal delivery,
exit). Mirrors agentcore's own `agents/base/chat_agent/cli.py` (the pattern
this file was copied from).

`namespace=` here is the Marketplace secret-provisioning namespace — a different
concept from `config/agent.yaml`'s AgentRegistry `namespace:` key that happens to
share its value. Mirrors `src/api/server.py`'s existing
`secrets_factory(namespace="cmn", agent_name="GovernmentProcedureQAAgent")` call
shape rather than a per-template value: one Marketplace Pod deploys exactly one
template, so there is no cross-template secret-path collision to guard against.
"""

from shared.bootstrap.marketplace_app import run_agent_marketplace
from src.graph.graph import GovernmentProcedureQAAgent, runtime_config

if __name__ == "__main__":
    run_agent_marketplace(
        GovernmentProcedureQAAgent,
        agent_name="GovernmentProcedureQAAgent",
        namespace="cmn",
        config=runtime_config(),
    )
