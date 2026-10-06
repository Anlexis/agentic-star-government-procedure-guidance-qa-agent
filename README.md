# Government Procedure Guidance Q&A Agent

AI agent for answering questions about government procedures and administrative guidance, built with Agentic Star.

> **Category**: Cat 1 (industry-agnostic building block)
> **Industry**: Common
> **Template ID**: CMN-C1-066

## Overview

Answers questions about government administrative procedures — what documents an
application needs, how long it takes, and where it is filed.

The agent scores the question against two JSONL corpora: a base corpus of published
procedures, and an optional local supplement that a tenant can use to override or extend
individual entries by procedure id. It answers only from what it retrieved, with a citation
for every record it used. When nothing scores above the confidence threshold it says so and
directs the user to a service counter, rather than composing an answer it cannot ground.

Two behaviours are deliberate and worth knowing before you adapt it:

- **Questions carrying an individual identification number are refused** before retrieval or
  generation runs. The refusal is the answer; nothing is looked up and no model is called.
- **It never rules on an individual's eligibility.** A question of the form "do I qualify?"
  returns the general procedure plus an explicit INDETERMINATE marker, because the record
  set cannot support a determination about a specific person.

Retrieval scores overlapping terms and does not segment Japanese text, so questions work
best as space-separated salient terms (`転入届 必要書類 提出窓口`) rather than full
sentences. Swapping in a segmenter or a vector store is a natural first adaptation.

This is an agent template built with the **AGENTIC STAR** development platform and the
**AgentCore Framework**. It is intended to be taken as a starting point: fork it, adapt it to
your own data and policies, and run it inside your own AGENTIC STAR deployment.

## Requirements

**This template does not run standalone.** It requires:

| Requirement | Notes |
|---|---|
| **AGENTIC STAR platform** | The agent connects to the platform at start-up. Without it, start-up fails immediately (see *Behaviour without the platform* below). Deployment guides and API documentation: [AGENTIC STAR Developers](https://developers.fd.agenticstar.tm.softbank.jp/) |
| **AgentCore Framework** (`agenticstar-agentcore`) | Installed from PyPI as a dependency. |
| Python | >=3.11 |

```bash
pip install -e .
```

### Behaviour without the platform

The framework is designed to run **only** on AGENTIC STAR. There is no fallback or degraded
mode. If the platform is unreachable or the SDK version does not match, the agent fails at graph
compile / start-up preflight rather than starting in a partially working state. This is intentional — a half-running agent is worse than one that refuses to start.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests run without a platform connection. Running the agent itself does not.

## Project Structure

```
src/          agent implementation (nodes, services, schemas)
tests/        unit, integration and boundary tests
config/       agent configuration
docs/         design and operational documentation
```

See `docs/` for the design and the test specification.

## Customising

1. Adjust `config/` for your own environment and policies.
2. Replace the knowledge sources and sample data with your own.
3. Review the node implementations under `src/nodes/` for domain-specific logic.
4. Re-run the test suite.

## License

MIT — see [LICENSE](LICENSE).

## Status of this repository

This template is published **as is**, by its individual author, under the MIT license. It carries
**no warranty and no support commitment**, and no organisation stands behind its behaviour or
fitness for any purpose. Issues and pull requests may or may not receive a response; that is at
the sole discretion of the repository owner.

---
