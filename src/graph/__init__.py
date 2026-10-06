"""Graph package.

Re-exports the agent class so ``from src.graph import GovernmentProcedureQAAgent``
resolves without reaching into the module.
"""

from .graph import GovernmentProcedureQAAgent

__all__ = ["GovernmentProcedureQAAgent"]
