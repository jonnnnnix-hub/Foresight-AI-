"""Specialist analysis agents."""

from flowedge.agents.base import AgentOutput, BaseAnalyst
from flowedge.agents.cartographer import RepoCartographer
from flowedge.agents.execution import ExecutionAnalyst
from flowedge.agents.ml import MLAnalyst
from flowedge.agents.product import ProductArchitect
from flowedge.agents.research import ResearchAnalyst
from flowedge.agents.risk import RiskAnalyst
from flowedge.agents.skeptic import Skeptic

ALL_ANALYSTS: list[type[BaseAnalyst]] = [
    RepoCartographer,
    ResearchAnalyst,
    ExecutionAnalyst,
    MLAnalyst,
    ProductArchitect,
    RiskAnalyst,
    Skeptic,
]

__all__ = [
    "AgentOutput",
    "BaseAnalyst",
    "RepoCartographer",
    "ResearchAnalyst",
    "ExecutionAnalyst",
    "MLAnalyst",
    "ProductArchitect",
    "RiskAnalyst",
    "Skeptic",
    "ALL_ANALYSTS",
]
