"""Epistemic Prism: simulation and testing suite for AI-mediated epistemic diversity."""

from __future__ import annotations

from apart_epistemics.backends import (
    BackendConfigError,
    BudgetExceededError,
    create_backend,
)
from apart_epistemics.datasets import generate_agent_population
from apart_epistemics.engine import (
    analyze_numeric_core,
    build_numeric_prompts,
    format_prompt_number,
    run_numeric_core,
)
from apart_epistemics.models import (
    AgentProfile,
    ConditionConfig,
    ConditionSummary,
    ExperimentSuiteConfig,
    HeterogeneityType,
    NationalityType,
    NumericEstimateRecord,
    RoundRecord,
    StatisticalComparison,
    TaskItem,
    TaskKind,
)

__version__ = "0.1.0"

__all__ = [
    "AgentProfile",
    "BackendConfigError",
    "BudgetExceededError",
    "ConditionConfig",
    "ConditionSummary",
    "ExperimentSuiteConfig",
    "HeterogeneityType",
    "NationalityType",
    "NumericEstimateRecord",
    "RoundRecord",
    "StatisticalComparison",
    "TaskItem",
    "TaskKind",
    "__version__",
    "analyze_numeric_core",
    "build_numeric_prompts",
    "create_backend",
    "format_prompt_number",
    "generate_agent_population",
    "run_numeric_core",
]
