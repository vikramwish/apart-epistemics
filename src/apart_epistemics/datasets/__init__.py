"""Stratified agent population and benchmark candidate validation."""

from __future__ import annotations

from apart_epistemics.datasets.population import generate_agent_population
from apart_epistemics.datasets.selection import validate_candidates

__all__ = [
    "generate_agent_population",
    "validate_candidates",
]
