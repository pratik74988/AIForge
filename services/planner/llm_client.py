"""
Provider-independent LLM client interface for AIForge.

This module defines the boundary between Planner logic and an actual
LLM provider.

Provider-specific implementations belong outside this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .models.planner_result import PlannerResult
from .prompt import PlannerPrompt


class LLMClient(ABC):
    """
    Interface for an LLM capable of structured Planner output.

    The client receives a fully constructed PlannerPrompt and must return
    a validated PlannerResult.
    """

    @abstractmethod
    def generate(self, prompt: PlannerPrompt) -> PlannerResult:
        """
        Generate a structured PlannerResult.

        Args:
            prompt: System and user messages prepared by the prompt
                construction layer.

        Returns:
            Validated PlannerResult.

        Raises:
            NotImplementedError:
                If a concrete implementation does not implement this
                method.
        """
        raise NotImplementedError