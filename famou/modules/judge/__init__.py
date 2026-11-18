"""
Judging modules for Famou 2.0.

Base class:
- JudgeModule: Inherit to create custom code review/feedback modules

Built-in implementations (import from files directly):
- llm.LLMJudge: LLM-based code review and feedback
- plan_judge.PlanJudge: Structured proof-plan judge with heuristic fallback
"""

from famou.modules.judge.base import JudgeModule
from famou.modules.judge.plan_judge import PlanJudge

__all__ = ["JudgeModule", "PlanJudge"]
