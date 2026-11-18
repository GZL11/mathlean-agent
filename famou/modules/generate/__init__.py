"""
Generation modules for Famou 2.0.

Base class:
- GenerateModule: Inherit to create custom generation strategies

Built-in implementations (import from files directly):
- mutation.MutationGenerate: LLM-based code mutation
- plan_mutation.PlanMutationGenerate: LLM-based structured proof-plan mutation
"""

from famou.modules.generate.base import GenerateModule
from famou.modules.generate.plan_mutation import PlanMutationGenerate
from famou.modules.generate.subgoal_repair import SubgoalRepairGenerate

__all__ = ["GenerateModule", "PlanMutationGenerate", "SubgoalRepairGenerate"]
