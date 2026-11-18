"""
MathProof Plan Island Strategy

Description: Plan-centric math-proof MVP strategy with structured plan mutation.
Author: Famou Framework
Tags: mathproof, plan, prm, mvp
"""

from typing import Any, Callable, Dict, Optional

from famou.config.settings import ModulesConfig
from famou.core.protocol import Rollout
from famou.modules.evaluate import EvaluateModule
from famou.modules.feature.embedding import EmbeddingFeature
from famou.modules.generate.plan_mutation import PlanMutationGenerate
from famou.modules.generate.subgoal_repair import SubgoalRepairGenerate
from famou.modules.judge.plan_judge import PlanJudge
from famou.modules.population.topk import TopKPopulation
from famou.modules.select.topk import TopKSelect


def create_strategy(
    evaluate_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
    params: Optional[ModulesConfig] = None,
):
    """
    Create a plan-centric math-proof strategy.

    This is the first compatibility MVP:
    - selection still operates over the current single-population abstraction
    - generation mutates structured proof plans
    - evaluation remains user-defined and can be made plan-aware
    - judge adds plan-progress and next-action feedback
    """
    params = params or ModulesConfig()

    select_params = {"k": 1, "num_inspirations": 2, **params.select}
    generate_overrides = dict(params.generate)
    local_island_budget = int(generate_overrides.pop("local_island_budget", 1) or 1)
    local_island_budget_mode = str(generate_overrides.pop("local_island_budget_mode", "fixed") or "fixed")
    local_island_budget_min = int(generate_overrides.pop("local_island_budget_min", 1) or 1)
    local_island_budget_max = int(generate_overrides.pop("local_island_budget_max", local_island_budget) or local_island_budget)
    local_island_return_mode = generate_overrides.pop("local_island_return_mode", "best")
    local_island_stop_on_solved = bool(generate_overrides.pop("local_island_stop_on_solved", True))
    global_stop_on_solved = bool(generate_overrides.pop("global_stop_on_solved", True))

    generate_params = {
        "template_generate": "generation/plan_mutation.txt",
        **generate_overrides,
    }
    inner_generate_params = {
        "template_generate": "generation/subgoal_repair.txt",
        **generate_overrides,
    }
    evaluate_params = {**params.evaluate}
    judge_params = {
        "template_judge": "evaluation/plan_judge.txt",
        **params.judge,
    }
    population_params = {**params.population}
    feature_params = {**params.feature}
    enable_embedding = bool(feature_params.pop("enable", False))

    modules = [
        TopKSelect(**select_params),
        PlanMutationGenerate(**generate_params),
        EvaluateModule(evaluate_fn=evaluate_fn, **evaluate_params),
        PlanJudge(**judge_params),
    ]
    if enable_embedding:
        modules.append(EmbeddingFeature(**feature_params))

    inner_modules = [
        TopKSelect(**select_params),
        SubgoalRepairGenerate(**inner_generate_params),
        EvaluateModule(evaluate_fn=evaluate_fn, **evaluate_params),
        PlanJudge(**judge_params),
    ]

    inner_rollout = Rollout(
        inner_modules,
        name="mathproof_fixed_plan_inner_repair_rollout",
    )

    rollout = Rollout(
        modules,
        name="mathproof_plan_island_rollout",
        execution_config={
            "local_island_budget": local_island_budget,
            "local_island_budget_mode": local_island_budget_mode,
            "local_island_budget_min": local_island_budget_min,
            "local_island_budget_max": local_island_budget_max,
            "local_island_return_mode": local_island_return_mode,
            "local_island_stop_on_solved": local_island_stop_on_solved,
            "global_stop_on_solved": global_stop_on_solved,
            "local_island_inner_rollout": inner_rollout,
        },
    )

    population_module = TopKPopulation(**population_params)

    return {
        "rollout": rollout,
        "population_module": population_module,
        "description": "Plan-centric math-proof strategy with PRM-style plan scoring",
        "tags": ["mathproof", "plan", "prm", "mvp"],
        "author": "Famou Framework",
    }
