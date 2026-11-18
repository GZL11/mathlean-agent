"""
MathProof Fixed-Plan Repair Strategy

Description: Plan-centric math-proof strategy with fixed-plan inner subgoal repair.
Author: Famou Framework
Tags: mathproof, plan, fixed-plan, repair
"""

from typing import Any, Callable, Dict, Optional

from famou.config.settings import ModulesConfig
from famou.core.protocol import Rollout
from famou.modules.evaluate import EvaluateModule
from famou.modules.feature.embedding import EmbeddingFeature
from famou.modules.generate.subgoal_repair import SubgoalRepairGenerate
from famou.modules.judge.plan_judge import PlanJudge
from famou.modules.population.topk import TopKPopulation
from famou.modules.select.topk import TopKSelect


def create_strategy(
    evaluate_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
    params: Optional[ModulesConfig] = None,
):
    """
    Create a fixed-plan inner-repair math-proof strategy.

    This Phase-1 migration strategy keeps the outer Famou abstractions intact
    while swapping the generation operator from whole-plan mutation to
    subgoal-local repair under a fixed topology.
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
        SubgoalRepairGenerate(**generate_params),
        EvaluateModule(evaluate_fn=evaluate_fn, **evaluate_params),
        PlanJudge(**judge_params),
    ]
    if enable_embedding:
        modules.append(EmbeddingFeature(**feature_params))

    rollout = Rollout(
        modules,
        name="mathproof_fixed_plan_repair_rollout",
        execution_config={
            "local_island_budget": local_island_budget,
            "local_island_budget_mode": local_island_budget_mode,
            "local_island_budget_min": local_island_budget_min,
            "local_island_budget_max": local_island_budget_max,
            "local_island_return_mode": local_island_return_mode,
            "local_island_stop_on_solved": local_island_stop_on_solved,
            "global_stop_on_solved": global_stop_on_solved,
        },
    )

    population_module = TopKPopulation(**population_params)

    return {
        "rollout": rollout,
        "population_module": population_module,
        "description": "Math-proof strategy with fixed-plan inner subgoal repair",
        "tags": ["mathproof", "plan", "fixed-plan", "repair"],
        "author": "Famou Framework",
    }
