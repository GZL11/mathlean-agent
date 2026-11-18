"""LLM-based judging module for Famou 2.0."""

from typing import Any, Dict, Optional

from famou.core.data import Context, Program, RolloutResult
from famou.prompts import prompt_registry
from famou.utils.code_parser import extract_json
from famou.modules.judge.base import JudgeModule
from famou.utils.color import Colors


class LLMJudge(JudgeModule):
    """
    LLM-based code evaluator providing structured feedback.

    **SIMPLIFIED: One Program Per Rollout**

    Uses LLM to generate structured evaluation feedback comparing the current
    program with its parent. Returns JSON with:
    - change_summary: Summary of changes from parent to current
    - error_eval: Error analysis and fix suggestions
    - key_features: Key features of the implementation
    - improvements: Potential improvements

    Config:
        temperature: float - LLM temperature (default: 0.3)
        max_tokens: int - Max tokens for feedback (default: 1500)
        validity_threshold: float - Skip feedback if validity < threshold (default: 0.5)

    Example:
        >>> judge = LLMJudge(name="evaluator", validity_threshold=0.7)
        >>> result = judge(context, result)
        >>> result.generated_program.llm_feedback  # JSON dict with evaluation
    """

    def judge(self, context: Context, result: RolloutResult) -> Dict[str, Any]:
        """Generate structured feedback using LLM."""
        program = result.generated_program
        parent = None
        if program.parent_ids:  # Use parent_ids (list) not parent_id
            parent = context.get_program_by_id(program.parent_ids[0])

        try:
            # Build evaluation prompt
            prompt = self._build_feedback_prompt(program, parent, context)
            # print(f"{Colors.BLUE}{Colors.BOLD}Feedback prompt is:{Colors.RESET}", prompt)

            # Call LLM
            self.log_info(f"Generating evaluation for {program.id}")

            # Only pass overrides if explicitly configured (uses LLMClient defaults otherwise)
            llm_kwargs = {}
            if "temperature" in self.config:
                llm_kwargs["temperature"] = self.config["temperature"]
            if "max_tokens" in self.config:
                llm_kwargs["max_tokens"] = self.config["max_tokens"]

            response = self.llm_client.generate(
                prompt=prompt,
                system=prompt_registry.get(
                    "evaluation/judge_system.txt",
                    task_constraint=context.task_constraint),
                **llm_kwargs,
            )

            # Extract JSON feedback
            feedback = extract_json(response.text)

            self.log_info(
                f"Generated evaluation for {program.id}",
                program_id=program.id,
                feedback_length=len(response.text),
            )
            # print(f"{Colors.BLUE}{Colors.BOLD}Feedback is:{Colors.RESET}", feedback)

            return feedback  # Return the feedback dict

        except Exception as e:
            self.log_error(
                f"Evaluation failed for {program.id}: {e}",
                program_id=program.id,
            )
            raise e

    def _build_feedback_prompt(
        self,
        program: Program,
        parent: Optional[Program],
        context: Context,
    ) -> str:
        """
        Build prompt for LLM evaluation.

        Uses template from famou/prompts/templates/evaluation/judge_feedback.txt

        Template expects:
        - language: Programming language
        - parent_program: Parent program code (or "N/A")
        - current_program: Current program code
        - error_info: Error messages (or "None")

        Args:
            program: Program to evaluate
            parent: Parent program for comparison (if available)
            context: Experiment context

        Returns:
            Formatted evaluation prompt
        """
        template_name = "evaluation/judge_feedback.txt"

        # Get parent code (or N/A if no parent)
        parent_code = parent.code if parent else "N/A (no parent available)"

        # Get error info
        error_info = program.error_info if program.error_info else "None"

        return prompt_registry.get(
            template_name,
            language=program.language,
            parent_program=parent_code,
            current_program=program.code,
            error_info=error_info,
        )
