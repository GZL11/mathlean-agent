"""Mutation-based generation module for Famou 2.0."""

from typing import List

from famou.core.data import Context, Program, RolloutResult
from famou.prompts import prompt_registry
from famou.modules.generate.base import GenerateModule
from famou.utils.color import Colors


class MutationGenerate(GenerateModule):
    """
    Generate improved programs through evolutionary mutation.

    **SIMPLIFIED: One Program Per Rollout**

    Uses a simple prompt template that asks the LLM to improve the parent code.
    Optionally includes error feedback if the parent had execution errors.

    This is the basic generation strategy for evolutionary code improvement,
    analogous to mutation in genetic algorithms.

    Config:
        prompt_template: Template for generation prompt (default: simple improvement)
        include_error_feedback: Include parent error messages in prompt (default: True)
        temperature: LLM temperature (default: 0.7)
        max_tokens: Maximum tokens for LLM response (default: 2000)

    Example:
        >>> generator = MutationGenerate(
        ...     name="mutation",
        ...     temperature=0.8,
        ...     include_error_feedback=True
        ... )
        >>> result = generator(context, result)
        >>> result.generated_program  # Single Program object
    """

    def build_prompt(self, parent: Program, context: Context, result: RolloutResult) -> str:
        """
        Build improvement prompt for parent program.

        Uses template from famou/prompts/templates/generation/mutation.txt

        Template expects:
        - language: Programming language
        - inspirations: Formatted evolution history string
        - parent_program: Current program code

        Args:
            parent: Parent program to improve
            context: Experiment context

        Returns:
            Formatted prompt string
        """
        # template_name = "generation/mutation2.txt"
        template_generate = self.config.get("template_generate", "generation/mutation2.txt")
        template_program = self.config.get("template_program", "base/program.txt")

        # ===== Here are all the information provided to LLM =====
        # === 1. Parent Information ===
        # The template is at prompts/templates/base/program.txt, and rendered using following kwargs
        # Parent information is passed in by parent: Program
        parent_program = prompt_registry.get(
            template_program,
            language=parent.language.to_scratch_language(),
            program_code=parent.code,
            explanation=parent.explanation,
            combined_score=parent.combined_score,
            error_info=parent.error_info,
            metrics=parent.metrics,
            llm_feedback=parent.llm_feedback,
        )

        # === 2. Inspiration Information ===
        # Format inspirations as evolution history
        inspirations_text = self._format_inspirations(context, result.inspiration_ids)

        # === 3. Combination of Parent and Inspiration ===
        # Render template with variables
        # print(f"Inspirations: {inspirations_text}")
        prompt = prompt_registry.get(
            template_generate,
            language=self.config.get("language", context.language),
            task_constraint=context.task_constraint,  # Currently not used in Generate Module, since task_description is already provided.
            inspirations=inspirations_text,
            parent_program=parent_program,
        )
        # print(f"Generated prompt: {prompt}")

        return prompt

    def _format_inspirations(self, context: Context, inspiration_ids: List[str]) -> str:
        """
        Format evolution history as inspiration text using base/program.txt template.

        Args:
            context: Experiment context
            parent: Current parent program
            max_count: Maximum number of inspirations to include

        Returns:
            Formatted string of evolution history
        """
        template_program = self.config.get("template_program", "base/program.txt")

        inspirations = [context.get_program_by_id(id) for id in inspiration_ids]
        formatted_programs = []
        for i, prog in enumerate(inspirations):
            program_text = prompt_registry.get(
                template_program,
                language=prog.language,
                program_code=prog.code,
                explanation=prog.explanation,
                combined_score=prog.combined_score,
                error_info=prog.error_info,
                metrics=prog.metrics,
            )
            formatted_programs.append(f"## Program {i} (Generation {prog.generation})\n{program_text}")

        return "\n".join(formatted_programs)

    def validate_output(self, context: Context, result: RolloutResult) -> None:
        """Validate that a program was generated."""
        if not result.generated_program:
            raise ValueError(
                f"{self.name}: Failed to generate a program. "
                "Check LLM client and prompt configuration."
            )
