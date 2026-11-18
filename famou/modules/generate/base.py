"""
Base generation modules for Famou 2.0.

All LLM-based generation modules:
- Read from result.selected_parent_id (str - single parent ID)
- Look up parent Program from context.population
- Write ONE Program to result.generated_program
"""

import time
from abc import abstractmethod
from typing import Optional, List

from famou.core.data import Context, Program, RolloutResult
from famou.core.protocol import Module, RequiresLLM
from famou.infrastructure.llm.base import LLMClient
from famou.utils.code_parser import extract_code_from_markdown, evolve_code, extract_required_packages
from famou.utils.id_gen import generate_program_id
from famou.utils.color import Colors


class GenerateModule(Module, RequiresLLM):
    """
    Base class for LLM-based code generation strategies.

    **SIMPLIFIED: One Program Per Rollout**

    All generation uses LLM to create exactly one new program from a selected parent.
    This class handles the common LLM calling logic, while subclasses customize
    how they extract information from context and build prompts.

    Data Flow:
    1. Read parent ID from result.selected_parent_id (str)
    2. Look up parent Program from context.population
    3. Generate ONE new Program (with code, prompts, responses)
    4. Write to result.generated_program (single Program)

    Design:
    - Subclasses implement build_prompt() to return a prompt string
    - Base class execute() handles:
      - Parent lookup from population
      - LLM calling and response handling
      - Code extraction and syntax validation
      - Program creation

    This makes it easy to implement new generation strategies by
    just changing the prompt building logic.
    """

    llm_client: LLMClient  # Injected by RolloutEngine

    def validate_input(self, context: Context, result: RolloutResult) -> None:
        """Validate that a parent program ID is selected."""
        if not result.selected_parent_id:
            raise ValueError(
                f"{self.name}: Cannot generate without a selected parent. "
                "Make sure SelectModule has run first and selected a parent."
            )

    @abstractmethod
    def build_prompt(self, parent: Program, context: Context, result: RolloutResult) -> str:
        """
        Build generation prompt for a parent program.

        This is the core method that subclasses must implement.
        Different generation strategies differ in how they build prompts:
        - Mutation: Ask to improve the code
        - Error-driven: Include error messages and ask to fix
        - Multi-objective: Ask for specific improvements (speed, readability, etc.)
        - Crossover: Combine multiple parents

        Args:
            parent: Parent program to generate from
            context: Experiment context (for user_prompt, config, etc.)

        Returns:
            Formatted prompt string to send to LLM

        Example:
            >>> def build_prompt(self, parent, context):
            ...     return f"Improve this code:\n{parent.code}"
        """
        pass

    def execute(self, context: Context, result: RolloutResult, **kwargs) -> RolloutResult:
        """
        Generate a single new program by prompting LLM.

        This method handles the common LLM calling logic.
        Subclasses customize behavior via build_prompt().

        Args:
            context: Experiment context (contains prompts, config, population)
            result: Rollout result (contains selected_parent_id)

        Returns:
            Updated result with generated_program populated
        """
        # Get parent Program from context using helper method
        parent = context.get_program_by_id(result.selected_parent_id)
        # print(f"{Colors.BLUE}{Colors.BOLD}Parent is:{Colors.RESET}", parent)

        if parent is None:
            raise ValueError(
                f"Parent program {result.selected_parent_id} not found in population. "
                "Cannot generate without valid parent."
            )

        try:
            # Build prompt using subclass strategy
            prompt = self.build_prompt(parent, context, result)
            # print(f"{Colors.BLUE}{Colors.BOLD}Context is:{Colors.RESET}", context)
            # print(f"{Colors.BLUE}{Colors.BOLD}Prompt is:{Colors.RESET}", prompt)

            # Call LLM
            self.log_info(f"Generating program from parent {parent.id}")

            # Only pass overrides if explicitly configured (uses LLMClient defaults otherwise)
            llm_kwargs = {}
            if "temperature" in self.config:
                llm_kwargs["temperature"] = self.config["temperature"]
            if "max_tokens" in self.config:
                llm_kwargs["max_tokens"] = self.config["max_tokens"]

            response = self.llm_client.generate(
                prompt=prompt,
                system=context.task_description,
                **llm_kwargs,
            )
            # print(f"{Colors.BLUE}{Colors.BOLD}Response is:{Colors.RESET}", response)
            # import sys
            # sys.exit(0)

            # Extract code from response
            code, explanation = extract_code_from_markdown(
                response.text, language=self.config.get("language", context.language)
            )
            print(f"{Colors.BLUE}{Colors.BOLD}Code is:{Colors.RESET}", code)
            # print(f"{Colors.BLUE}{Colors.BOLD}Explanation is:{Colors.RESET}", explanation)
            # if explanation is not None:
            #     print(f"{Colors.RED}{Colors.BOLD}Explanation Length:{Colors.RESET}", len(explanation))
            code = evolve_code(code, parent.code)

            # Extract required packages (for Python programs)
            required_packages = None
            if context.language == "python":
                required_packages = extract_required_packages(response.text)
                # Filter to empty list if no packages found
                if not required_packages:
                    required_packages = None

            if code:
                # Create valid program
                new_program = self._create_program(
                    code=code,
                    parent=parent,
                    context=context,
                    result=result,
                    prompt=prompt,
                    response=response.text,
                    explanation=explanation,
                    thinking=response.thinking,
                    required_packages=required_packages,
                )
            else:
                raise RuntimeError(
                    f"No code extracted from response for parent {parent.id}. "
                )

            result.generated_program = new_program

            self.log_info(
                f"Generated program {new_program.id}",
                parent_id=parent.id,
            )

        except Exception as e:
            self.log_error(f"Failed to generate program from parent {parent.id}: {e}")
            raise RuntimeError(
                f"Generation failed for parent {parent.id}: {e}"
            ) from e

        return result

    def _create_program(
        self,
        code: str,
        parent: Program,
        context: Context,
        result: RolloutResult,
        prompt: str,
        response: str,
        explanation: Optional[str] = None,
        thinking: Optional[str] = None,
        required_packages: Optional[List[str]] = None,
    ) -> Program:
        """
        Create a new Program instance.

        Args:
            code: Generated code
            parent: Parent program
            context: Experiment context
            result: Rollout result
            prompt: Prompt used to generate this code
            response: Full LLM response
            thinking: LLM thinking (for o1 models)
            required_packages: List of required Python packages (from LLM response)

        Returns:
            New Program instance

        Note:
            Program validity/error_info are set later by EvaluateModule, not here.
        """
        return Program(
            id=generate_program_id(
                generation=parent.generation + 1, iteration=result.iteration
            ),
            code=code,
            explanation=explanation,
            generation=parent.generation + 1,
            iteration=result.iteration,
            # language=context.language,
            language=self.config.get("language", context.language),
            parent_ids=[parent.id],
            system_prompt=context.task_description,
            prompt=prompt,
            response=response,
            thinking=thinking,
            required_packages=required_packages,
            meta={
                "experiment_id": context.experiment_id,
                "parent_score": parent.combined_score,
            },
            created_at=time.time(),
        )
