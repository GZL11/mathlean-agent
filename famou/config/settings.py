"""
Configuration models for Famou 2.0.

Provides a hierarchical config structure:
- FamouConfig: Top-level entry point
  - ExperimentConfig: What to optimize (problem definition, execution scope)
  - InfraConfig: How to run (infrastructure services)
  - ModulesConfig: Algorithm parameters (optional)

Each infrastructure service has its own config:
- LLMConfig, StorageConfig, LoggerConfig, EnvConfig, EmbeddingConfig
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from famou.core.types import Language


# =============================================================================
# Infrastructure Configs
# =============================================================================


class LLMConfig(BaseModel):
    """
    Configuration for LLM provider.

    Fields:
        provider: LLM provider name (e.g., "openai", "anthropic")
        api_base: API base URL
        model: Model identifier (e.g., "gpt-4o", "claude-3-5-sonnet")
        api_key: API key for authentication
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
        max_tokens: Maximum tokens in response
        timeout: Request timeout in seconds
    """

    provider: str = Field(default="openai", description="LLM provider (openai, anthropic, etc.)")
    api_base: str = Field(default="https://api.openai.com", description="LLM API base URL")
    model: str = Field(default="gpt-4o", description="Model identifier")
    api_key: str = Field(description="API key for authentication")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(default=2000, gt=0, description="Maximum response tokens")
    timeout: int = Field(default=60, gt=0, description="Request timeout (seconds)")

    class Config:
        """extra fields allowed"""
        extra = "allow"


class StorageConfig(BaseModel):
    """
    Configuration for storage service.

    Fields:
        type: Storage type (e.g., "local", "s3")
        base_path: Base path for local storage
        bucket: S3 bucket name (for s3 type)
    """

    type: str = Field(default="local", description="Storage type (local, s3, etc.)")
    base_path: str = Field(default="./famou_data", description="Base path for storage")
    # S3-specific (optional)
    bucket: Optional[str] = Field(default=None, description="S3 bucket name")
    region: Optional[str] = Field(default=None, description="S3 region")

    class Config:
        """extra fields allowed"""
        extra = "allow"


class LoggerConfig(BaseModel):
    """
    Configuration for logger.

    Fields:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        file_enabled: Whether to write logs to file
        log_dir: Directory for log files (optional, defaults to storage path)
    """
    type: str = Field(default="local", description="Logger type (local, remote)")
    level: str = Field(default="INFO", description="Logging level")
    file_enabled: bool = Field(default=True, description="Enable file logging")
    log_dir: Optional[str] = Field(default=None, description="Log directory (defaults to storage path)")


class EnvConfig(BaseModel):
    """
    Configuration for execution environment.

    Fields:
        type: Environment type (e.g., "local", "docker", "remote")
        timeout: Default execution timeout in seconds
        sandbox: Whether to use sandboxed execution
    """

    type: str = Field(default="local", description="Environment type (local, docker, remote)")
    timeout: int = Field(default=30, gt=0, description="Execution timeout (seconds)")
    sandbox: bool = Field(default=False, description="Use sandboxed execution")

    class Config:
        """extra fields allowed"""
        extra = "allow"


class EmbeddingConfig(BaseModel):
    """
    Configuration for embedding service.

    Fields:
        provider: Embedding provider (e.g., "openai")
        model: Model identifier
        api_key: API key (optional, defaults to LLM api_key)
        api_base: API base URL (optional)
    """

    provider: str = Field(default="openai", description="Embedding provider")
    model: str = Field(default="text-embedding-3-small", description="Model identifier")
    api_key: Optional[str] = Field(default=None, description="API key (defaults to LLM api_key)")
    api_base: Optional[str] = Field(default=None, description="API base URL")

    class Config:
        """extra fields allowed"""
        extra = "allow"


class InfraConfig(BaseModel):
    """
    Configuration for all infrastructure services.

    Groups LLM, storage, logger, env, and embedding configs.
    """

    llm: LLMConfig = Field(description="LLM service configuration")
    storage: StorageConfig = Field(default_factory=StorageConfig, description="Storage service configuration")
    logger: LoggerConfig = Field(default_factory=LoggerConfig, description="Logger configuration")
    env: EnvConfig = Field(default_factory=EnvConfig, description="Execution environment configuration")
    embedding: Optional[EmbeddingConfig] = Field(default=None, description="Embedding service configuration")


# =============================================================================
# Experiment Config
# =============================================================================


class IslandConfig(BaseModel):
    """
    Configuration for island model (multi-population evolution).

    Fields:
        num_islands: Number of evolutionary islands (1 = single population)
        island_size: Population size per island
        migration_interval: Migrate best programs every N iterations
        migration_size: Number of programs to migrate between islands
        migration_topology: Island topology (ring, full, star)
        reset_interval: reset worst-performing islands every N iterations
        reset_size: Programs to reset per island (defaults to island_size)
    """

    num_islands: int = Field(default=1, gt=0, description="Number of islands (1 = single population)")
    island_size: int = Field(default=5, gt=0, description="Population size per island")
    migration_topology: str = Field(default="ring", description="Topology: ring, full, star")
    migration_interval: Optional[int] = Field(default=None, ge=1, description="Migration interval (None disables)")
    migration_size: Optional[int] = Field(default=1, gt=0, description="Programs to migrate per interval")
    reset_interval: Optional[int] = Field(default=None, ge=1, description="reset interval (None disables)")
    reset_size: Optional[int] = Field(default=1, gt=0, description="Programs to reset per interval")


class ExperimentConfig(BaseModel):
    """
    Experiment configuration - defines WHAT to optimize.

    This contains problem definition and execution scope,
    but NOT infrastructure settings (those are in InfraConfig).

    Fields:
        name: Human-readable experiment name
        task_description: Description of the optimization problem/task
        language: Programming language
        strategy: Strategy name for rollout pipeline (e.g., "standard", "exploration")

        max_iterations: Maximum number of iterations to run
        max_workers: Maximum parallel workers for streaming evolution
        checkpoint_interval: Save checkpoint every N completed rollouts
        seed: Random seed for reproducibility

        island: Island model configuration
    """

    # Problem definition
    name: str = Field(description="Experiment name")
    task_description: str = Field(description="Description of the optimization problem/task")
    task_constraint: str = Field(description="Constraints of the optimization problem/task")
    language: Language = Field(default=Language.PYTHON, description="Programming language")
    strategy: str = Field(default="standard", description="Strategy name for rollout pipeline")

    # Execution scope
    max_iterations: int = Field(default=10, gt=0, description="Maximum iterations")
    max_workers: int = Field(default=4, gt=0, description="Maximum parallel workers")
    checkpoint_interval: int = Field(default=5, gt=0, description="Checkpoint interval (rollouts)")
    seed: int = Field(default=42, description="Random seed for reproducibility")

    # Island model
    island: IslandConfig = Field(default_factory=IslandConfig, description="Island model configuration")

    # Convenience properties for backwards compatibility
    @property
    def num_islands(self) -> int:
        """Number of islands (convenience accessor)."""
        return self.island.num_islands

    @property
    def island_size(self) -> int:
        """Island size (convenience accessor)."""
        return self.island.island_size

    @property
    def migration_interval(self) -> int:
        """Migration interval (convenience accessor)."""
        return self.island.migration_interval

    @property
    def migration_size(self) -> int:
        """Migration size (convenience accessor)."""
        return self.island.migration_size

    @property
    def migration_topology(self) -> str:
        """Migration topology (convenience accessor)."""
        return self.island.migration_topology

    @property
    def reset_interval(self) -> Optional[int]:
        """reset interval (convenience accessor)."""
        return self.island.reset_interval

    @property
    def reset_size(self) -> Optional[int]:
        """reset size (convenience accessor)."""
        return self.island.reset_size


# =============================================================================
# Modules Config
# =============================================================================


class ModulesConfig(BaseModel):
    """
    Optional configuration for module parameters.

    These parameters are passed to modules when instantiated.
    Modules are still instantiated in code (type-safe),
    but parameters can come from this config.

    Example YAML:
        modules:
          select:
            k: 1
            num_inspirations: 2
          population:
            max_size: 20

    Usage in code:
        rollout = Rollout([
            TopKSelect(**config.modules.select),
            ...
        ])
    """

    select: Dict[str, Any] = Field(default_factory=dict, description="Select module parameters")
    generate: Dict[str, Any] = Field(default_factory=dict, description="Generate module parameters")
    evaluate: Dict[str, Any] = Field(default_factory=dict, description="Evaluate module parameters")
    judge: Dict[str, Any] = Field(default_factory=dict, description="Judge module parameters")
    population: Dict[str, Any] = Field(default_factory=dict, description="Population module parameters")
    feature: Dict[str, Any] = Field(default_factory=dict, description="Feature module parameters")


# =============================================================================
# Top-Level Config
# =============================================================================


class FamouConfig(BaseModel):
    """
    Top-level configuration for Famou.

    Combines experiment definition, infrastructure services,
    and optional module parameters.

    Example:
        config = FamouConfig.from_yaml("config.yaml")
        infra = InfraFactory.create(config.infrastructure)
        evolver = Evolver(
            config=config.experiment,
            infrastructure=infra,
            ...
        )
    """

    experiment: ExperimentConfig = Field(description="Experiment configuration")
    infrastructure: InfraConfig = Field(description="Infrastructure configuration")
    modules: ModulesConfig = Field(default_factory=ModulesConfig, description="Module parameters")
    problem: Dict[str, Any] = Field(default_facotry=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert config to dictionary.

        Uses json mode to properly serialize enums (e.g., Language) as their string values
        instead of enum objects.
        """
        return self.model_dump(mode='json')

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FamouConfig":
        """Create config from dictionary."""
        return cls.model_validate(data)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "FamouConfig":
        """
        Load config from YAML file.

        Args:
            yaml_path: Path to YAML file

        Returns:
            FamouConfig instance
        """
        import yaml
        from pathlib import Path

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_yaml(self, yaml_path: str) -> None:
        """
        Save config to YAML file.

        Args:
            yaml_path: Path to save YAML file
        """
        import yaml
        from pathlib import Path

        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
