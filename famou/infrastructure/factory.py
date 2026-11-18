"""
Infrastructure Factory for Famou 2.0.

Provides factory methods to instantiate infrastructure services from config.
Supports both automatic instantiation (via type/provider field) and
explicit instantiation (direct class usage).

Example (automatic):
    config = FamouConfig.from_yaml("config.yaml")
    infra = InfraFactory.create_all(config.infrastructure)
    # infra.llm, infra.storage, infra.logger, infra.env, infra.embedding

Example (explicit):
    llm_client = OpenAIClient(
        api_key=config.infrastructure.llm.api_key,
        model=config.infrastructure.llm.model,
    )
"""

from dataclasses import dataclass
from typing import Optional

from famou.config.settings import (
    EmbeddingConfig,
    EnvConfig,
    InfraConfig,
    LLMConfig,
    LoggerConfig,
    StorageConfig,
)
from famou.infrastructure.embedding import EmbeddingClient, OpenAIEmbeddingClient, MockEmbeddingClient
from famou.infrastructure.env import ExecutionEnvironment, LocalEnv
from famou.infrastructure.llm import LLMClient, OpenAIClient, MockLLMClient
from famou.infrastructure.logger import LocalLogger, Logger
from famou.infrastructure.storage import DataService, LocalStorage


@dataclass
class Infrastructure:
    """
    Container for all infrastructure services.

    Created by InfraFactory.create_all() for convenience.
    """

    llm: LLMClient
    storage: DataService
    logger: Logger
    env: ExecutionEnvironment
    embedding: Optional[EmbeddingClient] = None


class InfraFactory:
    """
    Factory for creating infrastructure services from config.

    Supports automatic instantiation based on type/provider fields,
    while still allowing explicit instantiation when needed.
    """

    # ==========================================================================
    # LLM Client
    # ==========================================================================

    @staticmethod
    def create_llm(config: LLMConfig) -> LLMClient:
        """
        Create LLM client from config.

        Args:
            config: LLM configuration with provider field

        Returns:
            LLMClient instance

        Raises:
            ValueError: If provider is not supported
        """
        provider = config.provider.lower()

        if provider == "openai":
            return OpenAIClient(
                api_key=config.api_key,
                model=config.model,
                api_base=config.api_base,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout=config.timeout,
            )
        elif provider == "mock":
            return MockLLMClient()
        # Add more providers here as needed
        # elif provider == "anthropic":
        #     return AnthropicClient(...)
        else:
            raise ValueError(
                f"Unknown LLM provider: {config.provider}. "
                f"Supported providers: openai, mock"
            )

    # ==========================================================================
    # Storage Service
    # ==========================================================================

    @staticmethod
    def create_storage(config: StorageConfig) -> DataService:
        """
        Create storage service from config.

        Args:
            config: Storage configuration with type field

        Returns:
            DataService instance

        Raises:
            ValueError: If storage type is not supported
        """
        storage_type = config.type.lower()

        if storage_type == "local":
            return LocalStorage(base_path=config.base_path)
        # Add more storage types here as needed
        # elif storage_type == "s3":
        #     return S3Storage(bucket=config.bucket, region=config.region)
        else:
            raise ValueError(
                f"Unknown storage type: {config.type}. "
                f"Supported types: local"
            )

    # ==========================================================================
    # Logger
    # ==========================================================================

    @staticmethod
    def create_logger(
        config: LoggerConfig,
        name: str = "famou",
        experiment_id: Optional[str] = None,
        storage_path: Optional[str] = None,
    ) -> Logger:
        """
        Create logger from config.

        Args:
            config: Logger configuration
            name: Logger name (typically experiment name)
            experiment_id: Experiment ID for log file naming
            storage_path: Base storage path (used if log_dir not specified)

        Returns:
            Logger instance
        """
        # Determine log directory
        logger_type = config.type.lower()
        log_dir = config.log_dir
        # print(f"logger directory {log_dir}")
        if log_dir is None and storage_path and experiment_id:
            log_dir = f"{storage_path}/{experiment_id}"
        if logger_type == "local":
            # print("local logger")
            return LocalLogger(
                name=name,
                experiment_id=experiment_id,
                log_dir=log_dir,
                level=config.level,
                file_enabled=config.file_enabled,
            )
        else:
            raise ValueError(
                f"Unknown logger type: {config.type}. "
                f"Supported types: local"
            )

    # ==========================================================================
    # Execution Environment
    # ==========================================================================

    @staticmethod
    def create_env(config: EnvConfig) -> ExecutionEnvironment:
        """
        Create execution environment from config.

        Args:
            config: Environment configuration with type field

        Returns:
            ExecutionEnvironment instance

        Raises:
            ValueError: If environment type is not supported
        """
        env_type = config.type.lower()

        if env_type == "local":
            return LocalEnv()
        # Add more environment types here as needed
        # elif env_type == "docker":
        #     return DockerEnv(timeout=config.timeout, sandbox=config.sandbox)
        else:
            raise ValueError(
                f"Unknown environment type: {config.type}. "
                f"Supported types: local"
            )

    # ==========================================================================
    # Embedding Client
    # ==========================================================================

    @staticmethod
    def create_embedding(
        config: EmbeddingConfig,
    ) -> EmbeddingClient:
        """
        Create embedding client from config.

        Args:
            config: Embedding configuration with provider field
            llm_api_key: Fallback API key from LLM config

        Returns:
            EmbeddingClient instance

        Raises:
            ValueError: If provider is not supported
        """
        provider = config.provider.lower()

        # Use embedding-specific api_key, or fall back to LLM api_key
        api_key = config.api_key
        if not api_key:
            raise ValueError("Embedding API key required (either in embedding config or LLM config)")

        if provider == "openai":
            return OpenAIEmbeddingClient(
                api_key=api_key,
                model=config.model,
                api_base=config.api_base,
            )
        # Add more providers here as needed
        elif provider == "mock":
            return MockEmbeddingClient()
        else:
            raise ValueError(
                f"Unknown embedding provider: {config.provider}. "
                f"Supported providers: openai, mock"
            )

    # ==========================================================================
    # Create All Infrastructure
    # ==========================================================================

    @classmethod
    def create_all(
        cls,
        config: InfraConfig,
        experiment_name: str = "famou",
        experiment_id: Optional[str] = None,
    ) -> Infrastructure:
        """
        Create all infrastructure services from config.

        This is the main convenience method for creating all services at once.

        Args:
            config: Full infrastructure configuration
            experiment_name: Name for logger
            experiment_id: Experiment ID for logger and storage paths

        Returns:
            Infrastructure container with all services

        Example:
            config = FamouConfig.from_yaml("config.yaml")
            infra = InfraFactory.create_all(
                config.infrastructure,
                experiment_name=config.experiment.name,
                experiment_id=experiment_id,
            )
        """
        # Create LLM client
        llm = cls.create_llm(config.llm)

        # Create storage service
        storage = cls.create_storage(config.storage)

        # Create logger
        logger = cls.create_logger(
            config.logger,
            name=experiment_name,
            experiment_id=experiment_id,
            storage_path=config.storage.base_path,
        )

        # Create execution environment
        env = cls.create_env(config.env)

        # Create embedding client (optional)
        embedding = None
        if config.embedding:
            embedding = cls.create_embedding(
                config.embedding,
            )

        return Infrastructure(
            llm=llm,
            storage=storage,
            logger=logger,
            env=env,
            embedding=embedding,
        )
