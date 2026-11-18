"""
Utility functions for Famou 2.0.

ID Generation:
- generate_program_id: Generate unique program IDs
- generate_experiment_id: Generate unique experiment IDs
- generate_rollout_id: Generate unique rollout IDs
- get_timestamp: Get current timestamp

Code Parsing:
- extract_code_from_markdown: Extract code blocks from markdown
- evolve_code: Merge EVOLVE-BLOCK from child into parent code
- validate_python_syntax: Validate Python code syntax

Package Management:
- install_packages: Install Python packages to target directory
- get_pythonpath_with_target: Get PYTHONPATH with target directory prepended
- clear_package_cache: Clear global package installation cache

Serialization:
- model_to_dict, model_to_json: Pydantic model to dict/JSON
- dict_to_model, json_to_model: Dict/JSON to Pydantic model
- load_model_from_file, save_model_to_file: File I/O for models
"""

from famou.utils.id_gen import (
    generate_experiment_id,
    generate_program_id,
    generate_rollout_id,
    get_timestamp,
    set_global_seed,
    timestamp_to_str,
)
from famou.utils.code_parser import (
    evolve_code,
    extract_and_validate_code,
    extract_code_from_markdown,
    extract_required_packages,
    validate_python_syntax,
)
from famou.utils.package_manager import (
    clear_package_cache,
    install_packages,
)
from famou.utils.serialization import (
    dict_to_model,
    json_to_model,
    load_model_from_file,
    model_to_dict,
    model_to_json,
    save_model_to_file,
)

__all__ = [
    # ID Generation
    "generate_program_id",
    "generate_experiment_id",
    "generate_rollout_id",
    "get_timestamp",
    "set_global_seed",
    "timestamp_to_str",
    # Code Parsing
    "extract_code_from_markdown",
    "evolve_code",
    "extract_and_validate_code",
    "extract_required_packages",
    "validate_python_syntax",
    # Package Management
    "install_packages",
    "get_pythonpath_with_target",
    "clear_package_cache",
    # Serialization
    "model_to_dict",
    "model_to_json",
    "dict_to_model",
    "json_to_model",
    "load_model_from_file",
    "save_model_to_file",
]
