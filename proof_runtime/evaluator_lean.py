"""
Famou Lean 示例使用的证明评估器。

这个文件把通用的 Famou evaluator 接口适配成一个两阶段的 Lean 证明流程：

1. 外层模型先生成一个 `.scratch` 文件。
   它不是 Lean 代码，而是一份非形式化的证明提纲 / 证明计划。
2. 这个 evaluator 再把 `.scratch` 提纲转换成一个临时 Lean 工作目录，里面包含：
   - `definition.lean`
   - `step_1.lean`, `step_2.lean`, ...
3. 然后再次调用 Claude Code，把这些 step 文件里的 `sorry` 补成完整证明。
4. 最后检查生成结果，并把结果转成 Famou 需要的指标
   （例如 `combined_score`、`validity` 等）。

这个设计里有一个关键点：
- Famou 外层真正演化的对象是 `.scratch` 文件
- 真正的 Lean 构造和验证发生在这个 evaluator 内部

因此，这个 evaluator 同时承担两种职责：
- 把非形式化证明提纲“构造”成 Lean 证明产物
- 把 Lean 证明产物“验证并打分”后回传给 Famou
"""

import importlib.util
import numpy as np
import time
import os
import signal
import subprocess
import tempfile
import traceback
import sys
import pickle
import math
from famou.utils.color import Colors
import itertools
import random
from pathlib import Path
import yaml
import uuid
import shutil
import json


# 这些绝对路径在模块导入时就解析好，因为 evaluator 会在搜索过程中被反复调用，
# 后续所有辅助函数都会依赖这些路径。
current_path = Path(__file__).resolve()
parent_path = current_path.parent

#region === Get configuration ===
# evaluator 可以被不同示例复用。优先读取环境变量 `FAMOU_LEAN_CONFIG`
# 指向的配置；如果没有设置，则退回到当前目录下的默认运行时配置。
config_path_env = os.environ.get("FAMOU_LEAN_CONFIG", "").strip()
yaml_path = Path(config_path_env).resolve() if config_path_env else parent_path / "config_lean.yaml"
with open(yaml_path) as f:
    config = yaml.safe_load(f)
problem_config = config.get("problem")
if problem_config is None:
    raise ValueError("problem config not found in yaml file")

# `timeout_construct` 用于控制“生成 step 文件 + 补全证明”这个最耗时阶段，
# `timeout_verify` 用于最终验证阶段。`project_path` 指向 Lean 工程根目录，
# 后续 MCP / lake 都依赖它。
timeout_construct = problem_config.get("timeout_construct")
timeout_verify = problem_config.get("timeout_verify")
project_path = problem_config.get("project_path")
if project_path:
    project_path = str((yaml_path.parent / project_path).resolve()) if not Path(project_path).is_absolute() else project_path
#endregion

prompts_path = parent_path / "prompts"
prompt_generate_steps = prompts_path / "generate_steps.txt"

#region === Get Claude Settings ===
# Claude Code 在 root 环境和普通环境下的权限模式不完全一致。
# 在 root 环境里，`bypassPermissions` 可能不兼容，因此这里切到 `acceptEdits`。
permission_mode = "acceptEdits" if hasattr(os, "geteuid") and os.geteuid() == 0 else "bypassPermissions"
#endregion

score_amplify = 1

class TimeoutError(Exception):
    """本文件内部使用的超时异常类型。"""
    pass

def timeout_handler(signum, frame):
    """把系统级信号超时转换成 Python 异常。"""
    raise TimeoutError("Function execution timed out")


def _strip_code_fence(text: str) -> str:
    """Remove a surrounding markdown code fence if Claude returns one."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def _extract_structured_output(stdout: str) -> dict:
    """
    解析 Claude 返回的 JSON，并抽出真正的结构化内容。

    `generate_prove_steps` 阶段要求 Claude 按 JSON Schema 输出。
    不同 CLI 版本有时会把结果包在 `structured_output` 字段里，
    有时则直接返回对象，这里统一兼容。
    """
    payload = json.loads(stdout)
    if isinstance(payload, dict) and "structured_output" in payload:
        return payload["structured_output"]
    if isinstance(payload, dict):
        return payload
    raise ValueError("Claude JSON output is not an object")


def generate_prove_steps(scratch_path: Path, temp_folder: Path) -> tuple[bool, str]:
    """
    把用户生成的 `.scratch` 提纲转换成分阶段的 Lean 证明骨架。

    这是第一阶段的“构造”过程。它不会要求 Claude 直接完成最终证明，
    而是先把证明拆成结构化产物：

    - `definition.lean`：必要时放辅助定义 / 辅助声明
    - `step_i.lean`：每一步一个文件，证明体先保留 `sorry`

    也就是说，这个函数完成的是：

        informal outline (.scratch)
            ->
        按步骤拆开的 Lean 证明计划

    这样做的目的，是让后续第二阶段只专注于“补全每个 step 文件里的证明”。

    Args:
        scratch_path (Path): Path to the scratch file
        temp_folder (Path): Path to the temp folder where step files will be created

    Returns:
        (bool, str): (Success flag, Error message if failed)
    """
    try:
        # 每次评估都使用自己的临时目录，所有中间 Lean 文件和 prompt 快照
        # 都会写在这里，避免不同候选之间互相干扰。
        temp_folder.mkdir(parents=True, exist_ok=True)

        # 这个 prompt 定义了“如何把 .scratch 提纲转成 definition.lean +
        # step_i.lean”的规则。
        prompt_file = prompt_generate_steps
        if not prompt_file.exists():
            return False, f"Prompt file not found: {prompt_file}"

        # 外层模型直接输出的是 `.scratch` 文件。如果这里为空，说明根本没有
        # 可形式化的证明思路。
        scratch_content = scratch_path.read_text()
        if not scratch_content.strip() or scratch_content.strip() == "None":
            return False, "Scratch file is empty or contains no valid content"

        # 这里强制 Claude 按结构化 JSON 输出，而不是自由文本。
        # 这样 evaluator 才能稳定地把结果落成具体的 Lean 文件。
        prompt_content = prompt_file.read_text()
        schema = {
            "type": "object",
            "properties": {
                "definition_content": {"type": "string"},
                "steps": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                        },
                        "required": ["content"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["definition_content", "steps"],
            "additionalProperties": False,
        }

        full_prompt = f"""{prompt_content}

## Scratch File Content:
```
{scratch_content}
```

## Path and Reminder:
Parse the above scratch file and generate:
1. Content for `definition.lean` - All definitions needed
2. Content for `step_1.lean`, `step_2.lean`, etc. - Sequential proof steps
3. Avoid using `import Mathlib` in the definition file or proof steps! Instead, you can add specific imports like `import Mathlib.Data.Real.Basic` or `import Mathlib.Analysis.SpecialFunctions.Abs`
4. Use `sorry` as proof body without generating any actually proof besides the proof steps.
5. Do NOT write files yourself. Return the file contents through the structured JSON output only.

Remember to use the Lean MCP tools to search for relevant Mathlib theorems!
The Lean project path: {project_path}
"""

        # 把实际下发给 Claude 的 prompt 保留下来，方便后续排查失败样本。
        temp_prompt_file = temp_folder / "_generate_prompt.txt"
        temp_prompt_file.write_text(full_prompt)

        print(f"{Colors.BOLD}{Colors.CYAN}Calling Claude Code to generate proof steps...{Colors.RESET}")


        # 这里用 Claude 的 JSON 模式，Claude 只负责返回“应该写入文件的内容”，
        # 真正的文件写入由 evaluator 自己完成。
        base = ["claude", "--print", "--output-format", "json", "--json-schema", json.dumps(schema)]
        if permission_mode:
            base += ["--permission-mode", permission_mode]
        base.append(full_prompt)
        result = subprocess.run(
            base,
            capture_output=True,
            text=True,
            cwd=str(temp_folder),
            timeout=300
        )

        if result.returncode != 0:
            return False, (
                f"Claude Code failed with return code {result.returncode}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

        structured_output = _extract_structured_output(result.stdout)

        # 落盘 `definition.lean`。即使没有自定义定义，也会写一个占位文件，
        # 保证后续阶段看到的工作目录结构一致。
        definition_content = _strip_code_fence(structured_output.get("definition_content", ""))
        definition_file = temp_folder / "definition.lean"
        if definition_content.strip():
            definition_file.write_text(definition_content.rstrip() + "\n")
        else:
            definition_file.write_text("-- No custom definitions needed for this proof.\n")

        # 顺序落盘 step 文件。此时每个 step 文件应该都只是 theorem / lemma
        # 骨架，证明体先保留 `sorry`。
        for idx, step in enumerate(structured_output.get("steps", []), start=1):
            step_content = _strip_code_fence(step.get("content", ""))
            if not step_content.strip():
                return False, f"Claude returned empty content for step_{idx}.lean"
            step_file = temp_folder / f"step_{idx}.lean"
            step_file.write_text(step_content.rstrip() + "\n")

        # 合法的拆解结果至少要生成一个 step 文件。
        step_files = sorted(temp_folder.glob("step_*.lean"))

        if not step_files:
            return False, "No step files were generated"

        print(f"{Colors.BOLD}{Colors.GREEN}Generated {len(step_files)} proof steps in {temp_folder}{Colors.RESET}")
        return True, ""

    except subprocess.TimeoutExpired:
        return False, "Claude Code timed out while generating steps"
    except Exception as e:
        return False, f"Failed to generate proof steps: {str(e)}\n{traceback.format_exc()}"


def coherence_check(temp_folder: Path) -> tuple[bool, str]:
    """
    对生成出的 Lean step 文件做一次轻量级结构检查。

    这里刻意不做完整 Lean build，而是做一个便宜的“结构过滤器”：
    在进入昂贵的“补全所有 step 证明”阶段之前，先把明显不合理的拆解结果淘汰掉。

    Language: Lean,
    Input:
        temp_folder (Path): Path to the tempfolder containing the prove steps

    Output:
        (bool): Whether the prove steps are valid
        (str): Error message if the prove steps are invalid
    """
    try:
        # step 文件必须存在，并且遵循 `step_1.lean`, `step_2.lean`, ...
        # 这种编号模式。
        step_files = sorted(temp_folder.glob("step_*.lean"))

        if not step_files:
            return False, "No step files found in temp folder"

        # 后续 prompt 会把这些 step 当作有序证明链来处理，因此编号必须连续。
        for i, step_file in enumerate(step_files, start=1):
            expected_name = f"step_{i}.lean"
            if step_file.name != expected_name:
                return False, f"Step files are not sequential. Expected {expected_name}, found {step_file.name}"

        # 这里只做结构检查：
        # - 文件非空
        # - 有 import
        # - 至少像一个 theorem / lemma / def
        for step_file in step_files:
            content = step_file.read_text()

            # Basic checks
            if not content.strip():
                return False, f"{step_file.name} is empty"

            # Check for import statement
            if "import" not in content:
                return False, f"{step_file.name} missing import statement"

            # 这一阶段允许 `sorry` 存在；我们只要求 Claude 生成了一个像样的
            # theorem / lemma / def 骨架，而不是随便一段文本。
            has_theorem = any(keyword in content for keyword in ['theorem', 'lemma', 'def'])
            if not has_theorem:
                return False, f"{step_file.name} does not contain any theorem, lemma, or definition"

        # Check definition file if it exists
        definition_file = temp_folder / "definition.lean"
        if definition_file.exists():
            def_content = definition_file.read_text()
            if not def_content.strip():
                return False, "definition.lean exists but is empty"

        print(f"{Colors.BOLD}{Colors.GREEN}Coherence check passed for {len(step_files)} steps{Colors.RESET}")
        return True, ""

    except Exception as e:
        return False, f"Coherence check failed: {str(e)}\n{traceback.format_exc()}"


def construct_lean_solution(scratch_path: Path, temp_folder: Path, timeout: float) -> tuple[bool, str, Path]:
    """
    在 `temp_folder` 里构造完整的 Lean 解答工作目录。

    这个函数串起整个“构造阶段”：

    1. 从 `.scratch` 提纲生成 step 文件
    2. 对 step 文件做一次轻量 coherence check
    3. 再调用 Claude，把 `sorry` 补成真正的证明

    它不会把结果作为字符串返回，而是直接把 Lean 工作目录构造在磁盘上，
    后续验证阶段继续消费这个目录。

    Args:
        scratch_path (Path): Path to the scratch file
        temp_folder (Path): Path to the temporary folder for storing steps
        timeout (float): Timeout for the entire construction process in seconds

    Returns:
        (bool, str, Path): (Success flag, Error message if failed, Path to temp folder)
    """
    try:
        # === Step 1: Generate proof steps from scratch file ===
        success, error_msg = generate_prove_steps(scratch_path, temp_folder)
        if not success:
            return False, error_msg

        # === Step 2: Check coherence of generated steps ===
        success, error_msg = coherence_check(temp_folder)
        if not success:
            return False, error_msg

        # === Step 3: 用 Claude 补全证明 ===
        # 走到这里时，temp_folder 中已经有一个“证明计划”。
        # 现在要求 Claude 把这些 step 文件里的 `sorry` 真正补出来。
        prompt_file = Path(__file__).parent / "prompts" / "construct_solution.txt"
        if not prompt_file.exists():
            return False, f"Prompt file not found: {prompt_file}"

        # Get all step files
        step_files = sorted(temp_folder.glob("step_*.lean"))

        print(f"{Colors.BOLD}{Colors.CYAN}Using Claude Code to complete {len(step_files)} proof steps...{Colors.RESET}")

        # prompt 会显式列出所有 step 文件，要求 Claude 按这个拆解结果继续做，
        # 而不是重新从零开始构造证明。
        prompt_content = prompt_file.read_text()
        step_list = "\n".join([f"- {f.name}" for f in step_files])

        full_prompt = f"""{prompt_content}

## Current Situation:
Lean project path: {project_path}
Working directory: {temp_folder}
Step files to complete:
{step_list}

## Reminder:
- For each step file (step_1.lean, step_2.lean, etc.), replace the `sorry` placeholders with complete proofs.
- Moreover, avoid using `import Mathlib` in the definition file or proof steps! Instead, you can add specific imports like `import Mathlib.Data.Real.Basic` or `import Mathlib.Analysis.SpecialFunctions.Abs`.
If you see any `import Mathlib` in the definition file or proof steps, please change it to specific imports before any verification to avoid timeout.
- Work through each file methodically. Do not skip any steps!
"""

        temp_prompt_file = temp_folder / "_construct_prompt.txt"
        temp_prompt_file.write_text(full_prompt)

        # 这次 Claude 调用通常是 evaluator 里最耗时的一步，因此受
        # `timeout_construct` 保护。
        start_time = time.time()
        base = ["claude", "--print"]
        if permission_mode:
            base += ["--permission-mode", permission_mode]
        base.append(full_prompt)
        result = subprocess.run(
            base,
            capture_output=True,
            text=True,
            cwd=str(temp_folder),
            timeout=int(timeout)
        )

        # print(f"{Colors.RED}{Colors.BOLD}The stdout of constructing solution: {Colors.RESET}", result.stdout)
        elapsed = time.time() - start_time

        if result.returncode != 0:
            return False, (
                f"Claude Code failed with return code {result.returncode}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

        print(f"{Colors.BOLD}{Colors.GREEN}Claude Code completed in {elapsed:.1f}s{Colors.RESET}")
        return True, ""

    except subprocess.TimeoutExpired as e:
        # print(f"{Colors.RED}{Colors.BOLD}The stdout of constructing solution: {Colors.RESET}", result.stdout)
        # Try to kill the subprocess
        if hasattr(e, 'proc'):
            e.proc.kill() 
            e.proc.wait() # 必须 wait，否则会留下僵尸进程
        return False, f"Claude Code timed out after {timeout}s"
    except Exception as e:
        return False, f"Failed to construct Lean solution: {str(e)}\n{traceback.format_exc()}"


def verify_lean_solution(temp_folder: Path) -> tuple[bool, str, dict]:
    """
    验证补全后的 Lean step 文件，并汇总验证指标。

    设计目标上，这一步想做的是：
    - 让 Claude / MCP 检查每个 `.lean` 文件
    - 看 diagnostics
    - 确认没有 `sorry`
    - 汇总验证指标

    但当前实现上需要注意：
    - 仍然会先发起一次 Claude 验证 prompt
    - 本地 fallback 目前主要做的是“检查每个 step 文件里是否还含 `sorry`”
    - 所以它比纯文本检查更接近真实验证，但仍然弱于“逐文件完整本地编译”

    这意味着当前 `combined_score` 反映的是“这个 evaluator 目前实现的验证策略”，
    而不是一个理想化的最强 Lean 验证 oracle。

    Args:
        temp_folder (Path): Path to the temporary folder containing all proof steps

    Returns:
        (bool, str, dict): (Success flag, Error message if failed, Verification metrics)
    """
    try:
        # Get all step files
        step_files = sorted(temp_folder.glob("step_*.lean"))

        if not step_files:
            return False, "No step files to verify", {}

        # 这些 metrics 最后会直接回传给 Famou。
        metrics = {
            "total_steps": len(step_files),
            "verified_steps": 0,
            "failed_steps": [],
            "verification_time": 0.0,
        }

        start_time = time.time()

        # Get the verification prompt
        prompt_file = Path(__file__).parent / "prompts" / "verify_solution.txt"
        if not prompt_file.exists():
            return False, f"Prompt file not found: {prompt_file}", {}

        print(f"{Colors.BOLD}{Colors.CYAN}Using Claude Code to verify {len(step_files)} proof steps...{Colors.RESET}")

        # Prepare verification prompt
        prompt_content = prompt_file.read_text()
        step_list = "\n".join([f"- {f.name}" for f in step_files])

        full_prompt = f"""{prompt_content}

## Verification Task:
Lean project path: {project_path}
Working directory: {temp_folder}
Files to verify:
{step_list}

## Your Task:
1. Use lean_diagnostic_messages on each .lean file to check for errors
2. Verify no `sorry` keywords remain in any file
3. Check that all files compile successfully
4. Report any failures with specific error messages

**IMPORTANT**: Use lean_diagnostic_messages for EVERY file!

After verification, provide a summary of:
- Total steps: {len(step_files)}
- Verified steps: (count with no errors)
- Failed steps: (list with error details)
"""

        temp_prompt_file = temp_folder / "_verify_prompt.txt"
        temp_prompt_file.write_text(full_prompt)

        # 先让 Claude 驱动 MCP 去检查这些 step 文件。
        base = ["claude", "--print"]
        if permission_mode:
            base += ["--permission-mode", permission_mode]
        base.append(full_prompt)
        result = subprocess.run(
            base,
            capture_output=True,
            text=True,
            cwd=str(temp_folder),
            timeout=180
        )

        # 即使 Claude 验证阶段失败，仍然会走本地 best-effort fallback，
        # 尽量产出一份可用的验证指标。

        # 当前 fallback 策略：
        # - 读取每个文件
        # - 只要还残留 `sorry` 就判失败
        # 它足够简单，但也意味着这里目前没有独立对每个文件完整重跑 Lean 编译。
        for step_file in step_files:
            try:
                content = step_file.read_text()

                # 只要还含 `sorry`，就说明这一 step 仍未真正完成。
                if "sorry" in content:
                    metrics["failed_steps"].append({
                        "file": step_file.name,
                        "error": "Contains 'sorry' placeholder"
                    })
                    continue

                # 在当前 fallback 策略下，这一 step 视为通过。
                metrics["verified_steps"] += 1
                print(f"{Colors.BOLD}{Colors.GREEN}✓ {step_file.name} verified{Colors.RESET}")

            except Exception as e:
                metrics["failed_steps"].append({
                    "file": step_file.name,
                    "error": str(e)
                })

        metrics["verification_time"] = time.time() - start_time

        # 任务级 validity 目前是全有或全无：所有 step 都过，才算整个任务有效。
        if metrics["verified_steps"] == metrics["total_steps"]:
            print(f"{Colors.BOLD}{Colors.GREEN}All {metrics['total_steps']} steps verified successfully!{Colors.RESET}")
            return True, "", metrics
        else:
            failed_count = len(metrics["failed_steps"])
            error_msg = f"{failed_count}/{metrics['total_steps']} steps failed verification"
            print(f"{Colors.BOLD}{Colors.RED}{error_msg}{Colors.RESET}")
            return False, error_msg, metrics

    except subprocess.TimeoutExpired:
        return False, "Verification timed out", metrics
    except Exception as e:
        return False, f"Verification failed: {str(e)}\n{traceback.format_exc()}", {}


def run_with_timeout(program_path, timeout_seconds=20, function_name="construct_graph_coloring"):
    """
    旧的 Python evaluator 辅助函数。

    这个函数已经不属于当前 Lean workflow，基本可以视为遗留代码。
    它之所以还留在这里，是因为这个文件最早从更通用的 evaluator 模板演化而来。

    # Run the program in a separate process with timeout
    # using a simple subprocess approach

    # Args:
    #     program_path: Path to the program file
    #     timeout_seconds: Maximum execution time in seconds

    # Returns:
    #     centers, radii, sum_radii tuple from the program
    """
    # Create a temporary file to execute
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
        # Write a script that executes the program and saves results
        script = f"""
import sys
import numpy as np
import os
import pickle
import traceback
import inspect

# Add the directory to sys.path
sys.path.insert(0, os.path.dirname('{program_path}'))

# Debugging info
# print(f"Running in subprocess, Python version: {{sys.version}}")
# print(f"Program path: {program_path}")

try:
    # Import the program
    spec = __import__('importlib.util').util.spec_from_file_location("program", '{program_path}')
    program = __import__('importlib.util').util.module_from_spec(spec)
    spec.loader.exec_module(program)
    
    # Run the function
    # print("Calling {function_name}()...")
    
    {function_name} = getattr(program, "{function_name}", None)
    entry =  {function_name}
    if entry is None:
        raise AttributeError("new_born program.py has no {function_name} function")

    {result_name} = entry({graph_size}, {clique_size}, {independent_size})
    # print(f"{function_name} returned successfully: {result_name} = {{{result_name}}}")

    # Save results to a file
    results = {{
        '{result_name}': {result_name}
    }}

    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump(results, f)
    # print(f"Results saved to {temp_file.name}.results")
    
except Exception as e:
    tb = traceback.format_exc()
    # 尽量把源函数也带回
    try:
        if entry is not None:
            fn = inspect.unwrap(entry)
            function = inspect.getsource(fn)
        else:
            function = ""
    except Exception:
        function = ""
    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump({{'error': f'{{e}}', 'traceback': tb, 'function': function}}, f)
    # 关键：即使失败也正常退出，避免父进程看到非 0 退出码
    sys.exit(0)
""" 
        temp_file.write(script.encode())
        temp_file_path = temp_file.name

    results_path = f"{temp_file_path}.results"

    try:
        # Run the script
        print(f"{Colors.BOLD}{Colors.GREEN}Running evaluator{Colors.RESET}")
        process = subprocess.Popen(
            [sys.executable, temp_file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        try:
            # Main process wait for the subprocess's output with timeout.
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            exit_code = process.returncode

            # Always print output for debugging purposes
            #print(f"Subprocess stdout: {stdout.decode()}")
            #if stderr:
            #    print(f"Subprocess stderr: {stderr.decode()}")

            # Still raise an error for non-zero exit codes, but only after printing the output
            if os.path.exists(results_path):
                with open(results_path, "rb") as f:
                    results = pickle.load(f)
                print(f"{Colors.BOLD}{Colors.BLUE}Results from the program - 1: {Colors.CYAN}{results}{Colors.RESET}")

                if "error" in results:
                    # 把子进程错误包装后往上返回/抛出（由上层决定是否继续）
                    err = results.get("error", "")
                    tb = results.get("traceback", "")
                    fn = results.get("function", "")
                    raise RuntimeError(f"Program execution failed: {err}\nTraceback:\n{tb}\nFunction:\n{fn}")
                return results["adj_matrix"]

            # 如果没有结果文件，再根据退出码决定是否抛错，并附带 stdout/stderr
            if exit_code != 0:
                raise RuntimeError(
                    f"Process exited with code {exit_code}\n"
                    f"---- STDOUT ----\n{stdout.decode(errors='ignore')}\n"
                    f"---- STDERR ----\n{stderr.decode(errors='ignore')}"
                )
            else:
                raise RuntimeError("Results file not found although exit_code == 0")

        except subprocess.TimeoutExpired:
            # Kill the process if it times out
            process.kill()
            process.wait()
            raise TimeoutError(f"Process timed out after {timeout_seconds} seconds")

    finally:
        # Clean up temporary files
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        if os.path.exists(results_path):
            os.unlink(results_path)


def evaluate(path_user_scratch) -> dict | str | None:
    """
    评估外层 Famou 生成的一份 `.scratch` 候选。

    这是 `EvaluateModule` 期望调用的入口函数。它实现了 evaluator 的完整生命周期：

    1. 创建一个全新的临时 Lean 工作目录
    2. 从 `.scratch` 提纲构造 Lean 证明产物
    3. 验证这些 step 文件
    4. 把验证结果转换成 Famou 兼容的指标

    当前打分策略：
    - `validity` 只有当所有 step 都通过验证时才为 1.0
    - `verification_ratio = verified_steps / total_steps`
    - `combined_score = validity * verification_ratio`

    由于 validity 现在是二值的，因此 `combined_score` 实际上近似退化为：
    - 全部 step 验证通过时为 `1.0`
    - 否则为 `0.0`

    所以失败样本里更有信息量的通常不是 `combined_score`，而是：
    - `verified_steps`
    - `verification_ratio`
    - `error_info`
    - `metrics["failed_steps"]`

    Args:
        path_user_scratch: Scratch generated by Gemini-3-pro, should be ending with `.scratch`

    Before deleting the tempfolder (along with the steps inside), we should also save a copy of the tempfolder
    (along with the steps inside) under the same folder of `path_user_scratch`, named as `<prefix of path_user_scratch>`.

    Returns:
        Dictionary with evaluation results:
        - combined_score: Float (validity * verification_ratio) - REQUIRED overall fitness
        - total_steps: Number of proof steps
        - verified_steps: Number of successfully verified steps
        - verification_ratio: verified_steps / total_steps
        - validity: 1.0 if all steps valid, 0.0 otherwise
        - eval_time: Execution time in seconds
        - error_info: Dict with error details if any
    """
    scratch_path = Path(path_user_scratch)
    print(f"{Colors.BOLD}{Colors.CYAN}Scratch Path: {scratch_path}{Colors.RESET}")
    assert scratch_path.exists() and scratch_path.is_file(), f"Generated scratch file {scratch_path} does not exist, cannot do evaluation."
    assert scratch_path.suffix == ".scratch", f"Input file must be .scratch file, got {scratch_path.suffix}"

    # 每次评估都在 Lean 工程下创建一个独立 temp 目录，避免不同 rollout
    # 候选之间共享 / 污染生成文件。
    random_string = uuid.uuid4().hex[:8]
    temp_folder_name = f"temp_{scratch_path.stem}_{random_string}"  # Double randomness, is that necessary?
    temp_folder = Path(project_path) / "temp" / temp_folder_name if project_path else scratch_path.parent / "temp" / temp_folder_name

    # Path for saving the final copy
    # save_folder = scratch_path.parent / scratch_path.stem

    result = None
    try:
        start_time = time.time()

        # === Step 1: 构造 Lean 解答（先生成 step，再补全证明） ===
        # 如果这一阶段失败，就直接把这份 scratch 候选视为无效，记 0 分。
        print(f"{Colors.BOLD}{Colors.CYAN}Step 1: Constructing Lean solution...{Colors.RESET}")
        success, error_msg = construct_lean_solution(
            scratch_path=scratch_path,
            temp_folder=temp_folder,
            timeout=timeout_construct
        )
        print(f"{Colors.BOLD}{Colors.CYAN}Success: {success} & error_msg: {error_msg} & Temp Folder: {temp_folder}{Colors.RESET}")
        if not success:
            result = {
                "total_steps": 0,
                "verified_steps": 0,
                "verification_ratio": 0.0,
                "validity": 0.0,
                "eval_time": time.time() - start_time,
                "combined_score": 0.0,
                "error_info": "Construction failed: " + error_msg,
                "generated_folder": str(temp_folder),
            }
            return result

        # === Step 2: 验证完整解答 ===
        print(f"{Colors.BOLD}{Colors.CYAN}Step 2: Verifying Lean solution...{Colors.RESET}")
        success, error_msg, metrics = verify_lean_solution(temp_folder)

        eval_time = time.time() - start_time

        # 把验证结果整理成 Famou 通用 evaluator 接口需要的扁平指标格式。
        total_steps = metrics.get("total_steps", 0)
        verified_steps = metrics.get("verified_steps", 0)
        verification_ratio = verified_steps / total_steps if total_steps > 0 else 0.0
        validity = 1.0 if success else 0.0

        # 注意：在当前“全有或全无”的 validity 定义下，`combined_score`
        # 基本是二值的。这里仍保留显式公式，方便以后改成部分奖励。
        combined_score = verification_ratio * validity

        # === Save a copy of the temp folder to the permanent location ===
        # print(f"{Colors.BOLD}{Colors.CYAN}Step 3: Saving solution to {save_folder}...{Colors.RESET}")
        # if save_folder.exists():
        #     shutil.rmtree(save_folder)
        # shutil.copytree(temp_folder, save_folder)

        print(f"{Colors.BOLD}{Colors.GREEN}Evaluation complete: score={combined_score:.3f}, verified={verified_steps}/{total_steps}{Colors.RESET}")

        result = {
            "total_steps": total_steps,
            "verified_steps": verified_steps,
            "verification_ratio": float(verification_ratio),
            "validity": float(validity),
            "eval_time": float(eval_time),
            "combined_score": float(combined_score),
            "error_info": "Verification failed: " + error_msg if error_msg else None,
            "metrics": metrics,
            "generated_folder": str(temp_folder),
        }
        return result

    except Exception as e:
        print(f"{Colors.BOLD}{Colors.RED}Evaluation failed: {str(e)}{Colors.RESET}")

        result = {
            "total_steps": 0,
            "verified_steps": 0,
            "verification_ratio": 0.0,
            "validity": 0.0,
            "eval_time": 0.0,
            "combined_score": 0.0,
            "error_info": str(e) if e else None,
            "generated_folder": str(temp_folder),
        }
        return result

    finally:
        # 成功或正常返回的 run 会通过 `generated_folder` 把 temp 目录路径保留下来。
        # 下面这个清理分支主要是兜底：防止在 result 尚未完整构造时提前退出。
        if temp_folder.exists() and "generated_folder" not in result:
            try:
                shutil.rmtree(temp_folder)
                print(f"{Colors.BOLD}{Colors.BLUE}Cleaned up temp folder: {temp_folder}{Colors.RESET}")
            except Exception as e:
                print(f"{Colors.BOLD}{Colors.YELLOW}Warning: Could not clean up temp folder: {str(e)}{Colors.RESET}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python evaluator_lm.py path/to/your_program.py"
        )
        sys.exit(1)
    if ("json" in sys.argv[1]):
        json_path = Path(sys.argv[1])
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        code = data["code"]
        # code = "from scipy.optimize import differential_evolution, LinearConstraint\n" + code
        print(code)
        output_path = "temp.py"
        output = Path(output_path)
        with output.open("w", encoding="utf-8", newline="") as f:
            f.write(code)
    else:
        output_path = sys.argv[1]


    result = evaluate(output_path)
    print("---FINAL ERROR INFO---")
    print(result.get("error_info", "None"))
    print("--- RESULT ---")
    print(result)
