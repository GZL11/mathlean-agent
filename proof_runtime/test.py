import sys
print(sys.executable)
from numina_lean_agent.scripts.run_claude import ClaudeRunner
# import numina_lean_agent.scripts.run_claude
from pathlib import Path 
import os
# import mcp
cwd = os.getcwd()
# print(cwd)

file_path = Path(__file__).resolve()
# print(file_path.parent)

def run_with_timeout(program_path, timeout_seconds=20, function_name="construct_graph_coloring"):
    """
    Run the program in a separate process with timeout
    using a simple subprocess approach

    Args:
        program_path: Path to the program file
        timeout_seconds: Maximum execution time in seconds

    Returns:
        centers, radii, sum_radii tuple from the program
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
from numina_lean_agent.scripts import run_claude

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

if __name__ == "__main__":
    runner = ClaudeRunner()
    # target = file_path.parent / "examples" / "Basic.lean"
    # target = file_path.parent / "examples" / "algebra_sqineq_2atp2bpge2ab.lean"
    target = file_path.parent / "project" / "problems" / "simple.lean"
    prompt_file = file_path.parent / "examples" / "prompt_complete_file.txt"
    # print(f"{target}\n{prompt_file}")
    # prompt_file = 
    model_name = "sonnet"  # [haiku, sonnet, opus], from fast to slow, weak to strong ability
    max_rounds = 5
    # result = runner.run(target=target, prompt_file=prompt_file, model_name=model_name, max_rounds=max_rounds)
    
    permission_mode == None
    base = ["claude", "--print"]
    if permission_mode:
        base += ["--permission-mode", permission_mode]
    base.append(full_prompt)
    result = subprocess.run(
        base,
        capture_output=True,
        text=True,
        timeout=300  # 5 minute timeout
    )