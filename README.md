# mathlean-agent

LLM agent + evolutionary search for Lean 4 theorem proving: evolves structured proof plans, repairs subgoals within a fixed plan, and uses Lean compiler verification as feedback. Supports miniF2F & PutnamBench.

`mathlean-agent` 是一个面向 Lean 数学证明任务的 `agent + evolve + Lean verify` 实验代码库。  
当前仓库已经包含：

- 单题示例运行入口
- Lean 校验运行时
- benchmark 物化与批量执行工具
- `miniF2F-test` 与 `PutnamBench` 的适配代码


## 1. 仓库结构

- `examples/`
  存放单题案例和 benchmark 入口脚本。
- `examples/triangle_inequality/`
  最小数学证明案例。
- `examples/minif2f_test_smoke/`
  `miniF2F-test` benchmark 入口。
- `proof_runtime/`
  共享 Lean 证明运行时与 evaluator。
- `third_party/benchmarks/miniF2F/`
  `miniF2F` 数据。
- `third_party/benchmarks/PutnamBench/`
  `PutnamBench` 数据。
- `famou/`
  演化框架、mathproof 策略、benchmark runner 等核心逻辑。

## 2. 环境要求

当前仓库已经验证过的基础要求：

- Linux
- Python `>= 3.12`
- `conda` 可用
- 可访问 Lean / 模型服务所需网络
- `claude` CLI 可用

推荐做法：

```bash
conda activate famou
```

如果你还没有该环境，可以手动创建：

```bash
conda create -n famou python=3.12 -y
conda activate famou
pip install -r requirements.txt
```

说明：

- `requirements.txt` 使用默认 PyPI 源；如需镜像源，可在安装时追加 `-i <index-url>`。

## 3. 安装 Python 依赖

`third_party/benchmarks/` 下的 `miniF2F` 与 `PutnamBench` 以 git submodule 形式引入，克隆时请带上 submodule：

```bash
git clone --recurse-submodules https://github.com/GZL11/mathlean-agent.git
# 已克隆的仓库可补执行：git submodule update --init --recursive
```

进入仓库根目录后执行：

```bash
pip install -r requirements.txt
```

如果你只想验证安装是否完成，可以执行：

```bash
python -c "import famou; print('famou import ok')"
python -c "import openai, yaml, pydantic; print('deps import ok')"
```

## 4. 初始化 Lean 与 MCP

第一次运行前，先完成 Lean 工具链和 `lean-lsp` MCP 初始化：

```bash
bash proof_runtime/setup.sh project
```

初始化完成后，执行：

```bash
source "$HOME/.elan/env"
lake --version
claude mcp list
```

预期结果：

- `lake` 可以正常输出版本
- `claude mcp list` 中能看到 `lean-lsp`

如果 `lake` 找不到：

```bash
source "$HOME/.elan/env"
```

如果 `claude mcp list` 中没有 `lean-lsp`：

```bash
bash proof_runtime/setup.sh project
```

## 5. 运行前的配置说明

### 5.1 Lean 环境

多数入口脚本都会尝试自动执行：

```bash
source "$HOME/.elan/env"
```

但如果你是在新 shell 中手动运行命令，仍建议先显式执行一次。


### 5.2 LLM 配置

单题示例和 benchmark 模板都会从各自的 `config.yaml` 中读取模型配置。  
运行前请检查这些字段是否适合你的环境：

- `infrastructure.llm.provider`
- `infrastructure.llm.api_base`
- `infrastructure.llm.model`
- `infrastructure.llm.api_key`

相关文件：

- `examples/triangle_inequality/config.yaml`
- `examples/minif2f_test_smoke/config.yaml`

## 6. 最小可运行案例

仓库当前最直接的最小案例是 `examples/triangle_inequality`。

### 6.1 一条命令启动

在仓库根目录执行：

```bash
bash examples/triangle_inequality/run.sh
```

该脚本实际会使用：

- `examples/triangle_inequality/config.yaml`
- `examples/triangle_inequality/init.scratch`
- `examples/triangle_inequality/problem.lean`
- `proof_runtime/evaluator_lean.py`

### 6.2 等价的手动启动命令

```bash
source "$HOME/.elan/env"
python run_famou.py \
  --config examples/triangle_inequality/config.yaml \
  --programs examples/triangle_inequality/init.scratch \
  --evaluator proof_runtime/evaluator_lean.py
```

### 6.3 成功启动时你会看到什么

正常情况下日志中会出现：

- `Loading evaluator from: proof_runtime/evaluator_lean.py`
- `Loading configuration from: examples/triangle_inequality/config.yaml`
- `Loading initial program from: examples/triangle_inequality/init.scratch`
- `Starting evolution`

## 7. 运行 miniF2F-test

### 7.1 smoke：先验证 benchmark 链路是否通

最简单的 smoke 方式：

```bash
bash examples/minif2f_test_smoke/run.sh
```

默认行为：

- benchmark: `miniF2F`
- split: `test`
- `LIMIT=1`
- `RUN_MODE=prepare`

也就是说，它默认只做 benchmark 物化，不做完整证明。

### 7.2 跑完整的 `prepare + construct + summarize`

```bash
RUN_MODE=full LIMIT=1 bash examples/minif2f_test_smoke/run.sh
```

这条命令会：

- 从 `third_party/benchmarks/miniF2F` 读取题目
- 物化单题任务目录
- 调用 `agent + evolve + Lean verify`
- 输出 suite 级汇总结果

### 7.3 推荐的真实单题运行命令

一个典型的单题运行方式：

```bash
RUN_MODE=full \
LIMIT=1 \
PROBLEM_ID=mathd_algebra_478 \
OUTPUT_ROOT=./benchmark_runs/minif2f_real1 \
bash examples/minif2f_test_smoke/run.sh
```

### 7.4 常用环境变量

- `RUN_MODE`
  可选 `prepare`、`full`、`summarize`
- `LIMIT`
  物化并执行前多少题
- `PROBLEM_ID`
  指定单题，例如 `mathd_algebra_478`
- `OUTPUT_ROOT`
  benchmark suite 输出目录
- `LEAN_VERIFY_TIMEOUT`
  Lean 单次校验超时，默认 `300`
- `PER_TASK_TIMEOUT`
  单题整体执行超时，默认 `1200`

### 7.5 只重做汇总

如果题目已经跑过，只想重新做 suite 汇总：

```bash
RUN_MODE=summarize \
OUTPUT_ROOT=./benchmark_runs/minif2f_real1 \
bash examples/minif2f_test_smoke/run.sh
```

如果无法自动推断 `SUITE_ROOT`，可以显式指定：

```bash
RUN_MODE=summarize \
SUITE_ROOT=./benchmark_runs/minif2f_real1/<suite_dir> \
bash examples/minif2f_test_smoke/run.sh
```

## 8. 运行 PutnamBench

当前仓库还没有单独的 `examples/putnambench/run.sh`。  
因此 `PutnamBench` 目前推荐直接调用 benchmark runner。

### 8.1 运行 1 题真实 pilot

一个典型的 PutnamBench pilot 命令：

```bash
conda activate famou
source "$HOME/.elan/env"
python -B -m famou.utils.mathproof_benchmark_runner \
  --benchmark putnambench \
  --benchmark-root third_party/benchmarks/PutnamBench \
  --output-root ./benchmark_runs/putnambench_materialized \
  --limit 1 \
  --config-template examples/minif2f_test_smoke/config.yaml \
  --storage-base-path ./benchmark_runs/putnambench_suite_data \
  --run \
  --summarize \
  --per-task-timeout 900
```

### 8.2 指定某一道 Putnam 题

例如只跑 `putnam_1962_a1`：

```bash
conda activate famou
source "$HOME/.elan/env"
python -B -m famou.utils.mathproof_benchmark_runner \
  --benchmark putnambench \
  --benchmark-root third_party/benchmarks/PutnamBench \
  --output-root ./benchmark_runs/putnam_single \
  --problem-id putnam_1962_a1 \
  --config-template examples/minif2f_test_smoke/config.yaml \
  --storage-base-path ./benchmark_runs/putnam_single_suite_data \
  --run \
  --summarize \
  --per-task-timeout 900
```

### 8.3 只做物化，不做完整执行

```bash
python -B -m famou.utils.mathproof_benchmark_runner \
  --benchmark putnambench \
  --benchmark-root third_party/benchmarks/PutnamBench \
  --output-root ./benchmark_runs/putnam_prepare \
  --limit 1 \
  --config-template examples/minif2f_test_smoke/config.yaml
```

## 9. benchmark 输出目录说明

无论是 `miniF2F-test` 还是 `PutnamBench`，完整运行后通常会看到这些文件：

- `manifest.json`
  物化后的题目清单
- `runtime_manifest.json`
  每道题的运行时入口信息
- `batch_run_summary.json`
  每道题的批量执行结果
- `suite_result_summary.json`
  suite 级汇总
- `suite_result_summary.csv`
  便于表格查看的汇总

单题实验目录中常见文件：

- `experiment.log`
  该题完整运行日志
- `experiment_checkpoint_*.json`
  演化过程 checkpoint
- `programs/*.json`
  每个候选 proof object
- `programs/*.lean`
  对应导出的 Lean 文本

## 10. 证明成功的判定

对于当前 mathproof 策略，题目被视为证明成功，至少需要满足：

- `success_score = 1.0`
  或
- `theorem_solved = 1.0`

在当前实现中，这意味着最终 theorem 的 Lean 证明已经通过校验，而不是只生成了形式上看起来合理的计划。

如果只是：

- theorem statement 对齐
- 没有 `sorry`
- 但 Lean 没编译通过

那么仍然不会算作成功。

## 11. 新建一个自己的题目

复制模板：

```bash
cp -R examples/_template examples/my_new_problem
```

然后至少修改这三个文件：

- `examples/my_new_problem/problem.lean`
- `examples/my_new_problem/init.scratch`
- `examples/my_new_problem/config.yaml`

最后运行：

```bash
bash examples/my_new_problem/run.sh
```

模板参考：

- `examples/_template/README.md`
- `examples/_template/config.yaml`
- `examples/_template/problem.lean`
- `examples/_template/init.scratch`
- `examples/_template/run.sh`

## 12. 最小命令速查

### 12.1 最快确认仓库能跑

```bash
conda activate famou
pip install -r requirements.txt
bash proof_runtime/setup.sh project
bash examples/triangle_inequality/run.sh
```

### 12.2 最快确认 miniF2F benchmark 链路能跑

```bash
bash examples/minif2f_test_smoke/run.sh
```

### 12.3 跑 1 道真实 miniF2F

```bash
conda activate famou
source "$HOME/.elan/env"
RUN_MODE=full LIMIT=1 PROBLEM_ID=mathd_algebra_478 OUTPUT_ROOT=./benchmark_runs/minif2f_real1 bash examples/minif2f_test_smoke/run.sh
```

### 12.4 跑 1 道真实 PutnamBench

```bash
conda activate famou
source "$HOME/.elan/env"
python -B -m famou.utils.mathproof_benchmark_runner \
  --benchmark putnambench \
  --benchmark-root third_party/benchmarks/PutnamBench \
  --output-root ./benchmark_runs/putnambench_materialized \
  --limit 1 \
  --config-template examples/minif2f_test_smoke/config.yaml \
  --storage-base-path ./benchmark_runs/putnambench_suite_data \
  --run \
  --summarize \
  --per-task-timeout 900
```

## 13. 常见问题

### 13.1 `lake` 找不到

执行：

```bash
source "$HOME/.elan/env"
```

### 13.2 `claude mcp list` 中没有 `lean-lsp`

执行：

```bash
bash proof_runtime/setup.sh project
```

### 13.3 benchmark 运行很快结束，但没有真正开始证明

先检查：

- 是否只用了 `RUN_MODE=prepare`
- 是否激活了正确的 Python 环境
- Lean 临时 workspace 是否有写权限
- 模型服务和代理是否可用

### 13.4 `success_score=0`，但生成了看起来像证明的 Lean 代码

这通常表示以下某一种情况：

- Lean 没编译通过
- 最终 theorem 没有真正闭合
- 仍然残留 `sorry`
- 验证超时

只有最终 theorem 被 Lean 验证通过，才算真正成功。

## 14. 已验证的 benchmark pilot

截至当前仓库状态，下面两条真实 pilot 已经跑通过端到端链路：

- `miniF2F-test`，`limit=1`
- `PutnamBench`，`limit=1`

它们已经确认：

- benchmark 物化正常
- `agent + evolve + Lean verify` 链路正常
- suite 汇总正常

但在当前默认 budget 下，这两条 pilot 都还没有收敛到 `success_score=1.0`。

## 15. 维护说明

如需维护或扩展本仓库，建议优先阅读：

- `README.md`
- `docs/mathproof/current_codebase_status_and_runtime.md`
- `examples/_template/`
