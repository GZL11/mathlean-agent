# 最小运行说明

本文档只覆盖“如何把仓库跑起来”，不涉及证明质量优化。

## 1. 适用目录

仓库根目录：

```bash
<path-to>/mathlean-agent
```

以下命令默认都在仓库根目录执行。

## 2. 前置条件

需要具备：

- `conda` 环境：`famou`
- `claude` CLI 已可用
- 外网代理可用

推荐环境命令：

```bash
conda activate famou
export ALL_PROXY=http://<your-proxy>:<port>  # 如需代理
cd mathlean-agent
```

如果是新环境，先安装 Python 依赖：

```bash
pip install -r requirements.txt
```

## 3. 一次性初始化

初始化 Lean 工具链和 Claude MCP：

```bash
bash proof_runtime/setup.sh project
```

初始化完成后，建议检查：

```bash
source $HOME/.elan/env
lake --version
claude mcp list
```

预期至少看到：

- `lake` 可执行
- `lean-lsp` 为 `Connected`

如果想手动确认 Lean 项目可编译：

```bash
source $HOME/.elan/env
cd proof_runtime/project
lake build
cd ../..
```

## 4. 最小案例

当前内置了一个最小案例：实数绝对值三角不等式。

相关输入文件：

- `examples/triangle_inequality/problem.lean`
- `examples/triangle_inequality/init.scratch`
- `examples/triangle_inequality/config.yaml`
- `proof_runtime/evaluator_lean.py`

### 4.1 直接运行案例

推荐入口：

```bash
bash examples/triangle_inequality/run.sh
```

该脚本会：

1. 使用 `examples/triangle_inequality/config.yaml`
2. 使用 `examples/triangle_inequality/init.scratch`
3. 调用 `proof_runtime/evaluator_lean.py`
4. 启动一次完整的案例运行

### 4.2 手动运行同一个案例

```bash
python - <<'PY'
import importlib.util
import json
import os
from pathlib import Path

repo = Path.cwd()
os.environ["FAMOU_LEAN_CONFIG"] = str(repo / "examples" / "triangle_inequality" / "config.yaml")
path = repo / "proof_runtime" / "evaluator_lean.py"
spec = importlib.util.spec_from_file_location("evaluator_lean", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

scratch = repo / "examples" / "triangle_inequality" / "init.scratch"
result = mod.evaluate(str(scratch))
print(json.dumps(result, ensure_ascii=False, indent=2))
PY
```

### 4.3 新建题目

建议直接从模板复制：

```bash
cp -R examples/_template examples/my_new_problem
```

然后修改：

- `examples/my_new_problem/problem.lean`
- `examples/my_new_problem/init.scratch`
- `examples/my_new_problem/config.yaml`

最后运行：

```bash
bash examples/my_new_problem/run.sh
```

## 5. 输出位置

常见输出位置：

- `famou_data/`：实验日志与中间结果
- `proof_runtime/project/temp/`：单次 proof 构造与验证的临时目录

如果一次运行成功进入 proof 构造阶段，通常会看到：

- `definition.lean`
- `step_1.lean`
- `step_2.lean`
- `step_3.lean`

## 6. 当前推荐的“可运行基线”

如果你只想确认代码库“能正常跑起来”，建议按下面顺序检查：

```bash
conda activate famou
export ALL_PROXY=http://<your-proxy>:<port>  # 如需代理
cd mathlean-agent

bash proof_runtime/setup.sh project
bash examples/triangle_inequality/run.sh
```

这套流程关注的是：

- 环境是否完整
- Lean 工具链是否可用
- MCP 是否连通
- evaluator 是否能启动
- proof steps 是否能生成

不把“最终证明质量必须完美”作为这一步的验收标准。

## 7. 常见问题

### `claude mcp list` 里没有 `lean-lsp`

重新执行：

```bash
bash proof_runtime/setup.sh project
```

### `lake` 找不到

执行：

```bash
source $HOME/.elan/env
```

### 代理相关问题

当前默认使用：

```bash
export ALL_PROXY=http://<your-proxy>:<port>  # 如需代理
```

如果外网下载失败，优先检查这个变量是否仍然存在。
