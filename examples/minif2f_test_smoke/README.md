# miniF2F Test Smoke

这个目录提供一个最小 benchmark 入口，用来跑 `miniF2F` 的 `test` split。

默认行为是只 materialize 1 道题，不直接启动完整证明：

```bash
bash examples/minif2f_test_smoke/run.sh
```

如果要执行完整的 `construct + verify + summarize`：

```bash
RUN_MODE=full LIMIT=1 bash examples/minif2f_test_smoke/run.sh
```

默认会使用：

- `LEAN_VERIFY_TIMEOUT=300`
- `PER_TASK_TIMEOUT=1200`

如果你想覆盖它们：

```bash
RUN_MODE=full LIMIT=1 LEAN_VERIFY_TIMEOUT=600 PER_TASK_TIMEOUT=1800 \
bash examples/minif2f_test_smoke/run.sh
```

如果只想对已有 suite 重新汇总：

```bash
RUN_MODE=summarize bash examples/minif2f_test_smoke/run.sh
```

默认会取 `OUTPUT_ROOT` 下最新的一个 suite 目录。若你想指定某次 suite，也可以手动传：

```bash
RUN_MODE=summarize \
SUITE_ROOT=/path/to/minif2f_test_xxxxxxxx \
bash examples/minif2f_test_smoke/run.sh
```

常用环境变量：

- `LIMIT`: 题目数量，默认 `1`
- `PROBLEM_ID`: 指定某一道题，例如 `mathd_algebra_478`
- `LEAN_VERIFY_TIMEOUT`: 单次 Lean 校验超时秒数，默认 `300`
- `PER_TASK_TIMEOUT`: 单题总超时秒数，默认 `1200`
- `OUTPUT_ROOT`: suite 输出目录
- `BENCHMARK_ROOT`: miniF2F 数据目录

输出内容：

- `manifest.json`: suite 物化结果
- `runtime_manifest.json`: 每个 task 的运行入口
- `batch_run_summary.json`: 批量运行结果
- `suite_result_summary.json`: 汇总统计
- `suite_result_summary.csv`: 便于后续表格分析
