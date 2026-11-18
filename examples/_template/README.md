# Example Template

复制本目录即可创建一个新题目：

```bash
cp -R examples/_template examples/my_new_problem
```

然后至少修改：

1. `problem.lean`
2. `init.scratch`
3. `config.yaml`

最后直接运行：

```bash
bash examples/my_new_problem/run.sh
```

## 约定

- `problem.lean`: 目标定理
- `init.scratch`: 初始证明提纲
- `config.yaml`: 当前题目的实验配置
- `run.sh`: 当前题目的一键运行入口

`run.sh` 已经是通用脚本，不需要改路径逻辑。
