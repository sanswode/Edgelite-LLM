# results1

该目录用于集中整理“适合提交 / 适合上传 / 适合附论文展示”的实验结果材料。

## 目录说明

- `reports/`
  - 3 份核心自动报告
  - 包含参数化仿真报告、真实原型报告和统计增强报告

- `summaries/`
  - 关键汇总 CSV
  - 包含主实验、隐私实验、消融实验、真实原型、多随机种子统计与模型对比的核心统计结果

- `figures/main/`
  - 主实验与真实原型图表

- `figures/statistical/`
  - 统计增强实验图表

- `figures/model_compare/`
  - 模型规模与模型对比图表

## 有意未放入的内容

以下内容没有整理到 `results1`，这是刻意保留的结果：

- 原始大日志 `*raw_logs.csv`
- 运行时目录 `results/runtime/`
- 历史归档目录 `results/archive/`
- 统计实验生成的原始请求集 `results/statistical/datasets/`

原因是这些文件：

- 体积较大
- 对 GitHub 展示不友好
- 对论文正文和结果汇报不是必需材料

如果后续需要“完整版结果包”，可以再从原始 `results/` 目录中补充。
