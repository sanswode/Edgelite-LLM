# EdgeLite-LLM

面向资源受限边缘设备的 LLM 协同推理与缓存优化实验原型。

本项目围绕以下研究问题展开：在单台个人设备、有限显存和动态网络条件下，如何通过隐私门控、语义缓存、Prefix/KV 缓存以及状态感知路由优化，在时延、带宽、显存与回答质量之间取得更优平衡。


## 1. 仓库内容

本仓库包含：

- 参数化仿真实验代码
- 单机真实边缘 / 模拟云端原型实验代码
- 多随机种子统计增强实验代码
- 模型对比、图表绘制与报告生成脚本
- 论文正文所用图表与 Markdown 草稿

## 2. 目录结构

```text
edgelite_llm_experiment/
├── README.md
├── requirements.txt
├── .gitignore
├── client/                       # 请求加载与回放
├── configs/                      # 实验配置
├── data/                         # 示例请求数据
├── models/                       # 模型目录（默认不提交权重）
├── results/                      # 实验结果、报告、图表
├── scripts/                      # 运行脚本与绘图脚本
├── server/                       # 调度、缓存、隐私、模拟器核心逻辑
└── server_help.txt               
```

## 3. 环境安装

### 3.1 创建环境

```powershell
conda create -n edgeLLM python=3.12 -y
conda activate edgeLLM
```

### 3.2 安装 Python 依赖

```powershell
pip install -r requirements.txt
```

### 3.3 安装 GPU 版 PyTorch

本项目的参数化实验本身并不强依赖 GPU PyTorch，但如果你希望和当前实验环境尽量保持一致，建议单独安装 CUDA 12.1 对应版本：

```powershell
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
```

### 3.4 安装 `llama-cpp-python`（真实本地 GGUF 实验可选）

如果你只运行参数化仿真实验，这一步不是必须的。

如果你需要运行 `scripts/run_real_experiment_local.py`，则需要额外安装 `llama-cpp-python`。Windows + CUDA 的安装方式容易受本机编译环境影响，因此建议单独安装，不把它硬编码进 `requirements.txt`。

一个常见安装方式如下：

```powershell
$env:CMAKE_ARGS="-DGGML_CUDA=on"
$env:FORCE_CMAKE="1"
pip install llama-cpp-python
```

如果你已有可用版本，也可以直接复用现有环境。

## 4. 如何运行

### 4.1 参数化仿真实验

```powershell
& python scripts\run_experiment.py
```

该脚本会：

1. 生成实验请求
2. 运行主对比实验
3. 运行隐私与语义阈值分析
4. 运行消融实验
5. 生成汇总 CSV、图表和 `results/report.md`

### 4.2 多随机种子统计增强实验

```powershell
& python scripts\run_statistical_experiments.py
```

该脚本会额外生成：

- 多随机种子统计结果
- 置信区间
- 显著性检验
- 误差棒图
- 多模型四策略对比图

核心输出位于：

- `results/statistical/report.md`
- `results/statistical/main_significance.csv`
- `results/statistical/model_strategy_ci_summary.csv`
- `results/statistical/figures/`

### 4.3 真实原型实验

如果你已经准备好本地 GGUF 模型和对应服务，可运行：

```powershell
& python scripts\run_real_experiment.py
```

或更推荐的本地直接推理版本：

```powershell
& python scripts\run_real_experiment_local.py
```

常用配置文件：

- `configs/real_runtime_local.yaml`
- `configs/real_runtime_dual.yaml`
- `configs/real_runtime_emulated_cloud.yaml`

## 8. 当前设备说明

本项目当前实验主要围绕以下设备完成：

- CPU: Intel i9-14900HX
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- VRAM: 8 GB
- RAM: 32 GB

因此，项目在真实实验路径上更偏向：

- 1B - 4B 量化 GGUF 模型
- 边缘单模型优先
- 单机真实边缘 + 模拟云端
- 上下文长度控制在较稳妥范围

## 9. 论文相关文件

如果你要把这个仓库作为论文附带代码仓库，重点文件如下：

- 算法理论：[paper/section3_algorithm_theory.md](./paper/section3_algorithm_theory.md)
- 框架图（中文）：`paper/figures/figure3_framework_architecture_topconf_cn.png`
- 参数化实验报告：`results/report.md`
- 真实原型报告：`results/real_report.md`
- 统计增强报告：`results/statistical/report.md`