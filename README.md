# MELON：基于自适应掩码重执行的 LLM Agent 间接提示注入防御

> 基于 MELON 论文的防御机制改进实现——MELON-R（风险探针+分级路由）、MELON-SATM（语义感知定向掩码）、MELON-AT（自适应阈值）

- **仓库地址**：https://github.com/xwsdid/MELON
- **作者**：林昕然、谢玮珊
- **原论文**：[MELON: Provable Defense Against Indirect Prompt Injection Attacks in AI Agents](https://github.com/kaijiezhu11/MELON) (ICML 2025)
- **基础框架**：[AgentDojo](https://github.com/ethz-spylab/agentdojo) (NeurIPS 2024 Datasets & Benchmarks)

## 项目简介

本项目基于 MELON 论文提出的对比式提示注入检测框架，在 AgentDojo 基准测试环境中实现了三种改进方案：

| 改进方案 | 核心思路 | 代码位置 |
|----------|---------|---------|
| **MELON-R** | 轻量级风险探针 + 分级路由(PASS/SANITIZE/RECOVERY/BLOCK) | `src/agentdojo/agent_pipeline/pi_detector.py` → `MELONR` 类 |
| **MELON-SATM** | 语义感知定向掩码——七组正则规则替换高危片段 | `src/agentdojo/agent_pipeline/pi_detector.py` → `MELONSM` 类 |
| **MELON-AT** | 自适应阈值——仅用良性样本分数校准，动态平衡误报与漏报 | `src/agentdojo/agent_pipeline/pi_detector.py` → `MELONAT` 类 |

## 环境与依赖

### 运行环境

| 项目 | 版本 | 说明 |
|------|------|------|
| 操作系统 | Windows 11 | 开发与测试所用系统 |
| Python | 3.12 | 核心语言版本 |
| 包管理器 | uv | 用于依赖管理和虚拟环境 |
| GPU | 无 | 仅需 API 调用，无需本地推理 |

### 关键依赖

| 依赖名称 | 使用版本 | 说明 |
|----------|----------|------|
| openai | ≥1.59.7 | OpenAI API 客户端（LLM 调用 + 嵌入向量） |
| anthropic | ≥0.47.0 | Anthropic API 客户端（可选） |
| numpy | — | 余弦相似度计算 |
| pydantic | ≥2.7.1 | 数据模型定义 |

完整依赖清单见 `pyproject.toml`。

## 配置说明

### API Key 配置

在项目根目录创建 `.env` 文件（已加入 `.gitignore`）：

```
OPENAI_API_KEY=sk-your-key-here
```

本实验使用 OpenAI GPT-4o 的 Chat Completion API（用于 LLM 推理）和 text-embedding-3-large（用于工具调用嵌入向量）。

## 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/xwsdid/MELON.git
cd MELON

# 2. 安装依赖
uv sync --no-lock

# 3. 配置 API Key
cp .env.example .env
# 编辑 .env，填入你的 OpenAI API Key

# 4. 运行实验
uv run python -m agentdojo.scripts.benchmark \
    -s banking \
    --model gpt-4o-2024-05-13 \
    --defense melon_at \
    --attack important_instructions
```

可用的 `--defense` 选项：`melon`（原始）、`melon_r`、`melon_sm`、`melon_at`

可用的 `--attack` 选项：`direct`、`ignore_previous`、`important_instructions`

## 实验结果

实验基于 AgentDojo banking 场景，GPT-4o 模型，三类攻击（direct / important_instructions / ignore_previous）。

### 任务级指标

| 方法 | Direct UA | Direct ASR | Important UA | Important ASR | Ignore UA | Ignore ASR |
|------|:---------:|:----------:|:------------:|:-------------:|:---------:|:----------:|
| No Defense | 43.06% | 55.56% | 45.83% | 51.30% | 44.44% | 52.37% |
| MELON | 38.19% | 0.00% | 29.86% | 0.00% | 37.50% | 0.00% |
| MELON-R | 45.83% | 2.08% | 44.44% | 8.33% | 43.06% | 2.78% |
| MELON-SM | 52.08% | 0.00% | 33.33% | 2.08% | 52.78% | 0.00% |
| MELON-AT | 62.50% | 0.00% | 43.06% | 13.19% | 60.42% | 0.00% |

### 检测级指标（important_instructions）

| 指标 | MELON-SM (SATM) | MELON-AT |
|------|:---:|:---:|
| Precision | 100.0% | 100.0% |
| Recall | 81.4% | 78.6% |
| FPR | 0.0% | 0.0% |
| FNR | 18.6% | 21.4% |

完整实验结果保存在 `runs/` 目录。分析和解读详见 `report/MELON_SATM_AT_补充内容.md`。

## 项目结构

```
MELON/
├── src/agentdojo/agent_pipeline/
│   ├── pi_detector.py          # MELON 系列检测器核心代码（主要改动）
│   ├── agent_pipeline.py       # Pipeline 配置与 defense 注册
│   └── llms/openai_llm.py      # OpenAI LLM 后端
├── tests/                      # 测试脚本
│   ├── run_small_compare_melon.py
│   └── test_melonrp_probe_small.py
├── runs/                       # 实验结果
│   ├── gpt-4o-2024-05-13-melon*/
│   └── gpt-4o-mini-2024-07-18-melon*/
├── report/                     # 课程设计报告与文档
│   ├── 课程设计报告初版.docx
│   ├── MELON_SATM_AT_补充内容.md
│   ├── 答辩PPTv2.pptx
│   └── MELON Provable Defense Against Indirect Prompt Injection.pdf
├── pyproject.toml              # 依赖清单
├── .gitignore
└── README.md                   # 本文件
```

## 主要代码改动说明

| 文件 | 改动内容 |
|------|---------|
| `src/.../pi_detector.py` | 新增 `MELONR`（风险探针+分级路由）、`MELONSM`（语义掩码）、`MELONAT`（自适应阈值）三个检测器类；新增 `DetectionStats` 全局统计模块；修复 HIGH_RISK_PATTERNS 为纯英文正则 |
| `src/.../agent_pipeline.py` | 注册 `melon_r`、`melon_sm`、`melon_at` 三个 defense；修复阈值从 0.1 恢复为论文默认 0.8 |
| `src/.../llms/openai_llm.py` | 添加代理支持（`http://127.0.0.1:7890`） |

## 致谢

本项目基于以下开源工作：
- [MELON](https://github.com/kaijiezhu11/MELON) — 原始对比式检测框架
- [AgentDojo](https://github.com/ethz-spylab/agentdojo) — 提示注入攻击与防御评估环境
