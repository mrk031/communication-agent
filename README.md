# Communication Principles AI Agent

面向通信原理教学场景的智能问答 Agent。项目以确定性异步 Workflow 为主，组合
RAGFlow 教材检索、大模型流式回答、题库/教师答疑/案例/硬件实验服务，并提供离线
评测、端到端黑盒测试和阶段级 Trace。

> 本仓库是脱敏后的工程与评测代码快照。教材、教师真实答疑、服务器地址、账号、密钥、
> 本地运行输出及线上数据均未提交。运行前需要自行准备服务与测试数据。

## Architecture

```text
Text / Image question
        │
        ├── OCR and question fusion
        │
        ├── RAGFlow hybrid retrieval ── Top-10 candidates ── Top-4 context
        │
        └── rule-first intent routing
                │
                ├── main LLM answer (SSE)
                ├── question bank / teacher Q&A
                ├── interactive cases
                └── hardware experiments

All stages ── rag / intent / llm_answer / business / workflow Trace
```

当前系统更准确地属于“具备 Agent 能力的 Workflow”，而不是完全自主规划型 Agent：
路由、并发、超时、降级和结束条件由代码控制，LLM承担答案生成和结构化参数抽取。

## Highlights

- 规则优先的多标签意图路由，保留 `INTENT_ROUTER_MODE=llm` 回滚开关。
- RAGFlow 关键词与向量混合检索，冻结配置为向量权重 0.6、关键词权重 0.4。
- 基于统一 SSE 队列并发输出主答案、配置和业务 Section。
- 推荐追问与主回答解耦，附加服务失败不阻塞主答案。
- RAG、意图、模型回答及业务服务的阶段级 Trace。
- 覆盖检索层、答案层和完整 Agent 层的评测脚本。
- MCP 客户端工具发现与硬件设备健康检查示例。

## Evaluation Results

以下结果来自本地真实链路实验；原始私有数据和输出未提交。

| Layer | Result |
|---|---:|
| Frozen Gold-test retrieval | Hit@1 75%, Hit@4 85%, Hit@10 95% |
| Frozen Gold-test ranking | MRR 0.816, nDCG@4 0.813 |
| Answer comparison | RAG improved character F1 on 19/20 test cases |
| Agent success rate | 60/60 |
| Intent P50/P95 | 25.49/73.41 s → 0.078/0.185 ms |
| Full workflow P50/P95 | 27.34/73.96 s → 10.87/29.01 s |
| Unrelated business sections | 111 → 0 on 60 ordinary textbook questions |

字符 F1 仅用于自动筛查；正式结论需要结合语义正确性、证据忠实度和人工/Judge评审。

## Repository Layout

```text
tasks_distribute/     Agent入口、意图路由、RAG与业务编排、SSE Trace
evaluation/           数据构建、检索/答案/Agent评测及A/B分析
hardwareSet/          自然语言到硬件实验参数的结构化抽取示例
integrations/         MCP设备工具发现与健康检查示例
.env.example          脱敏配置模板
```

## Quick Start

### 1. Configuration

```powershell
Copy-Item .env.example .env
```

在本地填写必要地址和密钥。`.env`已被Git忽略，禁止提交真实凭据。

### 2. Start the Agent

```powershell
pip install -r tasks_distribute\requirements.txt
$env:RAGFLOW_API_KEY = "your-local-key"
$env:INTENT_ROUTER_MODE = "rules"
python tasks_distribute\tasks_distribute.py
```

默认入口：`POST http://127.0.0.1:9450/tasks_distribute`。

### 3. Run Tests

```powershell
python -m unittest discover -s evaluation -p "test_*.py"
```

### 4. Agent Black-box Evaluation

评测数据采用JSONL，每条至少包含：

```json
{"case_id":"example-1","question":"什么是QPSK？","reference_answer":"...","split":"dev"}
```

运行：

```powershell
python evaluation\run_agent_e2e.py `
  --cases path\to\gold.jsonl `
  --split dev `
  --agent-url http://127.0.0.1:9450/tasks_distribute `
  --output evaluation\output\agent_dev
```

## RAG Evaluation Design

- Gold：问题、参考答案、证据原文和 Gold Chunk ID 均经过审查。
- Silver：用于快速回归，不作为最终验收结论。
- Dev：允许反复调参。
- Test：参数和 Prompt 冻结后只运行一次，避免测试泄漏。
- 教师答疑困难集：用于外部验证，不与教材可回答 Gold 混为一体。

主要指标包括 Hit@K、Recall@K、Precision@K、MRR、nDCG、字符F1、TTFT、
P50/P95/P99、上下文Token估算和完整Workflow耗时。

## MCP Scope

`integrations/mcp_monitor.py`展示了基于 `MultiServerMCPClient` 的远程工具发现和设备
健康检查。`hardwareSet`当前仍通过普通 FastAPI/TCP 完成参数抽取与下发，本仓库不将其
宣称为完整 MCP Server。

若继续演进，可将实验检索、设备状态、参数解析和配置下发分别暴露为 MCP Tools；配置
下发属于写操作，应增加参数范围校验、权限控制、幂等键和人工确认。

## Security

- 所有服务地址和密钥通过环境变量注入。
- 教材、真实问答、会话、日志和评测输出不进入Git。
- RAG文本和工具结果应视为不可信输入，不能覆盖系统指令。
- 对硬件配置等写操作，应在生产环境启用审批、权限与审计。
- 如果历史版本中的密钥曾被暴露，应立即轮换；仅删除文件不能使旧密钥恢复安全。

## Known Limitations

- 仓库不包含线上业务服务和私有数据，因此无法一键复现完整生产拓扑。
- 当前没有通用上下文压缩，仅保留最近对话并为硬件实验维护专项实体记忆。
- 当前项目未配置独立Reranker。
- MCP Server和正式Agent Skill仍属于后续演进方向。

