# 本地离线评测工具

该目录只用于本地评测和调试，不导入、启动或修改 `tasks_distribute`、RAGFlow、
MySQL、Elasticsearch、硬件服务。默认运行路径没有网络访问能力，也不会删除任何
RAG 数据。

## 快速运行

```powershell
python evaluation\evaluate.py validate
python evaluation\evaluate.py run
```

结果写入 `evaluation/output/`：

- `summary.json`：聚合指标；
- `predictions.jsonl`：逐题检索结果；
- `traces.jsonl`：逐题、逐阶段耗时和上下文长度。

## 替换为真实数据

将从服务器下载的知识库副本转换成 JSONL 后放在本地，例如：

```json
{"doc_id":"textbook-001","content":"QPSK 是……","source_type":"textbook","topic":"数字调制"}
```

教师答疑评测集格式：

```json
{"case_id":"qa-001","question":"QPSK 为什么有四个相位？","reference_answer":"……","gold_doc_ids":["textbook-001"],"holdout_doc_ids":["teacher-qa-001"],"intent_labels":["大模型概念回答"],"split":"test"}
```

`holdout_doc_ids` 只在内存中排除对应文档，绝不会删除源文件。建议把教师答疑原文
作为 holdout，把教材/课件中的相关证据标为 `gold_doc_ids`。

运行真实副本：

```powershell
python evaluation\evaluate.py run `
  --cases evaluation\data\teacher_qa_cases.jsonl `
  --corpus evaluation\data\rag_export.jsonl `
  --output evaluation\output\teacher_qa_baseline `
  --top-k 4
```

如已有系统回答，可以在 case 中加入 `predicted_answer`，工具会额外计算精确匹配、
字符级 F1 和参考答案覆盖率。这里的文本指标只能作为自动化基线，不能替代人工或
经过校准的 LLM Judge。

## 数据拆分原则

1. 按知识主题或原始文档拆分，避免相邻问答随机拆分造成泄漏。
2. `dev` 用于选择 Top-K、阈值和 Prompt；`test` 只用于最终比较。
3. 主评测保留教材证据、排除教师答疑原文；另建 `no_answer` 子集测拒答和幻觉。
4. 原始下载文件保持只读，所有排除、切分和检索均在内存中完成。

## 半自动生成评测集

先只执行分层抽样和证据窗口构建，不调用模型：

```powershell
python evaluation\generate_eval_questions.py --dry-run --passages 150
```

在 `evaluation/.env` 中配置专用于本地数据生成的 OpenAI-compatible 模型：

```dotenv
EVAL_LLM_BASE_URL=https://你的服务地址/v1
EVAL_LLM_API_KEY=你的密钥
EVAL_LLM_MODEL=服务实际支持的模型名
```

每个证据窗口默认生成两题，150 个窗口约得到 300 条 Silver 候选：

```powershell
python evaluation\generate_eval_questions.py --passages 150 --questions-per-passage 2
```

中断后可继续；已经成功的证据窗口不会再次收费：

```powershell
python evaluation\generate_eval_questions.py --passages 150 --questions-per-passage 2 --resume
```

若中转服务不支持 `response_format=json_object`，加入 `--no-json-mode`。输出位于
`evaluation/data/generated/eval_v0/`：`review_queue.jsonl` 是自动校验通过的待审核题，
`rejected_cases.jsonl` 保留自动拦截的题，`selected_passages.jsonl` 用于核对原始证据。
人工审核时将 `review_status` 改为 `true` 或 `false`，必要时直接修正问题、答案和
`gold_chunk_ids`。生成目录默认不进入版本控制，也不会上传或修改 RAGFlow。

关键任务节点记录在 `evaluation/PROGRESS.md`，供调试复盘和面试项目总结使用。

### 自动形成 Gold / Silver

对生成结果进行保守的离线证据审查和分层抽样：

```powershell
python evaluation\adjudicate_eval_set.py
```

输出位于 `evaluation/data/adjudicated/eval_v0/`：

- `gold_v0.jsonl`：高置信正式集，固定划分为 `dev` 与 `test`；
- `silver_v0.jsonl`：适合调参和回归测试的扩展集；
- `needs_review.jsonl`：证据链、Gold ID 或答案充分性不确定的题；
- `needs_review.csv`：便于在 Excel 中填写 `human_decision` 和 `human_note`；
- `summary.json`：数量、类型、章节及风险统计。

调参只能使用 `dev` 和 Silver；不要查看测试结果后再反向修改参数，否则会造成测试集泄漏。
原始 `review_queue.jsonl`、`rejected_cases.jsonl` 和模型响应始终保持不变。

## 从服务器只读导出

两个脚本都不会删除或更新业务数据。教师答疑脚本结束时只会清理临时Scroll游标，
不会删除索引或文档。凭据建议只放在当前PowerShell进程的环境变量中，
不要写入代码、命令历史或导出文件。

也可以把 `evaluation/.env.example` 复制为 `evaluation/.env` 并填写凭据。`.env` 已被
Git忽略，两个导出脚本会自动读取它。模板可以提交，但真实 `.env` 不应提交或发送。

先仅查看RAGFlow“通原”文档清单：

```powershell
$env:RAGFLOW_API_KEY = "你的RAGFlow API Key"
python evaluation\export_ragflow.py --list-only
```

确认清单后下载原文并导出Chunk：

```powershell
python evaluation\export_ragflow.py
```

教师答疑先试导出10条：

```powershell
$env:TEACHER_QA_ES_USERNAME = "elastic"
$env:TEACHER_QA_ES_PASSWORD = "你的Elasticsearch密码"
python evaluation\export_teacher_qa.py --limit 10
```

检查JSONL字段正常后全量导出：

```powershell
python evaluation\export_teacher_qa.py
```

导出默认落在 `evaluation/data/local/`，该目录已被 `.gitignore` 排除。
