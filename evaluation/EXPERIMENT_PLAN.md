# RAG 对比试验框架

## 数据边界

- Silver：快速回归和发现失败模式，不用于最终结论。
- Gold-dev（60题）：反复调参、选择方案。
- Gold-test（20题）：参数冻结后只运行一次。
- 教师答疑（51题）：独立困难集，评价端到端专业问答，不与教材检索 Gold 混算。

## 第一阶段：当前检索 Baseline

固定现有线上配置：无 rerank、阈值0.2、向量权重0.3、返回Top-4。一次请求保留
Top-10候选，以同时计算 Recall@1/4/10、MRR、nDCG@K、P50/P95延迟、上下文长度。

```powershell
python evaluation\run_ragflow_baseline.py `
  --split dev --threshold 0.2 --vector-weight 0.3 `
  --top-k 4 --candidate-k 10 `
  --output evaluation\output\experiments\e00_baseline_dev

python evaluation\analyze_retrieval_results.py `
  --input evaluation\output\experiments\e00_baseline_dev --cutoff 4
```

## 第二阶段：低成本参数消融

每次只改变一个变量，先使用 Gold-dev：

1. Top-K：2、4、6、10；观察 Recall、上下文长度和冗余度。
2. 相似度阈值：0.1、0.2、0.3、0.4；关注无结果率和误召回。
3. 向量权重：0.3、0.5、0.7；判断当前0.7关键词权重是否过强。

选择原则不是单看最高 Recall，而是寻找 Pareto 点：Recall@4/MRR 提升，同时 P95 延迟、
上下文 Token 和无结果率处于可接受范围。每个实验使用独立输出目录并记录完整配置。

## 第三阶段：检索结构优化

按失败样本决定是否实验：

- 标题重复导致误召回：章节标题转结构化元数据、降低标题字段权重。
- Gold 常落在5～20名：混合召回Top-20，再用 reranker 精排到Top-4。
- Gold 相邻段落缺失：父子检索或命中后扩展相邻 Chunk。
- OCR/公式失败集中：单独处理公式与图表，不应先全库重切。

## 第四阶段：端到端回答对照

在检索方案冻结后比较同一个生成模型、Prompt和温度：

- A：无 RAG；
- B：当前 Baseline RAG；
- C：优化后 RAG；
- D（可选）：优化检索 + rerank。

回答指标包括正确性、faithfulness、引用准确率、拒答准确率、TTFT、总延迟、输入/输出
Token和单次成本。检索指标与回答指标分开报告，避免“召回正确但生成错误”被混为一谈。

### 受控回答链路 A/B/C

先在dev上进行3题冒烟，再跑完整60题：

```powershell
python evaluation\run_e2e_answer_eval.py `
  --split dev --limit 3 `
  --variant no_rag --variant e00 --variant e02 `
  --output evaluation\output\e2e\answer_dev_smoke

python evaluation\run_e2e_answer_eval.py `
  --split dev --variant no_rag --variant e00 --variant e02 `
  --output evaluation\output\e2e\answer_dev --resume

python evaluation\analyze_e2e_results.py `
  --input evaluation\output\e2e\answer_dev
```

`no_rag/e00/e02`使用相同模型、系统Prompt和问题，仅改变是否检索以及混合检索参数。
脚本轮换每题的变体执行顺序，减少服务缓存或时段对某个变体的系统性偏置。`--resume`
可复用已成功请求，避免中断后重复付费。字符F1只用于自动筛查，正式正确性和faithfulness
应在`paired_answer_review.csv`中进行盲审或由经校准的Judge评分。

### 已部署Agent黑盒指标

真实Agent还包括历史读取、意图识别、业务接口和下一问预测，因此另做小规模用户体验测试：

```powershell
python evaluation\run_agent_e2e.py `
  --split dev --limit 10 `
  --output evaluation\output\e2e\agent_dev_smoke
```

它测量首个SSE事件、答案TTFT、答案完成和完整流结束时间。当前Agent接口不返回内部usage，
所以无法从黑盒精确拆分各LLM调用Token与成本；受控脚本也会明确标记字符估算，而不冒充
供应商账单。Agent测试不传`conv_id`，不会读取或写入真实会话历史。

## 最终测试

只在方案和参数全部冻结后运行：

```powershell
python evaluation\run_ragflow_baseline.py `
  --split test --threshold <冻结值> --vector-weight <冻结值> `
  --top-k <冻结值> --candidate-k 10 `
  --output evaluation\output\experiments\final_test
```
