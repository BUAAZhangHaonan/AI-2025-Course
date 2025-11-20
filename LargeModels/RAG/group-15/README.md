# 🚀**检索增强生成**项目概述——第15组

本项目围绕大模型生成内容的三个关键阶段——**生成前**、**生成中**、**生成后**——分别复现并对比以下三类事实性提升与幻觉抑制技术：

- **SelfCheckGPT**：生成后事实性检测
- **CoVe（Chain-of-Verification）**：生成中验证链
- **SELF-RAG**：按需检索、反思、再生成的循环增强生成

项目最终提供：

- 幻觉检测与抑制的定量实验
- 可视化的“初稿 → 验证问题 → 证据 → 修订稿”Demo
- RAGAS 指标评测（适用于 CoVe，SELF-RAG）

------

## 🧩 1. 项目任务概述

本任务旨在系统性比较 **生成前、中、后**三类事实性增强技术在开放域问答和摘要任务中的表现，包括：

### ✔ SelfCheckGPT

- 通过多次采样生成并对比句段级内容一致性
- 输出句级/段级幻觉警告

### ✔ CoVe（Chain-of-Verification）

- 生成过程中构造验证问题
- 对验证问题求证并基于证据修订最终答案

### ✔ SELF-RAG

- 按需检索（Retrieve）
- 反思（Reflect）
- 再生成（Regenerate）
- 引入 RAGAS 进行可信度评价

------

## 📦 2. 功能模块

本项目包含以下核心模块：

```
├─ chain-of-verification/     # CoVe 验证链流水线
├─ datasets/                  # 任务数据
├─ demo/                      # demo 展示
├─ eval/                      # RAGAS 评测
├─ selfcheckgpt/              # SelfCheckGPT 实现
├─ self-RAG/                  # SELF-RAG 策略实现
├─ visualize/                 # 可视化实现
└─ README.md
```

------

## 🛠 3. 使用说明

### 3.1 数据准备

将数据放入 `datasets/` 目录：

- 开放域问答数据：Wikidata Category Dataset 的前 20 条问答任务

------

### 3.2 运行 SelfCheckGPT（生成后检测）

```
pip install selfcheckgpt
jupyter nbconvert --to notebook --execute \
    demo/experiments/probability-based-baselines.ipynb \
    --output prob-baseline-output.ipynb
```

------

### 3.3 运行 CoVe（生成中验证链）

```
>> bash scripts/wikidata.sh # for wikidata task, and Llama2-70b model
apptainer build my_container.sif my_apptainer.def
SBATCH apptainer_job.sh
apptainer exec --nv my_apptainer.sif ./scripts/wikidata.sh # for Wikidata task and Llama2-70b
```

如使用 RAGAS 评测：

```
python eval/eval.py 
```

------

### 3.4 运行 SELF-RAG（按需检索→反思→再生成）

```
python run_short_form.py \
  --model_name selfrag/selfrag_llama2_7b \
  --input_file eval_data/wiki_processed3.jsonl \
  --mode adaptive_retrieval --max_new_tokens 100 \
  --threshold 0.2 \
  --output_file outputfact \
  --metric match --ndocs 10 --use_groundness --use_utility --use_seqscore \
  --dtype half
```

如使用 RAGAS 评测：

```
python eval/eval.py 
```



------

### 3.5 Demo（生成全过程可视化）

<video src="demo/demo.mp4" controls width="600"></video>

- 初稿回答
- 验证问题与证据
- 修订稿与对比可视化

------

## 📊 4. 实验结果展示

### ✨ 一致性/事实性提升

### **Ragas+CoVE**

| Model     | Precision | Recall | Answer  Correctness |
| --------- | --------- | ------ | ------------------- |
| **GPT-3** | 1.0000    | 0.5999 | 0.7482              |
| **CoVe**  | 1.0000    | 1.0000 | 0.9492              |

### **Ragas+**Self-RAG****

| Model                     | Precision | Recall | Faithfulness | Semantic  Similarity | Factual  Correctness |
| ------------------------- | --------- | ------ | ------------ | -------------------- | -------------------- |
| **llama2-7B**             | 0.45      | 0.55   | 0.69         | 0.73                 | 0.78                 |
| **Self-RAG  (llama2-7B)** | 0.75      | 0.75   | 0.88         | 0.88                 | 0.95                 |
| **提升幅度**  **(%)**     | +66.7%    | +36.4% | +27.5%       | +20.5%               | +21.8%               |

------


## 📚 5. 参考文献

- **SelfCheckGPT**
   *Potsawee Manakul et al., "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models"*
- **Chain-of-Verification（CoVe）**
   *Shehzaad Dhuliawala et al., "Chain-of-Verification Reduces Hallucination in Large Language Models".*
- **SELF-RAG**
   *Akari Asai et al., "Learning to Retrieve, Generate, and Critique via Self-Reflection".*

------

## 🧭 6. 参考仓库

- SelfCheckGPT: https://github.com/potsawee/selfcheckgpt
- SELF-RAG: https://github.com/AkariAsai/self-rag