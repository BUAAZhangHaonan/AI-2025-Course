# CoVe Ragas评估

## 指标选择

precision、recall、answer_correctness

### 选择依据

选择这三个指标是为了**全面且鲁棒地**评估LLM在开放域问答中的表现：

1. **精确率 (Precision):** 评估回答的**准确性**和**忠实度**。它确保LLM的回答中不包含错误信息（幻觉）或与问题无关的“噪音”。（即：说的是对的吗？）
2. **召回率 (Recall):** 评估回答的**完整性**和**覆盖度**。它确保LLM的回答覆盖了“标准答案”中的所有关键信息点，没有遗漏。（即：该说的都说了吗？）
3. **Answer Correctness (RAGAS):** 评估回答在**语义和事实层面**与标准答案的一致性。这是对P/R的**关键升级**，因为它解决了传统P/R依赖“词汇重叠”的缺陷。LLM可能用完全不同的词句（释义）来表达相同且正确的意思，此指标能识别这一点，从而更准确地衡量回答的**真实正确性**。

**简言之：** Precision和Recall从“量”上保证了回答的**准确**和**完整**，而Answer Correctness从“质”上保证了回答在**意思**上是正确的，即使措辞不同。

## 数据集和任务

### 任务

开放域回答

### 数据集

Wikidata Category Dataset 的前 20 条

样例：

```json
{
    "question": "Name some Animals that can only be found in Oman",
    "answer": [
        "Arabian toad",
        "Asaccus gallagheri",
        "Asaccus platyrhynchus",
        "Acanthodactylus masirae",
        "Hemidactylus luqueorum",
        "Asaccus arnoldi",
        "Dhofar toad",
        "Hemidactylus masirahensis",
        "Asaccus margaritae"
    ]
}
```

## 模型

1. BaseModel (GPT-3)

2. CoVe

## 实验设置

### 数据预处理

- JSON 文件字段重命名：
   `question → user_input`，`answer → reference`
- CSV 文件字段重命名：
   `Question → user_input`
- 将两份数据通过 `user_input`（问题）进行合并。
- 生成两个评估数据集：
  - `dataset_base`：包含 `user_input`, `reference`, `response(BaseModelAnswer(GPT-3))`
  - `dataset_cove`：包含 `user_input`, `reference`, `response(CoVeAnswer)`

### 评估模型与嵌入模型

| 模块         | 类                                             | 配置                 |
| ------------ | ---------------------------------------------- | -------------------- |
| LLM 评估器   | `LangchainLLMWrapper(ChatOpenAI)`              | 模型: `"gpt-4o"`     |
| 向量嵌入模型 | `LangchainEmbeddingsWrapper(OpenAIEmbeddings)` | 默认 OpenAI 嵌入模型 |

### 评估指标（Metrics）

#### 自定义指标

1. **CustomPrecision（自定义精确率）**
   - 输入列：`user_input`, `response`, `reference`
   - 计算公式：
     $ [
      Precision = \frac{|response \cap reference|}{|response|}
      ]$
   - 异常处理：若无法解析为集合或为空集，返回 0 或 1（视边缘情况）。
2. **CustomRecall（自定义召回率）**
   - 输入列：`user_input`, `response`, `reference`
   - 计算公式：
     $ [
      Recall = \frac{|response \cap reference|}{|reference|}
      ]$
   - 异常处理逻辑同上。

#### 现有指标

3. **`answer_correctness`**（来自 `ragas.metrics`）

## 实验结果

|                   | precision | recall | answer correctness |
| ----------------- | --------- | ------ | ------------------ |
| BaseModel (GPT-3) | 1.0000    | 0.5999 | 0.7482             |
| CoVe              | 1.0000    | 1.0000 | 0.9492             |

## 结果分析

### 核心结论

**CoVe 模型在回答质量上全面且显著地优于 BaseModel (GPT-3)。**

CoVe 不仅能确保回答的准确性（更少幻觉），还能保证信息的完整性（更少遗漏），其回答在语义和事实上也与标准答案更一致。

### 详细分析

1. **精确率 (Precision)**
   - **BaseModel (GPT-3):** 1.0000
   - **CoVe:** 1.0000
   - **分析：** 两个模型在此项上均获得满分。这表明**两个模型都具有极高的忠实度**，它们的回答内容都是准确的、相关的，**没有产生“幻觉” (Hallucinations)** 或包含无关的“噪音”信息。
2. **召回率 (Recall)**
   - **BaseModel (GPT-3):** 0.5999
   - **CoVe:** 1.0000
   - **分析：** 这是两个模型表现差异最悬殊的指标。
     - **CoVe (1.0000)** 获得了满分，说明它**完美地覆盖了**标准答案中的所有关键信息点，做到了**“应答尽答”，没有任何遗漏**。
     - **BaseModel (0.5999)** 得分很低，表明它**存在严重的信息遗漏**。它的回答虽然正确（高Precision），但内容不完整，大约只覆盖了标准答案中60%的关键信息。
3. **答案正确性 (Answer Correctness)**
   - **BaseModel (GPT-3):** 0.7482
   - **CoVe:** 0.9492
   - **分析：** 此指标综合评估回答在语义和事实层面与标准答案的一致性。
     - **CoVe (0.9492)** 获得高分，再次印证了其高质量。结合满分的P和R，说明CoVe的回答在意思和事实上与标准答案几乎完全一致。
     - **BaseModel (0.7482)** 的得分相对较低，这主要是受其极低的召回率（信息遗漏）拖累。一个回答即使句句正确，但如果漏掉了关键信息，其在“语义”和“事实”的完整度上也是不正确的。

## 什么是Ragas

**Ragas**是一个专门为 LLM 应用评估流程设计的工具库，强调为LLM 系统提供定量、可操作的评估指标、自动化流程、测试数据生成等支持。

### 主要功能与特点

1. **多维度指标（Metrics）**
    Ragas 提供了一系列专用于 LLM 应用评估的指标。例如“faithfulness（忠实性）”、“answer relevancy（答案相关性）”、“context precision／recall（上下文精度／召回）”等。
    举例：
   - Context precision：检索出的上下文中有多少是“有用”的／相关的。
   - Context recall：检索是否覆盖了“所有”应该被检索的信息（对于有地面真实标注情形）
   - Faithfulness：生成的答案在多大程度上与提供的上下文／证据一致（是否出现“凭空编造”或“幻觉”）
   - Answer relevancy：答案有多切中用户提问／需求，有多相关。
   - 还有其它指标，例如 semantic similarity、answer correctness 等。
2. **自动／半自动测试数据生成**
    Ragas 支持生成“合成”测试集，以便在没有大量人工标注数据的情境下，也能开展评估。比如自动生成问题、上下文、答案／ground-truth。
3. **支持 “参考‐自由” (reference-free) 评估**
    一个比较重要的特点是：Ragas 尝试减少对人工标注“理想答案”（ground truth answer）或“理想上下文” 的依赖，从而加快评估流程。
4. **与主流工具／框架集成**
    Ragas 能与例如 Haystack、 LangChain、监控／可观测平台（如 Langfuse／Datadog）等进行配合。
5. **适用于生产监控 & 调优**
    不只是离线评估，也适合 “在生产环境／调用日志中” 抽样评估、持续监控模型性能、发现低质量回答、建立反馈回路。比如 Datadog 文档就提到：Ragas 用于生产轨迹（traces）里的 “faithfulness, answer relevancy, context precision” 的打分。 

### 典型使用场景

- 当你开发一个 RAG 系统（即检索 + LLM 生成）用于问答／总结／知识库查询，你想 **量化**其性能，不仅看 “用户觉得还行” 而是从多个维度打分。
- 当你正在比较多个 LLM 模型、或比较不同检索策略（比如不同向量库、不同检索器）的时候，希望有统一评估标准来选择 “哪一套组合更好”。（比如文档中有 “compare LLMs using Ragas” 的教程）
- 当你的服务上线后，希望监控其质量，比如：某些调用可能检索错误上下文、回答不关联提问、出现幻觉回答。Ragas 可帮你在日志中抽样评估、找出“低分答案”，用于反馈、迭代优化。
- 当你缺少大量人工标注的 ground-truth 数据时，想通过部分自动或辅助方式做评估（reference‐free 或少量标注）以加快测试流程。
- 与监控平台／分析平台集成（如 Langfuse, Datadog）来做 “模型可观测（model observability）” 的一环。

## 如何利用Ragas搭建大模型评估流水线

### 第 1 步：定义“评估者LLM”和“待测LLM”

这是最关键的区别。在 Ragas 流水线中，您需要两个（或更多）模型：

1. **待测模型 (Model-Under-Test):** 这是您真正想要评估性能的模型（例如 `gpt-3.5-turbo` 或您自己的微调模型）。它负责*生成* 。
2. **评估者模型 (Evaluator LLM):** 这是一个强大的“裁判”模型，Ragas 将使用它来*评判* `response` 的质量（例如，计算 `answer_correctness`）。

```python
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# 1. 评估者 LLM (裁判) - 用于 Ragas 评估
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

# 2. 待测 LLM (选手) - 我们要即时用它来生成答案
# 假设我们想测试 gpt-3.5-turbo
model_to_test = ChatOpenAI(model="gpt-3.5-turbo") 
```

### 第 2 步：加载“黄金标准”数据集

加载包含已有答案的数据集

```python
import pandas as pd
from ragas import EvaluationDataset

def load_ground_truth(json_path):
    print("正在加载黄金标准数据集...")
    try:
        df_gt = pd.read_json(json_path)
        df_gt = df_gt.rename(columns={'answer': 'reference', 'question': 'user_input'})
        df_gt['reference'] = df_gt['reference'].apply(str)
        print("黄金标准数据集加载完毕。")
        return df_gt
    except Exception as e:
        print(f"加载 JSON 文件失败: {e}")
        return None
```

### 第 3 步：即时生成

我们遍历“黄金标准”数据集，实时生成答案，并将结果（`user_input`, `reference`, `response`）收集起来。

```python
def generate_responses_on_the_fly(df_gt, llm_to_test):
    print(f"正在使用 {llm_to_test.model_name} 即时生成答案...")
    responses = []
    
    # 遍历问题
    for index, row in df_gt.iterrows():
        try:
            # "即时生成" 发生在这里
            response_content = llm_to_test.invoke(row['user_input']).content
            responses.append(response_content)
            print(f"Q: {row['user_input'][:30]}... -> A: {response_content[:30]}...")
        except Exception as e:
            print(f"为问题 '{row['user_input']}' 生成答案时出错: {e}")
            responses.append(None) # 记录失败

    # 将新生成的答案添加回 DataFrame
    df_gt['response'] = responses
    return df_gt
```

### 第 4 步：定义评估指标

自定义指标或者使用Ragas内置指标

```Python
# 自定义指标，CustomRecall 类的定义
@dataclass
class CustomPrecision(SingleTurnMetric):
    name: str = "custom_precision"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"user_input", "response", "reference"}
        }
    )

    def init(self, run_config: RunConfig):
        super().init(run_config)

    async def _single_turn_ascore(self, sample, callbacks: Callbacks) -> float:

        try:
            response_set = set(ast.literal_eval(sample.response))
            reference_set = set(ast.literal_eval(sample.reference))
        except (SyntaxError, ValueError):
            print("格式错误，无法解析 response 或 reference。")
            return 0.0

        if not response_set:
            print("Response 为空，处理边缘情况。")
            return 1.0 if not reference_set else 0.0

        intersection = response_set.intersection(reference_set)
        
        # 计算精确率
        precision = len(intersection) / len(response_set)

        return precision

# 以及内置指标
from ragas.metrics import answer_correctness, AnswerAccuracy

metrics_list = [
    CustomPrecision(),
    CustomRecall(),    
    answer_correctness,  
    AnswerAccuracy()
]
```

### 第 5 步：组装数据集并运行评估

将即时生成的结果封装成 Ragas 的 `EvaluationDataset` 并执行评估。

```python
from ragas import evaluate

def main_on_the_fly():
    # --- 第 1 步：定义模型 ---
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
    evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    model_to_test = ChatOpenAI(model="gpt-3.5-turbo") # 我们的“选手”

    # --- 第 2 步：加载黄金标准 ---
    JSON_FILE_PATH = "./datasets/wikidata_category_dataset_first20.json"
    df_ground_truth = load_ground_truth(JSON_FILE_PATH)
    if df_ground_truth is None:
        return

    # --- 第 3 步：即时生成答案 ---
    df_with_generated_responses = generate_responses_on_the_fly(df_ground_truth, model_to_test)

    # --- 第 4 步：定义指标 ---
    metrics_list = [
        CustomPrecision(),
        CustomRecall(),    
        answer_correctness,  
        AnswerAccuracy()
    ]

    # --- 第 5 步：组装并评估 ---
    print("\n--- 正在准备评估数据集 ---")
    
    # 过滤掉生成失败的条目
    df_final = df_with_generated_responses.dropna(subset=['response'])
    
    # 转换 Ragas 格式
    dataset_to_evaluate = EvaluationDataset.from_pandas(df_final)

    print(f"--- 正在使用 {evaluator_llm.llm.model_name} 评估 {model_to_test.model_name} ---")
    
    try:
        result = evaluate(
            dataset_to_evaluate, 
            metrics=metrics_list, 
            llm=evaluator_llm,  # 使用“裁判” gpt-4o
            embeddings=evaluator_embeddings
        )
        print(f"\n{model_to_test.model_name} 的评估结果:")
        print(result)
    except Exception as e:
        print(f"评估失败: {e}")

if __name__ == "__main__":
    # 运行新的“即时生成”流水线
    main_on_the_fly() 
    
    # (或者运行您原来的 'main()' 来进行离线评估)
    # main() 
```