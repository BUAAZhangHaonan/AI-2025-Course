import pandas as pd
from ragas import RunConfig, evaluate, EvaluationDataset
from ragas.metrics import answer_correctness, AnswerAccuracy
import typing as t
from dataclasses import dataclass, field
from ragas.metrics.base import SingleTurnMetric, MetricType
from ragas.callbacks import Callbacks
import ast  
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings


# --- 1. 自定义精确率 (Custom Precision) ---

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

# --- 2. 自定义召回率 (Custom Recall) ---

@dataclass
class CustomRecall(SingleTurnMetric):
    name: str = "custom_recall"
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

        if not reference_set:
            print("Reference 为空，处理边缘情况。")
            return 1.0 if not response_set else 0.0

        intersection = response_set.intersection(reference_set)

        recall = len(intersection) / len(reference_set)

        return recall

# --- 3. 加载和准备数据 ---
def load_and_prepare_data(json_path, csv_path):
    print("正在加载数据...")

    try:
        df_gt = pd.read_json(json_path)
    except Exception as e:
        print(f"加载 JSON 文件失败: {e}")
        return None, None

    df_gt = df_gt.rename(columns={'answer': 'reference', 'question': 'user_input'})
    df_gt['reference'] = df_gt['reference'].apply(str)

    try:
        df_llm = pd.read_csv(csv_path)
    except Exception as e:
        print(f"加载 CSV 文件失败: {e}")
        return None, None

    df_llm = df_llm.rename(columns={'Question': 'user_input'})

    csv_columns_to_use = ['user_input', 'BaseModelAnswer(GPT-3)', 'CoVeAnswer']
    merged_df = pd.merge(df_gt, df_llm[csv_columns_to_use], on='user_input')

    # BaseModel
    df_base = merged_df[['user_input', 'reference', 'BaseModelAnswer(GPT-3)']].rename(
        columns={'BaseModelAnswer(GPT-3)': 'response'}
    )
    dataset_base = EvaluationDataset.from_pandas(df_base)

    # CoVe
    df_cove = merged_df[['user_input', 'reference', 'CoVeAnswer']].rename(
        columns={'CoVeAnswer': 'response'}
    )
    dataset_cove = EvaluationDataset.from_pandas(df_cove)

    print("数据加载和准备完成。")

    return dataset_base, dataset_cove


# --- 4. 运行评估 ---
def main():
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
    evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    JSON_FILE_PATH = "./datasets/wikidata_category_dataset_first20.json"
    CSV_FILE_PATH = "./datasets/cove_simulation_first20.csv"

    dataset_base, dataset_cove = load_and_prepare_data(JSON_FILE_PATH, CSV_FILE_PATH)

    if dataset_base is None or dataset_cove is None:
        print("数据加载失败，程序退出。")
        return

    metrics_list = [
        CustomPrecision(),
        CustomRecall(),    
        answer_correctness,  
        AnswerAccuracy()
    ]

    print("\n--- 正在评估 BaseModel (GPT-3) ---")
    try:
        result_base = evaluate(dataset_base, metrics=metrics_list, llm=evaluator_llm, embeddings=evaluator_embeddings)
        print("BaseModel (GPT-3) 评估结果:")
        print(result_base)
    except Exception as e:
        print(f"评估 BaseModel 失败: {e}")

    print("\n--- 正在评估 CoVeAnswer ---")
    try:
        result_cove = evaluate(dataset_cove, metrics=metrics_list, llm=evaluator_llm, embeddings=evaluator_embeddings)
        print("CoVeAnswer 评估结果:")
        print(result_cove)
    except Exception as e:
        print(f"评估 CoVeAnswer 失败: {e}")


if __name__ == "__main__":
    main()