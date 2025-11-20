# Fine-tuning Multimodal Large Models

This project showcases the fine-tuning of multimodal large language models for specialized tasks using the LLaMA Factory framework. We primarily focus on two domains: medical image diagnosis and emotion recognition.

## Our Work

Our project is centered around the following key tasks:

*   **Medical Image Diagnosis:** We fine-tuned the `Qwen3-VL-8B-Thinking` model on the `MedTrinity-25M` dataset to improve its ability to generate accurate medical diagnoses from CT scan images.
*   **Emotion Recognition:** We also fine-tuned a model for emotion recognition  `Qwen2.5-Omni-7B`  using the `AffectNet` dataset.

## Environment Setup

 To set up the environment, please follow these steps:

1.  **Clone our repository:**
    
    ```bash
    git clone https://github.com/BUAAZhangHaonan/AI2025-MLLM.git
    ```
2.  **Create and activate a conda environment:**
    
    ```bash
    conda create -n llama_factory python=3.10
    conda activate llama_factory
    ```
3.  **Install the necessary dependencies:**
    
    ```bash
    cd multimodal_finetuning
    pip install -e .[torch,metrics]
    ```

## Datasets

The following datasets were used in this project:

*   [**MedTrinity-25M**](https://github.com/UCSC-VLAA/MedTrinity-25M): A large-scale multimodal dataset for medical image analysis. We used a 10k subset of the 25Mdemo for our experiments.
*   [**AffectNet**](https://www.kaggle.com/datasets/noamsegal/affectnet-training-data): A large dataset of facial expressions for emotion recognition. We used a 1k subset.

The prepared datasets are located in the `data/` directory.

## Model Weights

The pre-trained models can be downloaded from Hugging Face. The fine-tuned adapters will be made available soon.

- **Pre-trained Qwen3-VL-8B-Thinking:** [Qwen/Qwen3-VL-8B-Thinking](https://www.google.com/url?sa=E&q=https%3A%2F%2Fhuggingface.co%2FQwen%2FQwen3-VL-8B-Thinking)
- **Pre-trained Qwen2.5-Omni-7B:** [Qwen/Qwen2.5-Omni-7B](https://www.google.com/url?sa=E&q=https%3A%2F%2Fhuggingface.co%2FQwen%2FQwen2.5-Omni-7B)
- **Fine-tuned Adapters:** The LoRA adapters for medical diagnosis and emotion recognition will be released here soon.

## Key Commands

Here are the main commands used for fine-tuning and evaluation in our project:

### Model Fine-tuning

This command fine-tunes the `Qwen3-VL-8B-Thinking` model on the `MedTrinity-train` dataset using LoRA.

```bash
llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path [your_model_path] \
    --finetuning_type lora \
    --template qwen3_vl \
    --dataset_dir data \
    --dataset MedTrinity-train \
    --output_dir [your_save_dir] \
```

### Evaluation After Fine-tuning

This command evaluates the fine-tuned model on the `MedTrinity-test` dataset.

```bash
llamafactory-cli train \
    --stage sft \
    --model_name_or_path [your_model_path] \
    --finetuning_type lora \
    --template qwen3_vl \
    --dataset_dir data \
    --eval_dataset MedTrinity-test \
    --predict_with_generate True \
    --output_dir [your_save_dir] \
    --do_predict True \
    --adapter_name_or_path [your_adapter_path]
```