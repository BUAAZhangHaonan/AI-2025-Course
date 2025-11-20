# AI2025-MLLM 项目介绍

## 背景与目标

AI2025-MLLM 是北京航空航天大学《人工智能原理与应用》课程的大作业项目，围绕多模态大语言模型构建了一套从开源基线解析、能力评测到任务定制微调的完整实践体系。仓库聚合了多种模型（BLIP、Qwen2.5-VL、DeepSeek R1 Distill Qwen、Qwen3-VL 等）以及配套的环境配置、数据准备和实验脚本，目标是在统一的平台下探索视觉语言模型的推理、生成与下游任务适配能力。

## 仓库结构

| 目录                     | 说明                                                                                 |
| ------------------------ | ------------------------------------------------------------------------------------ |
| `blip_demo/`             | Salesforce BLIP 官方示例的本地化副本，提供图文检索、图像描述、VQA 等能力的参考实现。 |
| `model_benchmark/`       | 基于百炼平台 DeepSeek R1 Distill Qwen 系列的多模态对话与性能评测 Gradio 应用。       |
| `multimodal_benchmark/`  | 面向 Qwen2.5-VL 系列的前端演示与性能评测工具，支持更丰富的多模态任务。               |
| `multimodal_finetuning/` | 使用 LLaMA Factory 对 Qwen 系列模型进行医学影像诊断与情感识别微调的工程代码。        |
| `README.md`              | Git/GitHub 配置与提交规范。                                                          |

## 核心模块概览

### BLIP Demo (`blip_demo/`)

- 源自 Salesforce 开源实现，涵盖预训练、微调与推理脚本。
- 支持图像描述、开放式问答、图文匹配等基础任务，可直接运行 `demo.ipynb` 或 `predict.py` 体验模型效果。
- 适合作为入门基线或对比实验的参考。

### DeepSeek 多模态评测 (`model_benchmark/`)

- 通过 Gradio 搭建的可视化前端，面向 DeepSeek R1 Distill Qwen 7B/14B/32B 模型。
- 特色功能包括：文本+图像多模态对话、模型切换、响应时间监控及多任务性能测试。
- 提供数学推理、逻辑推理、创意写作、常识问答、代码生成等评测脚本，可快速对比不同模型规格。

### Qwen2.5-VL 多模态评测 (`multimodal_benchmark/`)

- 针对 Qwen2.5-VL 7B/32B/72B Instruct 模型设计，继承 DeepSeek 评测框架并扩展至图像描述、图像分析、视觉问答等任务。
- 集成 Plotly/Pandas 对结果进行可视化分析，便于观察响应时长与生成质量。
- 通过 `python multimodal_app.py` 即可启动本地服务。

### 多模态微调实践 (`multimodal_finetuning/`)

- 构建在 LLaMA Factory 之上，演示 LoRA 微调多模态模型的流程。
- 目前包含两个任务：医学 CT 诊断（Qwen3-VL-8B-Thinking + MedTrinity-25M 子集）与情感识别（Qwen2.5-Omni-7B + AffectNet 子集）。
- `data/` 目录给出预处理后的样例数据，`scripts/` 与 `examples/` 提供训练、推理命令模板。
- README 中列出了核心命令与环境依赖，可按需复现或扩展到自定义数据集。

## 数据与模型资源

- **模型权重**：推荐从 Hugging Face 获取 Qwen/Qwen3、DeepSeek R1 Distill 等模型的最新权重；BLIP 所需权重链接已在对应 README 中列出。
- **数据集**：
  - MedTrinity-25M：医学图像与文本描述，适用于医学问答或诊断。
  - AffectNet：面部表情数据集，适合情绪识别与多模态理解任务。
  - 其他通用图文数据可按 README 指引下载。
- 建议在首次运行前检查磁盘容量与网络带宽，模型下载与数据预处理会占用较多资源。

## 快速开始

1. **环境准备**
   - 推荐使用 Conda 或虚拟环境分别管理各子项目依赖。
   - `multimodal_finetuning/` 需要 Python 3.10 与 `pip install -e .[torch,metrics]`。
   - 各评测应用通过 `pip install -r requirements.txt` 安装依赖。
2. **运行前端评测**
   - 在 `model_benchmark/` 或 `multimodal_benchmark/` 目录下执行 `python multimodal_app.py`，浏览器访问 `http://localhost:7860`。
   - 按提示输入百炼平台 API Key，选择模型后即可进行多模态对话或性能测试。
3. **复现微调实验**
   - 参考 `multimodal_finetuning/README.md` 配置 LLaMA Factory 并准备数据。
   - 使用文档中的 `llamafactory-cli` 命令微调与评估 LoRA Adapter。

## 实验建议

- **硬件**：微调任务建议使用具备足够显存的 GPU（≥24GB）；评测应用可在 CPU 上运行，但多图像批量评测时建议启用 GPU。
- **日志管理**：评测脚本会生成响应时间与输出记录，可结合 Pandas/Plotly 进一步分析。
- **安全与合规**：医学图像与面部数据涉及隐私，请确保数据使用遵循相关协议。

## 贡献与协作

- 首次参与请阅读根目录 `README.md` 中的 Git 配置与提交信息规范。
- 建议使用分支进行功能开发，通过 Pull Request 发起合并并附带测试或演示说明。

## 致谢

本项目复用了 Salesforce BLIP、阿里通义实验室 Qwen、DeepSeek 等开源社区成果，感谢相关团队提供的模型与工具。也感谢课程教学组提供的任务设计与实验平台支持。

## github设置

### 设置github代理

```bash
git config --global https.proxy https://127.0.0.1:7890
git config --global http.proxy http://127.0.0.1:7890
```

### 设置github用户名和邮箱

```bash
git config --global user.name "YOUR_NAME"
git config --global user.email "YOUR_EMAIL"
```

### 关联github仓库

#### 如果本地仓库没有git初始化，初始化，关联仓库

```bash
git init
git remote add origin https://github.com/BUAAZhangHaonan/AI2025-MLLM.git
```

#### 如果本地仓库已经git初始化，删除已有.git文件夹，重新初始化，关联仓库

```bash
cd workspace
rm -rf ./AI2025-MLLM/.git
cd ..
git remote add origin https://github.com/BUAAZhangHaonan/AI2025-MLLM.git
```

#### 新建git分支

```bash
git checkout -b 分支名
```

### 从远程拉取代码仓库到本地

#### 将本地代码更改保存，解决冲突

```bash
git add .
git commit -m "save current changes"
git pull --no-rebase origin master # <-要修改pull的分支
```

之后在文档管理器中有冲突的文件会标红显示，在文档管理器中进行merge以完成不同版本commit的合并

#### 合并完成后保存文件，将合并后代码上传

```bash
git add .
git commit -m "commit message"
```

### 如果想让本地分支的内容与远程分支同步（用远程完全覆盖本地）

```bash
git fetch origin
git reset --hard origin/master
```

### 配置中科大镜像源

```bash
python -m pip install --upgrade pip
pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple/
```

## 提交类型说明

| 类型     | 描述                                                   |
| -------- | ------------------------------------------------------ |
| feat     | 新功能（feature）                                      |
| fix      | 修复 bug                                               |
| docs     | 文档变更                                               |
| style    | 代码格式（不影响功能，如空格、分号等）                 |
| refactor | 代码重构（既不是修复 bug 也不是添加新功能）            |
| perf     | 性能优化                                               |
| test     | 添加或修改测试用例                                     |
| build    | 构建相关（打包、CI 工作流等）                          |
| ci       | 持续集成配置修改                                       |
| chore    | 其他修改（不影响源代码或测试，如构建脚本、依赖更新等） |
| revert   | 回滚到上一个版本                                       |
