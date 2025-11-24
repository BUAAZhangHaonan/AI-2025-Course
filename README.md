# AI-2025-Course

本仓库用于 **北京航空航天大学《人工智能原理与应用》课程（2025年秋季学期）** 的作业提交与管理。  
请各小组严格按照以下规范进行提交。

---

## 提交前必看：你怎么能直接 commit 到我的 master 分支啊

你怎么能直接 commit 到我的 master 分支啊？！GitHub 上不是这样！你应该先 fork 我的仓库，然后从 develop 分支 checkout 一个新的 feature 分支，比如叫 feature/confession。然后你把你的心意写成代码，并为它写好单元测试和集成测试，确保代码覆盖率达到95%以上。接着你要跑一下 Linter，通过所有的代码风格检查。然后你再 commit，commit message 要遵循 Conventional Commits 规范。之后你把这个分支 push 到你自己的远程仓库，然后给我提一个 Pull Request。在 PR 描述里，你要详细说明你的功能改动和实现思路，并且 @ 我和至少两个其他的评审。我们会 review 你的代码，可能会留下一些评论，你需要解决所有的 thread。等 CI/CD 流水线全部通过，并且拿到至少两个 LGTM 之后，我才会考虑把你的分支 squash and merge 到 develop 里，等待下一个版本发布。你怎么直接上来就想 force push 到 main？！GitHub 上根本不是这样！我拒绝合并！

## 📂 仓库目录结构

```txt
LargeModels/                # 大模型主题
│   ├── FineTuning/         # 大模型微调
│   ├── MultiModal/         # 多模态
│   ├── RAG/                # 检索增强生成
│   ├── Agents/             # 智能体
│   ├── ICL/                # 上下文学习
│   └── Hallucination/      # 大模型幻觉

Robotics/                   # 机器人主题
│   └── Locomotion/         # 四足/人形机器人行走

ComputerVision/             # 计算机视觉主题
    ├── Segmentation/       # 图像分割
    ├── Generation/         # 图像生成
    └── Reconstruction3D/   # 三维重建
```

---

## 📌 提交规范

### 1. 分组目录

- 每组在对应子文件夹下新建自己的目录，命名为：```group-XX/```，示例：`LargeModels/FineTuning/group-05/`

### 2. 文件结构

每个组的目录下应包含以下内容：代码文件夹、说明文档`README.md`应该包含对自己项目的简要介绍、使用说明和结果展示等内容。

### 3. 分支命名

每次提交作业时，请在 fork 仓库中新建分支，命名规则：```group-XX```，示例：`group-05`，与文件夹名称一致。

### 4. 忽略文件

请在根目录下的 `.gitignore` 文件中添加不需要提交的文件或文件夹名称，如编译生成的文件、临时文件等。

---

## 📌 提交流程

1. **Fork 仓库** 到自己的 GitHub 账号。  
2. 在 fork 仓库中新建分支（如 `group-05`）。  
3. 在对应主题子文件夹下新建 `group-XX/` 目录，放置作业文件。  
4. 提交并推送到 fork 仓库。  
5. 在 GitHub 上发起 **Pull Request**，目标分支为本仓库的 `master`。  
6. 等待审核并合并。

---

## 📌 注意事项

- 请勿直接向 `master` 分支提交。  
- 每组仅由 **组长** 负责提交。  
- 保持目录结构和文件命名一致，否则可能被退回修改。  
