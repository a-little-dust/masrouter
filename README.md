# [ACL 2025] MasRouter: Learning to Route LLMs for Multi-Agent Systems

 (2025-2-16) Initial upload to arXiv [PDF](https://arxiv.org/abs/2502.11133).


## 🤔 Why MasRouter?

**MasRouter** expands LLM routing to the multi-agent systems (MAS) *for the first time*. It leverages the powerful reasoning capabilities of LLM MAS, while also making it relatively cost-effective.

![intro](assets/intro.png)

## 👋🏻 Method Overview

**MasRouter** integrates all components of MAS into a unified routing framework. It employs collaboration mode determination, role allocation, and LLM routing through a cascaded controller network, progressively constructing a MAS that balances effectiveness and efficiency.

![pipeline](assets/pipeline.png)

## 🏃‍♂️‍➡️ Quick Start

### 📊 Datasets

Please download the  `GSM8K`,  `HumanEval`, `MATH`, `MBPP`（代码）, `MMLU`（多学科问答） datasets and place it in the `Datasets` folder. The file structure should be organized as follows:
```
Datasets
└── gsm8k
    └── gsm8k.jsonl
└── humaneval
    └── humaneval-py.jsonl
└── MATH
    └── test
    └── train
└── mbpp
    └── mbpp.jsonl
└── MMLU
    └── data
```

### 🔑 Add API keys

Add API keys in `template.env` and change its name to `.env`. We recommend that this API be able to access multiple LLMs.
```python
URL = "" # the URL of LLM backend
KEY = "" # the key for API
```

### 🐹 Run the code

The code below verifies the experimental results of the `mbpp` dataset.

```bash
python experiments/run_mbpp.py
```

### 配置细节

根据gpt_chat.py，可以知道 具体是如何调用大模型的，可以知道 配置文件要怎么写

协作方式（collab / reasoning）是在 MAR/Agent/reasoning_profile.py 里集中声明的
可供选择的协作方式共有 6 种：
IO（单代理直接 I/O 回答）
CoT（单代理 Chain-of-Thought 分步推理）
Chain（多代理链式传递推理）
FullConnected（多代理全连接图协作推理）
Debate（多代理辩论式推理）
Reflection（反思式推理：代理自我审视并修正答案）

### 代码细节

在roles文件夹，每个任务有很多角色，角色列表被SentenceEncoder编码为向量

任务判别（TaskClassifier）
在任何后续模块之前，先用两个线性层把 query 和 task 编码到同一隐空间；
通过 L2-norm + 余弦相似度（softmax 后）挑出每条 query 最匹配的任务；
输出的 query_context 参与后续 Role 与 LLM 选择的上下文拼接。

角色数据加载（encoder_roles）
启动时一次性遍历 MAR/Roles/…/*.json，把每个任务可用角色与对应文本嵌入缓存到字典；
这样 RoleAllocation 阶段只做向量检索，不再读硬盘。

CollabDeterminer通过VAE，计算协作方法 和 query的嵌入相似度。这里只会选择一种协作方式

NumDeterminer根据query，用VAE编码，连接全连接层，获得任务难度，然后根据难度算出代理数量

RoleAllocation，前面已经获得了角色列表（是预先定义在roles的），以及代理数量，所以 依次选择每一个代理。角色编码器是VAE，上下文编码器是线性层。通过二者的相似度分数 选择代理 并添加到上下文
上下文 = [query_context ; collab_context]；

LLMRouter使用VAE编码LLM信息，使用线性层编码上下文信息，通过相似度获得要选择的基座模型
上下文 = [query_context ; collab_context ; role_context]；
体现“逐层细化、信息累积”的设计。

然后运行图
每轮 run() 会：清空旧边，采样新边，拓扑排序执行节点
每轮开始时重新构建空间连接，因此每一轮的组内架构是不一样的
运行完所有普通节点后，再把所有节点连到决策节点；

另外，发现：tools文件夹定义了很多工具；roles文件夹，给每个任务预先定义了很多role

发现，MasRouter反向传播的时候是一次性端到端的，只用了 一个 optimizer
token 

