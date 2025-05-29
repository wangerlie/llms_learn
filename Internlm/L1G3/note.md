# 本地部署课程的一些心得

## 1. HF镜像
- 国内访问HF官网受限，而且下载模型速度会非常慢，几乎不可能，只能从[镜像网站](https://hf-mirror.com/)下载。

- 需要现在官网注册登录账号，并且获得access token，然后在服务器输入`huggingface-cli login`，输入access token才能正常下载。

- 最好还是从share中加载模型，只需要将本地模型权重路径作为`lmdeploy.pipeline`的`model_path`参数输入。
## 2. 并行策略

You are absolutely right\! My apologies. The `<ul><li>` tags are indeed HTML list tags that didn't render correctly as markdown bullet points within the table.

Here's the corrected version of the table with proper markdown formatting for the lists:

-----

### 大语言模型并行策略对比

| 并行方式 (Parallelism Strategy) | 核心含义 (Core Meaning) | 优点 (Advantages) | 缺点 (Disadvantages) |
| :--- | :--- | :--- | :--- |
| **张量并行 (TP)**  Tensor Parallelism | **模型内部切分**。将模型中的单个大张量（如权重矩阵）沿特定维度切分到多个设备上。每个设备只持有模型的一部分，共同完成一个操作（如矩阵乘法）。 | - **降低单卡显存占用**：可以直接训练超过单卡显存的大模型。- **计算通信重叠**：可以将 AllReduce 通信与反向传播的计算重叠，隐藏通信开销。 | - **通信开销大**：在前向和后向传播中都需要进行 All-Reduce 或 All-Gather 操作，对通信带宽要求高。- **实现复杂**：需要修改模型代码，对特定算子进行切分和通信实现。 |
| **数据并行 (DP)**  Data Parallelism  | **模型复制，数据切分**。在每个设备上都保留一个完整的模型副本。将训练数据批次 (batch) 切分，每个设备独立计算梯度，然后通过 AllReduce 聚合梯度以更新所有模型。 | - **实现简单**：是最成熟、最容易实现的并行方式。- **训练速度快**：可以显著扩大总的批次大小，提升训练吞吐量。 | - **显存冗余**：每个设备都需要存储完整的模型参数、梯度和优化器状态，对显存不友好。- **无法扩展模型大小**：模型大小受限于单个设备（如GPU）的显存。 |
| **流水线并行 (PP)**  Pipeline Parallelism  | **按层切分**。将模型的不同层 (layers) 分布到不同的设备上，形成一条“流水线”。数据在一个设备上完成计算后，将输出（激活值）传递给下一个设备。 | - **极大扩展模型规模**：是解决模型层数过多、单卡无法容纳的最有效方式。- **通信量相对较小**：设备间只传递中间层的激活值。 | - **流水线气泡 (Bubbles)**：设备存在等待和空闲时间，导致硬件利用率降低。- **负载均衡困难**：需要精心设计层的划分，以确保每个设备的计算负载尽量均衡。 |
| **序列并行 (SP)**  Sequence Parallelism  | **按序列长度切分**。在 Transformer 模型中，将输入序列 (sequence) 在序列维度上进行切分，分配到不同设备上。主要用于处理那些 TP 无法并行化的部分（如 LayerNorm）。 | - **节省激活值显存**：在长序列场景下，可以显著降低激活值占用的显存。- **与TP互补**：可以进一步减少显存占用，支持更大的模型或更长的序列。 | - **适用范围有限**：主要针对 Transformer 架构中的特定操作。- **增加通信**：切分序列后，需要在前向和后向传播中增加额外的 All-Gather 通信。 |
| **优化器状态并行 (ZeRO-DP)**  Optimizer State Sharding  | **切分优化器状态**。在数据并行的基础上，将优化器状态、梯度甚至模型参数本身也进行切分，分布到不同的设备上。每个设备只负责更新自己那一部分参数。 | - **大幅降低显存占用**：在不改变模型代码的情况下，显著减少数据并行带来的显存冗余。- **扩展性好**：结合数据并行，可以高效训练非常大的模型。 | - **通信模式更复杂**：在参数更新阶段需要额外的通信来收集和分发参数。- **实现依赖特定框架**：通常需要 DeepSpeed 等特定框架的支持。 |
| **专家并行 (MoE)**  Expert Parallelism  | **激活部分模型**。在混合专家模型 (MoE) 中，将不同的“专家”子网络部署到不同设备上。对于每个输入，只由门控网络选择的一小部分专家被激活并进行计算。 | - **极大扩展参数量**：可以在计算成本（FLOPs）仅少量增加的情况下，将模型参数扩展到万亿级别。- **训练和推理速度快**：每个 token 只使用模型的一小部分，计算效率高。 | - **通信开销**：需要在专家之间进行 All-to-All 通信来路由 tokens，对网络带宽和延迟敏感。- **负载不均和实现复杂**：可能出现专家负载不均衡的问题，且模型实现和训练调优更复杂。 |

-----

`AllReduce` `AllGather`以及更多收集算子可以参考nvidia的[collective operations文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html).  

`AllReduce`的经典应用：数据并行
`AllGather`的经典应用：张量并行、序列并行、优化器状态并行