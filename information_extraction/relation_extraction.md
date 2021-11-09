### Dependency-driven Relation Extraction with Attentive Graph Convolutional Networks

> ACL 2021

#### 知识点

- 该论文采用了注意力图卷积神经网络模型 **A-GCN (Attentive Graph Convolutional Network)**，基于剪枝的依存句法知识，对词与词之间的依存关系以及关系类型进行上下文建模，通过注意力机制区分不同上下文特征的重要性，识别句法知识中的噪声，从而提升模型在关系抽取任务中的性能
- 利用关联矩阵**Adjacency Matrix**以及关联词间的点积计算权重，作为**A-GCN**的注意力矩阵**Attention Matrix**；将文本表征和依存类型矩阵**Dependency Type Matrix**作为**A-GCN**的输入
- 这篇论文并不是联合抽取，是知道了实体后的分类

#### 引用

无



### UniRE: A Unified Label Space for Entity Relation Extraction

> ACL 2021

#### 知识点

- 计算loss时额外考虑两个约束来规范表格中标签的关系

  - 对称性（**symmetry**）：同一实体的标签是对称的，无方向的关系标签是对称的
    $$
    L_{sym} = \frac{1}{|s|^2} \sum_{i=1}^{|s|}{\sum_{j=1}^{|s|}{\sum_{t\in\gamma_{sym}}{|P_{i,j,t}-P_{j,i,t}|}}}
    $$
  
- 牵连性（**Implication**）：如果一个关系存在，那么它的两个实体也一定存在，即关系的概率要小于每个实体的概率
    $$
    L_{imp} = \frac{1}{|s|} \sum_{i=1}^{|s|}{[\max_{l\in\gamma_r}{\{P_{i,;,l}, P_{:,i,l}\}-\max_{t\in\gamma_e}{\{P_{i,i,t}\}}}]}
    $$
  
- 提出了一个表格预测概率的解码方法，通过找出边界而不用对token逐一进行解码，提升了效率

#### 引用

- Timothy Dozat and Christopher D Manning. Deep biaffine attention for neural dependency parsing. ArXiv 2016.
- Zexuan Zhong and Danqi Chen. A frustratingly easy approach for joint entity and relation extraction. Arxiv 2020.



### PRGC: Potential Relation and Global Correspondence Based Joint Relational Triple Extraction

> ACL 2021

#### 知识点

- 指出了TPLinker存在的问题：为避免**曝光偏差（exposure bias）**，他利用了相当复杂的解码器，导致稀疏的标签，关系冗余，基于span的提取能力差
- 提出三步串行模型：
  1. **Potential Relation Prediction**：预测候选关系，减少第二步分关系类型序列标注的复杂度
  2. **Relation-Specific Sequence Tagging**：基于候选关系，分别进行特定关系头实体和尾实体的序列标注
  3. **Global Correspondence**：每个关系一个矩阵，根据阈值进行头尾的首词匹配

#### 引用

- Yucheng Wang et al. TPLinker: Single-stage joint extraction of entities and relations through token pair linking. COLING 2020.



### TPLinker: Single-stage Joint Extraction of Entities and Relations Through Token Pair Linking

> COLING 2020

#### 知识点

- 提出**曝光误差**问题：以往提出的一些方法已经可以解决重叠问题，如 CopyRE 、CopyMTL、CasRel（HBT）等，但它们在训练和推理阶段的不一致性导致存在曝光偏差。即在训练阶段，使用了golden truth 作为已知信息对训练过程进行引导，而**在推理阶段只能依赖于预测结果**。这导致中间步骤的输入信息来源于两个不同的分布，对性能有一定的影响。虽然这些方法都是在一个模型中对实体和关系进行了联合抽取，但从某种意义上它们“退化”成了“pipeline”的方法，即在解码阶段需要分多步进行,这也是它们存在曝光偏差的本质原因。
- 提出一种单阶段联合抽取模型 TPLinker，它能够发现共享一个或两个实体的重叠关系，同时不受暴露偏差的影响。TPLinker 将联合抽取描述为一个 token 对链接问题，并引入了一种新的握手标记方案，该方案将实体对中每个关系类型下的边界标记对齐。一共有三种握手标记，**紫色**代表两实体各自**内部的头尾握手**，**红色**代表两实体**头握手**，**蓝色**代表两实体的**尾握手**。同一种颜色的握手标记，会被表示在同一个矩阵。
- 模型结构：模型比较简单，整个句子过一遍 encoder，然后将 token 两两拼接输入到一个全连接层，再激活一下输出作为 token 对的向量表示，最后对 token 对进行分类即可。换句话说，这其实是一个较长序列的标注过程。

#### 引用

无



### FedED: Federated Learning via Ensemble Distillation for Medical Relation Extraction

> EMNLP 2020

#### 知识点

- 本论文主要解决的是存在数据隐私要求情形下的医学关系抽取问题，不能将所有的数据进行中心化的训练，对数据隐私的解决方案是**Federated Learning**（联邦学习）
- 本文的关系抽取是给定了实体判断两个实体之间的关系，结构是使用BERT进行编码，取[CLS]和实体部分表征进行组合，最后输出关系类型
- 本文针对的场景是水平联邦学习**horizontal federated learning**，即不同子数据场景的特征相同，但是样本不同。如：多个医院的电子病历数据，病人不同，但是存储的特征字段大部分相同。
- 一般的联邦学习方案，对每一个local model 使用本地数据训练模型，然后将模型梯度分别上传至central model，然后在收到所有local model的梯度后进行统一计算，之后更新central model。计算成本较高，对于BERT这种体量的深度模型，单模型的参数量巨大，传输时间较长。
- 本文提出了基于融合蒸馏的迁移学习方案：local model 与 central model使用相同的网络结构，对于每一个local model使用本地数据进行训练(这里模拟的场景是各个医院)，而central model 使用**验证数据集Dv**进行单独训练，central model 可以理解为在一个可信任的第三方机构进行训练，验证数据集Dv是一个单独的小体量数据集。每个local model 使用本地数据训练之后，在验证集Dv上给出各自的预测结果，这些local model 即为teacher model。基于各个local model的预测结果进行融合，使用central model 进行学习（学习每类的p，减小teacher和student预测结果的KL散度），central model 即为student model。

#### 问题

- central model只使用验证数据集单独训练，由于训练集很小，那不会过拟合吗？但又不能使用各个医院的数据，无法像一般的联邦学习方案一样基于所有本地数据进行训练，这样效果真的会好吗？

#### 引用

- 三个医疗关系抽取数据集：
  - **2010 i2b2/VA challenge dataset**: Ozlem Uzuner et al. 2010 i2b2/va challenge on concepts, assertions, and relations in clinical text. 2011.
  - **GAD**: Alex Bravo et al. Extraction of relations between genes and diseases from text and large-scale data analysis: implications for translational research. BMC bioinformatics. 2015.
  - **EU-ADR**: Erik M Van Mulligen et al. The eu-adr corpus: annotated drugs, diseases, targets, and their relationships. 2012.



### A Frustratingly Easy Approach for Entity and Relation Extraction

> ArXiv 2021.
>
> https://zhuanlan.zhihu.com/p/274938894

#### 知识点

- 主要贡献：
  - 设计了一种非常简单的end2end关系抽取方法，即采取2个独立的编码器分别用于实体抽取和关系识别，**使用相同的预训练模型就超越了之前所有的joint模型**
  - 分别学习实体和关系的不同上下文表示，比联合学习它们更有效
  - 在关系模型的输入层融合**实体类别信息**十分重要
  - 提出了一种新颖并且有效的近似方法，在精度下降很小的情况下，就实现8-16倍的推断提速
- 模型结构：
  - 实体模型：采用**Span-level NER**方式，提取所有可能的片段排列，通过SoftMax对每一个Span进行实体类型判断。
  - 关系模型：对所有实体pair进行关系分类，其中**最重要的一点改进，就是将实体边界和类型作为标识符加入到实体Span前后**，然后作为关系模型的input。对每个实体pair中第一个token的编码进concatenate，然后进行SoftMax分类，计算开销巨大。故提出一种加速的**近似模型**：可将实体边界和类型的标识符放入到文本之后，然后与原文对应实体**共享位置向量**。具体地，在attention层中，文本token只去attend文本token、不去attend标识符token，而标识符token可以attend原文token。综上，通过这种**近似模型**可以实现**一次编码文本就可以判断所有实体pair间的关系**。
  - 上述实体模型和关系模型采取的**两个独立的预训练模型进行编码**（不共享参数）。

#### 思考

- 并不认为pipeline误差传播问题不存在或无法解决，而需要探索更好的解决方案来解决此问题
- 两个trick：
  - 引入实体类别信息会让你的关系模型有提升
  - 对于pipeline实体关系抽取，2个独立的编码器也许会更好

#### 引用



### A Unified Multi-Task Learning Framework for Joint Extraction of Entities and Relations

> AAAI 2021

#### 知识点

- 使用了一个联合的框架将关系抽取的三步放在了一个框架中

  - **Type-attention Subject Extraction**
    $$
    Input: x_{subject} = [[CLS], t, [SEP], s_1, s_2, ..., s_{N_s}, [SEP]]
    $$
    给了一个可以**不依赖于预定义的实体类型**的网络结构，即将实体类型和句子一同输入BERT Encoder，进行语义融合后输出单纯的BIO标注

  - **Subject-aware Relation Prediction**

    局部：对每个预测出的entity，判断他是否存在某些关系类型；全局：对[CLS]进行分类，判断是否存在某些关系类型。训练时计算两个loss并合并，**预测时只使用局部模型**

  - **Global Relation Prediction**
    $$
    Input: x_{object} = [[CLS], q_i, [SEP], s_1, s_2, ..., s_{N_s}, [SEP]]
    $$
    

#### 引用

- Srivastava, R. K et al. Highway networks. Arxiv:1505.00387



### DDRel: A New Dataset for Interpersonal Relation Classification in Dyadic Dialogues

> AAAI 2021

#### 知识点

- 贡献
  - 提出了基于对话者对话的关系分类任务
  - 提出了一个数据集：13个预定义的关系，694对对话者的6300个对话段
  - 基于数据集提出了*session-level*和*pair-level*的关系分类任务

#### 引用

无



### Empower Distantly Supervised Relation Extraction with Collaborative Adversarial Training

> AAAI 2021

#### 知识点

- 贡献

  - 将数据利用率低作为**DS-MIL**（远程监督+多样本学习）的瓶颈
  - 提出**MULTICAST**（**MULT**i-**I**nstance **C**ollaborative **A**dver**S**arial **T**raining）来提升数据利用率，它在不同层次使用了**VAT**（虚拟对抗训练）和**AT**（对抗训练）

- **MULTICAST**

  - Inputs (Embedding): 每个`token`的表征由语义信息和与两个实体的相对位置信息组成
    $$
    m_i = [w_i; p_{i1}; p_{i2}]
    $$

  - Encoder (Piecewice CNN)

  - MIL (Multi-Instance Learning)：基于attention得到整个bag的表示并进行分类

  - IVAT (Instance-Level Virtual Adversarial Training): 对于bag中的噪声样本$x \in X_{noisy}$，加一个小的扰动$||d|| \le \varepsilon_x$，让它们的结果没有太大变化（使用LK-散度衡量），增加模型的**LDS** (Local Distributional Smothness)

  - BAT (Bag-Level Adversarial Training)：对整个bag的表示$z$加一个扰动，让它们的结果没有大的变化

- 问题

  - 对噪声样本使用了IVAT，对bag使用了AT，没太明白这两者有什么不同
  - P@N要去看一下具体怎么做的

#### 引用



### Curriculum-Meta Learning for Order-Robust Continual Relation Extraction

> AAAI 2021

#### 知识点

- **Continual Learning**（连续学习工作的两个问题）

  - **Catastrophic Forgetting**：当一个神经网络在一个任务的序列上学习，在后面的任务学习中可能降低之前任务的性能
  - **Order-sensitivity**：神经网络在这些任务上的表现取决于任务到达的顺序，十分多样

- 本文贡献

  - 提出一个新的连续学习方式，从memory区采样之前困难工作的样例，与当前样例同时训练，并引导模型学习当前任务和以前最相似任务之间的偏差以降低顺序敏感性

  - 引入了一种新的关系表示学习方法，通过关系的头尾实体的概念分布来量化每个关系提取任务的难度以构建课程（Curriculum）

  - 提出了两个新的评判标准

    **Average Forgetting Rate** -> Catatrophic Forgetting

    **Error Bound** -> Order-sensitivity

#### 引用



### Learning from Context or Names? An Empirical Study on Neural Relation Extraction

> EMNLP 2020

#### 知识点

- 实验发现
  - 上下文信息和实体信息对于relation extraction 提供了重要的信息, 其中实体类型提供了最有用的关系分类信息。上下文信息和实体信息对于正确的预测是非常必要的
  - 模型不能很好地理解上下文, 可能倾向于依赖mention的表面线索, 也促使我们思考要提升关系抽取的性能要很深刻地理解上下文信息而不能对mention 进行死记硬背
  - 从对比实验结果分析得知, 上下文信息和实体类型信息对于关系抽取模型比较重要, 然后现有的关系抽取模型不能更好从context中理解 relational patterns, 而仅仅依赖于实体的表面线索, 目的是为了更好的抓取实体类型信息和从文本中抽取relational facts, 故提出了一个**entity-masked contrastive pre-training framework** （看完MTB再回来看）
- **Matching the blanks (MTB)**：一个面向关系分类的BERT预训练模型，ACL 2019

#### 引用

- Matching the blanks: Distributional similarity for relation learning. Livio Baldini Soares et al. ACL 2019.



### Active Testing: An Unbiased Evaluation Method for Distantly Supervised Relation Extraction

> EMNLP 2020 findings

#### 知识点

- 指标——**Precision at top K (P@K)**

  所有实体对的置信分数降序排列：$s^{'} = \{s_1^{'}, s_2^{'}, ..., s_P^{'}\}$ ,P为实体对数乘关系数量

  对应顺序的预测标签和真实标签（**标签会被转化成one-hot，即标签为0或1**）分别为：
  $$
  y^{'} = \{y_1^{'}, y_2^{'}, ..., y_P^{'}\} \\
  z^{'} = \{z_1^{'}, z_2^{'}, ..., z_P^{'}\}
  $$
  指标为：
  $$
  P@K\{z_1^{'}, z_2^{'}, ..., z_P^{'}\} = \frac{1}{K}\sum_{i \le K}{z_i^{'}} \\
  R@K\{z_1^{'}, z_2^{'}, ..., z_P^{'}\} = \frac{\sum_{i \le K}{z_i^{'}}}{\sum_{i \le P}{z_i^{'}}}
  $$

- 以往的验证方法，将P@K中的z替换成y，明显导致了错误的结果（*不懂*）

- 提出一个新的验证方案，包含两个步骤：**Metric Estimator**和**Vetting Strategy**

#### 引用

无



### Minimize Exposure Bias of Seq2Seq Models in Joint Entity and Relation Extraction

> EMNLP 2020 findings

#### 知识点

- seq2seq模型，在test阶段用了树结构（可能）来生成关系集合，感觉没有**SPN**好。

#### 引用

- SPN: Joint Entity and Relation Extraction with Set Prediction Networks. DianboSui et al. 2020.



### How Knowledge Graph and Attention Help? A Qualitative Analysis into Bag-level Relation Extraction

> ACL 2021

#### 知识点

- 通过质量分析（**qualitative analysis**）和消融实验（**ablation study**）得出了四个结论：
  - 更高的注意力准确性可能导致更差的性能，这可能损害模型提取实体特征的能力
  - 注意力的表现很大程度上受各种噪声分布的影响，这些分布和现实的数据集密切相关
  - **KG-enhanced attention**确实提升了关系抽取的表现，但不是通过强化attention，而是结合实体先验
  - 注意力机制可能加剧训练数据不足的问题
- 提出两个描述噪声大小的指标
  - **Noise Ratio**：无效样本的比例
  - **Disturbing Bag Ratio**：纯粹地包含有效样本或无效样本地比例

#### 引用

无



### SENT: Sentence-level Distant Relation Extraction via Negative Training

> ACL 2021

#### 知识点

- 动机：在这篇论文中，作者提出在远程监督中使用负训练（**negative training**），这种方法通过选择给定标签的互补标签来训练模型，也就是从 “输入句子属于标签” 变为 “输入句子不属于互补标签”。由于选择正确的标签作为互补标签的概率很低，所以负训练能够降低提供噪声信息的风险，防止了模型对噪声数据的过拟合。
- 模型：
  - 从负训练数据数据中分离噪声
  - 对噪声数据进行过滤，并对部分置信度高地实例进行重新标记
  - 基于前两步的有效训练算法做迭代进一步提升表现
- 负训练：主要就是改造了以下交叉熵，随机选一个不属于标注标签k的负标签，使$1-p_k$最大化

#### 引用

无