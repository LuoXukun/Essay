### MRD-Net: Multi-Modal Residual Knowledge Distillation for Spoken Question Answering

> IJCAI 2021. 2021.08.12

#### 背景

- 任务：**Spoken Question Answering (SQA)**，输入为一个简单的**文本**问题和一段**语音**形式的文章，目标为根据问题从文章中截取答案。
- 通常是将该任务分解为两个子任务：**Automatic speech recognition (ASR) **和**Text question recognition (TQR) **。通过**ASR**模块将语音形式的文章转化为文本，再通过**TQR**模块从文本形式的问题和文章中得出答案。然而，这会发生**误差传播**的问题。
- 相关前置工作：
  - Schneider et al. [2019] 发表了一篇将语音信号转化为向量序列的工作**VQ-Wav2Vec**；
  - Houlsby et al. [2019] 提出在Transformer单元中添加**Adapter**，微调时固定其余参数，只调整Adapter，可以在使用少量额外参数的情况下达到和利用全部参数相同的效果。
  - Lee at al. [2019] 尝试将音频特征和文本特征映射到统一空间中加以利用，但会使训练不稳定；
  - **You et al. [2021]** 通过名为**RKD的知识蒸馏框架**提升了模型的效果，但并没有在**TQR**阶段引入音频信息，只关注文本信息；
  - 之前有使用类BERT模型融合语音和文本特征的工作，但只是用于语音情感识别任务、语音选择题任务。
- **RKD知识蒸馏框架**分为三个模型（模型结构基本相同）：
  - Teacher：使用文本形式的文章和问题作为输入，训练一个序列标注模型（即认为ASR的准确率100%）；
  - Student：使用语音形式的文章（**ASR**的输出）和文本形式的问题作为输入，尽可能拟合Teacher模型的训练结果，以逼近Teacher模型的效果；
  - Assistant：由于Teacher和Student的输入不同，需要一个助理模型从一些特征中学习两者的差距，结合Assistant和Student可以更好地拟合Teacher。

#### 文章工作

- 以BERT Tagger结构作为**RKD**中的子模型提出**MRD-NET**，与**You et al. [2021]** 相比有以下改进：
  - 将语音形式的文章通过**VQ-Wav2Vec**转化为向量后作为**RKD**中Assistant的输入特征，以达到在**TQR**模块利用音频特征的目的；
  - 训练时不仅拟合Teacher的训练结果，还使Student拟合Teacher的隐层输出（最后一层隐层或全部隐层）；
  - 在结合Student和Assistant模型的隐层输出时，模拟**Adapter**文章中的网络结构提出了一个**ST-Attention**机制。

#### 总结

- 主要创新点为：将语音信息作为**RDK**中Assistant的输入特征，多模态融合的一种尝试；模型训练时还拟合隐层的输出，使Student和Teacher的行为更加相似。感觉是顺着前人的工作自然而然地改进（但如果是我可能就想不到，哭...）。
- 对比实验做得很充分，包括了模型对比、消融实验、知识蒸馏策略对比、ST-Attention机制对比，感觉可能被challenge的地方都有相应的实验数据和分析。
- 实验效果是sota，其他没啥想法，毕竟这个领域并不了解，感觉能投IJCAI就有过人之处。

