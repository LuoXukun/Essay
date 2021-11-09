# BERT Family

## 1. 增加模型复杂度以优化BERT

### RoBERTa

核心思想：通过更好训练BERT可以达到超过其他新的预训练语言模型的效果

核心改动：

- 更大的batch size
- 去掉Next Sentence Prediction
- 采用更大的预训练语料
- Dynamic Masking

### XLNET

主要改动：

- Permutation Language Modeling
- Transformer-XL

### ERNIE

核心思想：使用Multi-Task Training提升模型效果

### T5

核心思想：

- 将Multi-task Learning的任务变为Seq2Seq的任务
- 测试了各类预训练语言模型设定的有效性

## 2. 同等参数下优化

### ALBERT

主要改进：

- 权重共享
- 输入层的优化
- Sentence Order Prediction

### ELECTRA

核心思想：采用对抗训练提升模型训练效果

- 通过MLM训练Generator
- Discriminator负责区分Generator生成的token是否被替代

其他改进：

- 权重共享



# 多任务学习

最大的问题：negative transfer

解决方法：部分权重共享；Soft Sharing（设置惩罚项，使不共享的参数差别不大）



