### Highway Networks

> Arxiv 2015

#### 知识点

- 基于门机制提出Highway Network，使用简单的SGD就可以训练很深的网络，而且optimization更简单，甚至收敛更快

- 具体思路：

  引入Transform Gate $$T$$和Carry Gate $$C$$
  $$
  y = H(x, W_H) \cdot T(x, W_T) + x \cdot C(x, W_C)
  $$
  为了简单，设置 $$C = 1 - T$$:
  $$
  y = H(x, W_H) \cdot T(x, W_T) + x \cdot (1 - T(x, W_T))
  $$
  其实就是对输入一部分进行处理，一部分直接通过

#### 引用