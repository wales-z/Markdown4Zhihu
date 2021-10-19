# Aspect-based Sentiment Classification with Aspect-specific Graph Convolutional Networks

## 摘要

由于其在 aspect 及其上下文词的语义对齐方面的固有能力，注意力机制和卷积神经网络 (CNN) 被广泛应用于基于 aspect 的情感分类。 然而，这些模型缺乏考虑相关句法 (relevant syntactical) 约束和长距离词依赖性的机制，因此可能会错误地将与句法无关的上下文词识别为判断方面情绪的线索。 为了解决这个问题，我们建议在句子的依赖树上构建一个图卷积网络 (GCN)，以利用句法信息和单词依赖关系。 在此基础上，提出了一种新颖的特定于 aspect 的情感分类框架。 在三个基准集合上的实验表明，我们提出的模型与一系列最先进的模型具有相当的有效性，并进一步证明了图卷积结构正确捕获了句法信息和长距离词依赖性。

## 1 引言

基于 aspect（也称为aspect-level）的情感分类旨在识别句子中明确给出的 aspect 的情感极性。 例如，在对笔记本电脑的评论中说“从速度到多点触控手势，该操作系统轻松击败了 Windows。”，操作系统和 Windows 两个 aspect 的情绪极性分别为积极和消极。 通常，此任务被表述为预测所提供的（句子，aspect）对的极性。

鉴于手动特征细化的低效率（Jiang 等人，2011），基于 aspect 的情感分类的早期工作主要基于神经网络方法（Dong 等人，2014 年；Vo 和 Zhang，2015 年）。 自从唐等人(2016a) 指出了建模上下文词和 aspect 之间的语义相关性的挑战，注意力机制和循环神经网络 (RNN) 相结合（Bahdanau 等人，2014 年；Lu ong 等人，2015 年；Xu 等人 (Wang et al., 2015) 开始在最近的模型中发挥关键作用（Wang et al., 2016; Tang et al., 2016b; Yang et al., 2017; Liu and Zhang, 2017; Ma et al., 2017;  Huang 等人，2018 年）。

虽然基于注意力的模型很有前景，但它们不足以捕捉上下文词和句子中的方面之间的句法依赖性。 因此，当前的注意力机制可能导致给定的 aspect 错误地将句法上不相关的上下文词作为描述符（限制 1）。 请看一个具体的例子：“它的尺寸是理想的，重量是可以接受的。”。 基于注意力的模型通常将“可接受的”定义为尺寸这个 aspect 的描述符，而实际上并非如此。 为了解决这个问题，He 等人。  (2018) 对注意力权重施加了一些句法限制，但句法结构的影响没有得到充分利用

除了基于注意力的模型，卷积神经网络 (CNN)（Xue 和 Li，2018 年；Li 等人，2018 年）已被用于挖掘针对一个aspect的描述性多词短语 (multi-word phrases) 。（基于 (Fan et al., 2018) 发现的，一个 aspect 的情绪通常由关键短语而不是单个词决定）。 尽管如此，基于 CNN 的模型只能将多词特征感知为对词序列进行卷积运算的连续词，但不足以确定由多个不相邻的词所描绘的情感（限制 2）  . 在以员工为 aspect 的“员工应该更友好一点”这个句子中，基于 CNN 的模型可以通过把“更友好的”检测为描述性短语来做出正确的预测，而忽略了“应该”的影响（一个距离了两个词的单词），但这个词切切实实扭转了情绪。

在本文中，我们旨在通过使用图卷积网络 (GCN)（Kipf 和 Welling，2017 年）来解决上述两个限制。  GCN 具有多层架构，每一层都使用直接邻居的特征来编码和更新图中节点的表示。 通过引用句法依赖树，GCN 有潜力能够将句法相关的词吸引到目标 aspect 上，并借助 GCN 层利用远程多词关系 (long-range multi-word relations) 和句法信息。  GCN 已被用在文档-词关系（Yao 等人，2018 年）和树结构（Marcheggiani 和 Titov，2017 年；Zhang 等人，2018 年）上，但它们如何有效地用于基于 aspect 的情感分类尚待探索。

为了填补这一空白，本文提出了 Aspect specific Graph Convolutional Network (ASGCN)，据我们所知，它是第一个基于 GCN 的基于 aspect 的情感分类模型。  ASGCN 从双向长短期记忆网络 (LSTM) 层开始，以捕获有关词序的上下文信息。 为了获得特定于 aspect 的特征，在 LSTM 输出之上实现了一个多层图卷积结构，然后是一个掩膜 (mask) 机制，过滤掉非 aspect 的词并只保留高级 aspect 的特定特征。 特定于 aspect 的特征被反馈到 LSTM 输出，用于检索关于 aspect 的信息特征，然后用于预测基于 aspect 的情绪。

在三个基准数据集上的实验表明，ASGCN 有效地解决了当前基于 aspect 的情感分类方法的局限性，并且优于一系列最先进的模型。

## 3 特定于apsect的图卷积网络

图2 给出了 ASGCN 的总览。ASGCN 的组件会在本节的其余部分分别介绍

![image-20210917143737599](https://gitee.com/Wales-Z/image_bed/raw/master/img/image-20210917143737599.png)

### 3.1 嵌入和双向LSTM

给定一个 n 字的句子 c = {wc 1 , wc 2 , · · · , wc τ+1 , · · · , wc τ+m, · · · , wc n−1 , wc n} 包含一个对应的 从第 (τ + 1) 个标记开始的 m 个词的aspect，我们使用嵌入矩阵 E ∈ R |V |将每个词嵌入到具有 的低维实值向量空间（Bengio 等人，2003 年）中。  ×de ，其中 |V | 是词汇量的大小，de 表示词嵌入的维度。 使用句子的词嵌入，构建双向 LSTM 以产生隐藏状态向量 Hc = {hc 1 , hc 2 , · · · , hc τ+1, · · · , hc τ+m, · · ·  , hcn−1 , hcn}，其中 hct ∈ R 2dh 表示来自双向 LSTM 的时间步 t 处的隐藏状态向量，dh 是单向 LSTM 输出的隐藏状态向量的维数。

### 3.2 获得面向aspect的特征

不同于一般的情感分类，基于aspect的情感分类的目标是从aspect的角度判断情感，因此需要一种面向aspect的特征提取策略。 在这项研究中，我们通过在句子的句法依赖树上应用多层图卷积并在其顶部施加特定于方面的掩蔽层来获得面向方面的特征。

#### 3.2.1 依赖树的图卷积

为了解决现有方法的局限性（如前几节所述），我们在句子的依赖树上利用图卷积网络。 具体来说，在给定句子的依存树构建之后，我们首先根据句子中的词得到一个邻接矩阵A∈Rn×n。 需要注意的是，依赖树是有向图。 虽然 GCN 通常不考虑方向，但它们可以适应方向感知的场景。 因此，我们提出了 ASGCN 的两种变体，即无向依赖图上的 ASGCN-DG 和关于有向依赖树的 ASGCN-DT。 实际上，ASGCN-DG 和 ASGCN-DT 的唯一区别在于它们的邻接矩阵：ASGCN-DT 的邻接矩阵比 ASGCN-DG 的邻接矩阵稀疏得多。 这种设置符合父节点受其子节点广泛影响的现象。 此外，遵循 Kipf 和 Welling (2017) 中自循环的思想，每个单词都被手动设置为与自身相邻，即 A 的对角线值都是 1。

ASGCN 变体以多层方式执行，在第 3.1 节中的双向 LSTM 输出之上，即 H0 = Hc 以使节点了解上下文（Zhang 等人，2018）。 然后使用具有归一化因子的图卷积操作更新每个节点的表示（Kipf 和 Welling，2017），如下所示：

<img src="https://gitee.com/Wales-Z/image_bed/raw/master/img/image-20210917145211337.png" alt="image-20210917145211337" style="zoom:80%;" />

其中gl−1 j ∈ R 2dh 是从前面的GCN 层演化而来的第j 个token 的表示，而hli ∈ R 2dh 是当前GCN 层的乘积，di = Pn j=1 Aij 是i-的度数 树中的第 th 个标记。 权重 Wl 和偏差 b l 是可训练的参数
