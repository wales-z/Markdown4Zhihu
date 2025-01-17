# Recommendation System Exploiting Aspect-based Opinion Mining with Deep Learning Method

## 摘要

随着电子商务网站的发展，用户文本评论已成为提高推荐系统性能的重要信息来源，因为它们包含细粒度的用户意见，通常反映他们对产品的偏好。 然而，大多数经典推荐系统 (RS) 经常忽略此类用户意见，因此无法准确捕捉用户对产品的特定情绪。 尽管有一些方法尝试利用细粒度的用户意见在一定程度上提高推荐系统的准确性，但这些方法中的大多数基本上都依赖于手工制作的和基于规则的方法，这些方法通常被认为是费时又费力 (labour-intensive) 的。 耗费体力。 因此，它们的应用在实践中受到限制。 因此，为了克服上述问题，本文提出了一种推荐系统，该系统利用基于深度学习技术的基于 aspect 的意见挖掘（aspect-based opinion mining, ABOM）来提高推荐过程的准确性。所提出的模型由两部分组成：ABOM 和评分预测。 在第一部分中，我们使用多通道深度卷积神经网络 (multichannel deep convolutional neural network, MCNN) 来更好地提取 aspect 并通过计算用户在各个 aspect 的情绪极性来生成 aspect 特定的评分。 在第二部分中，我们将特定于 aspect 的评分整合到张量分解 (tensor factorization, TF) 机器中以进行整体评分 (overall rating) 预测。 使用各种数据集的实验结果表明，与 baseline 方法相比，我们提出的模型取得了显着的改进。

## 1 引言

随着网络信息的爆炸式增长，RS在解决信息过载问题方-面发挥着至关重要的作用，已广泛应用于包括社交媒体和电子商务网站在内的许多在线服务中。 协同过滤 (CF) 是最广泛使用的 RS 技术。 这种技术的基本思想是，过去具有相似行为的人在未来倾向于具有相似的偏好。 尽管 CF 方法已经显示出良好的性能，但它们的主要挑战之一是数据稀疏问题，其特点是用户评分数量不足且项目数量较多。 然而，这会影响推荐系统的有效性

随着电子商务网站最近的发展，已经表明可以利用包含不同产品的丰富信息的用户文本评论来缓解数据稀疏问题，从而提高 RSs 的有效性。 通常，用户评论不仅包含用户对产品不同 aspect 的评论，还包含用户对产品各个 aspect 的细粒度意见。 从本质上讲，这些用户意见非常重要，因为它们反映了用户对产品的偏好，从而影响 RS 的准确性。 不幸的是，大多数传统的 RS [1]-[3] 经常忽略此类用户对评分预测的意见，从而在推荐过程中表现不佳。

最近，引入了少量作品 [4] [5] 来利用用户的意见来提高 RS 的准确性。 例如，[4] 对用户评论文本使用短语级别的意见挖掘来提取明确的产品 aspect 和相关用户的意见，以生成可解释的推荐。 这种方法的一个主要限制是它依赖于众所周知的耗时和劳动密集型的词典构建。  [5] 引入了一个统一的框架，该框架从文本评论中提取 aspect 并将它们集成到用于评分预测的扩展 CF 过滤方法中。 该方法使用双传播方法，依靠句法关系扩展意见词典，并基于依赖图共同提取 aspect 和相关的用户情绪。 这种方法的一个主要挑战是它严重依赖于容易产生错误的依赖解析器，尤其是在应用于在线评论时。 因此，为了克服上述问题，在本研究中，我们引入了一种利用深度学习技术利用 ABOM 的方法，以提高推荐系统的有效性。

我们提议的工作受到上述方法的启发； 然而，它在几个方-面与它们不同：

1）我们提出的方法不是使用先前方法中使用的词典或双重传播方法，而是专门探索使用深度学习技术从评论中提取产品的 aspect 。 具体来说，我们采用了一个深度 CNN 模型，该模型已被证明在几个 (NLP) 任务中是有效的。  CNN 是一种非线性模型，易于训练并且能够从评论文本中自动捕获显着特征。  CNN 模型先前已被用于 aspect 提取任务 [6]、[7]； 然而，与传统的基于 CNN 的方法不同，我们的 CNN 模型通常是一个多通道卷积神经网络 (MCNN)，它利用两个不同的输入层，即词嵌入层和词性 (POS) 标签嵌入层。 前者旨在更好地学习文档的语义信息，而后者旨在促进更好的顺序标记过程。  

2）与上述主要基于用户文本评论的评分预测任务相比，我们提出的方法不仅可以预测未知评分，还可以估计和评估用于评分估计的提取 aspect 的质量。

## 3 问题定义

假设有一组产品 $P =\{p_1,p_2,...,p_n\}$ ，在由一组用户 ${\rm U}= \{u_1,u_2,...u_m\}$ , 撰写的评论 $D_{ij}$ 中。 令 $\rm R$ 为大小 $I \times J$ 的整体评分矩阵，其中条目 $r_{ij}$ 表示用户对产品的整体评分。 假设有 $K$ 个aspect ${\rm A}= \{a_1, a_2,...a_k \}$和相关联的 $K$ 个 aspect 评分矩阵 $R^1, R^2,...R^k$ ，每个 aspect 都有一个。 在下文中，我们定义了我们在本研究中努力解决的研究问题。

- aspect 提取： aspect 提取的主要目的是提取评论文本中提到的产品 aspect 。  aspect 是描述文本文档中的一组词。 这些 aspect 的示例包括：笔记本电脑领域的“屏幕”、“电池”和“性能”。 假设评论中包含 $k$ 个 aspect ，并给出为 $a_1,a_2,...a_k$

- 基于 aspect 的评分：用户对产品的基于 aspect 的评分是显示用户对产品意见的数字评分。 令 $A= \{a_1,a_2,...a_k \}$ 为文本评论中的 $k$ 个 aspect ； 然后，用户对产品的基于 aspect 的评分可以表示为 $k$ 维向量。

  与整体评分 $R$ 类似，对于每个 aspect ，都有一个基于 aspect 的评分矩阵，可以表示为 $R^1, R_2, .... R^k$ 用户，其中 $r_{ijk}$ 可以写为 aspect 评分矩阵 $R^k$ 中的条目，表示用于 $u_i$ 对于产品 $P_j$ 的第 $k$  个aspect的评分

- 整体评分预测：我们的目标是估计用户尚未评分的产品的整体评分。 我们将用户的整体评分定义为反映用户对产品的综合 (general) 意见的数字评分。

考虑到用户对各个 aspect 的评价一般反映了他们对产品的总体评价，我们将总体评价矩阵 $\rm R$ 和 $k$ 个aspect 评分矩阵合并到一个 3 维张量 $\cal R$ 分解方法中，其中 $\cal R$ 的大小为 $I \times J \times K$  。 表1显示了本文中使用的一些重要符号。

<img src="https://gitee.com/Wales-Z/image_bed/raw/master/img/image-20211015205942096.png" style="zoom:80%;" />

<center>【表1】符号表示</center>

## 4 方法总览

在本节中，我们将介绍我们提出的 REAO 模型的详细信息，该模型利用使用深度学习方法构建 RS 的 aspect 提取方法。 图 1 描绘了所提出模型的整个过程，该模型由两个不同的组件组成。 首先，我们使用 MCNN 模型应用深度学习技术从评论文本中提取 aspect ，然后使用 LDA 模型生成潜在 aspect 的聚簇 (cluster) 。 然后，我们应用词典 (lexicon) 方法来计算用户对 aspect 的情绪，以生成 aspect 评分矩阵。 其次，我们将基于 aspect 的评分矩阵与整体评分结合到张量分解方法中进行评分估计。 我们将在以下小节中描述该方法的详细信息

<img src="https://gitee.com/Wales-Z/image_bed/raw/master/img/image-20211015211109095.png" alt="image-20211015211109095" style="zoom:67%;" />

<center>【图1】</center>

### 4.1 aspect 提取

我们的目标是发现 aspect 和相关联的意见词，然后根据评论估计用户对各个 aspect 的特定情绪。为此，我们首先开发了一个 MCNN 模型，以便更好地提取评论中的 aspect 。 然后，将 LDA 模型应用于意见总结，最后，使用词典方法计算相关用户的情绪（意见）分数。 该过程的细节描述如下。

#### 4.1.1 MCNN

我们提出的多通道 CNN 方法是 [35] 提出的 CNN 架构的扩展版本。 它通常包括输入通道、卷积层、最大池化层和全连接层。 在输入层，我们专门使用词嵌入和 POS 标签嵌入通道。 对于词嵌入通道，我们使用预训练的词嵌入 [36] 来更好地学习词的语义信息。 形式上，一条长为 $n$ 的文本可以表示为：$|X|_1^n=\{x_1,...,x_n \},X\in R^K$ 。

对于 POS 标签嵌入，我们使用 one-hot 向量。 具体来说，我们将每个标签转换为一个 $k$ 维向量。 依照 [37] ，我们使用带有 45 个标签的集合。 这可以表示为：$|S|_1^n=\{s_1,...s_n \},S \in R^{45}$ 。

然后，我们应用卷积运算来提取突出 (salient) 特征，对POS 标签和 word2vec 特征使用两种不同的过滤器大小，分别为 $P$ 和 $Z$ 。 让 $w_p \in R^{h\times k}$ 成为矩阵 $\bold P$ 的过滤器，$w_z \in R^{h\times 45}$ 成为矩阵 $\bold Z$ 的过滤器，其中 $h$ 是过滤器的高度。 然后，卷积通过以下方式生成特征：
$$
C_i=f(w\cdot x_{i+h} + b)
\tag1
$$
。。。

<img src="https://gitee.com/Wales-Z/image_bed/raw/master/img/image-20211016151620827.png" alt="image-20211016151620827" style="zoom: 50%;" />

<center>【图2 用于 aspect 提取的MCNN模型总览】</center>

#### 4.1.2 aspect 整合 (summarization)

在实词情况下，用户文本评论中包含了几个 aspect 和意见词； 然而，其中许多 aspect 可能具有相同的含义。 例如，service、services、performance和 performances 之类的 aspect term 都可以表示 service 这个 aspect 。 因此，为了聚合这些 aspect 的用户情绪极性，需要将提取的 aspect 映射到潜在 aspect 。 为了实现这一点，我们采用了先前相关研究 [38] 中使用过的潜在狄利克雷分配 (latent Dirichlet allocation) [8]。  LDA 方法的输入是包含 aspect term 的评论集合，输出是 aspect 的集合，其中每个 aspect 都由一组 aspect term 组成。 由于 LDA 的特性，一个 aspect 术语可以属于几个不同的组。本质上，可以通过实验估计 aspect 的数量。

#### 4.1.3 基于 aspect 的评分

本小节介绍了计算基于 aspect 的评分矩阵 $R^1, R^2, ... R^k$ 的方法，基于从前面的方法中提取的 aspect 和相关的意见词。为此，我们首先计算评论中各个 aspect 的情感分数，然后取意见词极性的比率。与 [22] 类似，我们采用基于词典的使用 Senti Wordnet 的方法 [39]。在这种方法中，每个基于 aspect 的评分都是根据与 aspect 相关联的意见词来估计的。 如前所述， aspect 通常是名词或名词短语，而意见词通常是形容词。 给定评论中的一个 aspect ，我们估计基于 aspect 的评分如下：
$$
r_{ijk}=\frac{\sum_{w \in W_k(D_{ij})}OP(w)}{|W_k(D_{ij})|}
\tag5
$$
其中 $W_k$ 表示包含在评论 $D_{ij}$ 中的在与 aspect 相关的词集，$OP(w)$ 表示基于 senti WordNet 的词的极性分数。 由于 aspect 评分值表达了用户对产品的态度，我们对 aspect 评分值进行了标准化，使它们与整体评级处于相同的范围内（通常在 1 -5 之间）。

### 4.2 整体评分预测

如前所述，RS 的最终目标是估计用户 $u_i$ 尚未评级的产品 $p_j$ 的整体评分 $r_{ij}$ 。 给定 k 个特定于 aspect 的评分 $R^1, R^2, ... R^k$ 。（如上面第 4.1 节中计算的），我们整合了 aspect 评级矩阵 。
并将总体评分矩阵 $R$ 转化为三阶 TF 机器以预测整体评分 $r_{ij}$ 。

许多不同的 TF 模型，例如 HOSVD [5] 和 CANDECOMP/PARAFAC(CP) [9] 可用于计算 TF。 在这项研究中，我们专门应用了 CP-WOPT [9]，它是 [9] 中使用的 CP 模型的变体，能够更好地将高阶张量可扩展地分解为 rank-one 张量的和。 张量 $\cal R$ 的 CP 分解示意图如图 3 所示。 形式上，CP分解可以分别由大小为 $I \times R,\ J \times R, \ K \times R $ 的因子矩阵 $X, Y, Z$ 定义，使得：
$$
r_{ijk} = \sum^R_{r=1}x_{ir}\ y_{jr}\ z_{kr}\\
{\rm for\ \  all\ \ i=1,2...I,\ j=1,2...J,\ k=1,2...K.}
$$
其中 $r_{ijk}$ 和 $R$ 分别是张量 $\cal R$ 的条目和秩 (rank) 。$I \times R,\ J \times R, \ K \times R $ 分别是 $X,Y,Z$ 的大小。图 3 描述了张量 $\cal R$ 的 CP 分解。

![image-20211016151758235](https://gitee.com/Wales-Z/image_bed/raw/master/img/image-20211016151758235.png)

<center>【图3 CP分解示意图】</center>

最后的预测评分
$$
\hat r_{ij}=\sum_{k=1}^R\sum^R_{r=1}x_{ir}\ y_{jr}\ z_{kr}
$$
