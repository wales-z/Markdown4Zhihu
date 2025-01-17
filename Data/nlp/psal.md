# Progressive Self-Supervised Attention Learning for Aspect-Level Sentiment Analysis

## 摘要

在 aspect 级情感分类 (ASC) 中，为了获得每个上下文词在给定 aspect 的重要性，普遍为主导 (dominant) 神经模型配备注意力机制。 然而，这样的机制往往会过度关注少数具有情感极性的常用词，而忽略不常用的词。 在本文中，我们提出了一种用于神经 ASC 模型的渐进式自监督注意力学习方法，该方法会自动从训练语料库中挖掘有用的注意力监督信息以改进注意力机制。 具体来说，我们迭代地对所有训练实例进行情感预测。 特别地，在每次迭代中，具有最大注意力权重的上下文词被提取为对每个实例的正确/不正确预测、具有 active/misleading 影响的词，然后该词本身被屏蔽 (masked) 以供后续迭代使用。 最后，我们使用正则化项来增强传统训练目标，这使 ASC 模型能够继续平等地关注提取的活动上下文词，同时降低那些误导性词的权重。 在多个数据集上的实验结果表明，我们提出的方法产生了更好的注意力机制，导致对两个最先进的神经 ASC 模型的实质性改进。 源代码和经过训练的模型也已开源。

## 1 引言

 aspect 级情感分类（ASC）作为情感分析中必不可少的任务，旨在推断输入句子在某个 aspect 的情感极性。 对此，以前的代表性模型大多是基于手动特征工程的判别分类器，例如支持向量机（Kir itchenko 等人，2014 年；Wagner 等人，2014 年）。 最近，随着深度学习的快速发展，占主导地位的 ASC 模型已经演变成基于神经网络 (NN) 的模型（Tang et al., 2016b; Wang et al., 2016; Tang et al., 2016a; Ma  et al., 2017; Chen et al., 2017; Li et al., 2018; Wang et al., 2018)，它们能够自动学习输入句子的 aspect 相关语义表示，从而表现出更好的表现。 通常，这些基于 NN 的模型都配备了注意力机制来学习每个上下文词对给定 aspect 的重要性。 不可否认，注意力机制在神经 ASC 模型中起着至关重要的作用。

然而，ASC 中现有的注意力机制存在一个主要缺点。 具体来说，它容易过分关注少数具有情感极性的高频词，而很少关注低频词。 因此，注意力神经 ASC 模型的性能还远不能令人满意。 我们推测这是因为广泛存在“明显模式”和“不明显模式”。 在这里，“明显模式”是具有强烈情感极性的高频词，“不明显模式”是训练数据中的低频词。如 (Li et al., 2018; Xu et al., 2018; Lin et al., 2017) 所述，NN 很容易受到这两种模式的影响：“明显模式”往往被过度学习，而“不明显模式”往往无法完全学习。

这里我们用表1中的句子来解释这个缺陷。 在前三个训练句子中，考虑到上下文词 “small” 经常出现负面情绪，注意力机制更关注它，并直接将包含它的句子与负面情绪联系起来。 这不可避免地导致另一个包含信息的上下文词 “crowded” 被部分忽略，尽管它也具有负面情绪。 因此，神经 ASC 模型错误地预测了最后两个测试句子的情绪：在第一个测试句子中，神经 ASC 模型未能捕捉到“拥挤”所暗示的负面情绪； 而在第二个测试句中，注意力机制直接关注“小”，尽管它与给定的 aspect 无关。 因此，我们认为 ASC 的注意力机制仍有很大的改进空间。

![image-20210916125404017](https://gitee.com/Wales-Z/image_bed/raw/master/img/image-20210916125404017.png)

【表1】

上述问题的一个潜在解决方案是监督注意力，然而，这应该是手动注释的，需要劳动密集型的工作。 在本文中，我们为神经 ASC 模型提出了一种新颖的渐进式自监督注意力学习方法。 我们的方法能够从训练语料库中自动增量地挖掘注意力监督信息，可用于指导 ASC 模型中注意力机制的训练。 我们方法背后的基本思想源于以下事实：**具有最大注意力权重的上下文词对输入句子的情感预测影响最大**。 因此，在模型训练期间应该考虑正确预测的训练实例的上下文词。 相比之下，**错误预测的训练实例中的上下文词应该被忽略**。 为此，我们迭代地对所有训练实例进行情感预测。 特别是，在每次迭代中，我们从每个训练实例中提取注意力权重最大的上下文词，形成注意力监督信息，可以用来指导注意力机制的训练：在正确预测的情况下，我们将继续考虑这个词； 否则，这个词的注意力权重会降低。然后，我们屏蔽到目前为止每个训练实例中所有提取的上下文词，然后重新遵循上述过程以发现更多注意力机制的监督信息。 最后，我们使用正则化器增强标准训练目标，这会强制这些挖掘的上下文词的注意力分布与其预期分布一致。

我们的主要贡献有三方面：

(1) 通过深入分析，我们指出了 ASC 注意力机制存在的缺陷。

(2) 我们提出了一种新的增量方法来自动提取神经 ASC 模型的注意力监督信息。 据我们所知，我们的工作是探索 ASC 自动注意监督信息挖掘的首次尝试。

 (3) 我们将我们的方法应用于两个主要的神经 ASC 模型：记忆网络 (MN) (Tang et al., 2016b; Wang et al., 2018) 和 Transformation Network (TNet) (Li et al., 2018) . 几个基准数据集的实验结果证明了我们方法的有效性。

## 2 背景

在本节中，我们简要介绍了 MN 和 TNet，它们都实现了令人满意的性能，因此被选为我们工作的基础。 这里我们引入一些符号以方便后续描述： $x= (x_1, x_2, ..., x_N ) $是输入句子， $t= (t_1, t_2, ..., t_T ) $是给定的目标 aspect , $y, yp\in \{\rm Positive, Negative, Neutral\}$ 分别表示真实情绪和预测情绪。

**MN** (Tang et al., 2016b; Wang et al., 2018)。MN 的框架说明如图 1 所示。 我们首先引入一个 aspect 嵌入矩阵，将每个目标 aspect 词 $t_j$ 转换为向量表示，然后将 $t$ 的最终向量表示 $v(t)$ 定义为这些词的平均 aspect 嵌入。 同时，另一个嵌入矩阵用于将每个上下文词 $x_i$ 投影到存储在记忆中的连续空间，用 $m_i$ 表示。 然后，应用内部注意力机制来生成句子 $x$ 的 aspect 相关语义表示 $x:o = \sum_i {\rm softmax}(v^T_t M m_i)h_i$ ，其中 $M$ 是注意力矩阵，$h_i$ 是最终的 $x_i$ 的语义表示，从上下文词嵌入矩阵中导出。 最后，我们使用一个全连接的输出层根据 $o$ 和 $v(t)$ 进行分类。

<img src="https://gitee.com/Wales-Z/image_bed/raw/master/img/image-20210916130244304.png" alt="image-20210916130244304" style="zoom:80%;" />

**TNet** 。(Li et al., 2018)。 图 2 提供了 TNet 的框架图解，主要由三个组件组成：

(1) 底层是一个 Bi-LSTM，它将输入 $x$ 转换为上下文化的词表示 $h^{(0)}(x)=(h^{(0)}_1 , h^{(0)}_2 , ..., h^{(0)}_N )$（即 Bi-LSTM 的隐藏状态）。

(2) 中间部分，作为整个模型的核心，包含 $L$ 层上下文保存变换 (Context-Preserving Transformation ，CPT)，其中词表示更新为 $h^{(l+1)}(x)  ={\rm CPT}(h^{(l)}(x))$。  CPT层的关键操作是 Target-Specific Transformation。 它包含另一个 Bi-LSTM，用来通过注意力机制生成 $v(t)$，然后将 $v(t)$ 合并到单词表示中。 此外，CPT 层还配备了上下文保存机制（Context-Preserving Mechanism，CPM）来保存上下文信息并学习更多抽象的单词级别的特征。 最后，我们获得了词级语义表示 $h(x)=(h_1,h_2...,h_N )$，其中 $h_i=h^{(L)}_i $。

(3) 最上面的部分是一个 CNN 层，用于为情感分类生成与 aspect 相关的句子表示 $o$。

<img src="https://gitee.com/Wales-Z/image_bed/raw/master/img/image-20210916130959714.png" alt="image-20210916130959714" style="zoom:80%;" />

在这项工作中，我们考虑了原始 TNet 的另一种替代方案，它用注意力机制替换了其最顶层的 CNN，以生成与 aspect 相关的句子表示为 $o={\rm Atten}(h(x), v(t))$。 在第 4 节中，我们将研究原始 TNet 和此变体的性能，表示为 **TNet-ATT** 。

**训练目标** 。上述两个模型都将 gold-truth 情感标签的negative log-likelihood 作为训练目标：
$$
\begin{align}
J(D;\theta)&=-\sum_{(x,t,y)\in D} J(x,t,y;\theta)\\
&=\sum_{(x,t,y)\in D} d(y)\cdot \log d(x,t;\theta) 
\end{align}
\tag1
$$
其中 D 是训练语料库，$d(y)$ 是 $y$ 的 one-hot 向量，$d(x,t;θ)$ 是模型预测的 $(x,t)$ 的情感分布，· 表示两个向量的点积。

## 3 我们的方法

在本节中，我们首先描述我们方法背后的基本直觉 (basic intuition)，然后提供其详细信息。最后，我们详细说明了如何将挖掘的注意力机制监督信息整合到神经 ASC 模型中。 值得注意的是，我们的方法只应用于神经ASC模型的训练优化，对模型测试没有任何影响。

### 3.1 基本直觉 (basic intuition)

我们方法的基本直觉源于以下事实：在注意力 ASC 模型中，每个上下文词在给定方面的重要性主要取决于其注意力权重。 因此，具有最大注意力权重的上下文词对输入句子的情感预测具有最重要的影响。 因此，对于一个训练句子，如果ASC模型的预测是正确的，我们认为继续关注这个上下文词是合理的。 如果不正确，则应该降低这个上下文词的注意力权重。

但是，如前所述，具有最大注意力权重的上下文词是具有强烈情感极性的十个词。 它通常在训练语料库中频繁出现，因此在模型训练过程中往往会被过度考虑。 这同时导致其他上下文词的学习不足，尤其是具有情感极性的低频词。 为了解决这个问题，一个直观可行的方法是先屏蔽这个最重要的上下文词的影响，然后再研究训练实例的剩余上下文词的影响。 在这种情况下，可以根据注意力权重发现其他具有情感极性的低频上下文词。

### 3.2 该方法的细节

基于上述分析，我们提出了一种新的增量方法，可以从训练实例中自动挖掘有影响力的 (influential) 的上下文词，然后可以将其用作神经 ASC 模型的注意力监督信息。

如算法1所示，我们首先使用初始训练语料$D$进行模型训练，然后得到初始模型参数$θ(0)$（第1行）。 然后，我们继续训练模型进行 $K$ 次迭代，其中可以迭代提取所有训练实例的有影响力的上下文词（第 6-25 行）。 在这个过程中，对于每个训练实例 $(x, t, y)$，我们引入两个初始化为 $\emptyset$ 的词集（第 2-5 行）来记录其提取的上下文词： (1) $s_a(x)$ 由对 $x$ 的情绪预测具有积极影响的上下文词组成。  $s_a(x)$ 的每个词都将被鼓励在 refined 模型训练中继续被考虑，并且 (2) $s_m(x)$ 包含具有误导性的上下文词组成，其注意力权重预期会(are expected to)降低。 具体来说，在第 $k$ 次训练迭代时，我们采用以下步骤处理 $(x, t, y)$：

在步骤 1 中，我们首先应用前一次迭代的模型参数 $θ^{(k−1)}$ 来生成 aspect 表示 $v(t)$（第 9 行）。 重要的是，根据 $s_a(x)$ 和 $s_m(x)$，我们然后屏蔽所有先前提取的 $x$ 的上下文词以创建一个新句子 $x'$ ，其中每个被屏蔽的词被重新放置一个特殊的标记“$<mask>$  ”（第 10 行）。这样，在 $x'$ 的情感预测过程中将屏蔽这些上下文词的影响，因此可以从 $x'$ 中潜在地提取其他上下文词。 最后，我们生成单词表示 $h(x')={h(x'_i )}^N_{i=1}$（第 11 行）。

在步骤 2 中，在 $v(t)$ 和 $h(x')$ 的基础上，我们利用 $θ^{(k−1)}$ 将 $x'$ 的情绪预测为 $y_p$（第 12 行），其中单词级别的注意力权重分布 $\alpha(x')={\alpha(x'_1 ), \alpha(x'_2 ), ..., \alpha(x'_N )} $，满足 $\sum^N_{i=1} \alpha(x'_i) = 1$。

在步骤 3 中，我们使用熵 $E(\alpha(x'))$ 来测量 $α(x')$ 的方差（第 13 行），这有助于确定对情感预测有影响的上下文词的存在 $x'$
$$
E(\alpha(x'))=-\sum^N_{i=1} \alpha(x'_i) \log (\alpha(x'_i))
\tag2
$$
如果 $E(\alpha(x'))$ 小于阈值 $\epsilon_\alpha$（第 14 行），我们认为至少存在一个对 $x'$ 的情绪预测有很大影响的上下文词。 因此，我们提取具有最大注意力权重（第 15-20 行）的上下文词 $x'_m$，将其用作注意力监督信息以改进模型训练。 具体来说，我们根据 $x'$ 上的不同预测结果，采用两种策略来处理 $x'_m$ ：如果预测正确，我们希望继续关注 $x'_m$ 并将其加入$s_a(x)$（第16-17行）  ; 否则，我们希望减少 $x'_m$ 的注意力权重，并将其包含在 $s_m(x)$ 中（第 18-19 行）。

在步骤 4 中，我们将 $x'$ 、$t$ 和 $y$ 组合为一个三元组，并将其与收集到的三元组合并，形成一个新的训练语料库 $D^{(k)}$（第 22 行）。 然后，我们利用 $D^{(k)}$ 继续更新下一次迭代的模型参数（第 24 行）。 这样做，我们使我们的模型自适应以发现更多有影响力的上下文词。

![image-20210916220547060](https://gitee.com/Wales-Z/image_bed/raw/master/img/image-20210916220547060.png)

【算法1】

通过上述步骤的 $K$ 次迭代，我们设法提取了所有训练实例的有影响的上下文词。 表2说明了表1所示第一句的上下文词挖掘过程。在这个例子中，我们依次迭代提取三个上下文词：“small”, “crowded”和“quick”。前两个词包含在 $s_a(x)$ 中，而最后一个词包含在 $s_m(x)$ 中。最后，将每个训练实例提取的上下文词放入$D$中，形成带有注意力监督信息的最终训练语料 $D_s$（第26-29行），用于进行最后的模型训练（第30行）. 详细信息将在下一小节中提供。

![image-20210916220704572](https://gitee.com/Wales-Z/image_bed/raw/master/img/image-20210916220704572.png)

### 3.3 使用注意力监督信息的模型训练

为了利用上述提取的上下文词来改进 ASC 模型注意力机制的训练，我们提出了一个软注意力正则化器 $\Delta(\alpha(s_a(x)\cup s_m(x)), \hat \alpha(s_a(x)\cup s_m(x)); \theta)$ 共同最小化标准训练目标，其中 $\alpha(∗)$ 和 $\hat \alpha(∗)$ 分别表示 $s_a(x)\cup s_m(x)$ 中单词的model-induced和预期的注意力权重分布。 更具体地说，$\Delta(\alpha(*), \hat \alpha(*);\theta)$ 是一种欧几里得距离风格损失，它惩罚了 $\alpha(∗)$ 和 $\hat \alpha(∗)$  之间的分歧。

如前所述，我们希望在最终模型训练期间同样继续关注 $s_a(x)$ 的上下文词。 为此，我们将他们的预期注意力权重设置为相同的值 $\frac{1}{|s_a(x)|} $  . 这样做可以减少先提取的词的权重，增加后提取的词的权重，避免具有情感极性的高频上下文词的过拟合和低频词的欠拟合。另一方面，对于 $s_m(x)$ 中对 $x$ 的情绪预测有误导作用的词，我们希望减少它们的影响，因此直接将它们的期望权重设置为 0。 回到表 2 所示的句子，“small”和“crowded” $\in s_a(x)$被赋予相同的期望权重0.5，“quick”$\in s_m(x)$的期望权重为0

最后，我们在带有注意力监督信息的训练语料库 Ds 上的目标函数变为
$$
J_s(D_s;\theta)=-\sum_{(x,t,y)\in D_s} \{J(x,t,y;\theta)+\gamma\Delta(\alpha(s_a(x)\cup s_m(x)),\hat\alpha(s_a(x)\cup s_m(x));\theta\}
\tag3
$$
其中 $J(x,t,y;θ)$ 是等式 1 中定义的常规训练目标，$γ>0$ 是平衡传统损失函数和正则化项之间的超参数。 除了利用注意力监督信息之外，我们的方法还有一个优势：通过将这些信息添加到整个网络的中间层（Szegedy et al., 2015），更容易解决梯度消失问题，因为 $\hat \alpha (∗)$ 的监督比 $y$ 更接近于 $\alpha(∗)$。

