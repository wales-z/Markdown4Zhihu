# Exploiting BERT for End-to-End Aspect-based Sentiment Analysis

## 摘要

在本文中，我们研究了来自预训练语言模型的上下文嵌入在E2E-ABSA 任务中的建模能力，例如 BERT。 具体来说，我们构建了一系列简单而富有洞察力的神经基线来处理 E2E-ABSA。 实验结果表明，即使使用简单的线性分类层，我们基于 BERT 的架构也能胜过SOTA。此外，我们还通过始终使用 hold-out 开发数据集进行模型选择来标准化比较研究，这在很大程度上被以前的工作所忽略。 因此，我们的工作可以作为 E2E-ABSA 基于 BERT 的基准 (benchmark)。

## 1 引言

基于 aspect 的情感分析 (ABSA) 是从用户生成的自然语言文本（Liu，2012 ）中，发现用户对某个 aspect 的情感或意见，通常以明确提及的 aspect 术语的形式（Mitchell 等人，2013 ；Zhang 等人，2015 ）或隐式 aspect 类别（Wang 等人，2016 ）。最受欢迎的 ABSA 基准数据集来自 SemEval ABSA 挑战（Pontiki 等人，2014、2015、2016），其中提供了数千条带有黄金标准 aspect 情感注释的评论语句。

表 1 总结了与 ABSA 相关的三个现有研究问题。 第一个是原始的 ABSA，旨在预测句子对给定 aspect 的情感极性。 与这个分类问题相比，第二个和第三个，即面向 aspect 的意见词提取（Aspect-oriented Opinion Words Extraction, AOWE）（Fan t al.，2019）和端到端基于 aspect 的情感分析（E2E-ABSA）(Ma et al., 2018a; Schmitt et al., 2018; Li et al., 2019a; Li and Lu, 2017, 2019)，与序列标记问题有关。 准确地说，AOWE 的目标是从给定 aspect 的句子中提取特定于 aspect 的意见词。  E2E-ABSA 的目标是联合检测 aspect 术语/类别和相应的 aspect 情绪。

![image-20211011231100336](https://gitee.com/Wales-Z/image_bed/raw/master/img/image-20211011231100336.png)

许多由与任务无关的预训练词嵌入层和特定于任务的神经架构组成的神经模型已被提出用于原始 ABSA 任务（即 aspect 级情感分类）（Tang 等人，2016 ；Wang 等人） ，但改进 这些模型的准确率或 F1 分数已达到瓶颈。 一个原因是与任务无关的嵌入层，通常是一个用 Word2Vec (Mikolov et al., 2013) 或 GloVe (Pennington et al., 2014) 初始化的线性层，只提供与上下文无关的词级特征，这是不足以捕获句子中复杂的语义依赖关系。 同时，现有数据集的大小太小，无法训练复杂的特定于任务的架构。 因此，对于使用有标签的数据微调轻量级的特定于任务的网络，引入在具有深度 LSTM（McCann 等人，2017；Peters 等人，2018；Howard 和 Ruder，2018）或 Transformer（Rad Ford 等人2018, 2019；Devlin 等人，2019；Lample 和 Conneau，2019；Yang 等人，2019；Dong 等人，2019）的大规模数据集上预训练的上下文感知词嵌入层，具有良好的进一步提高性能的潜力

Xu等人 (2019);，Sun等人， (2019); Song等人(2019)，Yu和Jiang（2019）， Rietzler等人  (2019)，Huang和Carley（2019）， Hu等人 (2019a) 进行了一些初步尝试，将深度上下文化的词嵌入层与下游神经模型结合起来，用于原始 ABSA 任务，并建立新的最先进的结果。 它鼓励我们探索将这种上下文化嵌入用于更困难但更实际的任务的潜力，即 E2E-ABSA（表 1 中的第三个设置）。注意我们的目标不是开发一个特定于任务的架构，相反，我们的重点是检查上下文化嵌入用于 E2E-ABSA的潜力，配合预测 E2E-ABSA 标签的各种简单层。

在本文中，我们研究了 BERT（Devlin 等人，2019 ）在 E2E-ABSA 任务上的建模能力，BERT 是最受欢迎的预训练语言模型之一，配备了 Transformer（Vaswani 等人，2017 ）。 具体而言，受 Li 等人对 E2E-ABSA 的研究启发(2019a)，它使用单个序列标记器预测 aspect 边界和 aspect 情感，我们为序列标记问题构建了一系列简单但 insightful 的neural baselines，并使用 BERT 或deem BERT 作为特征提取器微调特定于任务的组件。 此外，我们通过始终使用 hold-out 开发数据集进行模型选择来标准化比较研究，这在大多数现有的 ABSA 工作中被忽略

## 2 模型

在本文中，我们专注于 aspect term 级别的、端到端的、基于 aspect 的情感分析（E2E-ABSA）问题设置。 这个任务可以表述为一个序列标记问题。我们模型的整体架构如图 1 所示。 给定长度为 $T$ 的输入 token 序列 ${\rm x} = \{x_1, · · , x_T \}$，我们首先对输入的 tokens 使用具有 $L$ 个转换器层的 $\rm BERT$ 组件来计算相应的上下文化表示 $H^L = \{h^L_1,···,h^L_T\}\in \mathbb R^{T\times dim_h} $，其中 $dim_h$ 表示表示向量的维度。 然后，上下文表示被馈送到任务特定层以预测标签序列 ${\rm y} = \{y_1,···, y_T\}$。 标签 $y_t$ 的可能值为 $\rm B-\{POS,NEG,NEU\}, I-\{POS,NEG,NEU\}, E-\{POS,NEG,NEU\}$, $\rm S-\{POS,NEG,NEU\} $或 $\rm O$， 表示 aspect 的开始， aspect 的内部， aspect 的结束，单个词的 aspect ，分别带有积极，消极或中性的情绪，以及 aspect 外部（非 aspect ）。

![image-20211008202347568](https://gitee.com/Wales-Z/image_bed/raw/master/img/image-20211008202347568.png)

### 2.1 BERT作为嵌入层

与传统的基于 Word2Vec 或 GloVe 的嵌入层仅为每个标记提供一个独立于上下文的表示相比，BERT 嵌入层将句子作为输入并使用整个句子的信息计算 token 级表示 。 首先，我们将输入特征打包为 $H^0 = \{e_1,···,e_T\}$，其中 $e_t (t \in [1,T])$ 是，对应于输入标记 $x_t$ 的token 嵌入、位置 (position) 嵌入和 segment 嵌入的线性组合。然后引入 $L$ 个transformer 层来逐层细化 token 级特征。 具体来说，在第 $l(l \in [1, L]) $层的表示 $H^l = \{h^l_1,···,h^l_T \}$ 计算如下：
$$
H^l = {\rm Transfomer}(H^{l-1})
\tag1
$$
我们将 $H^L$ 视为输入 tokens 的上下文化的表示，并使用它们来执行下游任务的预测。

### 2.2 下游模型的设计

在获得 BERT 表示后，我们在 BERT 嵌入层之上设计了一个神经层，在图1中称为 E2E-ABSA 层，用于解决 E2E-ABSA 的任务。 我们研究了 E2E-ABSA 层的几种不同设计，即线性层、循环神经网络、自注意力网络和条件随机场层。

#### 2.2.1 线性层

获得的 tokens 表示可以直接馈送到具有 softmax 激活函数的线性层，以计算 token 级别的预测：
$$
P(y_t|x_t) = {\rm softmax} (W_o h^L_t +b_o)
\tag2
$$

#### 2.2.2 RNN

考虑到其序列标记公式，循环神经网络 (RNN) (Elman, 1990) 是 E2E-ABSA 任务的自然解决方案。 在本文中，我们采用 GRU（Cho 等人，2014 ），其与 LSTM（Hochreiter 和 Schmid huber，1997 ）和基本 RNN 相比的优越性已在 Jozefowicz 等人中得到验证。  (2015)。 在第 $t$ 个时间步, 特定任务的隐藏表示 $h^{\cal T}_t \in \mathbb R ^{{\rm dim}_h}$ 的计算公式如下所示：

<img src="https://gitee.com/Wales-Z/image_bed/raw/master/img/image-20211011170121540.png" alt="image-20211011170121540" style="zoom: 80%;" />

其中 $\sigma$ 是 sigmoid 激活函数，$r_t$ , $z_t$ , $n_t$ 分别表示 reset 门、update 门和 new 门。$W_x, W_h \in \mathbb R^{2{\rm dim}_h\times {\rm dim}_h}, W_{xn}, W_{hn} \in \mathbb R^{{\rm dim}_h\times {\rm dim}_h}$ 是GRU的参数。由于直接在 Transformer 的输出（即BERT 表示 $h^L_t$）上应用 RNN，可能会导致训练不稳定（Chen 等人，2018 ；Liu，2019 ），在计算门时，我们添加了额外的层归一化 (layer-normalization)（Ba 等人, 2016)，表示为 $\rm LN$。 然后，通过引入 softmax 层获得预测：
$$
p(y_t|x_t) = {\rm softmax} (W_o h^{\cal T}_t +b_o)
\tag4
$$

#### 2.2.3 自注意力网络

在自我注意（Cheng et al., 2016; Lin et al., 2017）的帮助下，自我注意网络（Vaswani et al., 2017; Shen et al., 2018）是除了RNN和CNN之外的另一个有效的特征提取器。 在本文中，我们介绍了两种 SAN 变体来构建特定于任务的 tokens 表示 $H^{\cal T} = \{h^{\cal T}_1,···, h^{\cal T}_T\}$。 有一种变体由简单的自注意力层和残差连接组成（He 等，2016），称为“SAN”。  SAN的计算过程如下：

<img src="https://gitee.com/Wales-Z/image_bed/raw/master/img/image-20211011172953054.png" alt="image-20211011172953054" style="zoom: 50%;" />

其中 SLF-ATT 与自我注意缩放点积注意力 (self-attentive scaled dot-product attention) 相同（Vaswani 等，2017）。 另一种变体是 transformer 层（称为“TFM”），它与 BERT 中的 transformer 编码器层具有相同的架构。TFM的计算过程如下：

<img src="https://gitee.com/Wales-Z/image_bed/raw/master/img/image-20211011173145145.png" alt="image-20211011173145145" style="zoom:50%;" />

其中 FFN 指的是 point-wise 前馈网络（Vaswani 等，2017）。同样，在 SAN/TFM 层上堆叠一个具有 softmax 激活的线性层以输出预测（与等式（4）中的相同）

#### 2.2.4 条件随机场

条件随机场 (CRF) (Lafferty et al., 2001) 在序列建模中很有效，并被广泛用于与神经模型一起解决序列标记任务 (Huang et al., 2015; Lample et al.，2016 ；Ma and Hovy，2016)。 在本文中，我们在 BERT 嵌入层之上引入了一个 linear-chain CRF 层。 与上面提到的最大化 token级似然 $p(y_t|x_t)$ 的神经模型不同，基于 CRF 的模型旨在找到全局最可能 (most probable) 的标签序列。具体来说，${\rm y} = {y_1,···,y_T}$ 的句子级分数 $s(\rm x, y)$ 和似然 $p(\rm y|x) $计算如下：

<img src="https://gitee.com/Wales-Z/image_bed/raw/master/img/image-20211011174208987.png" alt="image-20211011174208987" style="zoom:67%;" />

其中 $M_A \in \mathbb R^{\cal |Y|×|Y|}$ 是随机初始化的转移矩阵，用于建模相邻预测的依赖性 (dependency) ， $M^P \in \mathbb R^{T \times \cal|Y|}$ 表示从 BERT 表示  $H^L$ 线性变换得到的 emission 矩阵。 这里的 softmax 是在所有可能的标签序列上进行的。至于解码，我们将得分最高的标签序列作为输出：
$$
y^∗ = {\rm arg} \max _{\rm y} s(\rm x, y)
 \tag8
$$
其中solution是通过 Viterbi search 获得的。

