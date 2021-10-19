# Transferable End-to-End Aspect-based Sentiment Analysis with Selective Adversarial Learning

## 摘要

aspect 和情感的联合提取可以有效地表述为序列标记问题。 然而，由于在许多领域缺乏带注释的序列数据，这种公式阻碍了监督方法的有效性。 为了解决这个问题，我们首先为这个任务探索一个无监督的域适应设置。 先前的工作只能使用 aspect 词和意见词之间的共同句法关系来弥合域差距，这高度依赖于外部语言资源。 为了解决这个问题，我们提出了一种新颖的选择性对抗学习 (Selective Adversarial Learning, SAL) 方法来对齐自动捕获其潜在关系的、推断的相关性向量 (inferred correlation vectors)。 SAL 方法可以动态地学习每个词的对齐权重，这样更重要的词可以拥有更高的对齐权重，以实现细粒度（词级）自适应。 根据经验，广泛的实验证明了所提出的 SAL 方法的有效性。

## 1 引言

End-to-End Aspect-Based Sentiment Analysis (E2E-ABSA) 旨在联合检测句子中明确提到的 aspect 术语并预测它们的情感极性（Liu，2012；Pontiki 等，2014）。 例如，在“The **AMD Turin Processor** seems to always perform much better than **Intel**”这句话中，用户提到了“AMD Turin Processor”和“Intel”两个 aspect term，并分别表达了对它们的正面和负面情绪。

通常，先前的工作将 E2E-ABSA 表述为统一标记方案上的序列标记问题（Mitchell 等，2013；Zhang 等，2015；Li 等，2019a）。统一标注方案连接一组 aspect 边界标签（例如，{B, I, E, S, O} 表示开始、内部、结束、单个词和无 aspect 术语）和情感标签（ 例如，{POS, NEG, NEU} 表示积极、消极或中性情绪）一起构成每个单词的联合标签空间。因此，“AMD Turin Processor”和“Intel”应分别用 {B-POS, I-POS, E-POS} 和 {S-NEG} 标记，而其余的词则用 O 标记。 这种形式使得两个子任务联合建模更容易，而且意味着，往往是低资源的。 每个新域通常几乎没有带标注的 (annotated) 数据，用统一标签标记每个单词可能更耗时且成本更高。

为了减轻对域监督的依赖，我们探索了 E2E-ABSA 的无监督域适应设置，旨在利用来自有标记的源域的知识来改进无标记的目标域中的序列学习 (sequence learning) 。 实现此设置的挑战有两个：（1）域之间存在较大的特征分布转移，因为不同域中的 aspect term 通常是 disjoint 的。 例如，用户通常在餐厅领域提到“pizza”，而在 Laptop 领域中经常提到“camera”； (2) 与学习共享的句子或文档表示的传统情感分类中的域适应 (Blitzer et al., 2007) 不同，我们需要学习细粒度（词级）表示以在序列预测中实现域不变

考虑第一个问题，即迁移什么 (what to transfer) ？ 尽管来自不同领域的 aspect term 表现 (behave) 不同，但 aspect 和意见词之间的一些关联模式 (association pattern) 在跨领域中是常见的； 例如，餐厅领域的“The **pizza** is great。” 和Laptop领域的“The **camera** is excellent。” 。 它们都具有相同的句法模式（aspect words→nsubj (normal subject)→opinion words）。 受此启发，现有研究使用一般句法关系作为跨域 aspect 提取（Jakob and Gurevych, 2010; Ding et al., 2017）或 aspect 和意见共同提取（Li 等，2012；Wang 和 Pan，2018）。 不幸的是，这些方法高度依赖先验知识（例如，手动设计的规则）或外部语言资源（例如，依赖解析器），它们不灵活且容易引入知识错误。 相反，我们引入了一种多跳 (multi-hop) 对偶记忆交互（Dual Memory Interaction, DMI）机制来自动捕获 aspect 和意见词之间的潜在关系。  DMI 通过将其局部记忆（LSTM 隐藏状态）与全局 aspect 和意见记忆进行交互来迭代地推断每个词的相关性向量，从而获得 aspect 和意见之间的相互关联 (inter-correlation)，以及 aspect 或 aspect 的内部相关性 (intra-correlation) 。

第二，如何为这个序列预测任务进行迁移 (how to transfer)？ 一种直接的方法是应用域自适应方法来对齐句子中的所有单词，但是，已经发现它不会产生显着的改进。 实际上，尽管需要细粒度的适应，但并非所有的词对域不变特征空间的贡献都相同。因此，我们提出了一种新颖的选择性对抗学习（SAL）方法来动态学习每个单词的对齐权重，其中更重要的单词可以拥有更高的对齐权重，以实现基于对抗训练的局部语义对齐。根据经验，所提出的模型在四个基准数据集上大大优于最先进的细粒度适应方法。 我们还进行了广泛的消融研究，以定量和定性地证明对抗性学习选择性的有效性。

总的来说，我们的主要贡献总结为：

(1) 据我们所知，第一次为 E2E-ABSA 探索了一种无监督的域适应设置；

(2) 提出了一种有效的 SAL 方法来进行局部语义对齐以进行细粒度域适应；

(3) 大量实验验证了所提出的 SAL 方法的有效性。

## 2 任务定义

**单域**：E2E-ABSA 涉及 aspect 检测 (AD) 和 aspect 情感 (AS) 分类任务，它们被定义为统一的序列标记问题。 给定输入的单词序列 $x=\{w_1, w_2, ..., w_T \}$ 及其词嵌入 $e=\{e_1, e_2, ..., e_T \}$，目标是预测标签序列 $y=\{y_1,  y_2, ..., y_T \}$在统一标签上，其中 $y_i∈\cal Y^U$ ={B-POS, I-POS, E-POS, S-POS, B-NEG, I-NEG, E-NEG, S-NEG  , B-NEU, I-NEU, E-NEU, S-NEU, O}。

**跨域**：这里我们在更具挑战性的无监督域自适应设置中执行。 给定来自源域的一组标记数据 $D_s ={({\rm x}^i_s , {\rm y}^i_s)}^{N_s}_{i=1}$ 和来自目标域的一组未标记数据 $ D_t = {({\rm x}^j_t)} ^ {N_t} _ {i = 1} $，我们的目标是迁移 $D_s$ 的知识以改进 $D_t$ 中的序列学习。

## 3 模型描述

**总览**：如图 1 所示，我们采用两个堆叠的双向 LSTM 作为 E2E-ABSA 的基本模型 (Li et al., 2019a)。 每个 LSTM$^\cal U$ 的上层用于高级 ADS (AD+AS) 任务，它预测统一标签作为输出，而较低的 LSTM$\cal ^B$ 用于低级 AD 任务，并预测 aspect 边界标签作为指导。为了适应基本模型，我们分别针对两个问题设计了不同的组件，即传输什么和如何传输。

![image-20211016165913624](https://gitee.com/Wales-Z/image_bed/raw/master/img/image-20211016165913624.png)

(1) 为了自动捕获 aspect 和观点词之间的潜在关系作为跨域的可迁移知识，我们在两个 LSTM 之间引入了多跳对偶记忆交互（DMI）机制。 在每一跳，例如第 1 跳，每个局部记忆 $h_i^\cal B$ 将与全局 aspect 和意见记忆，即基于 DMI 的 ${\rm m}^1_a$ 和  ${\rm m}^1_o$ 交互，以产生 aspect 和意见词的两个相关性向量用于 aspect 和意见词共同检测，其中意见检测用作AD任务的辅助任务。“局部”记忆表示句子中每个单词的隐藏表示（LSTM$\cal ^B$ 隐藏状态）。 而这两个“全局”记忆由所有输入句子全局共享，这在记忆网络中常用（Sukhbaatar 等人，2015 年；Kumar 等人，2016 年），可以分别看作是 aspect 和意见词的高级表示。然后，A-attention 和 O-attention 聚合最相关的 aspect 或意见词信息，以细化下一跳的两个全局记忆。

(2) 为了将 这些关系实现跨域适应，我们提出了一种选择性对抗学习 (SAL) 方法，以动态地专注于对齐域之间的 aspect 词。 这是因为具有信息的 aspect 的词比句子中用 O 标记的无意义词对共享特征空间的贡献更大（Zhou et al., 2019b）。因此，在源域上训练的 aspect 标记器在应用于目标域时可以很好地工作。 具体来说，在最后一跳，我们为每个单词采用域判别器，并带有梯度反转层 (Ganin et al., 2016) 以对其相关性向量进行域对抗学习（对齐）。而 A-attention 模块提供了一个 aspect 注意力分布作为选择器来控制每个单词的可学习对齐权重（选择性）。 最后，每个对齐的相关性向量将用于预测 aspect 边界标签（AD 任务）并馈送到 LSTM$\cal ^U$ 进行统一标签预测（ADS 任务）。 在以下部分中，我们将详细介绍每个组件。

### 3.1 基础模型

我们采用两个堆叠的双向 LSTM 作为基本模型。 我们将这两个 LSTM 层连接起来，以便由 LSTM$\cal ^B$ 生成的隐藏表示可以作为形成指导提供给 LSTM$\cal ^U$。 具体来说，它们在第 $i$ 个时间步 ($i \in [1, T]$) 的隐藏表示 ${\rm h}^{\cal B}_i \in \mathbb R ^{{\rm dim}^{\cal B}_h}$ 和  ${\rm h}^{\cal U}_i \in \mathbb R ^{{\rm dim}^{\cal U}_h}$ 计算如下

<img src="https://gitee.com/Wales-Z/image_bed/raw/master/img/image-20211016173356042.png" alt="image-20211016173356042" style="zoom:67%;" />

aspect 边界标签 $\cal Y^B$={B, I, E, S, O} 的概率分数 $z^{\cal B}_i \in \mathbb R ^{\cal |Y^B|}$ 由全连接的 softmax 层计算：
$$
{\rm z}^{\cal B}_i=p({\rm y}^{\cal B}_i | {\rm h}^{\cal B}_i)
={\rm Softmax}({\rm W}_{\cal B} {\rm h}^{\cal B}_i + {\rm b}_{\cal B})
$$
类似地，第二节中定义的统一标签 $\cal Y^U$ 的分数  $z^{\cal U}_i \in \mathbb R ^{\cal |Y^U|}$  通过如下方式得到：
$$
{\rm z}^{\cal U}_i=p({\rm y}^{\cal U}_i | {\rm h}^{\cal U}_i)
={\rm Softmax}({\rm W}_{\cal U} {\rm h}^{\cal U}_i + {\rm b}_{\cal U})
$$

### 3.2 全局-局部记忆交互

在详细介绍 DMI 模块之前，我们首先介绍全局-局部记忆交互（Global-Local Memory Interaction, GLMI），它描述了局部记忆 ${\rm h}_i∈\mathbb R^{{\rm dim}_h}$ 和全局内存 ${\rm m} \in \mathbb R^{{\rm dim}_h}$ 之间的交互。 形式上，我们用 $\Theta={\rm W, b}$ 以及 $\rm G$将 GLMI $f({\rm h}_i, \rm m; \Theta, G)$ 参数化，由残差变换和张量乘积运算组成。具体来说，我们首先将全局记忆信息 $\rm m$ 合并到每个具有残差变换 ${\rm \hat h}={\rm h}_i+{\rm ReLU}({\rm W}[{\rm h}_i:\rm m]+b)$ 的局部位置，其中 [:] 表示向量拼接 (concatenation) 。 因此，全局记忆可以提取更多相关的局部信息，并将它们映射到相同的空间。
然后我们计算一个相关性向量 ${\rm r}_i ∈ \mathbb R^K$ ，它通过一个张量乘积运算对全局记忆 $\rm m$ 和转换后的局部记忆 ${\rm \hat h}_i$ 之间的相关性强度进行编码：
$$
{\rm r}_i = {\rm m}^T {\rm G} {\rm \hat h}^{\cal B}_i
$$
其中张量 ${\rm G}\in \mathbb R ^{{\rm dim}_h \times {\rm dim}_h \times K}$ 可以看作是建模了两个对象之间 K 种潜在关系的多重双线性 (multiple bilinear) 矩阵。 G 的第 $k$ 个切片，即  ${\rm G}_k \in \mathbb R ^{{\rm dim}_h \times {\rm dim}_h}$ 表示一种潜在关系，它与 2 个向量相互作用以构成一种组合。

### 3.3 对偶记忆交互DMI

遵循第 3.2 节中的符号，我们进一步定义了全局 aspect 记忆 ${\rm m}_a \in \mathbb R ^{{\rm dim}^{\cal B}_h}$ 、全局意见记忆 ${\rm m}_o \in \mathbb R ^{{\rm dim}^{\cal B}_h}$ 和 LSTM$\cal ^B$ 的隐藏状态 ${\rm H}^{\cal B} =\{{\rm h}_i^{\cal B}\}^T_{i=1}$ 作为局部记忆。全局 aspect 和意见记忆能够分别从局部记忆中捕获高度相关的 aspect 或意见词。
根据观察， aspect 词经常与跨域的意见词并置 (collocate)，因此它们的关联可以作为弥合域差距的枢纽信息。 为了自动捕获它们在句子中的潜在关系，在第 $l$ 跳，每个局部内存 ${\rm h}^{\cal B}_i$ 将通过图 2 所示的对偶记忆交互（DMI）与全局记忆 ${\rm m}^l_a$ 和 ${\rm m}^l_a$ 交互，来为 aspect 和意见共同检测产生两个相关向量：

<img src="https://gitee.com/Wales-Z/image_bed/raw/master/img/image-20211016194731648.png" alt="image-20211016194731648" style="zoom: 80%;" />

其中 ${\rm G}_a, {\rm G}_o, {\rm G}_{ao}$ 分别表示对 aspect 和 aspect 、意见和意见、 aspect 和意见之间的潜在关系进行建模的合成张量。

<img src="https://gitee.com/Wales-Z/image_bed/raw/master/img/image-20211016200944157.png" alt="image-20211016200944157" style="zoom:67%;" />

<center>【图 2 对偶记忆交互DMI】</center>

相关性向量衡量局部和全局记忆之间的相关性强度； 例如，如果单词 $w_i$ 的 ${\rm h}^{\cal B}_i$ 既与 aspect 记忆 ${\rm m}_a$ 高度内相关 (intra-correlated)，又与观点记忆 ${\rm m}_o$ 高度相关 (inter-correlated)，则 $w_i$ 更有可能是 aspect 项。 然后可以将两个相关向量分别转换为标量 aspect 注意力（A-attention）和意见注意力（O-attention）权重 $\alpha^  l_{p,i}$，其中 $p\in \{a, o\}$ 表示 aspect 或意见， 表示句子中每个词是 aspect 词或观点词的可能性为：

<img src="https://gitee.com/Wales-Z/image_bed/raw/master/img/image-20211016195632345.png" alt="image-20211016195632345" style="zoom: 50%;" />

其中 ${\rm W}_p$ 是注意力模块的权重。
 aspect 或意见注意力权重  $\alpha^  l_{p,i}$ 将整合局部记忆以分别更新下一跳的全局 aspect 和意见记忆，即 ${\rm m}^{l+1}_p ={\rm m}^{l}_p + \sum^T_{i=1} \alpha^l_{p,i} {\rm h}_i^{\cal B}$  。 更新会逐渐细化全局记忆，以基于注意力机制纳入更多相关候选者。 在 DMI 中，所有参数在不同跳和域中共享。

在最后的第 $L$ 跳，我们将 ${\rm r}^L_{a,i}$ 用于 AD 任务，并将其提供给 LSTM$\cal ^U$ 用于 ADS 任务。对于辅助意见检测任务，我们将${\rm r}^L_{o,i}$ 输入一个 softmax 层，用于预测在意见标签$\cal Y^O$上的概率分数 $z^{\cal O}_i \in \mathbb R ^{\cal |Y^O|}$ ，即一个词是否是意见词，如： 
$$
{\rm z}^{\cal O}_i=p({\rm y}^{\cal O}_i | {\rm r}^{\cal O}_{o,i})
={\rm Softmax}({\rm W}_{\cal O} {\rm r}^{\cal O}_{o,i} + {\rm b}_{\cal O})
$$

### 3.4 选择性对抗学习

为了使捕获的关系具有域不变性，我们提出了一种选择性对抗学习（SAL）方法来动态对齐具有高概率的词以落入 aspect 边界，即成为 aspect 词。 具体来说，我们为每个词引入了一个域判别器，旨在识别域标签 ${\rm y}_i^{\cal D}\in \mathbb R^{\cal |Y^D|}$ 输入词的，即句子中的词来自源域还是目标域。 而特征提取器将产生域不变相关性向量 ${\rm r}^L_{a,i}$ ，且域判别器无法通过梯度反转层 (GRL) 区分该向量 (Ganin et al., 2016)。 在数学上，我们将 GRL 公式化为具有反向梯度 $\frac {∂Rλ(x)} {∂x} = −λI$ 的“伪函数”$R_λ(x)=x$，其中 λ 是适应率 (adaptation rate)。 相关性向量 ${\rm r}^L_{a,i}$  将在域判别器之前馈入 GRL，用于预测在域标签 $\cal Y^D$ 上e概率分数  $z^{\cal D}_i \in \mathbb R ^{\cal |Y^D|}$：
$$
{\rm z}^{\cal D}_i=p({\rm y}^{\cal D}_i | {\rm r}^{\cal L}_{a,i})
={\rm Softmax}({\rm W}_{\cal D} R_\lambda({\rm r}^{\cal D}_{a,i}) + {\rm b}_{\cal D})
$$
同时，最后一跳的 aspect 注意力权重 $\alpha^L_{a,i}$ 作为选择器，作为每个单词的可学习的对齐权重。 因此，选择性域对抗性损失是来自有标记源数据 $D_s$ 和无标记目标数据 $D_t$ 的所有词的加权交叉熵损失：

<img src="https://gitee.com/Wales-Z/image_bed/raw/master/img/image-20211016202429354.png" alt="image-20211016202429354" style="zoom: 67%;" />

现有研究（Yosinski 等人，2014 年；Mou 等人，2016 年）已经表明低级神经层特征（即低级任务）更容易迁移到不同的任务或领域。 因此，我们选择要对齐的低级 AD 任务中的 ${\rm r}^L_{a,i}$  而不是高级 ADS 任务中的特征 ${\rm h}_i^{\cal U}$ 进行传输。 我们的消融研究也证实了这一假设。

### 3.5 交替训练

主要任务损失由有标记源数据 $Ds$ 的有指导的 AD 和主要 ADS 任务的交叉熵损失组成：

<img src="https://gitee.com/Wales-Z/image_bed/raw/master/img/image-20211016203013661.png" style="zoom:67%;" />

辅助意见检测损失是有标记源数据 $Ds$ 和无标记目标数据 $Dt$ 的交叉熵损失如下：

<img src="https://gitee.com/Wales-Z/image_bed/raw/master/img/image-20211016203154047.png" alt="image-20211016203154047" style="zoom:67%;" />

传统上，我们可以直接优化方程  (1)-(3) 的联合损失，即 $E=\cal L_M+\rho L_O + \gamma L_D$ 以获得具有判别性 (discriminative) 且域不变的词表示，其中 $ρ$ 和 $γ$ 是权衡因子。 然而，我们发现优化过程往往不稳定，因为可能很难联合优化许多目标。 因此，我们提出了一种经验性交替策略来迭代地训练 $\cal L_M+\rho L_O $和 $\cal L_D$，将整个词表示学习分为判别阶段和域不变阶段。 令$\theta_f、\theta_w、\theta_d$分别表示AD、ADS和意见检测任务的词预测器，以及域判别器每个词的特征学习参数。基于我们的策略，我们正在寻找在两个阶段中传递 E 的鞍点 (saddle point) 的参数 $(\hat \theta_f^{(1)}, \hat \theta_f^{(2)},\hat \theta_w, \hat \theta_d)$：

<img src="https://gitee.com/Wales-Z/image_bed/raw/master/img/image-20211016203926020.png" alt="image-20211016203926020" style="zoom: 50%;" />

在鞍点，特征学习参数 $θ_f$ 最小化第一阶段的词预测损失（即特征具有判别性）。
对于第二阶段，域分类损失通过域判别器参数 $θ_d$ 最小化，而通过 GRL（即特征具有域不变性）通过特征学习参数 $θ_f$ 最大化。因此，我们可以实现更简单、更稳定的特征学习优化。

