# Exploiting Coarse-to-Fine Task Transfer for Aspect-Level Sentiment Classification

## 摘要

aspect 级情感分类 (ASC) 旨在识别句子中 aspect 的情感极性，其中 aspect 可以表现为一般的 aspect 类别 (Aspect Category, AC) 或特定的 aspect 项 (Aspect Term, AT)。 但是，由于标注特别昂贵和费力，现有的AT级公共语料库都比较小。 同时，以前的大多数方法都依赖于具有给定稀缺数据的复杂结构，这在很大程度上限制了神经模型的有效性。 在本文中，我们开发了一个新的方向，称为从粗到细的任务迁移 (coarse-to-fine task transfer) ，旨在利用从粗粒度 AC 任务的资源丰富的源域中学习到的知识（该领域更易于访问），以改进学习细粒度 AT 任务的缺少资源的目标域。 为了解决领域之间的 aspect 粒度不一致和特征不匹配，我们提出了一个多粒度对齐网络（Multi-Granularity Alignment Network, MGAN）。在 MGAN 中，由辅助任务引导的新型 Coarse2Fine 注意力可以帮助 AC 任务进行与 AT 任务相同的细粒度级别的建模。为了减轻特征错误对齐，采用对比特征对齐 (contrastive feature alignment) 方法在语义上对齐特定于 aspect 的特征表示。此外，还提供了用于 AC 任务的大规模多域数据集。从经验上讲，广泛的实验证明了 MGAN 的有效性。

## 1 引言

 aspect-level 情感分类 (ASC) 旨在推断分布在句子中的 aspect 类别 (AC) 或 aspect 项 (AT) 的情感极性 (Pang, Lee 等人 2008; Liu 2012)。 一个 aspect 类别隐含地出现在句子中，它描述了实体的一般类别。 例如，在“三文鱼很好吃，服务员很粗鲁”这句话中，用户分别对“食物”和“服务”两个 aspect 类别进行了肯定和否定的评价。  aspect 项表征明确出现在句子中的特定实体。 考虑到同一句话“鲑鱼很好吃，服务员很粗鲁”， aspect 词是“鲑鱼”和“服务员”，用户分别对它们表达了积极和消极的情绪。在 aspect 粒度方-面，AC 任务是粗粒度的，而 AT 任务是细粒度的。

为了对面向 aspect 的情感分析建模，为循环神经网络 (RNN) 配备注意力机制已成为主流方法（Tang 等人 2015；Wang 等人 2016；Ma 等人 2017；Chen 等人 2017)，其中 RNN 旨在捕获序列模式，而注意机制是强调适当的上下文特征以对特定于 aspect 的表示进行编码。通常，基于注意力的 RNN 模型只有在大型语料库可用时才能获得良好的性能。然而，AT 级数据集需要从句子中全面手动标记或通过序列标记算法提取 aspect 项，这尤其昂贵。 因此，现有的公共 AT 级数据集都相对较小，这限制了神经模型的潜力。

尽管如此，我们观察到大量的 AC 级语料库更容易获取。 这是因为 aspect 类别通常在可以预定义的一小组一般 aspect 中。 例如，评论网站或社交媒体等商业服务可以针对特定领域中的产品或事件（例如，“食物”、“服务”、“速度”和“价格”）定义一组有价值的 aspect 类别。餐厅域）。结果，对不同 aspect 类别的用户偏好的大量收集变得可行。受此观察的启发，我们提出了一个新问题，即同时跨域和跨粒度的、从粗到细的任务迁移，目的是从粗粒度 AC 任务的丰富源域中借用知识到小规模的细粒度 AT 任务的目标领域。

实现这一设置的挑战有两个：（1）任务差异：这两个任务涉及不同粒度的 aspect 。 源 aspect 是粗粒度的 aspect 类别，缺乏上下文中的先验位置信息。 而目标 aspect 是细粒度的，具有准确的位置信息。 因此， aspect 的粒度不一致导致了任务之间的差异；  (2) 特征分布差异：通常两个任务中的域不同，这会导致 aspect 及其上下文在域之间的分布偏移。 例如，在源域：Restaurant 域中，tasty 和 delicious 用于表达对aspect类别“food”的积极情绪，而 lightweight 和 responsive 通常表示对目标域：Laptop域中的aspect term “mouse”的积极情绪。

为了解决这些挑战，我们提出了一个名为多粒度对齐网络 (MGAN) 的新框架，以同时对齐跨域的 aspect 粒度和 aspect 特定的特征表示。具体来说，MGAN 由两个网络组成，分别用于学习两个域的 aspect-specific 的表示。首先，为了减少域之间的任务差异，即在相同的细粒度级别对两个任务进行建模，我们提出了一种新颖的 Coarse2Fine (C2F) 注意力模块来帮助源任务自动捕获相应的 aspect 给定 aspect 类别的上下文中的术语（例如，“salmon”到“food”）。 没有任何额外的标签，C2F 注意力模块可以通过一个辅助任务来学习从粗到细的过程。 实际上，更具体的 aspect 词及其位置信息与情感表达最直接相关。C2F模块为源任务补全这些缺失的信息，有效缩小任务之间的 aspect 粒度差距，方便后续的特征对齐。

其次，考虑到一个句子可能包含具有不同情感的多个 aspect ，因此针对该 aspect 捕获不正确的情感特征会误导特征的对齐。为了防止错误对齐，我们采用对比特征对齐 (CFA) (Motiian et al. 2017) 在语义上对齐 aspect-specific 的表示。 CFA 通过最大限度地确保来自不同领域但同一类的等价分布来考虑语义对齐，并通过保证来自不同类别和领域的分布尽可能不相似来考虑语义分离。此外，我们构建了一个名为 YelpAspect 的大规模多域数据集，每个域有 10 万个样本，作为非常有益的源域。根据经验，大量实验表明，所提出的 MGAN 模型可以在来自 SemEval'14 ABSA 挑战的两个 AT 级数据集和一个不符合语法的 AT 级 Twitter 数据集上实现卓越的性能。

我们本文的贡献有四方-面：（1）据我们所知，首次提出了一种跨领域和粒度的新型迁移设置，用于 aspect 级别的情感分析；  (2) 构建新的大规模、多领域 AC 级数据集；  (3) 提出了新颖的 Coarse2Fine 注意力以有效减少任务之间的 aspect 粒度差距；  (4) 实证研究验证了所提出模型在三个 AT 级基准测试上的有效性。

## 2 相关工作

传统的监督学习算法高度依赖广泛的手工特征来解决 aspect 级别的情感分类（Jiang et al. 2011; Kiritchenko et al.2014）。这些模型无法捕捉 aspect 与其上下文之间的语义相关性。 为了克服这个问题，已成功应用于许多 NLP 任务的注意力机制（Bahdanau、Cho 和 Bengio 2014；Sukhbaatar 等人 2015；Yang 等人 2016 ；Shen 等人 2017 ）可以提供帮助，该模型明确捕获了内在的 aspect-context 关联（Tang et al. 2015; Tang, Qin, and Liu 2016; Wang et al. 2016; Ma et al. 2017; Chen et al. 2017; Ma, Peng, and Cambria 2018;  Li 等人，2018a)。 然而，这些方法大多高度依赖数据驱动的 RNN 或量身定制的结构来处理复杂的案例，这需要大量的 AT 级数据来训练有效的神经模型。与它们不同的是，我们提出的模型可以从 AC 级任务的相关资源丰富的领域中学到的有用知识中并从中受益匪浅。

现有的用于情感分析的领域适应任务侧重于传统的情感分类而不考虑 aspect （Blitzer、Dredze 和 Pereira 2007；Pan 等人。2010；Glorot、Bordes 和 Bengio 2011；陈等人。  2012；  Bollegala、Weir 和 Carroll 2013 ；于和江 2016；李等人。2017;  2018b)。 在数据稀缺性和任务的价值方-面 ，迁移学习对于表征用户不同偏好的 aspect 级情感分析更为紧迫。 据我们所知，只有少数研究探索了基于对抗训练从单个 aspect 类别转移到同一领域中的另一个 aspect 类别（Zhang、Barzilay 和 Jaakkola 2017）。 与此不同的是，我们探索了一个积极且具有挑战性的设置，旨在跨 aspect 粒度和跨领域进行迁移。

## 3 Multi-Granularity Alignment Network

在本节中，我们介绍所提出的 MGAN 模型。我们首先介绍问题定义和符号，然后是模型的概述。 然后我们详细介绍每个组件的模型。

### 3.1 问题定义和符号表示

#### 从粗到细的任务迁移

假设我们在源域 $D_s$ 中有足够的 AC 级标记数据 $X^s =\{(x^s_k, a^s_k), y^s_k \}^{N^s}_{k=1}$，其中 $y^s_k$ 是第 $k$ 个sentence-aspect pair $(x^s_k, a^s_k)$ 的情感标签。此外，在目标域 $D_t$ 中只有少量的 AT 级有标签数据 $X^t =\{(x^t_{k'}, a^t_{k'}), y^t_{k'} \}^{N^t}_{k'=1}$ 可用。 请注意，每个源 aspect  $a^s_k$ 属于一组预定义的 aspect 类别 $C$，而每个目标 aspect  $a^t_{k'}$ 是 $x^t_{k'}$ 的一个子序列，即 aspect 项。 这个任务的目标是学习一个准确的分类器来预测目标测试数据的情感极性

### 3.2 MGAN 模型总览

MGAN 的目标是从 AC 任务的资源丰富的源域迁移到 AT 任务的缺少资源的目标域。提出的 MGAN 的架构如图 1 所示。具体来说，MGAN 由两个网络组成，分别用于处理两个 aspect 级别的任务。为了减少任务差异，两个网络包含不同数量的注意力跳跃 (attention hop) ，以便它们可以保持一致的粒度和对 aspect 的对称信息。在 MGAN 中，两个基本跳跃单元的使用与常见的基于注意力的 RNN 模型类似，其中 Context2Aspect（C2A）注意力旨在衡量每个 aspect 词的重要性，并借助每个上下文词生成 aspect 表示，以及位置感知情感 (Position-aware Sentiment，PaS) 注意力利用获得的 aspect 表示和 aspect 的位置信息来捕获上下文中的相关情感特征，以对 aspect-specific 的表示进行编码。

![image-20211027215122209](https://gitee.com/Wales-Z/image_bed/raw/master/img/image-20211027215122209.png)

【图 1】

此外，我们在 C2A 模块上建立了 Coarse2Fine (C2F) 注意力，以专门模拟源 aspect ，然后再馈送到 PaS 模块。C2F 模块使用源 aspect 表示来注意上下文中相应的 aspect 项，然后将注意到的上下文特征反向预测为源 aspect 的类别（伪标签）。在获得 aspect-specifc 的表示后，两个任务之间的知识迁移是通过对比特征对齐进行的。综上所述，源网络充当“老师”，由三级注意力跳（C2A+C2F+PaS）组成，用于AC任务，而目标网络就像一个“学生”，只使用了两个基本的注意力跳 (C2A+PaS) 用于 AT 任务。下面我们详细介绍MGAN的各个组成部分。

### 3.2 双向 LSTM 层

给定来自源域或目标域的 sentence-aspect 对 $(x, a)$，我们假设句子由 $n$ 个单词组成，即 $x=\{w_1, w_2, ..., w_n\}$，并且 aspect 包含 $m$ 个单词，即 $a=\{w^a_1 , w^a_2 , ..., w^a_m\}$。 然后我们将它们分别映射到其嵌入向量 $e=\{e_i\}^n_{i=1}\in \mathbb R ^{n\times d_w}$ 和 $e^a = \{e^a_j \} ^m _{j=1} \in \mathbb R^{m\times d_w}$。 为了在上下文中捕获 phrase 级别的情感特征（例如，“not satisfactory”），我们采用双向 LSTM (Bi-LSTM) 来保留输入句子中每个单词的上下文信息。  Bi-LSTM 将输入 $e$ 转换为上下文化的词表示 $h=\{h_i\}^n_{i=1}\in \mathbb R^{n \times 2d_h}$（即 Bi-LSTM 的隐藏状态）。 为简单起见，我们将 LSTM 单元在 $e_i$ 上的操作表示为 ${\rm LSTM}(e_i)$。 因此，上下文化的词表示 $h_i \in \mathbb R ^{2d_h}$ 由以下式子获得：

<img src="https://gitee.com/Wales-Z/image_bed/raw/master/img/image-20211028165409397.png" alt="image-20211028165409397" style="zoom: 67%;" />

其中 $;$ 表示向量拼接 (concatenation)。

### 3.3 Context2Aspect (C2A) 注意力

并非所有 apsect 词都对 apsect 的语义有同等贡献。 例如，在 apsect 术语“techs at HP”中，情绪通常通过标题“techs”表达，但很少通过诸如品牌名称“HP”之类的修饰词表达。 因此

“techs”比“at”和“HP”更重要。 这也适用于 apsect 类别（例如，“food seafood fish”）。 因此，我们提出了 Context2Aspect (C2A) 注意力来衡量 apsect 词对于每个上下文词的重要性。 形式上，我们计算上下文和 apsect 之间的成对对齐矩阵 $M \in \mathbb  R ^{n×m}$，其中第 $i$ 个上下文词和第 $j$ 个 apsect 词之间的对齐分数 $M(i, j)$ 由以下式子获得：
$$
M(i,j) = \tanh (W_a[h_i;e^a_j] + b_a)
\tag2
$$
其中 $W_a$ 和 $b_a$ 是可学习的参数。 然后，我们应用一个 row-wise softmax 函数来获得每一行的概率分布。 通过将 $\delta(i)\in \mathbb R^ m$ 定义为第 $i$ 个上下文词的单个 aspect 级别注意力，我们对所有 $\delta(i)$ 求平均值以获得 C2A 注意力为 $\alpha = \frac{1}{n} \sum^n_{i=1} \delta(i)$。  C2A 注意力通过 $h^a_* = \sum^m _{j=1}\alpha_j e^a_j$ 进一步贡献了上下文感知的 apsect 的表示，其中 $∗\in \{s, t\}$ 表示源域或目标域。我们以不同的方式处理这两个任务的 apsect 表示 $h^a_∗$，其中 $h ^a_s$ 被馈送到 C2F 模块，而 $h ^a _t$ 直接馈送到 PaS 模块。

### 3.4 Coarse2Fine (C2F) 注意力

作为真正的“意见实体”的 apsect 项与情感表达最直接相关。 然而，源任务涉及在上下文中缺乏详细位置信息的粗粒度 apsect 类别。我们希望实现任务对齐，以便目标任务可以在相同的细粒度级别上利用从源任务中学到的更多有用的知识。据观察，源 apsect 的数量要少得多，许多实例包含相同的 apsect 类别，但底层实体可以在不同的上下文中表现不同。 例如， apsect 类别“食用海鲜鱼”可以实例化为“鲑鱼”、“金枪鱼”、“味道”等。

基于这一观察，我们可以捕获源 apsect 的更具体的语义及其以上下文为条件的位置信息。受自动编码器（Ben gio et al. 2007）的启发，我们为源任务引入了一个辅助伪标签预测任务。 在这个任务中，源 apect $a^s$ 不仅被视为一个 apsect 词的序列，还作为伪标签（ apsect 的类别）$y ^c$ ，其中 $c \in C$ ， $C$ 是一组 apsect 类别。我们利用获得的 apsect 表示 $h ^a _s$ 来关注上下文，然后注意力分数聚合上下文信息以反过来预测 $a^s$ 本身的伪类别标签。如果上下文包含与源 apsect 密切相关的 apsect 项，那么注意力机制可以强调它以更好地预测。我们将这种机制表示为 Coarse2Fine 注意力，计算如下：

<img src="https://gitee.com/Wales-Z/image_bed/raw/master/img/image-20211028174111761.png" alt="image-20211028174111761" style="zoom:50%;" />

其中 $W_f \in \mathbb R^{d_u \times (2d_h + d_e)} , b_f \in \mathbb R^{d_u} $和 $u_f \in \mathbb R^{d_u}$ 是层的权重。我们将带注意力的表示 $v^a$ 提供给辅助任务预测的 softmax 层，该层通过最小化预测伪标签 $\hat y^c_k$ 与其真实值 $y^c_k$ 之间的交叉熵损失来训练，如下所示：
$$
{\cal L}_{aux} = -\frac{1}{N_s}\sum^{N_s}_{k=1} \sum_{c\in C}
y^c_k \log \hat y^c_k
\tag6
$$
然而，当上下文隐含地表达对 apsect 类别的情感时，可能不存在相应的 apsect 项。为了克服这个问题，类似于 RNN 变体（Jozefowicz、Zaremba 和 Sutskever 2015）中的门机制，我们采用融合门 (fusion gate) $F$ 来自适应地控制 $h^a_s$ 和 $v^a$ 的传递比例，以实现更具体的源 apsect 表示 $r^a_s$ ：

<img src="https://gitee.com/Wales-Z/image_bed/raw/master/img/image-20211028174727959.png" alt="image-20211028174727959" style="zoom:55%;" />

其中 $W \in \mathbb R^{d_e \times (d_e+2d_h)}$ 和 $b \in \mathbb R^{d_e}$ 是门的权重，$W' \in \mathbb R^{d_e \times 2d_h}$ 执行降维，$\odot$表示元素乘法。

### 3.5 位置感知的情感（PaS）注意力

根据 (Tang, Qin, and Liu 2016; Chen et al. 2017) 中发现的一项重要观察，更接近的情感词更有可能是 apsect 项的实际修饰语（例如，在“great food but the service is dreadful” ，“great”比“service”更接近“food”），我们在设计 PaS 注意力时考虑了 apsect 项的位置信息。 对于目标域，我们采用邻近策略来计算第 $i$ 个上下文词和 apsect 项之间的目标位置相关性，如下所示：

<img src="https://gitee.com/Wales-Z/image_bed/raw/master/img/image-20211028175133426.png" alt="image-20211028175133426" style="zoom:58%;" />

其中 $m_0$ 是第一个 apsect 词的索引，$n$ 和 $m$ 分别是句子和 apsect 的长度。

不幸的是，在给出 apsect 类别的源域中，无法直接访问相应 apsect 项的确切位置。相反，C2F 注意力向量 $\beta^f \in \mathbb R^n$，表示每个上下文词是 apsect 词的概率，可以帮助建立位置相关性。 我们首先定义一个位置矩阵 $L \in \mathbb R ^{n \times n}$ 来表示句子中每个词的接近程度 (proximity):
$$
L_{ii'}=1-\frac{|i-i'|}{n}, i,i' \in [1,n]
\tag{10}
$$
然后我们借助 C2F 注意力权重，通过 $p^s_i = L_i \beta^f$ 计算第 $i$ 个上下文词的源位置相关性。显然，更接近 $\beta^f$ 中具有较大值的可能的 aspect 项的第 $i$ 个上下文词将具有更大的位置相关性 $p^s_i$ 。最后，PaS 注意力由两个域的一般形式计算：

<img src="https://gitee.com/Wales-Z/image_bed/raw/master/img/image-20211028180101524.png" alt="image-20211028180101524" style="zoom:67%;" />

其中 $p ^∗ _i$ 是位置相关性，$r ^a _∗$ 是输入的 aspect 表示，$∗ \in \{s, t\}$ 表示源域或目标域（注意 $r ^a_ t = h ^a _t$ ）。 然后我们将 aspect-specific 的表示 $v^o$ 传递给一个全连接层和一个用于情感分类的 softmax 层。 通过分别最小化两个交叉熵损失 $\cal L ^s _{sen}$ 和 $\cal L ^t _{sen}$ 来训练两个域的情感分类任务。

### 3.6 对比特征对齐 (Contrastive Feature Alignment)

在以相同的粒度获得两个领域的特定 apsect 的表示后，我们将进一步弥合跨域的分布差距。流行的无监督域适应方法（Gretton et al. 2007; Ganin et al. 2016）需要大量未标记的目标数据才能达到令人满意的性能，这在我们的问题中是不切实际的，我们的问题中收集未标记的数据需要对句子中所有 apsect 项进行费力地注释。因此，受 (Motiian et al. 2017) 的启发，我们通过充分利用有限的目标标记数据在跨域语义上对齐表示来执行对比特征对齐 (CFA)。 在数学上，我们用 $g_s$ 和 $g_t$ 参数化两个网络，并用 $\mathbb P$ 表示概率分布。具体来说，CFA 包括语义对齐（SA）和语义分离（SS）。SA 旨在确保特征表示 $\mathbb P(g_s(X^s ))$ 和 $\mathbb P(g_t(X^t ))$ 的相同分布以不同域但同一类为条件，而 SS 通过保证 $\mathbb P(g_s(X^s ))$ 和 $\mathbb P(g_t(X^t ))$ 在不同的域和类上都尽可能不同。考虑到只有少量目标标记数据可用，我们使用足够的数据将 CFA 表征分布还原为 pair-wise surrogates：
$$
{\cal L}_{cfa} = \sum_{k,k'}\omega(g_s(x^s_k, a^s_k),g_t(x^t_k, a^t_k))
\tag{14}
$$
其中 $\omega(·,·)$ 是一个对比函数，它根据来自两个域的监督信息执行语义对齐或分离。形式上，$ \omega (·,·)$被定义为：

<img src="https://gitee.com/Wales-Z/image_bed/raw/master/img/image-20211028203134314.png" alt="image-20211028203134314" style="zoom: 58%;" />

其中 $D$ 是决定分离程度的参数，在我们的实验中设置为 1。

### 3.7 交替训练

将我们之前引入的损失与 $l_2$ 正则化相结合，我们将源网络和目标网络的整体损失构成为

<img src="https://gitee.com/Wales-Z/image_bed/raw/master/img/image-20211028203321727.png" alt="image-20211028203321727" style="zoom: 50%;" />

其中 $\lambda, \rho$ 分别平衡 CFA 损失和 $l_2$ 正则化损失的影响。源网络与目标网络相比，多了一个辅助损失 $\cal L_{aux}$ 来实现任务对齐。 整个训练过程包括两个阶段：

(1) 为了防止目标域的早期过拟合，通过优化 ${\cal L^s_{sen} }+ {\cal L_{aux }}+ {\cal\rho L^s_{reg}}$ 在源域上单独训练源网络 $S$。
然后，$S$ 和 $S$ 的 $\rm BiLSTM$、$\rm C2A $和 $\rm PaS $ 模块分别用于初始化 MGAN 的源网络和目标网络。  

(2) 我们交替优化源网络的 $\cal L_{src}$ 和目标网络的 $\cal L_{tar}$。

