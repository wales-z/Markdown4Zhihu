# 【推荐系统论文笔记】TDAR: Semi-supervised Collaborative Filtering by Text-enhanced Domain Adaptation

TDAR：**T**ext-enhanced **D**omain **A**daptation **R**ecommendation

## 摘要

------

数据稀疏性是推荐系统的固有挑战，其中大部分数据是从用户的隐式反馈中收集的。这给设计有效算法带来了两个困难：

- 第一，大多数用户与系统的交互很少，没有足够的数据可供学习；
- 其次，隐式反馈中没有负样本，执行负采样生成负样本是一种常见的做法。然而，这会导致许多潜在的正样本被错误标记为负样本，而且数据稀疏会加剧错误标记（mislabeling）问题。

为了解决这些困难，我们将稀疏隐式反馈的推荐问题视为半监督学习任务，并探索域自适应（domain adaptation）来解决它。我们将从密集数据中学到的知识转移到稀疏数据中，而且我们重点关注最具挑战性的情况——没有用户或物品重叠。

在这种极端情况下，直接对齐（align）两个数据集的embeddings是次优的（sub-optimal），因为两个潜在空间（latent space）的编码信息差别很大。因此，我们采用域不变（domain-invariant）的文本特征作为锚点来对齐潜在空间。为了对齐embeddings，我们提取每个用户和物品的文本特征，并将它们输入到具有用户和物品的embeddings的域分类器中。这些embeddings被训练用于迷惑（puzzle）分类器，文本特征被固定为锚点。通过域自适应，源域中的分布模式被迁移到目标域。由于目标域的部分可以通过域适应来监督，我们放弃了目标域数据集中的负采样以避免标签噪声。

我们采用三对真实世界的数据集来验证我们的迁移策略的有效性。结果表明，我们的模型明显优于现有模型。

------

## 关键词

迁移学习、域自适应、协同过滤、物品推荐、用户评论、对抗性学习。

## 1 引言  INTRODUCTION

由于在电子商务和视频网站等各种在线平台上的普遍应用，推荐系统获得了广泛的研究关注。在实际应用中，隐式反馈数据（一类数据，如点击和购买等）被广泛使用，因为它易于收集且普遍适用（generally applicable）。实际应用中的推荐通常存在严重的稀疏性问题，这会导致两个困难：

(1) 没有足够的交互来为模型学习提供信息。此外，数据高度不平衡：大多数用户和物品与系统只有很少的交互，这使得推荐任务更加困难。 
(2) 由于我们在隐式反馈中只观察到一小部分正样本，现有的负采样策略是将未观察到的样本视为负样本 [6, 18, 21, 25, 27]。
然而，通过这种方式，许多潜在的正样本被错误地标记为负样本，并且模型被标签噪声严重误导，尤其是在稀疏数据上。

为了解决上述挑战，我们采用迁移学习来丰富稀疏数据集上的信息，并且我们专注于没有用户和物品重叠的跨域推荐。具体来说，我们分别在稠密和稀疏数据上训练源模型和目标模型，并探索域自适应 [4, 5] 以对齐embeddings（即潜在因子 latent factors），这是协同过滤 (CF) 模型中对用户偏好进行编码的关键组成部分。我们将从密集数据中学到的知识（如分布模式）迁移到稀疏数据中，以学习它更好的embeddings。考虑到目标模型同时受到正样本和域适应机制的监督，我们不对稀疏数据进行负采样以避免标签噪声问题。正如我们所见，学习目标模型是一项半监督学习任务。

为了说明我们的策略，我们首先简要介绍域自适应。Ganin等人[5] 提出了用于无监督图像分类任务的域对抗性神经网络 (DANN)。在 DANN 中，训练一个特征提取器来提取视觉特征，同时训练一个域分类器来辨别当前特征来自哪个域（分类）。通过对抗地训练特征提取器来迷惑域分类器，两个域的视觉特征被对齐，而且从源域中学到的知识被迁移到了目标域。域自适应的更多细节可以在第 2.1.1 小节中找到。

在本文中，我们的目标是在推荐任务中通过域适应来对齐embeddings，但面临一个关键问题。我们在图 1 中说明了 vanilla DANN 和我们的策略。 DANN 被提出并被用于图像分类任务，它在视觉空间中对齐高级的（high-level）图像表示。由于两个域共用相同的特征提取器，来自两个域的图像被映射到相同的空间，因此语义相似的图像分布在此空间中的相似位置。通过域自适应，具有相似语义的集群（clusters）被对齐在一起，且分布模式被迁移以改善（refine）目标域上的表示。以图 1 (a) 中展示的视觉空间为例，负半轴和正半轴分别编码猫和狗，因此来自不同域的猫都映射到负半轴并通过域自适应对齐在一起。

<img src="https://raw.githubusercontent.com/wales-z/Markdown4Zhihu/master/Data/tdar_for_zhihu/figure1.png" alt="figure1" style="zoom:80%;" />

<center>图1 计算机视觉任务和推荐任务中的域适应示例。我们使用不同的颜色来区分两个域，使用实线和虚线来区分不同的类别。为了简洁，视觉空间、潜在空间和文本空间都表示为一维空间。</center>

然而，在基本的 CF 模型中，没有具有特定语义的数据（例如图像和文本），我们通过将用户和物品嵌入（embedding）到潜在空间中来提取高级的密集特征。通过这种方式，我们将来自不同领域的用户和物品映射到不同的潜在空间。以图1 (b)中的电影为例，实线和虚线分别表示恐怖片和喜剧片。如图1 (b) 所示，直接对齐embeddings可能会引起误导——来自“橙色”域的恐怖电影和来自“蓝色”域的喜剧被聚集在一起，并且分布模式被错误地迁移移。原因是这些embeddings被映射到不同的潜在空间——在“橙色”域上，负半轴和正半轴分别编码恐怖和有趣，而在“蓝色”域上，我们面临相反的情况。为了解决这个间隔 / 差距（gap），我们需要在同一个空间中进行域适应，即我们既对齐空间又对齐embeddings。

为了对齐潜在空间，我们探索域不变特征作为锚点。在本文中，我们利用了可以从用户评论（review）中轻松提取的文本特征。一个例子如图 1 (c) 所示，我们将文本特征与embedding连接（concatenate）起来，从而将空间扩展为文本潜在空间（横轴表示潜在空间，纵轴表示文本空间）。在图 1(b) 所示的潜在空间中，不同的类别是不可分离的，而在图 1(c) 中，不同的类别可以通过扩展文本维度来分离。对于域自适应，我们使用连接的embeddings和文本特征作为域分类器的输入。embeddings使用分类器进行对抗训练，而文本特征则固定。

正如我们所见，在我们的策略中，文本特征应该是域不变的，例如，来自所有域的恐怖电影都映射到文本空间的负半轴。现有的许多模型 [2, 30] 提取文本特征用于推荐，而这些特征不是域不变的。为了缩小这一差距，首先，我们提出了一种称为文本记忆网络 ( Text Memory Network，TMN) 的记忆结构，通过将每个用户和物品映射到单词语义空间（word semantic space）来提取文本特征。​然后，我们将这些特征注入 CF 模型以生成预测。这个由文本特征和CF模块组成的模型被命名为文本协同过滤（Text Collaborative Filtering，TCF）模型。最后，我们在源域和目标域上同步训练两个 TCF 模型，并通过自适应网络将它们连接起来。这个迁移学习模型被命名为文本增强域自适应推荐 (Text-enhanced Domain Adaptation Recommendation，TDAR) 方法。

具体来说，我们的贡献如下：

- 我们通过将embeddings对齐到相同的潜在空间来提出一种域自适应推荐方法（TDAR），这大大提高了稀疏数据集的性能。为了对齐空间和embeddings，我们使用文本特征作为锚点。
- 作为 TDAR 中的一个重要模块，我们设计了一个记忆网络来提取域不变的文本特征，并将这些特征注入到 CF 模型中以提出基于文本的 CF 模型。
- 我们在三对真实世界数据集上设计了综合实验，以证明我们提出的方法的有效性。代码可在 https://github.com/Wenhui-Yu/TDAR 上获得。



## 2 相关工作

最近，推荐系统（RS）因其在各种在线平台上的广泛应用而受到越来越多的关注。我们根据历史交互对用户偏好进行建模，并将个性化推荐返回给每个用户。在各种 RS 模型中，CF 模型以直接 [7, 22] 和高级 [13, 20] 方式从用户-物品交互图中挖掘协同信息（collaborative infromation）。隐语义模型 / 潜在因素模型（latent factor models） [8, 13, 20] 被认为是一种特殊的 CF 模型，使用embeddings对用户偏好和物品属性进行编码，并测量embeddings在潜在空间中的距离。为了提高表示能力，已经提出了许多变体 [6, 24, 26, 28, 30]。虽然进行了广泛的研究，但是仍然存在一个关键问题：RS 存在严重的稀疏问题，并且在稀疏数据上的性能还有很多不足之处。在本文中，我们的目标是利用迁移学习来丰富稀疏数据的信息，并使用文本特征作为锚点，因此我们在本节中介绍了最相关的方面：跨域推荐和基于文本的推荐。

### 2.1 跨域推荐

在本小节中，我们首先介绍核心技术——域自适应，然后介绍一些关于有重叠和无重叠的跨领域推荐的相关工作。

#### 2.1.1 域自适应

为了学习只有残缺的标签甚至没有标签的数据，迁移学习被提出，将从标记良好的源数据集学到的知识迁移到目标数据集 [1, 4, 5, 15] 。Ganin等人[5]提出了用于无监督图像分类任务的 DANN。假设 <img src="https://www.zhihu.com/equation?tex=\{{\bf x}_i^d, y_i^d\}_{i=1,...,m;d=S}" alt="\{{\bf x}_i^d, y_i^d\}_{i=1,...,m;d=S}" class="ee_img tr_noresize" eeimg="1"> 是带标记的源数据， <img src="https://www.zhihu.com/equation?tex=\{{\bf x}_i^d\}_{i=1,...,n;d=T}" alt="\{{\bf x}_i^d\}_{i=1,...,n;d=T}" class="ee_img tr_noresize" eeimg="1"> 是无标记的目标数据。DANN有三部分：一个特征提取器 <img src="https://www.zhihu.com/equation?tex=G_f(,\theta_f)" alt="G_f(,\theta_f)" class="ee_img tr_noresize" eeimg="1"> （CNN的卷积层），一个标签预测器 <img src="https://www.zhihu.com/equation?tex=G_y(,\theta_y)" alt="G_y(,\theta_y)" class="ee_img tr_noresize" eeimg="1"> （CNN的全连接层）和一个域分类器 <img src="https://www.zhihu.com/equation?tex=G_d(,\theta_d)" alt="G_d(,\theta_d)" class="ee_img tr_noresize" eeimg="1"> 。模型训练的时候，分别在在源域更新  <img src="https://www.zhihu.com/equation?tex=\theta_f" alt="\theta_f" class="ee_img tr_noresize" eeimg="1">  和   <img src="https://www.zhihu.com/equation?tex=\theta_y" alt="\theta_y" class="ee_img tr_noresize" eeimg="1">  来最小化标签预测损失 <img src="https://www.zhihu.com/equation?tex=\sum_iL(G_y(G_f({\bf x}_i^S,\theta_f),\theta_y),y_i^S)" alt="\sum_iL(G_y(G_f({\bf x}_i^S,\theta_f),\theta_y),y_i^S)" class="ee_img tr_noresize" eeimg="1"> 和训练  <img src="https://www.zhihu.com/equation?tex=\theta_d" alt="\theta_d" class="ee_img tr_noresize" eeimg="1">  和  <img src="https://www.zhihu.com/equation?tex=\theta_f" alt="\theta_f" class="ee_img tr_noresize" eeimg="1">  来最小化域预测损失 <img src="https://www.zhihu.com/equation?tex=\sum_{i,d}L(G_d(G_f({\bf x}_i^d,\theta_f),\theta_d),d)" alt="\sum_{i,d}L(G_d(G_f({\bf x}_i^d,\theta_f),\theta_d),d)" class="ee_img tr_noresize" eeimg="1"> 。通过对抗地训练  <img src="https://www.zhihu.com/equation?tex=\theta_d" alt="\theta_d" class="ee_img tr_noresize" eeimg="1">  和  <img src="https://www.zhihu.com/equation?tex=\theta_f" alt="\theta_f" class="ee_img tr_noresize" eeimg="1">  ，两个域的视觉特征  <img src="https://www.zhihu.com/equation?tex=G_f({\bf x}_i^S, \theta_f)" alt="G_f({\bf x}_i^S, \theta_f)" class="ee_img tr_noresize" eeimg="1">  和  <img src="https://www.zhihu.com/equation?tex=G_f({\bf x}_i^T, \theta_f) " alt="G_f({\bf x}_i^T, \theta_f) " class="ee_img tr_noresize" eeimg="1">  被对齐了，而且从源域学到的知识也被迁移到了目标域。



#### 2.1.2 有重叠的跨域推荐

有许多模型被提出用于有用户和物品重叠的跨域推荐。

- 潘等人[19] 通过直接最小化源embeddings和目标embeddings之间的距离的 Frobenius 范数来减小二者的差异。
- 卢等人[16] 提出了一种选择性迁移学习方法，该方法基于 boost 算法决定要迁移的内容。
- 胡等人[9] 在源域和目标域上提出了两个深度神经网络，通过共享用户embeddings层，将所有用户和物品映射到同一个潜在空间，通过构建两个网络之间的交叉连接（cross connections），实现参数跨域迁移。
- 袁等人[29] 通过自动编码器对每个用户进行编码，并通过域自适应对齐用户表示。
- 胡等人[10]使用在源域上与当前用户有交互的物品来增强目标域上的当前物品的表示，并使用用户评论来提高模型性能。



我们可以看到迁移学习很容易实现，因为存在用户和物品重叠。在这种情况下，两个域的用户-物品二分图是整个图的不同部分，我们可以通过简单地在两个域之间共享重叠用户和物品的embeddings来迁移知识。然而，如果没有重叠，两个图是完全独立的，embeddings是不可共享的，因此我们必须使用更高级的方法，例如域自适应。



#### 2.1.3 无重叠的跨域推荐

无重叠的跨域推荐有多种模型。

- Kanagawa等人[11] 介绍了一个有趣的任务——基于文本将源域中的物品推荐给目标域中的用户。为了实现这个目标，用自动编码器提取文本特征并通过域自将它们对齐。 
- Wang 等人 [23] 提出了长短期记忆（LSTM）来构建用户和物品的文本表示，并将两个域的它们对齐用于迁移学习。

正如我们所看到的，这些模型 [11, 23] 在文本空间中实现了域自适应，因为embeddings很难在没有重叠的情况下对齐。在这种情况下，[11, 23] 中的跨域推荐更接近于自然语言处理 (NLP) 任务而不是推荐任务。 

- Li 等人[14] 提出了一种“码本（codebook）”方法，该方法在集群级别（cluster-level）迁移评分模式，但过于粗糙和经验主义。此外，该方法基于用户的评分模式，因此难以扩展到隐式反馈的情形。

在本文中，我们的目标是改进embeddings，这是 CF 模型的关键表示。据我们所知，本文是第一篇专注于无用户和物品重叠的跨域推荐任务的embeddings对齐的工作。



### 2.2 基于文本的推荐

由于我们想使用文本特征作为锚点来对齐embeddings，我们为每个用户和物品提取域不变的文本特征。在本小节中，我们将介绍一些基于文本的推荐模型。 [11, 23]中介绍的基本模型都是基于文本的模型。

- Kanagawa等人[11] 设计了一个自动编码器，而且利用 LSTM 来提取用户和物品的文本表示。
- 郑等人[30] 为每个用户和物品聚集评论，并通过卷积神经网络 (CNN) 从这些评论中提取文本特征。
- Chen [2] 等人进一步增加了注意力机制。 

然而，这些现有模型提取的文本特征不是域不变的。受 [10] 的启发，我们在本文中提出了一种用于文本特征的记忆网络。



## 3 文本记忆网络

在本文中，粗体大写字母表示矩阵。假设总共有  <img src="https://www.zhihu.com/equation?tex=M" alt="M" class="ee_img tr_noresize" eeimg="1">  个用户和  <img src="https://www.zhihu.com/equation?tex=N" alt="N" class="ee_img tr_noresize" eeimg="1">  个物品，我们使用矩阵  <img src="https://www.zhihu.com/equation?tex={\bf R} ∈ \mathbb R ^{M×N}" alt="{\bf R} ∈ \mathbb R ^{M×N}" class="ee_img tr_noresize" eeimg="1">  来表示用户和物品之间的交互。 如果用户  <img src="https://www.zhihu.com/equation?tex=u" alt="u" class="ee_img tr_noresize" eeimg="1">  投票给物品  <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1"> ，则  <img src="https://www.zhihu.com/equation?tex={\bf R}_{ui} = 1" alt="{\bf R}_{ui} = 1" class="ee_img tr_noresize" eeimg="1"> ，否则  <img src="https://www.zhihu.com/equation?tex={\bf R}_{ui} = 0" alt="{\bf R}_{ui} = 0" class="ee_img tr_noresize" eeimg="1"> 。我们的任务是对缺失值（ <img src="https://www.zhihu.com/equation?tex={\bf R}" alt="{\bf R}" class="ee_img tr_noresize" eeimg="1">  中的  <img src="https://www.zhihu.com/equation?tex=0" alt="0" class="ee_img tr_noresize" eeimg="1"> ）进行预测（表示为  <img src="https://www.zhihu.com/equation?tex=\hat {\bf R}" alt="\hat {\bf R}" class="ee_img tr_noresize" eeimg="1"> ）。

在本节中，我们根据评论构建特定用户和特定物品（user- and item- specific）的文本表示。请注意，本节设计的模型是针对信号域的，我们分别提取源域和目标域的文本特征。以用户  <img src="https://www.zhihu.com/equation?tex=u" alt="u" class="ee_img tr_noresize" eeimg="1">  为例，我们构造了一个评论集  <img src="https://www.zhihu.com/equation?tex=R_u = \bigcup ^N _{i=1} r_{ui}" alt="R_u = \bigcup ^N _{i=1} r_{ui}" class="ee_img tr_noresize" eeimg="1">  ，其中  <img src="https://www.zhihu.com/equation?tex=r_{ui}" alt="r_{ui}" class="ee_img tr_noresize" eeimg="1">  是一个包含  <img src="https://www.zhihu.com/equation?tex=u" alt="u" class="ee_img tr_noresize" eeimg="1">  对  <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1">  的评论的词的集合。如果  <img src="https://www.zhihu.com/equation?tex=u " alt="u " class="ee_img tr_noresize" eeimg="1"> 与  <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1">  没有交互，则  <img src="https://www.zhihu.com/equation?tex=r_{ui} = \emptyset" alt="r_{ui} = \emptyset" class="ee_img tr_noresize" eeimg="1"> 。类似地，物品  <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1">  的评论集是  <img src="https://www.zhihu.com/equation?tex=R_i = \bigcup ^M _{u=1} r_{ui}" alt="R_i = \bigcup ^M _{u=1} r_{ui}" class="ee_img tr_noresize" eeimg="1">  。我们用  <img src="https://www.zhihu.com/equation?tex=W" alt="W" class="ee_img tr_noresize" eeimg="1">  来表示词集： <img src="https://www.zhihu.com/equation?tex=W = \bigcup ^M _{u=1} \bigcup ^N _{i=1} r_{ui}" alt="W = \bigcup ^M _{u=1} \bigcup ^N _{i=1} r_{ui}" class="ee_img tr_noresize" eeimg="1">  ，用  <img src="https://www.zhihu.com/equation?tex=H" alt="H" class="ee_img tr_noresize" eeimg="1">  来表示词的总数： <img src="https://www.zhihu.com/equation?tex=H = |W|" alt="H = |W|" class="ee_img tr_noresize" eeimg="1"> 。由于我们想要提取域不变的文本特征（即来自所有域的特征都在同一空间中），我们通过线性组合评论的词语义(word semantic)向量将所有用户和物品映射到词语义空间。 <img src="https://www.zhihu.com/equation?tex={\bf S}\in \mathbb R ^{H\times K_1}" alt="{\bf S}\in \mathbb R ^{H\times K_1}" class="ee_img tr_noresize" eeimg="1">  是word2vec[17]在GoogleNews语料库上预训练的词语义矩阵， <img src="https://www.zhihu.com/equation?tex={\bf S}_w" alt="{\bf S}_w" class="ee_img tr_noresize" eeimg="1"> 表示词 <img src="https://www.zhihu.com/equation?tex=w" alt="w" class="ee_img tr_noresize" eeimg="1"> 的语义特征。我们使用  <img src="https://www.zhihu.com/equation?tex={\bf E} ∈ \mathbb R ^{M×K_1}" alt="{\bf E} ∈ \mathbb R ^{M×K_1}" class="ee_img tr_noresize" eeimg="1">  和  <img src="https://www.zhihu.com/equation?tex={\bf F} ∈ \mathbb R ^{N×K_1}" alt="{\bf F} ∈ \mathbb R ^{N×K_1}" class="ee_img tr_noresize" eeimg="1">  分别表示我们为用户和物品构建的文本特征。以用户  <img src="https://www.zhihu.com/equation?tex=u" alt="u" class="ee_img tr_noresize" eeimg="1">  为例， <img src="https://www.zhihu.com/equation?tex={\bf E}_u = \sum_{w\in R_u} a_{uw} {\bf S}_w" alt="{\bf E}_u = \sum_{w\in R_u} a_{uw} {\bf S}_w" class="ee_img tr_noresize" eeimg="1">  ，其中  <img src="https://www.zhihu.com/equation?tex=a_{uw}" alt="a_{uw}" class="ee_img tr_noresize" eeimg="1">  是词  <img src="https://www.zhihu.com/equation?tex=w" alt="w" class="ee_img tr_noresize" eeimg="1">  基于  <img src="https://www.zhihu.com/equation?tex=u" alt="u" class="ee_img tr_noresize" eeimg="1">  的语义偏好的权重。我们提出了一个文本记忆网络 (TMN) 来计算用户 {  <img src="https://www.zhihu.com/equation?tex=a_{uw}" alt="a_{uw}" class="ee_img tr_noresize" eeimg="1">  } 和物品 {  <img src="https://www.zhihu.com/equation?tex=a_{iv}" alt="a_{iv}" class="ee_img tr_noresize" eeimg="1">  } 的权重，以根据单词语义构建文本特征。

对一个喜欢恐怖电影的用户  <img src="https://www.zhihu.com/equation?tex=u" alt="u" class="ee_img tr_noresize" eeimg="1"> ， <img src="https://www.zhihu.com/equation?tex=u" alt="u" class="ee_img tr_noresize" eeimg="1">  可能更喜欢相关的词（比如“horrible”、“frightened”、“terrifying”），对不相关的词（比如“this”、“is”、“a  ”）和相反的词（如“funny”、“relaxing”、“comical”）没兴趣。 对于  <img src="https://www.zhihu.com/equation?tex=u" alt="u" class="ee_img tr_noresize" eeimg="1">  偏爱的词  <img src="https://www.zhihu.com/equation?tex=w" alt="w" class="ee_img tr_noresize" eeimg="1"> ，我们需要为  <img src="https://www.zhihu.com/equation?tex=w" alt="w" class="ee_img tr_noresize" eeimg="1">  设置一个很大的权重  <img src="https://www.zhihu.com/equation?tex=a_{uw}" alt="a_{uw}" class="ee_img tr_noresize" eeimg="1"> 。在物品方面，对于恐怖片  <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1"> ， <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1">  的评论中的相关词提供了很多关于  <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1">  的信息，而不相关或相反的词提供的信息很少。针对对  <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1">  很重要的词  <img src="https://www.zhihu.com/equation?tex=v" alt="v" class="ee_img tr_noresize" eeimg="1"> ，我们需要设置一个很大的权重  <img src="https://www.zhihu.com/equation?tex=a_{iv}" alt="a_{iv}" class="ee_img tr_noresize" eeimg="1">  。容易看出，在这个任务中，我们的目标是向用户和物品推荐偏好词 (preferred words)。

受矩阵分解 [13] 的启发，我们分别为用户、物品和单词声明了三个矩阵  <img src="https://www.zhihu.com/equation?tex={\bold P} \in {\mathbb R} ^{M\times K_2},{\bold Q} \in {\mathbb R} ^{N \times K_2}" alt="{\bold P} \in {\mathbb R} ^{M\times K_2},{\bold Q} \in {\mathbb R} ^{N \times K_2}" class="ee_img tr_noresize" eeimg="1">  和  <img src="https://www.zhihu.com/equation?tex={\bold T} \in {\mathbb R} ^{H\times K_2}" alt="{\bold T} \in {\mathbb R} ^{H\times K_2}" class="ee_img tr_noresize" eeimg="1">  。 以用户为例，我们使用  <img src="https://www.zhihu.com/equation?tex=e_{uw} = {\bold P}_u {\bold T}^{\rm T}_w" alt="e_{uw} = {\bold P}_u {\bold T}^{\rm T}_w" class="ee_img tr_noresize" eeimg="1">  来建模  <img src="https://www.zhihu.com/equation?tex=u" alt="u" class="ee_img tr_noresize" eeimg="1">  对单词  <img src="https://www.zhihu.com/equation?tex=w" alt="w" class="ee_img tr_noresize" eeimg="1">  的偏好。 为了进一步强调重要的词，我们将  <img src="https://www.zhihu.com/equation?tex=e_{uw} " alt="e_{uw} " class="ee_img tr_noresize" eeimg="1">  输入到 softmax 函数中以获得  <img src="https://www.zhihu.com/equation?tex=\{a_{uw}\}:a_{uw} =\frac{\exp(e_{uw})} {\sum_{w^′∈R_u} \exp(e_{uw^′})}" alt="\{a_{uw}\}:a_{uw} =\frac{\exp(e_{uw})} {\sum_{w^′∈R_u} \exp(e_{uw^′})}" class="ee_img tr_noresize" eeimg="1">  。 针对物品  <img src="https://www.zhihu.com/equation?tex=a_{iv}" alt="a_{iv}" class="ee_img tr_noresize" eeimg="1">  的权重以相同的方式构造。 我们最终通过  <img src="https://www.zhihu.com/equation?tex=\hat {\bold R} = \sigma(\bold E \bold F^{\rm T} )" alt="\hat {\bold R} = \sigma(\bold E \bold F^{\rm T} )" class="ee_img tr_noresize" eeimg="1">  来预测用户对物品的偏好，其中  <img src="https://www.zhihu.com/equation?tex=\sigma ( ) " alt="\sigma ( ) " class="ee_img tr_noresize" eeimg="1"> 是element-wise的 sigmoid 函数，我们使用交叉熵损失作为我们的损失函数：

<img src="https://www.zhihu.com/equation?tex=L=-\sum_{u,i}{\bold R_{ui}}\log \hat{\bold R}_{ui}
+(1-{\bold R_{ui}})log(1-\hat{\bold R}_{ui})
+\lambda reg
\tag1
" alt="L=-\sum_{u,i}{\bold R_{ui}}\log \hat{\bold R}_{ui}
+(1-{\bold R_{ui}})log(1-\hat{\bold R}_{ui})
+\lambda reg
\tag1
" class="ee_img tr_noresize" eeimg="1">
其中

<img src="https://www.zhihu.com/equation?tex=\hat{\bold R}_{ui}=\sigma \Big[ 
\Big(
\sum_{w \in R_u} \frac{\exp({\bold P}_u {\bold T}^{\rm T}_w)} {\sum_{w^′∈R_u} \exp({\bold P}_u {\bold T}^{\rm T}_{w^′})}
{\rm S}_w
\Big)

\Big(
\sum_{v \in R_i} \frac{\exp({\bold Q}_i {\bold T}^{\rm T}_v)} {\sum_{v^′∈R_i} \exp({\bold Q}_i {\bold T}^{\rm T}_{v^′})}
{\rm S}_v
\Big)

\Big]
" alt="\hat{\bold R}_{ui}=\sigma \Big[ 
\Big(
\sum_{w \in R_u} \frac{\exp({\bold P}_u {\bold T}^{\rm T}_w)} {\sum_{w^′∈R_u} \exp({\bold P}_u {\bold T}^{\rm T}_{w^′})}
{\rm S}_w
\Big)

\Big(
\sum_{v \in R_i} \frac{\exp({\bold Q}_i {\bold T}^{\rm T}_v)} {\sum_{v^′∈R_i} \exp({\bold Q}_i {\bold T}^{\rm T}_{v^′})}
{\rm S}_v
\Big)

\Big]
" class="ee_img tr_noresize" eeimg="1">
正则化项  <img src="https://www.zhihu.com/equation?tex=reg" alt="reg" class="ee_img tr_noresize" eeimg="1">  是模型参数  <img src="https://www.zhihu.com/equation?tex=\{{\bold P}, {\bold Q}, {\bold T}\}" alt="\{{\bold P}, {\bold Q}, {\bold T}\}" class="ee_img tr_noresize" eeimg="1">  的 Frobenius 范数，它是通过使用 Adam [12] 最小化  <img src="https://www.zhihu.com/equation?tex=\cal L" alt="\cal L" class="ee_img tr_noresize" eeimg="1">  来学习的。 在这个基于文本的推荐任务中，我们得到了用户、物品和单词的三分图。 在构建权重时，我们仅利用用户单词和物品单词的联系。在为用户和物品构建文本表示后，我们通过用户-物品连接来监督模型。 正如我们所见，我们确实在 TMN 中使用了三个二分图。

与现有的 CNN 和 RNN 推荐模型 [2, 23, 30] 相比，我们的文本特征提取器无法对序列信息进行建模，而擅长突出重要关键字。我们认为，为了提取特定于交互的文本信息（例如 [3] 中的任务），序列信息很重要。 然而，为了提取特定于用户和物品的文本信息，例如 [2, 23, 30] 以及本文中的任务，关键字更为重要，因为我们希望用文本特征总结每个用户和物品的偏好元素。 实验还表明 TMN 在我们的任务中表现非常好，尤其是在稀疏情况下。

## 4 文本增强跨域推荐

在通过 TMN 提取文本特征后，我们将在本节中介绍我们的文本增强域自适应推荐 (TDAR) 模型。 首先，我们将文本特征注入到 CF 模型中以提出基本的 TCF 模型。 然后，我们同时在目标域和源域上训练两个 TCF 模型（共享相同的交互函数，请参见图 2），并通过域自适应对齐用户和物品嵌入。

### 4.1 文本协同过滤 TCF

在本小节中，我们设计了我们的基本模型。   <img src="https://www.zhihu.com/equation?tex={\bold U} ∈ {\mathbb R}^{M\times K_3}" alt="{\bold U} ∈ {\mathbb R}^{M\times K_3}" class="ee_img tr_noresize" eeimg="1">  和  <img src="https://www.zhihu.com/equation?tex={\bold V} ∈ {\mathbb R}^{N\times K_3}" alt="{\bold V} ∈ {\mathbb R}^{N\times K_3}" class="ee_img tr_noresize" eeimg="1"> 分别是用户和物品的embedding。 如图 1(c) 所示，我们连接了embedding和文本特征，因此用户和物品的表示是  <img src="https://www.zhihu.com/equation?tex=[{\bold U} , {\bold E}]" alt="[{\bold U} , {\bold E}]" class="ee_img tr_noresize" eeimg="1">  和  <img src="https://www.zhihu.com/equation?tex=[{\bold V} , {\bold F}]" alt="[{\bold V} , {\bold F}]" class="ee_img tr_noresize" eeimg="1">  。我们用以下公式预测用户偏好：

<img src="https://www.zhihu.com/equation?tex=\hat {\bold R}_{ui} = f([{\bold U} , {\bold E}]_u, [{\bold V} , {\bold F}]_i, \Theta),
" alt="\hat {\bold R}_{ui} = f([{\bold U} , {\bold E}]_u, [{\bold V} , {\bold F}]_i, \Theta),
" class="ee_img tr_noresize" eeimg="1">
其中  <img src="https://www.zhihu.com/equation?tex=f(,\Theta)" alt="f(,\Theta)" class="ee_img tr_noresize" eeimg="1">  是结合用户和物品 embedding 并返回偏好预测的交互函数，例如深度结构 [8, 24, 26]， <img src="https://www.zhihu.com/equation?tex=\Theta" alt="\Theta" class="ee_img tr_noresize" eeimg="1">  表示参数。 我们还通过最小化方程（1）中给出的损失函数来学习模型。 在这个模型中， <img src="https://www.zhihu.com/equation?tex={\bold U}" alt="{\bold U}" class="ee_img tr_noresize" eeimg="1"> 、 <img src="https://www.zhihu.com/equation?tex={\bold V}" alt="{\bold V}" class="ee_img tr_noresize" eeimg="1">  和  <img src="https://www.zhihu.com/equation?tex=\Theta" alt="\Theta" class="ee_img tr_noresize" eeimg="1">  是可训练的参数， 而  <img src="https://www.zhihu.com/equation?tex={\bold E}" alt="{\bold E}" class="ee_img tr_noresize" eeimg="1">  和  <img src="https://www.zhihu.com/equation?tex={\bold F}" alt="{\bold F}" class="ee_img tr_noresize" eeimg="1">  是固定的。等式 (1) 中的  <img src="https://www.zhihu.com/equation?tex=reg" alt="reg" class="ee_img tr_noresize" eeimg="1">  表示  <img src="https://www.zhihu.com/equation?tex={\bold U}" alt="{\bold U}" class="ee_img tr_noresize" eeimg="1">  和  <img src="https://www.zhihu.com/equation?tex={\bold V}" alt="{\bold V}" class="ee_img tr_noresize" eeimg="1">  的 Frobenius 范数。

### 4.2 文本增强域自适应推荐

在本小节中，我们使用上标  <img src="https://www.zhihu.com/equation?tex=s、t、u" alt="s、t、u" class="ee_img tr_noresize" eeimg="1"> 和  <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1">  分别表示源域、目标域、用户和物品。  <img src="https://www.zhihu.com/equation?tex={\bold R}^s \in {\mathbb R} ^{M_s×N_s} " alt="{\bold R}^s \in {\mathbb R} ^{M_s×N_s} " class="ee_img tr_noresize" eeimg="1"> 和  <img src="https://www.zhihu.com/equation?tex={\bold R}^t \in {\mathbb R} ^{M_t×N_t} " alt="{\bold R}^t \in {\mathbb R} ^{M_t×N_t} " class="ee_img tr_noresize" eeimg="1">  表示源域和目标域上的交互。 我们在两个域上训练两个 TCF，同时共享相同的交互函数，因此对两个数据集的预测由下式给出：

<img src="https://www.zhihu.com/equation?tex=\hat {\bold R}^s_{u^s i^s}=f([{\bold U}^s,{\bold E}^s]_{u^s},[{\bold V}^s,{\bold F}^s]_{i^s}, \Theta)
\tag 2
" alt="\hat {\bold R}^s_{u^s i^s}=f([{\bold U}^s,{\bold E}^s]_{u^s},[{\bold V}^s,{\bold F}^s]_{i^s}, \Theta)
\tag 2
" class="ee_img tr_noresize" eeimg="1">


<img src="https://www.zhihu.com/equation?tex=\hat {\bold R}^t_{u^t i^t}=f([{\bold U}^t,{\bold E}^t]_{u^t},[{\bold V}^t,{\bold F}^t]_{i^t}, \Theta)
\tag 3
" alt="\hat {\bold R}^t_{u^t i^t}=f([{\bold U}^t,{\bold E}^t]_{u^t},[{\bold V}^t,{\bold F}^t]_{i^t}, \Theta)
\tag 3
" class="ee_img tr_noresize" eeimg="1">

然后我们在两个 TCF 上添加自适应网络以实现迁移学习。 考虑到用户和物品 embedding 的分布模式可能不同，我们分别对用户和物品进行域自适应。 这里我们以用户为例。假设有两个用户嵌入分布  <img src="https://www.zhihu.com/equation?tex=dist(\bold U^s)" alt="dist(\bold U^s)" class="ee_img tr_noresize" eeimg="1">  和  <img src="https://www.zhihu.com/equation?tex=dist(\bold U^t )" alt="dist(\bold U^t )" class="ee_img tr_noresize" eeimg="1"> ，我们使用一个二元变量  <img src="https://www.zhihu.com/equation?tex=d^u_u" alt="d^u_u" class="ee_img tr_noresize" eeimg="1">  作为域 label，表示  <img src="https://www.zhihu.com/equation?tex=\bold U_u" alt="\bold U_u" class="ee_img tr_noresize" eeimg="1">  是来自目标分布还是来自源分布： <img src="https://www.zhihu.com/equation?tex=d^u_u = 1" alt="d^u_u = 1" class="ee_img tr_noresize" eeimg="1">   ，如果  <img src="https://www.zhihu.com/equation?tex=\bold U_u  ∼ dist(\bold U^t )" alt="\bold U_u  ∼ dist(\bold U^t )" class="ee_img tr_noresize" eeimg="1">  ，以及  <img src="https://www.zhihu.com/equation?tex=d^u_u = 0" alt="d^u_u = 0" class="ee_img tr_noresize" eeimg="1"> ，如果  <img src="https://www.zhihu.com/equation?tex=\bold U_u ∼ dist(\bold U^s )" alt="\bold U_u ∼ dist(\bold U^s )" class="ee_img tr_noresize" eeimg="1"> 。   <img src="https://www.zhihu.com/equation?tex=d^u_u" alt="d^u_u" class="ee_img tr_noresize" eeimg="1">  上标表示域标签用于用户embedding，下标表示域标签用于当前用户  <img src="https://www.zhihu.com/equation?tex=u" alt="u" class="ee_img tr_noresize" eeimg="1"> 。

对于自适应网络，我们利用一个表示为  <img src="https://www.zhihu.com/equation?tex=g(,\Phi u)" alt="g(,\Phi u)" class="ee_img tr_noresize" eeimg="1">  的域分类器，该分类器针对域分类进行了训练： <img src="https://www.zhihu.com/equation?tex=\hat d^u_u=g([{\bf U},{\bf E]}_u,\Phi^u)" alt="\hat d^u_u=g([{\bf U},{\bf E]}_u,\Phi^u)" class="ee_img tr_noresize" eeimg="1"> 。 为了对齐 embedding ，我们希望分布  <img src="https://www.zhihu.com/equation?tex=dist(\bold U^s)" alt="dist(\bold U^s)" class="ee_img tr_noresize" eeimg="1">  和  <img src="https://www.zhihu.com/equation?tex=dist(\bold U^t)" alt="dist(\bold U^t)" class="ee_img tr_noresize" eeimg="1">  相似。 最广泛使用的方法是训练域分类器来区分两种分布，并训练 embedding 来迷惑分类器 [4, 5]。 具体来说，我们更新  <img src="https://www.zhihu.com/equation?tex=\Phi^u" alt="\Phi^u" class="ee_img tr_noresize" eeimg="1">  以最小化  <img src="https://www.zhihu.com/equation?tex=g(,\Phi^u)" alt="g(,\Phi^u)" class="ee_img tr_noresize" eeimg="1">  的损失，然后更新  <img src="https://www.zhihu.com/equation?tex=\bold U^s" alt="\bold U^s" class="ee_img tr_noresize" eeimg="1">  和  <img src="https://www.zhihu.com/equation?tex=\bold U^t" alt="\bold U^t" class="ee_img tr_noresize" eeimg="1">  来最大化这个损失。 通过这种方式，两个域的用户 embedding 变得不可分离，从而对齐到相同的分布。物品 embedding 以相同的方式对齐。

我们使用  <img src="https://www.zhihu.com/equation?tex={\cal L}^s" alt="{\cal L}^s" class="ee_img tr_noresize" eeimg="1">  和   <img src="https://www.zhihu.com/equation?tex={\cal L}^t" alt="{\cal L}^t" class="ee_img tr_noresize" eeimg="1">   表示源域主域和目标域的预测损失，并使用   <img src="https://www.zhihu.com/equation?tex={\cal L}^u" alt="{\cal L}^u" class="ee_img tr_noresize" eeimg="1">   和  <img src="https://www.zhihu.com/equation?tex={\cal L}^i" alt="{\cal L}^i" class="ee_img tr_noresize" eeimg="1">  分别表示用户和物品的域分类损失。  <img src="https://www.zhihu.com/equation?tex={\cal L}^s" alt="{\cal L}^s" class="ee_img tr_noresize" eeimg="1">   、   <img src="https://www.zhihu.com/equation?tex={\cal L}^u" alt="{\cal L}^u" class="ee_img tr_noresize" eeimg="1">   和  <img src="https://www.zhihu.com/equation?tex={\cal L}^i" alt="{\cal L}^i" class="ee_img tr_noresize" eeimg="1">  都是二元预测变量的交叉熵损失。 对于   <img src="https://www.zhihu.com/equation?tex={\cal L}^t" alt="{\cal L}^t" class="ee_img tr_noresize" eeimg="1">   ，我们只使用正标签作为对目标域的监督。 损失函数如下：

<img src="https://raw.githubusercontent.com/wales-z/Markdown4Zhihu/master/Data/tdar_for_zhihu/formula4.png" alt="formula4" style="zoom: 80%;" />

其中， <img src="https://www.zhihu.com/equation?tex=u^s" alt="u^s" class="ee_img tr_noresize" eeimg="1">  和  <img src="https://www.zhihu.com/equation?tex=i^s" alt="i^s" class="ee_img tr_noresize" eeimg="1">  是源域的用户和物品， <img src="https://www.zhihu.com/equation?tex=u^t" alt="u^t" class="ee_img tr_noresize" eeimg="1">  和  <img src="https://www.zhihu.com/equation?tex=i^t" alt="i^t" class="ee_img tr_noresize" eeimg="1">  是目标域的用户和物品。 <img src="https://www.zhihu.com/equation?tex=\hat {\bold R}^s_{u^s i^s}" alt="\hat {\bold R}^s_{u^s i^s}" class="ee_img tr_noresize" eeimg="1">   和  <img src="https://www.zhihu.com/equation?tex=\hat {\bold R}^t_{u^t i^t}" alt="\hat {\bold R}^t_{u^t i^t}" class="ee_img tr_noresize" eeimg="1">  已在公式(2)和(3) 中给出。 <img src="https://www.zhihu.com/equation?tex=\hat d^u_u=g([{\bf U},{\bf E]}_u,\Phi^u)" alt="\hat d^u_u=g([{\bf U},{\bf E]}_u,\Phi^u)" class="ee_img tr_noresize" eeimg="1">  ， <img src="https://www.zhihu.com/equation?tex=\hat d^i_i=g([{\bf U},{\bf E]}_i,\Phi^i)" alt="\hat d^i_i=g([{\bf U},{\bf E]}_i,\Phi^i)" class="ee_img tr_noresize" eeimg="1">  。请注意分类器   <img src="https://www.zhihu.com/equation?tex=g(,\Phi^u)" alt="g(,\Phi^u)" class="ee_img tr_noresize" eeimg="1">   和   <img src="https://www.zhihu.com/equation?tex=g(,\Phi^i)" alt="g(,\Phi^i)" class="ee_img tr_noresize" eeimg="1">   共享相同的结构，但具有不同的参数。   <img src="https://www.zhihu.com/equation?tex=reg^s" alt="reg^s" class="ee_img tr_noresize" eeimg="1">  和  <img src="https://www.zhihu.com/equation?tex=reg^t" alt="reg^t" class="ee_img tr_noresize" eeimg="1">  分别表示  <img src="https://www.zhihu.com/equation?tex=\{{\bold U^s} , {\bold V^s}\}" alt="\{{\bold U^s} , {\bold V^s}\}" class="ee_img tr_noresize" eeimg="1">  和  <img src="https://www.zhihu.com/equation?tex=\{{\bold U^t} , {\bold V^t}\}" alt="\{{\bold U^t} , {\bold V^t}\}" class="ee_img tr_noresize" eeimg="1">  的Frobenius 范数， <img src="https://www.zhihu.com/equation?tex=\lambda_s" alt="\lambda_s" class="ee_img tr_noresize" eeimg="1">   和  <img src="https://www.zhihu.com/equation?tex=\lambda_t" alt="\lambda_t" class="ee_img tr_noresize" eeimg="1">  是对应的正则化系数。 我们更新模型参数如下：

<img src="https://raw.githubusercontent.com/wales-z/Markdown4Zhihu/master/Data/tdar_for_zhihu/formula5678.png" alt="formula5678" style="zoom:80%;" />

其中  <img src="https://www.zhihu.com/equation?tex=\eta^s, \eta^t, \eta^+, \eta^-" alt="\eta^s, \eta^t, \eta^+, \eta^-" class="ee_img tr_noresize" eeimg="1">  是学习率， <img src="https://www.zhihu.com/equation?tex=\grad_{\bf X}=f(\bf X)" alt="\grad_{\bf X}=f(\bf X)" class="ee_img tr_noresize" eeimg="1">  是  <img src="https://www.zhihu.com/equation?tex=f(\bf X)" alt="f(\bf X)" class="ee_img tr_noresize" eeimg="1">  关于 <img src="https://www.zhihu.com/equation?tex=\bf X" alt="\bf X" class="ee_img tr_noresize" eeimg="1">  的梯度。TDAR中的所有参数都用Adam进行训练。TDAR的结构如图2所示。

![figure2](https://raw.githubusercontent.com/wales-z/Markdown4Zhihu/master/Data/tdar_for_zhihu/figure2.png)

<center>图2：TDAR 的图示。U 和V是用户和物品嵌入，E和F是用户和物品文本特征。f是交互函数， g是域分类器。线条表示模型的前向传播，我们使用实线和虚线来区分不同的样本。粗箭头表示反向传播。为了简洁，我们只展示了源域上的预测损失 Ls 和物品分类损失以 Li 及相应的梯度。</center>

在 TDAR 中，两个基本模型由标签和域分类器共同监督。 正如引言中所讨论的，目标域上的负标签受噪声严重污染，但正标签是纯净的，因此我们放弃了负样本。不幸的是，在隐式反馈案例中仅使用正样本进行监督会导致一个新问题——该模型倾向于将所有项目预测为正项目。为了解决这个问题，我们利用域自适应机制来监督目标域上的基本模型以及正样本。 但是，作为双域系统，需要对整个系统进行负采样。 考虑到源域通常比目标域密集得多，负标签的质量要高得多，我们采用源域进行负监督，并通过域适应将负监督转移到目标域。

对于文本特征  <img src="https://www.zhihu.com/equation?tex=\bf E" alt="\bf E" class="ee_img tr_noresize" eeimg="1">  和  <img src="https://www.zhihu.com/equation?tex=\bf F" alt="\bf F" class="ee_img tr_noresize" eeimg="1"> ，我们可以在 TDAR 训练期间预训练并固定它们，也可以从头开始与 TDAR 联合训练。 实验表明，联合训练使模型更难以调优，且没有实现性能提升，因此我们选择了前一种策略。 实验还表明，与学习用户和物品 embedding 相比，学习到的文本特征对标签噪声的鲁棒性要强得多（请参见图 3，在图 3(c) 和 3(d) 中尤其明显）。 因此，对于文本特征提取器——TMN，我们在没有前面提到的迁移学习策略的情况下根据两个域对其进行训练。 动机是我们的目标是使用文本特征来指导 embedding 对齐，因此不想盲目地对齐它们。 例如，两个不同领域（如电影和衣服）的文本表示应该是不同的。

