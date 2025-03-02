# 【论文笔记】CATN

## 摘要

在大型推荐系统中，产品（或物品）可能位于许多不同的类别（category）或域（domain）中。给定两个相关的域（例如，书和电影），用户可能与一个域中的物品进行交互，而与另一域中的物品没有交互。后者被视为冷启动用户。**跨域推荐中的关键问题是如何根据用户的将他们的偏好有效地从一个域迁移到另一个相关域。**受基于评论的推荐的进步的启发，我们提议对用户偏好的迁移进行aspect级的建模，aspect从评论中得出。为此，我们提出了一种通过aspect迁移网络针对冷启动用户的跨域推荐框架（命名为CATN）。 CATN旨在从他们的评论文档中为每个用户和每个物品提取多个aspect，并使用注意力机制来学习跨域的aspect相关性（correlation）。此外，我们进一步利用志趣相投（like-minded）的用户的辅助评论来增强用户的aspect表示。然后，使用端到端优化框架来增强我们模型的鲁棒性。在实际数据集上，CATN在评分预测精度（accuracy）方面明显优于SOTA模型。进一步的分析表明，我们的模型能够细粒度地揭示跨域的用户aspect连接，从而使推荐结果可以解释（explainable）。

## 关键词

冷启动推荐系统；基于aspect的推荐；深度学习

## 1 介绍

- 推荐系统在各种电子商务平台中起着至关重要的作用。传统的协作过滤方法主要根据用户的历史反馈向他们推荐物品。但是，这些方法在面对没有历史反馈的新用户，即冷启动用户时效果不佳。最近，跨域推荐已引起广泛关注[18，40]。给定两个相关的域（例如，书和电影），用户可以在一个域（即，源域）中具有历史交互，而在另一个域（即，目标域）中没有历史交互。对于目标域，这些用户被视为冷启动用户。但是，由于这两个域是相关的，因此可以利用源域中的反馈为目标域提供有意义的（meaningful）推荐。

- 跨域推荐的核心任务是两个相关域之间的用户偏好映射。为了实现映射，现有的方法，例如EMCDR [22]，CDLFM [29]和RC-DFM [9]，将用户的偏好编码为单个向量，然后整体进行跨域映射。如图1所示，现有解决方案分别在源域和目标域中学习用户/项表示。然后，基于重叠（overlapping）的用户来学习跨域表示映射。请注意，源域和目标域的用户表示之间的直接映射无法明确捕获用户在不同域中的各种细粒度的偏好。例如，喜欢中国功夫小说的用户更有可能喜欢中国古代戏剧。

<img src="https://raw.githubusercontent.com/wales-z/Markdown4Zhihu/master/Data/catn_note_for_zhihu/figure1.png" alt="figure1" style="zoom:80%;" />

<center>图1 现有的跨域推荐中针对冷启动用户的工作流程（以彩色显示最佳）


在我们的研究中，我们假设用户的偏好是多方面的，例如，“书”和“电影”域中的情节，文字描述和场景。对这些细粒度的语义aspect进行建模，并探索它们在各个域之间的相互关系（mutual relationships），将会导致更好的用户偏好理解和可解释的推荐。为此，我们的目标是利用用户/物品评论探索跨域aspect的关联。近年来，利用用户/物品评论进行基于aspect的推荐（例如，给定用户-物品对的评分预测）的方法激增[3-5、19]。受他们encouraging表现的启发，我们提出根据跨域评论生成的aspect来探索用户的偏好。

- 在本文中，我们提出了一个针对冷启动用户的基于aspect迁移的跨域推荐框架，名为CATN。在源域中，我们用包含该用户写过的所有评论的用户文档表示一个用户，并通过包含该物品收到过的所有评论的物品文档代表一个物品。在目标域中也是如此。因此，重叠的用户将拥有两个用户文档，一个在源域中，另一个在目标域中。
  - 为了提取用户和物品文档中提到（mention）的aspect，我们在卷积层上使用了aspect-specific的门机制。然后，通过注意力机制识别全局跨域aspect的相关性并对其进行加权，以进行偏好估算。
  - 为了支持基于评论的的知识迁移，我们引入了具有两个学习流程的新颖的基于跨域评论的偏好匹配流程。这两个学习流程的图示如图2所示。具体来说，对于给定的重叠用户和目标域中的物品，我们利用源域中的用户评论文档和目标域中的物品评论文档来进行评分预测，反之亦然。这两个学习流程在全局跨域aspect相关性的指导下轮流进行。考虑到评论的稀缺性[31]以及重叠的用户的数量很小[17]，我们通过为每个用户添加一个附加的用户辅助评论来进一步增强用户表示。一个辅助文档包含志趣相投的用户写的所有评论，即对相同物品给予了与当前用户相同评分的用户。辅助文档也被用于aspect提取。

<img src="https://raw.githubusercontent.com/wales-z/Markdown4Zhihu/master/Data/catn_note_for_zhihu/figure2.png" alt="figure2" style="zoom: 67%;" />

<center>图2 CATN中的两个学习流程（以彩色显示最佳）</center>

- 我们将主要贡献总结如下。我们为冷启动用户提出了一种新颖的深度推荐模型，它通过不同领域的评论来桥接（bridge）多个用户的固有特征。据我们所知，这是以端到端的学习方式来学习跨域aspect级偏好匹配的第一次尝试。通过在三对现实数据集上进行的广泛实验，我们证明了CATN的性能明显优于SOTA。我们还进行了详细的分析，以验证CATN的每个组件所带来的好处，并展示了CATN如何在细粒度的语义级别上发挥作用。

## 2 相关工作

我们的工作与推荐系统的两个子领域有关：跨域推荐和基于aspect的推荐。接下来，我们简要回顾这两个领域中的作品。

### 2.1 跨领域推荐

- 通过利用相关的源域作为辅助信息，一系列解决方案被提出，以解决数据稀疏问题和目标域的冷启动问题。在一开始，CMF [25]提出通过连接多个评分矩阵并跨域共享用户因子（factor）来实现跨域知识整合。然后，Temporal-Domain CF [18]跨时域共享静态组级（group-level）评分矩阵。后来，提出了CDTF [16]，通过张量分解来捕获用户-物品-域的三重关系。当将不同的域当作一个整体来考虑时，这些基于协作过滤的方法遭受了数据稀疏性问题的严重困扰。
- 近年来，随着深度学习技术的复兴，许多基于深度学习来增强知识迁移的模型被提出。 
  EMCDR [22]通过多层完全连接的神经网络显式地映射来自不同域的用户表示。 DCDCSR [40]通过生成基准因子（benchmark factors）来解决跨域和跨系统问题，从而进一步拓展了EMCDR。CoNet[15]训练一个深层的cross-stitch网络，以同时增强对两个域的推荐。 PPGN [38]利用用户-物品交互图来捕获用户偏好传播的过程。 DARec [35]配备了对抗性学习过程，用于用户-物品评分预测。  <img src="https://www.zhihu.com/equation?tex=π" alt="π" class="ee_img tr_noresize" eeimg="1"> -Net[21]用于共享帐户（shared-account）的跨域顺序（sequential）推荐。
- 为了避免用户隐私泄露，NATR [10]选择仅跨域迁移物品embedding。 SSCDR [17]研究了实际情况下跨域重叠用户的分布，并提出了一种半监督映射方法来为冷启动用户进行推荐。 
  CDLFM [29]通过利用用户邻域（neighborhoods）来修改矩阵分解和映射过程。跨域推荐器系统的另一条路线是基于聚类的，也取得了良好的性能。  <img src="https://www.zhihu.com/equation?tex=C^3R" alt="C^3R" class="ee_img tr_noresize" eeimg="1">  [8]利用用户的多个社交媒体资源来提高场所（venue）推荐的效果。 CDIE-C [30]通过跨域联合聚类（co-clustering）增强了物品embedding学习。
- 尽管如此，上述解决方案大多仅考虑评分记录，而忽略了其他补充性且丰富的信息，例如评论。 
  MVDNN [7]将用户和商品的辅助信息映射到一个潜在空间（latent space），在该空间中，用户与其偏好商品之间的相似性被最大化。为了同时利用评分和评论，RB-JTF [26]通过从评论得出的联合（joint）张量分解来迁移用户的偏好。 RC-DFM [9]使用**评论-fused的SDAE**来训练用户或物品因子，这达到了冷启动用户推荐的SOTA性能。
- 现有的基于评论的迁移方法相对于传统的基于交互的方法已获得了实质性的进展。但是，这些方法仍然有许多缺点需要克服，正如第一小节里讨论的和图1所展示的。现有方法分别学习源域和目标域中的用户和物品表示（图1中的步骤1和2）。然后他们学习基于重叠用户的跨域表示映射（cross-domain representation mapping）（图1中的步骤3）。但是此映射无法明显区分细粒度的语义特征。此外，流水线学习过程很容易累积和放大在中间步骤中由次优（sub-optimal）学习产生的噪声信息。因此，我们提出了一种完全不同的网络体系结构，以端对端的方式，通过评论在aspect级别上捕获并对齐（align）源域和目标域之间的细粒度用户偏好。

### 2.2 基于aspect的推荐

- 评论反映了用户的购买体验，且已被证明可以有效地解决推荐中的稀疏性问题。如今，基于审阅的推荐系统已成为单域推荐的重要组成部分[4，19，20，24，27，32，34，36，39]。在基于评论的推荐系统中，基于aspect的推荐系统对用户偏好和物品特征之间的细粒度关系进行建模，这类推荐系统最近引起了极大的关注。通常，基于aspect的推荐系统的解决方案可以分为两个主要类别。

- 第一类解决方案通过使用外部NLP工具包从评论中提取aspect和情感（sentiments）。解决方案样例有MTER [37]，TriRank [12]，LRPPM [2]，SULM [1]和EFM [28]等。此类解决方案的性能高度依赖于过程中使用的外部工具包的质量。
- 第二类解决方案通过内部模型组件实现aspect的自动提取。例如，JMARS [5]利用主题建模来学习多种aspect表示。继JMARS之后，提出了FLAME [33]和AFLM [3]通过集成的隐藏主题学习过程对方面级别的用户偏好和物品特征进行建模。但是，通过这些方法学习的静态表示不能对用户与物品之间动态的、复杂的关系进行建模。为了动态地建模由不同用户-物品对（user-item pair）编码出来的关系，ANR [4]使用协同注意机制（co-attention mechanism）来推断给定用户-物品对的不同aspect的重要性。最近，CARP [19]提出了一种胶囊（capsule）网络来进行评分预测并以细粒度的方式提供可解释性。

- 请注意，这两类解决方案都侧重于单域推荐。这些方法无法处理其历史交互在目标域中不可用的冷启动用户。在我们的研究中，我们在这方面做了首次尝试，通过设计跨域aspect迁移网络来实现针对目标域中的冷启动用户的推荐。

## 3 CATN框架

我们从针对冷启动用户的跨域推荐问题设定开始。然后，我们概述了CATN以及其每个组成部分背后的动机。介绍完所有组件后，我们将进行优化过程。

### 3.1 问题提出

我们使用 <img src="https://www.zhihu.com/equation?tex=\mathcal D_s" alt="\mathcal D_s" class="ee_img tr_noresize" eeimg="1">  和  <img src="https://www.zhihu.com/equation?tex=\mathcal D_t" alt="\mathcal D_t" class="ee_img tr_noresize" eeimg="1">  分别代表源领域和目标领域。请注意，一个域包括其用户、物品以及用户和物品之间的交互（例如评分和评论）。
令 <img src="https://www.zhihu.com/equation?tex=U^o" alt="U^o" class="ee_img tr_noresize" eeimg="1"> 为重叠用户的集合，这些用户与 <img src="https://www.zhihu.com/equation?tex=\mathcal D_s" alt="\mathcal D_s" class="ee_img tr_noresize" eeimg="1"> 和t <img src="https://www.zhihu.com/equation?tex=\mathcal D_t" alt="\mathcal D_t" class="ee_img tr_noresize" eeimg="1"> 中的物品都有历史交互。  <img src="https://www.zhihu.com/equation?tex=U^{cs}" alt="U^{cs}" class="ee_img tr_noresize" eeimg="1"> 表示冷启动用户的集合，他们与 <img src="https://www.zhihu.com/equation?tex=\mathcal D_s" alt="\mathcal D_s" class="ee_img tr_noresize" eeimg="1"> 中的物品有交互但与 <img src="https://www.zhihu.com/equation?tex=\mathcal D_t" alt="\mathcal D_t" class="ee_img tr_noresize" eeimg="1"> 中的物品没有交互。
对于给定的冷启动用户 <img src="https://www.zhihu.com/equation?tex=u∈U^{cs}" alt="u∈U^{cs}" class="ee_img tr_noresize" eeimg="1"> ，我们的任务是估算用户 <img src="https://www.zhihu.com/equation?tex=u" alt="u" class="ee_img tr_noresize" eeimg="1"> 将给 <img src="https://www.zhihu.com/equation?tex=\mathcal D_t" alt="\mathcal D_t" class="ee_img tr_noresize" eeimg="1">  中的物品 <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1"> 打的评分 <img src="https://www.zhihu.com/equation?tex=\hat r_{u,i}" alt="\hat r_{u,i}" class="ee_img tr_noresize" eeimg="1"> 。

### 3.2 CATN总览

- CATN的总体结构如图3所示。它的结构由三个部分组成：aspect提取，辅助评论增强和跨域aspect相关性学习。因为我们的任务是实现基于评论的跨域偏好迁移，所以评分预测的总体流程与单域中现有的基于评论的推荐系统有根本区别。此处，来自 <img src="https://www.zhihu.com/equation?tex=\mathcal D_s" alt="\mathcal D_s" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=\mathcal D_t" alt="\mathcal D_t" class="ee_img tr_noresize" eeimg="1"> 的重叠用户 <img src="https://www.zhihu.com/equation?tex=U^o" alt="U^o" class="ee_img tr_noresize" eeimg="1"> 的评分和评论被用于模型训练。

![figure3](https://raw.githubusercontent.com/wales-z/Markdown4Zhihu/master/Data/catn_note_for_zhihu/figure3.png)

<center>图3 CATN的架构（以彩色显示最佳）</center>

- 在源域，我们用用户文档 <img src="https://www.zhihu.com/equation?tex=D_u" alt="D_u" class="ee_img tr_noresize" eeimg="1"> 表示用户，用物品文档 <img src="https://www.zhihu.com/equation?tex=D_i" alt="D_i" class="ee_img tr_noresize" eeimg="1"> 表示物品。类似地，目标域中的每个用户和每个物品分别具有一个用户文档和一个物品文档。重叠的用户将有两个用户文档，一个来自源域 <img src="https://www.zhihu.com/equation?tex=D^s_u" alt="D^s_u" class="ee_img tr_noresize" eeimg="1"> ，另一个来自目标域 <img src="https://www.zhihu.com/equation?tex=D^t_u" alt="D^t_u" class="ee_img tr_noresize" eeimg="1"> 。我们使用上标 <img src="https://www.zhihu.com/equation?tex=“ s”" alt="“ s”" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=“ t”" alt="“ t”" class="ee_img tr_noresize" eeimg="1"> 来表示源域和目标域，以进行清晰的演示。
  回想一下，重叠的用户与源域和目标域中的物品都有交互。对于给定的一个重叠用户 <img src="https://www.zhihu.com/equation?tex=u∈U^o" alt="u∈U^o" class="ee_img tr_noresize" eeimg="1"> ，如图2所示，我们设计了一种基于跨域评论的偏好匹配过程，该过程具有两个学习流程：
  1. 用他在源域中的用户文档 <img src="https://www.zhihu.com/equation?tex=D^s_u" alt="D^s_u" class="ee_img tr_noresize" eeimg="1"> 和在目标域中的物品 <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1"> 的物品文档 <img src="https://www.zhihu.com/equation?tex=D^t_i" alt="D^t_i" class="ee_img tr_noresize" eeimg="1"> 进行模型训练，来匹配在目标域 <img src="https://www.zhihu.com/equation?tex=u" alt="u" class="ee_img tr_noresize" eeimg="1"> 对 <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1"> 的评分 <img src="https://www.zhihu.com/equation?tex=r^t_{u,i}" alt="r^t_{u,i}" class="ee_img tr_noresize" eeimg="1"> 。
  2. 用他在目标域中的用户文档 <img src="https://www.zhihu.com/equation?tex=D^t_u" alt="D^t_u" class="ee_img tr_noresize" eeimg="1"> 和在源域中的物品 <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1"> 的物品文档 <img src="https://www.zhihu.com/equation?tex=D^s_i" alt="D^s_i" class="ee_img tr_noresize" eeimg="1"> ，来匹配源域中的 <img src="https://www.zhihu.com/equation?tex=u" alt="u" class="ee_img tr_noresize" eeimg="1"> 对 <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1"> 的评分 <img src="https://www.zhihu.com/equation?tex=r^s_{u,i}" alt="r^s_{u,i}" class="ee_img tr_noresize" eeimg="1"> 。

- 源域和目标域之间的用户偏好匹配是在aspect级别实现的，aspect从用户和物品文档中获取，如图3所示。请注意，除了用户文档，我们还为每个用户使用了辅助评论文档。该辅助评论文档包含志趣相投的用户写的评论，稍后将详细介绍。从两种用户文档获取的aspect将合并。然后，跨域aspect相关性学习将区分两个域间相关性更高的aspect对，以进行评分预测。接下来，我们详细介绍aspect提取过程。

### 3.3 aspect提取

为了提取aspect，我们将相同的过程应用于源域和目标域中的用户文档 <img src="https://www.zhihu.com/equation?tex=D_u" alt="D_u" class="ee_img tr_noresize" eeimg="1"> 和项目文档 <img src="https://www.zhihu.com/equation?tex=D_i" alt="D_i" class="ee_img tr_noresize" eeimg="1"> 。因为过程相同，我们用 <img src="https://www.zhihu.com/equation?tex=D_u" alt="D_u" class="ee_img tr_noresize" eeimg="1"> 来举例。

#### 文本卷积（Text Convolution）

给定用户文档 <img src="https://www.zhihu.com/equation?tex={D}_u = [{\rm w}_1，{\rm w}_2，..，{\rm w}_l]" alt="{D}_u = [{\rm w}_1，{\rm w}_2，..，{\rm w}_l]" class="ee_img tr_noresize" eeimg="1"> ，我们首先将每个单词投影到其embedding表示： <img src="https://www.zhihu.com/equation?tex={\rm E}_u = [{\rm e}_1，{\rm e}_2，..，{\rm e}_l]，{\rm e}_j∈\mathbb R^d" alt="{\rm E}_u = [{\rm e}_1，{\rm e}_2，..，{\rm e}_l]，{\rm e}_j∈\mathbb R^d" class="ee_img tr_noresize" eeimg="1"> ，其中 <img src="https://www.zhihu.com/equation?tex=l" alt="l" class="ee_img tr_noresize" eeimg="1"> 是文档长度， <img src="https://www.zhihu.com/equation?tex=d" alt="d" class="ee_img tr_noresize" eeimg="1"> 是词的embedding维度。为了捕获每个单词周围的上下文信息，我们执行卷积运算，配合 <img src="https://www.zhihu.com/equation?tex=ReLU" alt="ReLU" class="ee_img tr_noresize" eeimg="1"> 激活函数。我们在矩阵 <img src="https://www.zhihu.com/equation?tex={\rm E}_u" alt="{\rm E}_u" class="ee_img tr_noresize" eeimg="1"> 上应用 <img src="https://www.zhihu.com/equation?tex=n" alt="n" class="ee_img tr_noresize" eeimg="1"> 个卷积滤波器（都用相同的大小为 <img src="https://www.zhihu.com/equation?tex=s" alt="s" class="ee_img tr_noresize" eeimg="1"> 的滑动窗口），以提取上下文特征。结果特征矩阵（resultant feature matrix）为 <img src="https://www.zhihu.com/equation?tex={\rm C}_u = [{\rm c}_{1,u}，{\rm c}_{2,u}，.. {\rm c}_{l,u}]" alt="{\rm C}_u = [{\rm c}_{1,u}，{\rm c}_{2,u}，.. {\rm c}_{l,u}]" class="ee_img tr_noresize" eeimg="1"> ，其中 <img src="https://www.zhihu.com/equation?tex={\rm c}_{j,u}∈\mathbb R^n" alt="{\rm c}_{j,u}∈\mathbb R^n" class="ee_img tr_noresize" eeimg="1"> 是第 <img src="https://www.zhihu.com/equation?tex=j" alt="j" class="ee_img tr_noresize" eeimg="1"> 个单词的潜在上下文（latent contextual）特征向量。

#### Aspect门控（Aspect Gate Control）

为第 <img src="https://www.zhihu.com/equation?tex=j" alt="j" class="ee_img tr_noresize" eeimg="1"> 个单词提取的上下文特征可 <img src="https://www.zhihu.com/equation?tex={\rm c}_{j,u}" alt="{\rm c}_{j,u}" class="ee_img tr_noresize" eeimg="1"> 以看作是多个语义aspect的组合。这里，我们进一步利用aspect-specific的门控机制来识别出与每个aspect相关的有哪些特征。具体来说，对第 <img src="https://www.zhihu.com/equation?tex=m" alt="m" class="ee_img tr_noresize" eeimg="1"> 个aspect，单词 <img src="https://www.zhihu.com/equation?tex={\rm w}_j" alt="{\rm w}_j" class="ee_img tr_noresize" eeimg="1"> 的aspect-specific特征 <img src="https://www.zhihu.com/equation?tex={\rm g}_{m,j,u}" alt="{\rm g}_{m,j,u}" class="ee_img tr_noresize" eeimg="1"> 的提取如下：

<img src="https://www.zhihu.com/equation?tex={\rm g}_{m,j,u}=({\rm W}_m {\rm c}_{j,u}+{\rm b}_m)⊙\sigma({\rm W}^g_m {\rm c}_{j,u}+{\rm b}^g_m)\tag1
" alt="{\rm g}_{m,j,u}=({\rm W}_m {\rm c}_{j,u}+{\rm b}_m)⊙\sigma({\rm W}^g_m {\rm c}_{j,u}+{\rm b}^g_m)\tag1
" class="ee_img tr_noresize" eeimg="1">
其中其中 <img src="https://www.zhihu.com/equation?tex=\sigma" alt="\sigma" class="ee_img tr_noresize" eeimg="1"> 是sigmoid型激活函数， <img src="https://www.zhihu.com/equation?tex=⊙" alt="⊙" class="ee_img tr_noresize" eeimg="1"> 是按元素相乘运算。 <img src="https://www.zhihu.com/equation?tex={\rm W}_m" alt="{\rm W}_m" class="ee_img tr_noresize" eeimg="1"> ， <img src="https://www.zhihu.com/equation?tex={\rm W}^g_m∈\mathbb R^{k×n}，{\rm b}_m，{\rm b}^g_m∈\mathbb R^k" alt="{\rm W}^g_m∈\mathbb R^{k×n}，{\rm b}_m，{\rm b}^g_m∈\mathbb R^k" class="ee_img tr_noresize" eeimg="1"> 分别表示第 <img src="https://www.zhihu.com/equation?tex=m" alt="m" class="ee_img tr_noresize" eeimg="1"> 个aspect的变换矩阵和偏置向量。  <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1"> 是aspect表示的潜在维度（latent dim）。这里，等式1右侧的第二项的作用是软开关，它控制哪个潜在特征与该aspect有关。作为结果，我们获得了 <img src="https://www.zhihu.com/equation?tex=M" alt="M" class="ee_img tr_noresize" eeimg="1"> 个aspect-specific的词上下文特征 <img src="https://www.zhihu.com/equation?tex=\rm G_u" alt="\rm G_u" class="ee_img tr_noresize" eeimg="1"> ，这些特征被用于进一步的方面提取。

<img src="https://www.zhihu.com/equation?tex={\rm G_u}=[{\rm G}_{1,u},{\rm G}_{2,u},...,{\rm G}_{M,u}]\tag2
" alt="{\rm G_u}=[{\rm G}_{1,u},{\rm G}_{2,u},...,{\rm G}_{M,u}]\tag2
" class="ee_img tr_noresize" eeimg="1">


<img src="https://www.zhihu.com/equation?tex={\rm G}_{m,u}=[{\rm g}_{m,1,u},{\rm g}_{m,2,u},...,{\rm g}_{m,l,u}]\tag3
" alt="{\rm G}_{m,u}=[{\rm g}_{m,1,u},{\rm g}_{m,2,u},...,{\rm g}_{m,l,u}]\tag3
" class="ee_img tr_noresize" eeimg="1">

#### Aspect注意力（Aspect Attention）

来自不同领域的评论对侧重于（put emphasis on）不同的aspect。例如，“书”域倾向于包含情节和人物，而“电影”域倾向于包含演员和特效。因此，我们在 <img src="https://www.zhihu.com/equation?tex=\mathcal D_s" alt="\mathcal D_s" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=\mathcal D_t" alt="\mathcal D_t" class="ee_img tr_noresize" eeimg="1"> 中设计了两个全局共享的aspect表示的矩阵。对于源域和目标域，它们分别表示为 <img src="https://www.zhihu.com/equation?tex={\rm V}_s = [{\rm v}_{1s}，...，{\rm v}_{M,s}]" alt="{\rm V}_s = [{\rm v}_{1s}，...，{\rm v}_{M,s}]" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex={\rm V}_t = [{\rm v}_{1,t}，...，{\rm v}_{M,t}]" alt="{\rm V}_t = [{\rm v}_{1,t}，...，{\rm v}_{M,t}]" class="ee_img tr_noresize" eeimg="1"> 。  <img src="https://www.zhihu.com/equation?tex={\rm V}_s" alt="{\rm V}_s" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex={\rm V}_t" alt="{\rm V}_t" class="ee_img tr_noresize" eeimg="1"> 用作指导方面提取的查询（query）。具体地，从 <img src="https://www.zhihu.com/equation?tex={\rm G}_{m,u}" alt="{\rm G}_{m,u}" class="ee_img tr_noresize" eeimg="1"> 提取的第m个方面的表示 <img src="https://www.zhihu.com/equation?tex={\rm a}_{m,u}" alt="{\rm a}_{m,u}" class="ee_img tr_noresize" eeimg="1"> 的推导如下：

<img src="https://www.zhihu.com/equation?tex={\rm a}_{m,u}=\sum^l_{j=1}\beta_{m,j,u} {\rm g}_{m,j,u}\tag4
" alt="{\rm a}_{m,u}=\sum^l_{j=1}\beta_{m,j,u} {\rm g}_{m,j,u}\tag4
" class="ee_img tr_noresize" eeimg="1">


<img src="https://www.zhihu.com/equation?tex=\beta_{m,j,u}=\frac{exp({\rm g}^\top_{m,j,u}{\rm v}_{m,s})}
{\sum^l_{i=1}exp({\rm g}^\top_{m,i,u}{\rm v}_{m,s})}\tag5
" alt="\beta_{m,j,u}=\frac{exp({\rm g}^\top_{m,j,u}{\rm v}_{m,s})}
{\sum^l_{i=1}exp({\rm g}^\top_{m,i,u}{\rm v}_{m,s})}\tag5
" class="ee_img tr_noresize" eeimg="1">

在此， <img src="https://www.zhihu.com/equation?tex=β_{m,j,u}" alt="β_{m,j,u}" class="ee_img tr_noresize" eeimg="1"> 表示单词 <img src="https://www.zhihu.com/equation?tex={\rm w}_j" alt="{\rm w}_j" class="ee_img tr_noresize" eeimg="1"> 对第 <img src="https://www.zhihu.com/equation?tex=m" alt="m" class="ee_img tr_noresize" eeimg="1"> 个aspect的重要性。作为结果，我们可以从 <img src="https://www.zhihu.com/equation?tex=D_u" alt="D_u" class="ee_img tr_noresize" eeimg="1"> 中获得 <img src="https://www.zhihu.com/equation?tex=M" alt="M" class="ee_img tr_noresize" eeimg="1"> 个aspect的表示，构成aspect矩阵 <img src="https://www.zhihu.com/equation?tex={\rm A}u = [{\rm a}_{1,u},..{\rm a}_{M,u}]" alt="{\rm A}u = [{\rm a}_{1,u},..{\rm a}_{M,u}]" class="ee_img tr_noresize" eeimg="1"> 。按照相同的步骤，我们从 <img src="https://www.zhihu.com/equation?tex=D_i" alt="D_i" class="ee_img tr_noresize" eeimg="1"> 中提取 <img src="https://www.zhihu.com/equation?tex=M" alt="M" class="ee_img tr_noresize" eeimg="1"> 个aspect： <img src="https://www.zhihu.com/equation?tex=A_i = [{\rm a}_{1,i},..,{\rm a}_{M,i}]" alt="A_i = [{\rm a}_{1,i},..,{\rm a}_{M,i}]" class="ee_img tr_noresize" eeimg="1"> 。值得强调的是，尽管 <img src="https://www.zhihu.com/equation?tex=D_u" alt="D_u" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=D_i" alt="D_i" class="ee_img tr_noresize" eeimg="1"> 是使用不同领域的评论构建的，但 <img src="https://www.zhihu.com/equation?tex=D_u" alt="D_u" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=D_i" alt="D_i" class="ee_img tr_noresize" eeimg="1"> 的aspect提取参数在每个学习流程中都是共享的。而且在每个学习流程中也使用一组独立的参数。由于我们的目标是跨域映射aspect，因此 <img src="https://www.zhihu.com/equation?tex={\rm V}_s" alt="{\rm V}_s" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex={\rm V}_t" alt="{\rm V}_t" class="ee_img tr_noresize" eeimg="1"> 分别在其对应的域中共享。

### 3.4 辅助评论增强

- 请注意，跨域重叠用户的比例通常很小[17]。数据稀疏性问题由于缺少评论而进一步加剧，因为用户文档包含不完整和简短的评论[31]。

- 为了克服这些限制，我们选择充分利用相似的非重叠用户的交互。我们按照[31]的方法从志趣相投的用户中提取辅助评论。具体来说，对于给定的用户-物品对，辅助评论是由另一位用户写的一条评论（对该物品），且这个用户与目标用户对该物品的评分相同。对于用户 <img src="https://www.zhihu.com/equation?tex=u" alt="u" class="ee_img tr_noresize" eeimg="1"> ，他的辅助文档 <img src="https://www.zhihu.com/equation?tex=D_{u_{aux}}" alt="D_{u_{aux}}" class="ee_img tr_noresize" eeimg="1"> 是通过合并用户 <img src="https://www.zhihu.com/equation?tex=u" alt="u" class="ee_img tr_noresize" eeimg="1"> 在相同域中购买的历史物品的辅助评论而形成的。请注意，我们**仅考虑非重叠用户的辅助评论**，这可以增加训练数据的多样性。通过这种数据增强策略，即使重叠的用户很少，我们的模型仍可以优化到一个良好的状态。
- 利用辅助文档的一种很自然的方法是按照相同的aspect提取过程，然后将其与 <img src="https://www.zhihu.com/equation?tex={\rm A}_u" alt="{\rm A}_u" class="ee_img tr_noresize" eeimg="1"> 简单合并。
  但是，这种解决方案忽略了以下事实：辅助文档是由具有不同语言风格、与目标用户具有不同偏好重点的不同用户组成的，因此可能导致特征不兼容。在[32]中有报告，将CNN网络堆叠在上下文矩阵的顶部对于评分预测非常有效，尤其是当文档中的语义不连贯时。因此，在前面的aspect提取过程中使用的文本卷积的顶层，我们在处理辅助文档时添加了另一个卷积层，如图3所示。


<img src="https://www.zhihu.com/equation?tex=c^i_{h,u_{aux}}=RELU({\rm W}^i _{aux}*{\rm H}_{u_{aux}}[h-\frac{s-1}{2}:h+\frac{s-1}{2}]+b^i_{aux})\tag6
" alt="c^i_{h,u_{aux}}=RELU({\rm W}^i _{aux}*{\rm H}_{u_{aux}}[h-\frac{s-1}{2}:h+\frac{s-1}{2}]+b^i_{aux})\tag6
" class="ee_img tr_noresize" eeimg="1">

其中 <img src="https://www.zhihu.com/equation?tex=*" alt="*" class="ee_img tr_noresize" eeimg="1"> 表示卷积操作， <img src="https://www.zhihu.com/equation?tex={\rm W}^i _{aux} \in \mathbb R^{s\times n}" alt="{\rm W}^i _{aux} \in \mathbb R^{s\times n}" class="ee_img tr_noresize" eeimg="1"> 时卷积的权重矩阵， <img src="https://www.zhihu.com/equation?tex=b^i_{aux}" alt="b^i_{aux}" class="ee_img tr_noresize" eeimg="1"> 是偏置项， <img src="https://www.zhihu.com/equation?tex={\rm H}_{u_{aux}}" alt="{\rm H}_{u_{aux}}" class="ee_img tr_noresize" eeimg="1"> 是用3.3节中的文本卷积提取出的特征矩阵。类似地，我们生成抽象特征矩阵 <img src="https://www.zhihu.com/equation?tex={\rm C}_{u_{aux}}=	[{\rm c}_{1,u_{aux}},{\rm c}_{2,u_{aux}},...,{\rm c}_{l,u_{aux}}]" alt="{\rm C}_{u_{aux}}=	[{\rm c}_{1,u_{aux}},{\rm c}_{2,u_{aux}},...,{\rm c}_{l,u_{aux}}]" class="ee_img tr_noresize" eeimg="1"> ，其中 <img src="https://www.zhihu.com/equation?tex={\rm c}_{j,u_{aux}} \in \mathbb R^n" alt="{\rm c}_{j,u_{aux}} \in \mathbb R^n" class="ee_img tr_noresize" eeimg="1"> 。然后进行同样的aspect门控和aspect注意力处理过程以从 <img src="https://www.zhihu.com/equation?tex=D_{u_{aux}}" alt="D_{u_{aux}}" class="ee_img tr_noresize" eeimg="1"> 获得 <img src="https://www.zhihu.com/equation?tex={\rm A}_{u_{aux}}" alt="{\rm A}_{u_{aux}}" class="ee_img tr_noresize" eeimg="1"> 矩阵。为了有效地用 <img src="https://www.zhihu.com/equation?tex={\rm A}_{u_{aux}}" alt="{\rm A}_{u_{aux}}" class="ee_img tr_noresize" eeimg="1"> 更新 <img src="https://www.zhihu.com/equation?tex={\rm A}_{u}" alt="{\rm A}_{u}" class="ee_img tr_noresize" eeimg="1"> ，我们采用了基于对应aspect的逐元素交互的门机制：

<img src="https://www.zhihu.com/equation?tex={\rm g}_{axu}=\sigma ({\rm W}^1_f [({\rm A}_{u}-{\rm A}_{u_{aux}}) \oplus
({\rm A}_{u}\odot{\rm A}_{u_{aux}})]+{\rm b}^1_f)\tag7
" alt="{\rm g}_{axu}=\sigma ({\rm W}^1_f [({\rm A}_{u}-{\rm A}_{u_{aux}}) \oplus
({\rm A}_{u}\odot{\rm A}_{u_{aux}})]+{\rm b}^1_f)\tag7
" class="ee_img tr_noresize" eeimg="1">


<img src="https://www.zhihu.com/equation?tex={\rm A}_u=tanh({\rm W}^2_f [{\rm A}_u \oplus ({\rm g}_{aux} \odot {\rm A}_{u_{aux}})]+b^2_f)\tag8
" alt="{\rm A}_u=tanh({\rm W}^2_f [{\rm A}_u \oplus ({\rm g}_{aux} \odot {\rm A}_{u_{aux}})]+b^2_f)\tag8
" class="ee_img tr_noresize" eeimg="1">

其中， <img src="https://www.zhihu.com/equation?tex=\oplus" alt="\oplus" class="ee_img tr_noresize" eeimg="1"> 是连接（concatenation）操作， <img src="https://www.zhihu.com/equation?tex={\rm W}^1_f, {\rm W}^2_f \in \mathbb R^{k\times2k}" alt="{\rm W}^1_f, {\rm W}^2_f \in \mathbb R^{k\times2k}" class="ee_img tr_noresize" eeimg="1"> 是转换（transform）矩阵， <img src="https://www.zhihu.com/equation?tex={\rm b}^1_f,{\rm b}^2_f \in \mathbb R^k" alt="{\rm b}^1_f,{\rm b}^2_f \in \mathbb R^k" class="ee_img tr_noresize" eeimg="1"> 是偏置向量。这样就更新了aspect表示矩阵 <img src="https://www.zhihu.com/equation?tex={\rm A_u}" alt="{\rm A_u}" class="ee_img tr_noresize" eeimg="1"> 以更好地描述用户 <img src="https://www.zhihu.com/equation?tex=u" alt="u" class="ee_img tr_noresize" eeimg="1"> 。

### 3.5 跨域Aspect相关性（correlation）学习

- 现在，对于用户 <img src="https://www.zhihu.com/equation?tex=u" alt="u" class="ee_img tr_noresize" eeimg="1"> 和项目 <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1"> ，我们分别有抽象aspect特征 <img src="https://www.zhihu.com/equation?tex={\rm A}_u" alt="{\rm A}_u" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex={\rm A}_i" alt="{\rm A}_i" class="ee_img tr_noresize" eeimg="1"> 。直观上，评级预测可以分别是 <img src="https://www.zhihu.com/equation?tex={\rm A}_u" alt="{\rm A}_u" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex={\rm A}_i" alt="{\rm A}_i" class="ee_img tr_noresize" eeimg="1"> 中两个aspect之间语义匹配的聚合（aggregation）。但是，对于特定的用户-物品对，匹配分数仅反映两个aspect之间的语义相关性。由于并非所有aspect对都同样重要，因此识别全局跨域aspect相关性能带来一定好处。然后，我们可以在域级别突出显示重要的aspect对，以便更好地进行评级预测。为此，我们设计了一种简单但有效的跨域偏好匹配方法。回想一下，我们利用一组全局aspect表示 <img src="https://www.zhihu.com/equation?tex={\rm V}_s" alt="{\rm V}_s" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex={\rm V}t" alt="{\rm V}t" class="ee_img tr_noresize" eeimg="1"> 来指导aspect提取。在这里，我们利用这些静态aspect表示来计算全局跨域aspect相关矩阵 <img src="https://www.zhihu.com/equation?tex=\rm S" alt="\rm S" class="ee_img tr_noresize" eeimg="1"> ，如下所示：


<img src="https://www.zhihu.com/equation?tex={\rm S}=LeakyReLU({\rm V}^\top_s {\rm W} {\rm V}_t)\tag9
" alt="{\rm S}=LeakyReLU({\rm V}^\top_s {\rm W} {\rm V}_t)\tag9
" class="ee_img tr_noresize" eeimg="1">

其中 <img src="https://www.zhihu.com/equation?tex=S(p,q)" alt="S(p,q)" class="ee_img tr_noresize" eeimg="1"> 反映了基于源域中的aspect  <img src="https://www.zhihu.com/equation?tex=p" alt="p" class="ee_img tr_noresize" eeimg="1"> 和目标域中的aspect  <img src="https://www.zhihu.com/equation?tex=q" alt="q" class="ee_img tr_noresize" eeimg="1"> 的偏好迁移的重要性。  <img src="https://www.zhihu.com/equation?tex={\rm S}∈\mathbb R^{M×M}，{\rm W}∈\mathbb R^{k×k}" alt="{\rm S}∈\mathbb R^{M×M}，{\rm W}∈\mathbb R^{k×k}" class="ee_img tr_noresize" eeimg="1"> 是可学习的用于**亲和力投影（affinity projection）**的矩阵。通过将相应的 <img src="https://www.zhihu.com/equation?tex=\alpha" alt="\alpha" class="ee_img tr_noresize" eeimg="1"> 设置为非常小的值（例如0.01），可以采用 <img src="https://www.zhihu.com/equation?tex=LeakyReLU" alt="LeakyReLU" class="ee_img tr_noresize" eeimg="1"> 激活函数来支持跨域的稀疏aspect相关性。

- 然后，我们计算 <img src="https://www.zhihu.com/equation?tex={\rm A}_u" alt="{\rm A}_u" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex={\rm A}_i" alt="{\rm A}_i" class="ee_img tr_noresize" eeimg="1"> 之间每个aspect对之间的语义匹配，如下所示：


<img src="https://www.zhihu.com/equation?tex={\rm S}_{u,i}={\rm A}^\top_u {\rm W} {\rm A}_i\tag{10}
" alt="{\rm S}_{u,i}={\rm A}^\top_u {\rm W} {\rm A}_i\tag{10}
" class="ee_img tr_noresize" eeimg="1">

与公式9相似， <img src="https://www.zhihu.com/equation?tex={\rm S}_{u,i}(p,q)" alt="{\rm S}_{u,i}(p,q)" class="ee_img tr_noresize" eeimg="1"> 反映了对应aspect之间的匹配程度；  <img src="https://www.zhihu.com/equation?tex=\rm W" alt="\rm W" class="ee_img tr_noresize" eeimg="1"> 被共享用于亲和力投影。最后，我们使用 <img src="https://www.zhihu.com/equation?tex=\rm S" alt="\rm S" class="ee_img tr_noresize" eeimg="1"> 作为注意力权重，来汇总逐对的aspect匹配作为最终的评分预测。

<img src="https://www.zhihu.com/equation?tex={\rm S}^r_{u,i}={\rm S} \odot {\rm S}_{u,i}\tag{11}
" alt="{\rm S}^r_{u,i}={\rm S} \odot {\rm S}_{u,i}\tag{11}
" class="ee_img tr_noresize" eeimg="1">


<img src="https://www.zhihu.com/equation?tex=\hat r_{u,i}=\frac{1}{M*N} \sum^M_{p=1} \sum^M_{q=1} {\rm S}^r_{u,i}(p,q)+b_u+b_i \tag{12}
" alt="\hat r_{u,i}=\frac{1}{M*N} \sum^M_{p=1} \sum^M_{q=1} {\rm S}^r_{u,i}(p,q)+b_u+b_i \tag{12}
" class="ee_img tr_noresize" eeimg="1">

其中， <img src="https://www.zhihu.com/equation?tex=b_u" alt="b_u" class="ee_img tr_noresize" eeimg="1"> ， <img src="https://www.zhihu.com/equation?tex=b_i" alt="b_i" class="ee_img tr_noresize" eeimg="1"> 分别是用户偏置和物品偏置。

### 3.6 优化策略

对于模型训练，我们利用源域和目标域中重叠用户的交互进行参数优化。令 <img src="https://www.zhihu.com/equation?tex=\mathcal O_s" alt="\mathcal O_s" class="ee_img tr_noresize" eeimg="1"> 或 <img src="https://www.zhihu.com/equation?tex=\mathcal O_t" alt="\mathcal O_t" class="ee_img tr_noresize" eeimg="1"> 分别是在 <img src="https://www.zhihu.com/equation?tex=\mathcal D_s" alt="\mathcal D_s" class="ee_img tr_noresize" eeimg="1"> 或 <img src="https://www.zhihu.com/equation?tex=\mathcal D_t" alt="\mathcal D_t" class="ee_img tr_noresize" eeimg="1"> 中观察到的一批（a batch of）用户物品评分对，仅限于 <img src="https://www.zhihu.com/equation?tex=U^o" alt="U^o" class="ee_img tr_noresize" eeimg="1"> 。  <img src="https://www.zhihu.com/equation?tex=\mathcal L_s" alt="\mathcal L_s" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=\mathcal L_s" alt="\mathcal L_s" class="ee_img tr_noresize" eeimg="1"> 的损失函数可以定义如下：

<img src="https://www.zhihu.com/equation?tex=\mathcal L_*=\frac{1}{|\mathcal O_*|} \sum_{(u,i)\in {\mathcal O_*}}
(r_{u,i}-\hat r_{u,i})^2+\lambda ||\Theta_*||\tag{13}
" alt="\mathcal L_*=\frac{1}{|\mathcal O_*|} \sum_{(u,i)\in {\mathcal O_*}}
(r_{u,i}-\hat r_{u,i})^2+\lambda ||\Theta_*||\tag{13}
" class="ee_img tr_noresize" eeimg="1">
其中符号 <img src="https://www.zhihu.com/equation?tex=*" alt="*" class="ee_img tr_noresize" eeimg="1"> 表示 <img src="https://www.zhihu.com/equation?tex=s" alt="s" class="ee_img tr_noresize" eeimg="1"> 或 <img src="https://www.zhihu.com/equation?tex=t" alt="t" class="ee_img tr_noresize" eeimg="1"> ， <img src="https://www.zhihu.com/equation?tex=\lambda" alt="\lambda" class="ee_img tr_noresize" eeimg="1"> 是正则化系数， <img src="https://www.zhihu.com/equation?tex=\Theta_∗" alt="\Theta_∗" class="ee_img tr_noresize" eeimg="1"> 是可训练的参数。每次一个batch，轮流执行两个学习流程（即，通过使用源域中的 <img src="https://www.zhihu.com/equation?tex=D_u" alt="D_u" class="ee_img tr_noresize" eeimg="1"> 和目标域中的 <img src="https://www.zhihu.com/equation?tex=D_i" alt="D_i" class="ee_img tr_noresize" eeimg="1"> 来预测 <img src="https://www.zhihu.com/equation?tex=r_{u,i}" alt="r_{u,i}" class="ee_img tr_noresize" eeimg="1"> 在目标域中，然后通过使用目标域中的 <img src="https://www.zhihu.com/equation?tex=D_u" alt="D_u" class="ee_img tr_noresize" eeimg="1"> 和源域中的 <img src="https://www.zhihu.com/equation?tex=D_i" alt="D_i" class="ee_img tr_noresize" eeimg="1"> 来在源域中预测 <img src="https://www.zhihu.com/equation?tex=r_{u,i}" alt="r_{u,i}" class="ee_img tr_noresize" eeimg="1"> ）。每个训练batch都由经过乱序的 <img src="https://www.zhihu.com/equation?tex=\mathcal O_s" alt="\mathcal O_s" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=\mathcal O_t" alt="\mathcal O_t" class="ee_img tr_noresize" eeimg="1"> 按固定比例组成。w.r.t.  <img src="https://www.zhihu.com/equation?tex=|\mathcal O_s|/|\mathcal O_t|=|R_s|/|R_t|" alt="|\mathcal O_s|/|\mathcal O_t|=|R_s|/|R_t|" class="ee_img tr_noresize" eeimg="1"> ，其中 <img src="https://www.zhihu.com/equation?tex=|R_s|" alt="|R_s|" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=|R_t|" alt="|R_t|" class="ee_img tr_noresize" eeimg="1"> 分别表示 <img src="https://www.zhihu.com/equation?tex=U^o" alt="U^o" class="ee_img tr_noresize" eeimg="1"> 在 <img src="https://www.zhihu.com/equation?tex=\mathcal D_s" alt="\mathcal D_s" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=\mathcal D_t" alt="\mathcal D_t" class="ee_img tr_noresize" eeimg="1"> 中进行的评分数量。我们采用Adam作为优化器来更新参数。

## 4 实验

### 4.1 数据集

- 为了与SOTA的基准相比较评估我们的模型，我们在Amazon Review数据集[11]上进行了实验。在最大的类别中，我们选择三个相关的类别作为三个域，分别是书，电影（在亚马逊称为“电影和电视”）和音乐（在亚马逊称为“ CD和唱片（vinyl）”）。在每个域中，我们删除没有评论文本的交互记录，然后根据早期研究[14，17]过滤掉交互少于10次的用户和交互少于30次的物品。表1报告了每个域的详细统计信息。

<center>表1 Amazon上3个数据集的统计信息</center>

<img src="https://raw.githubusercontent.com/wales-z/Markdown4Zhihu/master/Data/catn_note_for_zhihu/table1.png" alt="table1" style="zoom: 67%;" />

- 由于这三个域彼此相关，因此我们构建了三个成对的跨域方案。在每种情况下，我们选择具有更多用户的域作为 <img src="https://www.zhihu.com/equation?tex=\mathcal D_s" alt="\mathcal D_s" class="ee_img tr_noresize" eeimg="1"> ，另一个作为 <img src="https://www.zhihu.com/equation?tex=\mathcal D_t" alt="\mathcal D_t" class="ee_img tr_noresize" eeimg="1"> 。按照[17]中的设置，我们随机抽取50％的重叠用户为冷启动用户，即，他们在 <img src="https://www.zhihu.com/equation?tex=\mathcal D_t" alt="\mathcal D_t" class="ee_img tr_noresize" eeimg="1"> 中的交互无法被模型看到，但用于验证和测试（具体来说，设定30％用于测试，20％用于验证）。剩下的50％的重叠用户用于训练。为了模拟不同比例的重叠用户，我们从剩余的50％重叠用户随机抽取的一定比例 <img src="https://www.zhihu.com/equation?tex=η∈\{100％,50％,20％,10％,5％\}" alt="η∈\{100％,50％,20％,10％,5％\}" class="ee_img tr_noresize" eeimg="1"> 来构建训练集。表2报告了每个跨域方案的详细统计信息。

<center>表2 三种跨域推荐场景的统计信息。 
    η表示训练集中包含的重叠用户的比率。</center>

![table2](https://raw.githubusercontent.com/wales-z/Markdown4Zhihu/master/Data/catn_note_for_zhihu/table2.png)

### 4.2 基准方法 (Baseline Methods)

我们与以下基准进行比较，包括传统的和最近的SOTA。

- **CMF [25]**是一种简单且众所周知的跨域推荐方法，通过共享用户因子（factors）并分解跨域的联合评级矩阵（joint rating matrix）进行跨域推荐。
- **EMCDR [22]**率先提出了三步优化范例，通过在两个域中轮流（successively）训练矩阵分解，然后利用多层感知器来映射用户潜在因子。
- **CDLFM [29]**通过融合三种用户相似度作为正则项（基于其评分行为）来改进矩阵分解。
  通过考虑相似的用户和基于梯度提升树（gradient boosting trees, GBT）的集成学习方法，他们使用基于邻域的映射方法用于替换以前的多层感知机。
- **DFM [9]**是RC-DFM [9]的简单版本。它利用aSDAE [6]的工作从评分矩阵生成用户表示，并使用多层感知器进行映射。
- **R-DFM [9]**是RC-DFM [9] 的另一个变体。它通过扩展的aSDAE将评分记录和评论结合在一起，以增强用户/物品表示。映射部分也是多层感知机。
- **ANR [4]**是一种基于评论的SOTA单域方法，它通过对用户-物品对进行aspect匹配进行推荐。在此，我们仅在目标域上训练了模型后，直接在源域中利用对应的评论来进行推荐。

### 4.3 实验设置

- 依照相关研究[32]，我们会预处理所有数据集中的用户文档和项目文档

1. 删除停用词和文档频率（document frequency）较高的单词（即相对文档频率高于0.5）；
2. 根据他们的tf-idf分数，选择前20,000个单词作为词汇表，并从原始文档中删除其他单词； 
3. 截断（填充）长（短）文档，使其长度等于500词。我们利用Google News [23]中预先训练的300维word embedding来获取每个单词的embedding向量。

- 我们根据他们论文中报告的设定策略，应用网格搜索（grid search）来调整所有方法的超参数。所有方法的最终性能报告都至少经过5次运行。

- 对于CATN，卷积滤波器的数量 <img src="https://www.zhihu.com/equation?tex=n = 50" alt="n = 50" class="ee_img tr_noresize" eeimg="1"> ，窗口大小 <img src="https://www.zhihu.com/equation?tex=s =3" alt="s =3" class="ee_img tr_noresize" eeimg="1"> 。batch size（ <img src="https://www.zhihu.com/equation?tex=\mathcal O_s∪\mathcal O_t" alt="\mathcal O_s∪\mathcal O_t" class="ee_img tr_noresize" eeimg="1"> 的数量）为256。在训练过程中，采用dropout策略随机忽略aspect表示中的一小部分值。dropout的keep概率设置为0.8，对于模型训练，我们选择的学习率为0.001。潜在维度大小 <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1"> 从 <img src="https://www.zhihu.com/equation?tex=\{16,32,64,128\}" alt="\{16,32,64,128\}" class="ee_img tr_noresize" eeimg="1"> 中优化得出，而aspect数量 <img src="https://www.zhihu.com/equation?tex=M" alt="M" class="ee_img tr_noresize" eeimg="1"> 从 <img src="https://www.zhihu.com/equation?tex=\{3,5,7,9\}" alt="\{3,5,7,9\}" class="ee_img tr_noresize" eeimg="1"> 中优化得出。
- 对于评估指标，我们将 <img src="https://www.zhihu.com/equation?tex=MSE" alt="MSE" class="ee_img tr_noresize" eeimg="1"> 用作性能指标，它在许多相关工作中被广泛用于性能评估[4，19，27，32]，其公式为：


<img src="https://www.zhihu.com/equation?tex=MSE=\frac{1}{\mathcal O} \sum_{(u,i)\in \mathcal O} (r_{u,i}-\hat r_{u,i})^2
" alt="MSE=\frac{1}{\mathcal O} \sum_{(u,i)\in \mathcal O} (r_{u,i}-\hat r_{u,i})^2
" class="ee_img tr_noresize" eeimg="1">

其中 <img src="https://www.zhihu.com/equation?tex={\mathcal O}" alt="{\mathcal O}" class="ee_img tr_noresize" eeimg="1"> 是用于参数选择的冷启动用户验证集或用于性能比较的测试集。

### 4.4 结果和讨论

- 表3报告了在三种跨域推荐方案中所有方法的总体结果。我们从结果中得出以下结论。

<center>表3 考虑MSE的三种推荐方案的效果比较。最佳结果和次佳结果分别以黑体和下划线突出显示。黑色三角形％表示相对于最佳的SOTA算法，CATN的相对性能提升。与基准方法相比，所有已报告的在0.05水平的提升具有重要的统计学意义。</center>

![table3](https://raw.githubusercontent.com/wales-z/Markdown4Zhihu/master/Data/catn_note_for_zhihu/table3.png)

- 首先，就所有跨领域推荐而言，以及在所有设置中重叠用户的比例不同，CATN均明显优于所有基准。该结果证明了我们的基于评论的推荐对于跨域设置中的冷启动用户的优越性。
- CMF在所有评估中始终表现最差，这在意料之中。仅通过分解联合（joint）矩阵来学习用户表示是不够的，这也与早期研究中观察到的一致[9，22，29]。 CDLFM对用户因子学习和跨域映射过程进行了一些改进，获得了相对于EMCDR的显著提升。 R-DFM通过融合用户评论来进一步修改DFM。但是，它们都无法获得最佳结果，这证明了如图1所示的简单的三个优化过程的缺点。


- 对于DFM和R-DFM，根据我们的实验，与EMCDR相比，结果下降了。这是因为，aSDAE将原始评级向量作为输入，在我们的数据集中可能超过十万个维度。在这种情况下，需要优化数百万个训练参数，这使得该模型收敛会变得相当复杂，并且产生的效果也较差。尽管ANR并非针跨领域场景而设计，但它在其他基线上的结果仍然很有竞争力，从而确认了评论信息对推荐任务的实用性。


- 从结果可以看出，尽管基于三步优化的方法对 <img src="https://www.zhihu.com/equation?tex=η" alt="η" class="ee_img tr_noresize" eeimg="1"> 敏感，尤其是当比率较低（10％或5％）时，我们的CATN仍然表现出更强大的性能。随着 <img src="https://www.zhihu.com/equation?tex=η" alt="η" class="ee_img tr_noresize" eeimg="1"> 降低，重叠的用户会减少。由于缺少训练实例，现有的跨域映射无法得到很好的训练，这导致效果不佳。相反，CATN利用一种简单而有效的方式来强调跨域aspect的迁移，而不是直接使用用户表示。这样，CATN可以在很大程度上降低 <img src="https://www.zhihu.com/equation?tex=η" alt="η" class="ee_img tr_noresize" eeimg="1"> 的影响。

## 5 模型分析

现在我们对CATN模型进行详细分析。我们首先研究超参数设置（即 <img src="https://www.zhihu.com/equation?tex=M" alt="M" class="ee_img tr_noresize" eeimg="1"> ）对CATN性能的影响。接下来，我们进行了三项消融研究（ablation studies），以分析我们提出的模型中的不同组成部分如何对总体结果做出贡献。最后，研究案例显示了跨域aspect转移过程的可解释性分析。

### 5.1	Aspect 数字敏感性

- 图4绘制了在具有不同预设 <img src="https://www.zhihu.com/equation?tex=η" alt="η" class="ee_img tr_noresize" eeimg="1"> 值的多个评估设置中，变化M∈{3，5，7，9}对CATN的影响。
  通常，较小的M会导致粗糙的aspects，而较大的M会导致细粒度的aspects。但是，正如我们前面所讨论的，并非源域和目标域中的所有aspects都可以匹配并参与偏好转移，并且注意力机制将学习匹配aspect之间的最佳权重。从这个意义上讲，M的变化只会影响源域和目标域中aspects的数量，而对偏好转移的影响不大。如图所示，在给定相同设置（即特定跨域推荐任务中的固定 <img src="https://www.zhihu.com/equation?tex=η" alt="η" class="ee_img tr_noresize" eeimg="1"> ）的情况下，由不同的M值引起的性能波动非常小，这表明CATN对于此参数设置具有鲁棒性。

![figure4](https://raw.githubusercontent.com/wales-z/Markdown4Zhihu/master/Data/catn_note_for_zhihu/figure4.png)

<center>图4 aspect的数量M对CATN的影响</center>

- 另一方面， <img src="https://www.zhihu.com/equation?tex=η" alt="η" class="ee_img tr_noresize" eeimg="1"> 的设置直接影响系统学习跨域偏好匹配的重叠用户的数量。显然，更多重叠的用户可以更好地理解跨域的偏好，从而提高推荐的准确性。


### 5.2 消融研究

- 为了反映CATN的直觉（intuition），我们设计了一种具有两个学习流的基于跨域评论的偏好匹配流程。学习流包括全局共享aspect表示 <img src="https://www.zhihu.com/equation?tex=V_s" alt="V_s" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=V_t" alt="V_t" class="ee_img tr_noresize" eeimg="1"> ，用以指导aspect提取。用全局跨域aspect相关性S给出最终预测。另外，出于减轻数据稀疏性的目的，还利用了来自志趣相投的用户和不重叠的用户的辅助评论来增强用户aspect的提取。
  因此，我们提出了CATN的三种变体，如下所示：
  - **CATN-basic**：作为CATN的基本变体，它在两个学习流中共享aspect提取参数。我们通过用等式4中的**简单平均**（simple average）运算代替注意力机制来排除（exclude） <img src="https://www.zhihu.com/equation?tex=V_s" alt="V_s" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=V_t" alt="V_t" class="ee_img tr_noresize" eeimg="1"> 。预测仅考虑了aspect匹配。此变体中没有使用辅助评论。
  - **CATN-attn**：与CATN-basic相比，我们引入了全局共享aspect表示来实现aspect提取，并考虑了全局跨域aspect的相关性。换句话说，CATN-attn是CATN的简化版本，没有特定领域的aspect提取和辅助评论。
  - **CATN-separate**：与CATN-attn相比，我们在两个学习流中利用了两个独立的aspect提取参数。换句话说，CATN-separate是不包括辅助评论的CATN的简化版本。

- 表4中报告了所有评估设置下的消融研究结果。我们得出以下观察结果：

  1. 参考表3中的结果，CATN-basic在所有推荐场景中均优于大多数基准，证明了基于aspect的跨域迁移方法的有效性；
  2. CATN-attn比CATN-basic有了一些改进，这表明了包括全局共享aspect表示的好处； 
  3. CATN-separate胜过上述变体，这表明了独立（distinct）aspect提取的有用性；
  4. 作为集成模型，CATN通过利用志趣相投的用户和非重叠用户的辅助评论来进一步提高性能。

  观察结果表明，辅助评论对于缓解数据稀疏性问题至关重要。

<center>表4 三种推荐方案下三种模型变体的性能比较。</center>

![table4](https://raw.githubusercontent.com/wales-z/Markdown4Zhihu/master/Data/catn_note_for_zhihu/table4.png)

### 5.3 优化器效率

- 我们提议的CATN被设计成端到端的学习框架，不仅克服了有缺陷的三步优化过程，而且还可以通过仅优化重叠用户的评分来缩短训练时间。
- 具体地，在现有方法中，只有在前两个步骤达到其最佳状态后，才能进行第三步跨域转移过程，这相当耗时。此外，DFM和R-DFM的自编码器组件包含大量参数，从而阻碍了收敛速度。
- 在我们的实验中，仅使用一个Nvidia 1080 GPU，CATN花费了大约600秒的时间在 <img src="https://www.zhihu.com/equation?tex=η= 50\%" alt="η= 50\%" class="ee_img tr_noresize" eeimg="1">  的情况下达到了Book→Movie中的最佳验证性能。相比之下，CMF为300s，ANR为400s，EMCDR为1000s，CDLFM为1200s，DFM和R-DFM为1小时以上。我们的CATN在取得了最佳性能的同时，在其他基线上仍保持了有竞争力的训练时间，尤其是在基于评论的方法R-DFM方面。

### 5.4 可解释性分析

我们进一步调查CATN能否发现领域间有意义的（meaningful）aspect迁移。为了更好地可视化aspect，我们检索了权重为文档中注意力得分平均值（即 <img src="https://www.zhihu.com/equation?tex=β_{m，j，u}" alt="β_{m，j，u}" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=β_{m，j，i}" alt="β_{m，j，i}" class="ee_img tr_noresize" eeimg="1"> ）的top-5单词。然后，我们展示包含这些信息性单词的句子，以便更好地理解。

- 来自每个跨域推荐（设定 <img src="https://www.zhihu.com/equation?tex=η= 50\%" alt="η= 50\%" class="ee_img tr_noresize" eeimg="1"> ）的用户-物品项对被随机采样并显示在表5中并提供语义解释。如图5所示，跨域的全局aspect相关性非常稀疏（通常集中在一个特定的block上）。由于篇幅所限，我们选择了最相关的跨域aspect对（即，对应于矩阵S中的最大值），并列出了分别从用户文档， <img src="https://www.zhihu.com/equation?tex=\mathcal D_s" alt="\mathcal D_s" class="ee_img tr_noresize" eeimg="1"> 的用户辅助文档和 <img src="https://www.zhihu.com/equation?tex=D_t" alt="D_t" class="ee_img tr_noresize" eeimg="1"> 的物品文档中提取的aspect。
  在 <img src="https://www.zhihu.com/equation?tex=\mathcal Dt" alt="\mathcal Dt" class="ee_img tr_noresize" eeimg="1"> 中还列出相应的用户文档（即 <img src="https://www.zhihu.com/equation?tex=\mathcal D^t_u" alt="\mathcal D^t_u" class="ee_img tr_noresize" eeimg="1"> ）。信息短语（informative phrases）以橙色突出显示，包括上下文中的停止词。作为参考，我们使用红色下划线将目标评论 <img src="https://www.zhihu.com/equation?tex=d_{u,i}" alt="d_{u,i}" class="ee_img tr_noresize" eeimg="1"> 中的相应部分标出。

![table5](https://raw.githubusercontent.com/wales-z/Markdown4Zhihu/master/Data/catn_note_for_zhihu/table5.png)

<center>表5 η= 50％时来自三个推荐方案的三个用户-物品对的示例研究</center>

![figure5](https://raw.githubusercontent.com/wales-z/Markdown4Zhihu/master/Data/catn_note_for_zhihu/figure5.png)

<center>图5 在η= 50％的三个推荐场景下的全局aspect相关性矩阵S。</center>

- **例1**：第一个示例显示了从Book域到Movie域的aspect迁移过程。为了更好地解释，我们在表6中列出了每个aspect的top-5单词，并根据我们的检查总结了“aspect标签”。
  我们观察到Book的第二和第三aspect（即情节，场景）与movie的第三aspect（即内容）关系最密切。这与图5所示一致。

  具体地说，从 <img src="https://www.zhihu.com/equation?tex=A_u[2]" alt="A_u[2]" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=A_{u_{aux}} [2]" alt="A_{u_{aux}} [2]" class="ee_img tr_noresize" eeimg="1"> 中可以推断出用户喜欢有趣的情节或故事。根据 <img src="https://www.zhihu.com/equation?tex=A_i [3]" alt="A_i [3]" class="ee_img tr_noresize" eeimg="1"> ，该作品被描述为一部有趣的电影，有着演地很好的角色。因此，可以合理地迁移用户更偏爱的aspect，从而得到了较高的预测精度。请注意， <img src="https://www.zhihu.com/equation?tex=D^t_u" alt="D^t_u" class="ee_img tr_noresize" eeimg="1"> 中针对Au [3]提及的信息和目标评论中用户喜欢喜剧的表示都暗示了aspect迁移的正确性。

- **例2**：第二个示例显示了从“Movie”域到“Music”域的aspect 迁移。根据图5，最相关的aspect对是 <img src="https://www.zhihu.com/equation?tex=\mathcal D_s" alt="\mathcal D_s" class="ee_img tr_noresize" eeimg="1"> 的第一个aspect和 <img src="https://www.zhihu.com/equation?tex=\mathcal D_t" alt="\mathcal D_t" class="ee_img tr_noresize" eeimg="1"> 的第五个aspect。对应地，从 <img src="https://www.zhihu.com/equation?tex=A_u[1]" alt="A_u[1]" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=A_{u_{aux}}[1]" alt="A_{u_{aux}}[1]" class="ee_img tr_noresize" eeimg="1"> 中我们可以推断出该用户更喜欢电影中的和平而温柔的情节。根据 <img src="https://www.zhihu.com/equation?tex=A_i[5]" alt="A_i[5]" class="ee_img tr_noresize" eeimg="1"> 的说法，该物品确实被描述为有一些负面评论的舒缓鲁特琴（lute）音乐。考虑到用户的偏爱和物品的声誉，我们的模型给出了一个中等得分的预测，而该预测的结果参考目标评论可知是准确的。

- **例3**：第三个示例显示了从Book域到Music域的aspect迁移。类似地，根据图5，最相关的aspect对是Book的第七aspect和Music的第六aspect。从 <img src="https://www.zhihu.com/equation?tex=A_u[7]" alt="A_u[7]" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=A_{u_{aux}}[7]" alt="A_{u_{aux}}[7]" class="ee_img tr_noresize" eeimg="1"> 可以推断出该用户喜欢漂亮的文字和故事。根据 <img src="https://www.zhihu.com/equation?tex=A_i[6]" alt="A_i[6]" class="ee_img tr_noresize" eeimg="1"> ，这是一首流行歌曲，因其旋律而受到赞誉，但因其歌词而受到批评。考虑到用户对故事的偏爱，我们的CATN仍会根据物品的特点和aspect迁移趋势给出高分预测。

总的来说，这三组示例表明，CATN在冷启动用户的跨域推荐中非常有效，并且通过合理的aspect迁移来支持语义解释。

## 6 结论

在本文中，我们研究了针对冷启动用户的基于评论的跨领域推荐问题。我们的主要重点是从源域到目标域的用户偏好迁移，以提供有效且可解释的建议。我们提出了一种端到端的学习策略，而不是遵循现有的框架来首先学习源域和目标域中的用户/物品表示形式，然后学习映射。更重要的是，我们考虑到以下事实：用户的偏好是多方面的，并且两个域中只有aspects的子集相匹配。因此，在我们的框架中，我们从评论文档中获取aspect，且把目标定为：通过带注意力的全局aspect表示来找到它们之间的相关性。我们证明了我们的CATN模型优于所有现有的跨域推荐任务模型。我们相信，CATN对这个有趣而关键（critical）的任务提供了另一种观点。我们的研究还将触发对跨不同领域的用户偏好迁移建模的更有效方法的研究。受[13]的启发，我们将来可能会研究基于图的CDR的更多可能性。

