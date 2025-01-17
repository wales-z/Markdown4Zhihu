# SULM

## Aspect Based Recommendations: Recommending Items with the Most Valuable Aspects Based on User Reviews

## 摘要

在本文中，我们提出了一种推荐技术，它不仅可以像传统推荐系统一样向用户推荐感兴趣的物品，而且可以推荐物品消费的特定 aspect ，以进一步增强用户对这些物品的体验。例如，它可以推荐用户去特定的餐厅（物品），并在那里订购一些特定的食物，例如海鲜（消费的一个 aspect ）。 我们的方法称为情绪效用Logistic模型（Sentiment Utility Logistic Model, SULM）。 顾名思义，SULM 使用用户评论的情感分析。 它首先根据用户可能对物品的各个 aspect 表达的内容来预测他/她可能对物品的情绪，然后确定用户对该物品的潜在体验中最有价值的 aspect 。 此外，该方法可以推荐物品以及用户可以控制的那些最重要的 aspect ，并且可以潜在地选择它们，例如去餐厅的时间，例如 午餐与晚餐，以及在那里点什么，例如海鲜。 我们在三个应用场景（餐厅、酒店和美容 & 水疗中心）上测试了所提出的方法，并通过实验表明，那些在消费物品时遵循我们对最有价值的 aspect 的建议的用户，拥有更好的体验，正如总体评价 (overall rating) 所定义的 。

## 1 引言

在过去的十年中，人们对利用用户评论提供基于这些评论的个性化推荐产生了极大的兴趣 [6]。 该领域的大部分工作都集中在尝试根据用户评论和其他相关信息来改进对物品用户评分的估计 [6]，并解释为什么根据评论信息向用户提供某些特定推荐 [25] 。

这些方法旨在根据用户和物品特征预测和解释评分，而不考虑其他因素，例如环境和用户对消费物品的个人选择。 例如，考虑用户在咖啡馆中选择订购提拉米苏或 Cannoli。 根据用户在访问期间选择品尝什么，她可以对机构给出不同的评分。 因此，通过推荐一些额外的 aspect 和个人用户选择消费该物品，例如在该咖啡馆订购提拉米苏，可以进一步改善特定物品的用户体验。

请注意，并非用户体验的所有 aspect 都可以由用户选择，以改善她对物品的体验。 例如，在电影的情况下，电影情节或演员等 aspect 超出了用户的控制，这与上述餐厅示例中选择特定菜肴形成对比。

在本文中，通过既推荐特定物品又推荐用户控制的消费的最重要 aspect ，我们关注用户控制 aspect 的后者的情况，例如在咖啡馆中订购提拉米苏或 Cannoli。 此外，我们可以向机构（物品）的管理人员推荐某些操作，这些操作可以个性化用户在消费物品（例如，访问机构）时的体验。 例如，我们可能会向水疗沙龙的管理层推荐向用户推荐免费饮料，因为我们的方法预估用户会特别喜欢该沙龙中的饮料，这将增强她在那里的体验

在本文中，我们提出了一种方法，该方法可以识别用户尚未尝试过的物品的、可能是用户体验中最有价值的 aspect ，并推荐这些物品，同时提供建议：使用这些最有价值的、用户控制的 aspect ，这些 aspect 我们已识别为对该用户有益。 更进一步，我们开发了情感效用Logistic模型 (SULM)，它获取用户评论和评分，提取 aspect ，并对用户评论中的 aspect 的情感进行分类，并推荐物品以及可能增强用户体验的最重要 aspect  与物品。 为了实现这一点，模型学习如何预测未知的评分、用户对物品各个 aspect 的看法，并确定这些 aspect 对物品整体评分的影响。 此外，我们使用这些估计的影响向用户推荐最有价值的 aspect ，以增强他们对推荐物品的体验。因此，SULM 更进了一步，通过为传统评分预测和推荐任务提供所有这些附加功能，显着增强了当前推荐系统的功能。

我们在本文中做出了以下贡献：

(1) 提出了一种新颖的方法来增强当前推荐系统的功能，不仅推荐商品本身，还推荐消费的特定 aspect ，以进一步增强的用户对该物品体验。

(2) 开发一种新方法 (SULM)，使用细粒度 aspect 级别的情感分析来识别未来用户体验中最有价值的 aspect ，该分析会自动发现用户在评论中指定的 aspect 和相应的情感。

(3) 在三个现实生活应用场景的实际评论中测试所提出的方法，并通过提供改善用户体验的最有价值 aspect 的建议，表明我们的方法在这些应用场景中表现良好。此外，我们表明，所提出的方法还可以预测用户评论的未知评分以及用户在评论中提到的一组 aspect 。

本文的其余部分组织如下。 我们在第 2 节讨论相关工作，并在第 3 节介绍所提出的方法。 第 4 节描述了三个实际应用的实验，第 5 节介绍了结果。第 6 节总结了我们的发现并总结了论文

## 2 文献回顾

在过去的几年中，有几篇论文试图通过从用户评论中提取有用信息并利用这些信息来改进对物品的未知评分的估计 [6]。 例如，[10] 的作者在餐厅评论中发现了六个 aspect ，并训练分类器在评论中识别它们以提高评分预测质量。 在 [20] 中，作者计算了整个评论的情绪，并将这些信息整合到矩阵分解技术中。

除了这些基于用户评论的直接评分预测方法之外，还有几种建议的方法可以根据从用户评论中推断出的潜在 aspect 来预测用户评分。
特别是，[26] 提出了一种潜在 aspect 评分分析方法来发现主题 aspect 的相对重要性。  [18] 使用基于 LDA 的方法结合矩阵分解来预测未知评分。  他们为潜在评分维度获得了高度可解释的文本标签，这有助于他们使用评论文本“证明” (justify) 特定评分值。最近，[8]、[16] 和 [27] 通过使用更复杂的图形模型来预测基于协同过滤和用户评论主题建模的未知评分，从而超越了 [18]。  他们的模型能够捕捉可解释的 aspect 以及评论每个 aspect 的情绪。 在 [22] 中，作者提出了基于 Aspect 的潜在因子模型 (ALFM)，该模型结合了评分和评论文本以改进评分预测。 最近的一项工作 [28] 提出了一个框架，该框架结合了用户意见和不同 aspect 的偏好。 特别是，他们将张量分解技术应用于使用 LDA 方法聚类的术语。 此外，[28] 使用 LDA terms 主题来构建用户画像 (profile) 并过滤要显示给用户的评论。

另一个研究流侧重于利用情感分析技术从评论中提取有用的 aspect 。 特别是，[31] 提出了显式因子模型 (EFM) 以根据特定产品 aspect 生成推荐，并且 [7] 应用张量因子分解技术来学习用户对物品各个 aspect 的偏好排名 . 此外，[12] 将顶点排序方法应用于用户-物品- aspect 的三方图，以使用评论提供更好的物品推荐。 最后，[29] 开发了一种算法来推断 aspect 对于用户对历史评论的整体意见的重要性。  这个方法无法像 SULM 那样预测新的潜在评论的 aspect 重要性。

此外，我们的工作还与上下文感知推荐系统 (CARS) [4] 有关。请注意，我们的 aspect 还可能包括用户体验的上下文变量，但不仅限于这些变量。 在 CARS 中已经有很多相关工作，包括处理用户评论的论文 [1, 11, 15]。 大部分工作开发了从用户生成的评论中提取上下文信息的新方法，并使用这些信息来估计物品的未知评分。 例如，在 [1] 中，作者根据分类和文本挖掘技术确定了评论中包含上下文信息的句子。  他们将此方法应用于具有旅行目标上下文变量的酒店应用程序。[15] 的作者提出了一种基于 NLP 技术在餐厅应用中提取同伴、场合、时间和位置上下文变量的方法。 此外，[11] 使用 Labeled-LDA 方法根据旅行类型上下文变量对酒店评论进行分类。 所有这些论文都展示了如何使用提取的上下文变量来改进物品的评分预测。最后，[32] 提出了一个“上下文建议”系统，该系统基于收集的有关上下文变量的数据，但不像我们那样处理用户评论。

也有关于很多关于多标准评分预测的文献 [3]，其中此类多标准系统使用少量预定义的 aspect 的评分（例如餐厅应用中的食品质量、服务质量和氛围）提供适当的物品建议。与这种类型的研究相比，我们使用从一个评论变为另一个评论的自动提取的广泛的 (a wide range of) aspect ，因此，我们不限于预先定义的固定的一组标准。 例如，评论 A 可能会提到  <img src="https://www.zhihu.com/equation?tex=x1" alt="x1" class="ee_img tr_noresize" eeimg="1"> 、 <img src="https://www.zhihu.com/equation?tex=x2" alt="x2" class="ee_img tr_noresize" eeimg="1"> 、 <img src="https://www.zhihu.com/equation?tex=x3" alt="x3" class="ee_img tr_noresize" eeimg="1">  这些 aspect ，而评论 B 可能会提到  <img src="https://www.zhihu.com/equation?tex=x3" alt="x3" class="ee_img tr_noresize" eeimg="1"> 、 <img src="https://www.zhihu.com/equation?tex=x4" alt="x4" class="ee_img tr_noresize" eeimg="1"> 、 <img src="https://www.zhihu.com/equation?tex=x5" alt="x5" class="ee_img tr_noresize" eeimg="1"> ，而多标准方法在所有评论中使用相同（通常很小）的一组固定的 aspect 。

与之前的所有工作相比，我们不仅像上面回顾的先前工作中所做的那样，根据用户评论预测物品的未知评分，而且还估计用户在评论中将在各个 aspect 表达的情绪，并确定  aspect 对有关物品评论的总体预测评分的影响。 此外，我们使用这些估计的影响向用户推荐最有价值的 aspect ，以增强他们对推荐物品的体验。 最后，我们不仅向用户提供推荐，还向管理人员推荐用户消费的有价值的 aspect ，帮助他们经营业务并为用户提供更好的服务。
在下一节中，我们将介绍我们提出的方法

## 3 方法总览

在本节中，我们提出的方法可以识别用户尚未尝试过的物品的可能用户体验中最有价值的 aspect ，并推荐这些物品以及我们已经确定的那些最重要的用户控制的 aspect ，这些 aspect 我们已识别为对用户有益。 特别是，我们的方法包括用户评论的情感分析和我们称为情感效用Logistic模型（SULM）的模型的后续训练。 SULM 模型不仅可以预测评论的评分，还可以识别每个 aspect 对整体评分的影响。 更具体地说，SULM 构建用户和物品画像，用于估计情绪效用以及评论的整体评级。 因此，SULM 可用于提供物品推荐，并提供建议以体验模型认为对用户有益的最有价值的 aspect 。

SULM 模型依赖于第 3.2 节和第 3.3 节中使用的Logistic函数 (logistic function) 。 在关注模型本身之前，我们在此提供一些有关它的背景信息。Logistic函数将实数映射到区间 [0, 1] ，其定义为

<img src="https://www.zhihu.com/equation?tex=g(t)=\frac{1}{1+e^{-t}}
\tag1
" alt="g(t)=\frac{1}{1+e^{-t}}
\tag1
" class="ee_img tr_noresize" eeimg="1">
Logistic函数可以应用于分类问题，因为我们可以估计向量  <img src="https://www.zhihu.com/equation?tex=x" alt="x" class="ee_img tr_noresize" eeimg="1">  具有类标签  <img src="https://www.zhihu.com/equation?tex=y \in {0, 1} " alt="y \in {0, 1} " class="ee_img tr_noresize" eeimg="1"> 之一的概率为

<img src="https://www.zhihu.com/equation?tex=\begin {cases}
P(y=1|x;\theta)=g(f(x,\theta))\\
P(y=0|x;\theta)=1-g(f(x,\theta))
\end {cases}
" alt="\begin {cases}
P(y=1|x;\theta)=g(f(x,\theta))\\
P(y=0|x;\theta)=1-g(f(x,\theta))
\end {cases}
" class="ee_img tr_noresize" eeimg="1">
其中  <img src="https://www.zhihu.com/equation?tex=f (x,\theta )" alt="f (x,\theta )" class="ee_img tr_noresize" eeimg="1">  是一个函数， <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1"> 是我们将要估计的模型的一组参数。   <img src="https://www.zhihu.com/equation?tex=f (x, θ ) = a_0·x_0 +···+a_n · x_n" alt="f (x, θ ) = a_0·x_0 +···+a_n · x_n" class="ee_img tr_noresize" eeimg="1">  的线性情况构成Logistic回归[5]。 假设训练样例  <img src="https://www.zhihu.com/equation?tex=x" alt="x" class="ee_img tr_noresize" eeimg="1">  是独立生成的，我们可以将参数  <img src="https://www.zhihu.com/equation?tex=θ" alt="θ" class="ee_img tr_noresize" eeimg="1">  的似然写为：

<img src="https://gitee.com/Wales-Z/image_bed/raw/master/img/image-20211020175720177.png" alt="image-20211020175720177" style="zoom:67%;" />

为了找到最大化  <img src="https://www.zhihu.com/equation?tex={\cal L}(\theta) " alt="{\cal L}(\theta) " class="ee_img tr_noresize" eeimg="1"> 的  <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1"> ，我们将随机梯度下降 [30] 应用于对数似然函数，其中梯度步骤是基于偏导数计算的：

<img src="https://gitee.com/Wales-Z/image_bed/raw/master/img/image-20211020175850500.png" alt="image-20211020175850500" style="zoom:67%;" />

我们将使用下面的等式 (4) 来训练 SULM 模型。

在本节的其余部分，我们描述了所提出方法的细节。

### 3.1 提取 aspect-sentiment 对

在这一步中，我们利用最先进的 “industrial-strength” 情感分析系统 Opinion Parser (OP) 从评论文本中提取 aspect 表达。  OP 是一种无监督的基于 aspect 的情感分析系统。 它执行两个关键功能， aspect 提取和 aspect 情感分类。  aspect 提取旨在提取已被表达某些情感的情感目标。  这些目标通常是实体（例如，产品或服务）的不同 aspect ，它们是我们上下文中的物品。 aspect 情感分类对在 aspect 表达的情感是积极的、中性的还是消极的进行分类。 例如，从句子“The food is great”中，“food”应该被 aspect 提取子系统提取为一个 aspect 或目标，并且“food”的意见应该被 aspect 情感分类子系统归类为positive 。  Opinion Parser 中使用的 aspect 提取算法称为双重传播 (double propagation, DP) [21]。 它基于这样一种思想，即情感总是有一个目标 aspect ，它们在句子中的表达具有某种句法关系。例如，情感词“great”的目标 aspect 是“food”。 给定情感词“great”，依赖解析器 (DEPENDENCY PARSER) 可用于识别用于提取“食物”的关系。 因此，DP 的工作原理如下：给定一组种子情感词  <img src="https://www.zhihu.com/equation?tex=S" alt="S" class="ee_img tr_noresize" eeimg="1">  ，执行引导程序以使用  <img src="https://www.zhihu.com/equation?tex=S" alt="S" class="ee_img tr_noresize" eeimg="1">  提取 aspect 和更多情感词，并且结果 aspect 和情感词可用于更迭代地提取。 该算法的详细信息可以在 [21] 中找到。 Aspect 情感分类基于一组情感表达（称为情感词典 sentiment lexicon）、语法分析和上下文分析，以确定一个句子对一个 aspect 是正面还是负面。 更多细节可以在[17]中找到。 我们将在第 4.2 节报告其性能。

在我们的研究中，我们将 OP 应用于给定应用场景（例如，餐厅）的 aspect 级别的评论集合  <img src="https://www.zhihu.com/equation?tex=R" alt="R" class="ee_img tr_noresize" eeimg="1"> 。  OP 建立一组出现在 R 中的 aspect   <img src="https://www.zhihu.com/equation?tex=\mathbb A" alt="\mathbb A" class="ee_img tr_noresize" eeimg="1"> ，对于每个评论  <img src="https://www.zhihu.com/equation?tex=r \in R" alt="r \in R" class="ee_img tr_noresize" eeimg="1"> ，OP 识别一组出现在  <img src="https://www.zhihu.com/equation?tex=r" alt="r" class="ee_img tr_noresize" eeimg="1">  中的 aspect   <img src="https://www.zhihu.com/equation?tex=A_r" alt="A_r" class="ee_img tr_noresize" eeimg="1">  和相应的情绪意见  <img src="https://www.zhihu.com/equation?tex=o^k_{ui} \in \{0, 1\}" alt="o^k_{ui} \in \{0, 1\}" class="ee_img tr_noresize" eeimg="1"> ，其中 1 是正面的（例如 ) 和 0 是负数。

我们使用识别到的 aspect 和情感来训练我们的模型，如本节其余部分所述。

### 3.2 Aspect 情感

情感效用Logistic模型 (SULM) 假设对于消费物品  <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1">  的每个 aspect   <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1"> ，用户  <img src="https://www.zhihu.com/equation?tex=u" alt="u" class="ee_img tr_noresize" eeimg="1">  可以给出情感效用值  <img src="https://www.zhihu.com/equation?tex=s^k_{u,i} \in \mathbb R" alt="s^k_{u,i} \in \mathbb R" class="ee_img tr_noresize" eeimg="1">  表示对物品  <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1">  的 aspect   <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1">  的满意度。这些效用值无法从用户评论中观察到。取而代之的是，我们观察 OP 的输出，该输出仅识别所表达情感的二进制值  <img src="https://www.zhihu.com/equation?tex=o^k_{ui} \in \{0, 1\}" alt="o^k_{ui} \in \{0, 1\}" class="ee_img tr_noresize" eeimg="1"> 。因此，我们估计实际情感效用值  <img src="https://www.zhihu.com/equation?tex=s^k_{u,i}" alt="s^k_{u,i}" class="ee_img tr_noresize" eeimg="1">  ，通过以下方式：使它们在应用Logistic函数 (1) 后拟合从评论中提取二元情感值。

此外，SULM 使用矩阵分解方法 [13] 估计每个 aspect   <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1">  的情感效用值：

<img src="https://www.zhihu.com/equation?tex=\hat s^k_{u,i}(\theta_s)=\mu^k + b^k_u + (q^k_i)^T \cdot p^k_u
\tag5
" alt="\hat s^k_{u,i}(\theta_s)=\mu^k + b^k_u + (q^k_i)^T \cdot p^k_u
\tag5
" class="ee_img tr_noresize" eeimg="1">
其中  <img src="https://www.zhihu.com/equation?tex=\mu^k" alt="\mu^k" class="ee_img tr_noresize" eeimg="1">  是与 aspect   <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1">  相关的常数， <img src="https://www.zhihu.com/equation?tex=b^k_u" alt="b^k_u" class="ee_img tr_noresize" eeimg="1">  和  <img src="https://www.zhihu.com/equation?tex=b^k_i" alt="b^k_i" class="ee_img tr_noresize" eeimg="1">  是的用户  <img src="https://www.zhihu.com/equation?tex=u" alt="u" class="ee_img tr_noresize" eeimg="1">  和物品  <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1">  的  aspect   <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1">  的偏置， <img src="https://www.zhihu.com/equation?tex=p^k_u" alt="p^k_u" class="ee_img tr_noresize" eeimg="1">  和  <img src="https://www.zhihu.com/equation?tex=q^k_i" alt="q^k_i" class="ee_img tr_noresize" eeimg="1">  是对应于 aspect   <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1">  的用户  <img src="https://www.zhihu.com/equation?tex=u" alt="u" class="ee_img tr_noresize" eeimg="1">  和物品  <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1">  的  <img src="https://www.zhihu.com/equation?tex=m" alt="m" class="ee_img tr_noresize" eeimg="1">  维隐向量。 我们用  <img src="https://www.zhihu.com/equation?tex=\theta_s = (\mu, B_u, B_i, P,Q)" alt="\theta_s = (\mu, B_u, B_i, P,Q)" class="ee_img tr_noresize" eeimg="1">  表示所有这些系数。
此外，我们估计参数  <img src="https://www.zhihu.com/equation?tex=θ_s" alt="θ_s" class="ee_img tr_noresize" eeimg="1">  以使情感的估计值  <img src="https://www.zhihu.com/equation?tex=\hat o^K_{u,i}(\theta_s)=g(\hat s^k_{u,i}(\theta_s))" alt="\hat o^K_{u,i}(\theta_s)=g(\hat s^k_{u,i}(\theta_s))" class="ee_img tr_noresize" eeimg="1">  拟合由 OP 提取的真实二元情感，如上所述。

特别地，假设训练样例是独立生成的，我们搜索  <img src="https://www.zhihu.com/equation?tex=\theta_s" alt="\theta_s" class="ee_img tr_noresize" eeimg="1">  来最大化对数似然

<img src="https://gitee.com/Wales-Z/image_bed/raw/master/img/image-20211020193700150.png" alt="image-20211020193700150" style="zoom:67%;" />

其中  <img src="https://www.zhihu.com/equation?tex=S" alt="S" class="ee_img tr_noresize" eeimg="1">  是用户在训练评论集中表达的所有情感的集合。

在本小节中，我们描述了如何估计模型的参数以评估用户评论中的情绪。 在下一小节中，我们将重点讨论评分估计问题。最后，我们将这两个模型组合成整体 SULM 模型，以在 3.4 节中估计这两个组件。

### 3.3 总体满意度

与单个 aspect 的情况一样，SULM 假设用户  <img src="https://www.zhihu.com/equation?tex=u" alt="u" class="ee_img tr_noresize" eeimg="1">  可以定义消费物品  <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1">  的总体满意度，该满意度由效用值  <img src="https://www.zhihu.com/equation?tex=d_{u,i} \in \mathbb R" alt="d_{u,i} \in \mathbb R" class="ee_img tr_noresize" eeimg="1">  衡量。我们将此效用估计为评论中所有 aspect 的单个情绪效用值的线性组合 

<img src="https://www.zhihu.com/equation?tex=\hat d_{u,i}(\theta)=\sum_{k \in \mathbb A}\hat s^k_{u,i}(\theta_s)
\cdot (z^k + w^k_u + v^k_i)
\tag7
" alt="\hat d_{u,i}(\theta)=\sum_{k \in \mathbb A}\hat s^k_{u,i}(\theta_s)
\cdot (z^k + w^k_u + v^k_i)
\tag7
" class="ee_img tr_noresize" eeimg="1">
其中  <img src="https://www.zhihu.com/equation?tex=z^k" alt="z^k" class="ee_img tr_noresize" eeimg="1">  是表示aspect  <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1">  在应用场景（例如餐馆）中的相对重要性的一般系数。 此外，每个用户  <img src="https://www.zhihu.com/equation?tex=u" alt="u" class="ee_img tr_noresize" eeimg="1">  对于整体满意度可能具有个人偏好和不同 aspect 的特定重要性值 (specific values of importance of aspects)，因此，系数  <img src="https://www.zhihu.com/equation?tex=w^k_u" alt="w^k_u" class="ee_img tr_noresize" eeimg="1">  表示 aspect  <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1"> 对于用户  <img src="https://www.zhihu.com/equation?tex=u" alt="u" class="ee_img tr_noresize" eeimg="1">   的这种个人的重要性值。 类似地，每个物品  <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1">  都有自己的特性，系数  <img src="https://www.zhihu.com/equation?tex=v^k_i" alt="v^k_i" class="ee_img tr_noresize" eeimg="1">  决定了物品  <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1">  的aspect  <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1">  的重要性值。 我们用  <img src="https://www.zhihu.com/equation?tex=\theta_r = (Z,W ,V )" alt="\theta_r = (Z,W ,V )" class="ee_img tr_noresize" eeimg="1">  表示这些新系数，用  <img src="https://www.zhihu.com/equation?tex=θ = (θ_r, θ_s ) " alt="θ = (θ_r, θ_s ) " class="ee_img tr_noresize" eeimg="1"> 表示模型中所有系数的集合。

此外，在我们的模型中，我们没有估计用户会给物品的评分并最小化 RMSE 性能指标，而是遵循以前工作（例如 [2]）中提倡的替代方法，并将评分分为“喜欢” (like) 和“不喜欢” (dislike)。 在传统的五星评分设定中，我们会将“喜欢”评分映射到  <img src="https://www.zhihu.com/equation?tex=\{4, 5\}" alt="\{4, 5\}" class="ee_img tr_noresize" eeimg="1"> ，将“不喜欢”评分映射到  <img src="https://www.zhihu.com/equation?tex=\{1, 2, 3\}" alt="\{1, 2, 3\}" class="ee_img tr_noresize" eeimg="1"> 。 因此，我们将推荐回归转化为分类问题。

最后，我们估计参数  <img src="https://www.zhihu.com/equation?tex=θ" alt="θ" class="ee_img tr_noresize" eeimg="1">  以使整体效用值  <img src="https://www.zhihu.com/equation?tex=\hat d_{u,i} (θ)" alt="\hat d_{u,i} (θ)" class="ee_img tr_noresize" eeimg="1">  的Logistic变换 (1) 可以拟合用户  <img src="https://www.zhihu.com/equation?tex=u" alt="u" class="ee_img tr_noresize" eeimg="1">  为物品  <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1">  指定二进制评级  <img src="https://www.zhihu.com/equation?tex=r_{u,i} \in \{0, 1\}" alt="r_{u,i} \in \{0, 1\}" class="ee_img tr_noresize" eeimg="1">  

<img src="https://www.zhihu.com/equation?tex=\hat r_{u,i}(\theta)=g \big( \hat d_{u,i}(\theta)  \big)
\tag8
" alt="\hat r_{u,i}(\theta)=g \big( \hat d_{u,i}(\theta)  \big)
\tag8
" class="ee_img tr_noresize" eeimg="1">
特别是，假设训练样例是独立生成的，我们在评论训练集上搜索最大化对数似然函数的  <img src="https://www.zhihu.com/equation?tex=θ" alt="θ" class="ee_img tr_noresize" eeimg="1"> ：

<img src="https://gitee.com/Wales-Z/image_bed/raw/master/img/image-20211020201121957.png" alt="image-20211020201121957" style="zoom:67%;" />

在本小节中，我们描述了如何估计模型的参数以获取用户提供的二元评分。在下一小节中，我们将两个模型（6）和（9）组合成整体 SULM。

### 3.4 SULM模型

SULM 模型由 3.2 和 3.3 节中描述的两部分组成。SULM 的主要目标是估计系数  <img src="https://www.zhihu.com/equation?tex=θ" alt="θ" class="ee_img tr_noresize" eeimg="1"> ，使得模型的两个部分同时处理从评论中提取的情绪和用户提供的评分。
更具体地说，SULM 优化标准由来自模型的情感效用部分（等式 (6) ）和模型的评分预测部分（等式 (9)）的标准组成。 此外，我们还应用正则化来避免过拟合。 结合所有这些考虑，我们寻找  <img src="https://www.zhihu.com/equation?tex=θ" alt="θ" class="ee_img tr_noresize" eeimg="1">  来最小化：

<img src="https://gitee.com/Wales-Z/image_bed/raw/master/img/image-20211020201414100.png" alt="image-20211020201414100" style="zoom:67%;" />

其中  <img src="https://www.zhihu.com/equation?tex=\alpha" alt="\alpha" class="ee_img tr_noresize" eeimg="1">  是模型的参数，它定义了优化标准的 aspect 和评分部分的相对重要性，而  <img src="https://www.zhihu.com/equation?tex=\lambda_r" alt="\lambda_r" class="ee_img tr_noresize" eeimg="1">  、 <img src="https://www.zhihu.com/equation?tex=\lambda_s" alt="\lambda_s" class="ee_img tr_noresize" eeimg="1">  是正则化参数。

### 3.5 拟合SULM模型

我们应用随机梯度下降 [30] 来估计参数  <img src="https://www.zhihu.com/equation?tex=θ" alt="θ" class="ee_img tr_noresize" eeimg="1">  来最小化 (10)。 特别地，我们计算偏导数  <img src="https://www.zhihu.com/equation?tex=\frac{\partial Q}{\partial \theta_j}" alt="\frac{\partial Q}{\partial \theta_j}" class="ee_img tr_noresize" eeimg="1">  以执行梯度步骤。

首先，将评分的真实值与预测值之间的差表示为  <img src="https://www.zhihu.com/equation?tex=\Delta^r_{u,i} = r_{u,i} − \hat r_{u,i} " alt="\Delta^r_{u,i} = r_{u,i} − \hat r_{u,i} " class="ee_img tr_noresize" eeimg="1"> ，将情感的真实值与预测值之间的差表示为  <img src="https://www.zhihu.com/equation?tex=\Delta^s_{u,i,k}= o^k_{ui}−\hat o^k_{ui} " alt="\Delta^s_{u,i,k}= o^k_{ui}−\hat o^k_{ui} " class="ee_img tr_noresize" eeimg="1"> 。 此外，我们用  <img src="https://www.zhihu.com/equation?tex=I^k_{u,i} \in \{0,1\}" alt="I^k_{u,i} \in \{0,1\}" class="ee_img tr_noresize" eeimg="1">  表示指示函数，显示用户  <img src="https://www.zhihu.com/equation?tex=u" alt="u" class="ee_img tr_noresize" eeimg="1">  是否在她的评论中表达了对物品  <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1">  的aspect  <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1">  的情绪。

基于 (4)， <img src="https://www.zhihu.com/equation?tex=Q" alt="Q" class="ee_img tr_noresize" eeimg="1">  对  <img src="https://www.zhihu.com/equation?tex=\mu_k" alt="\mu_k" class="ee_img tr_noresize" eeimg="1">  的偏导数将是

<img src="https://gitee.com/Wales-Z/image_bed/raw/master/img/image-20211020202108413.png" alt="image-20211020202108413" style="zoom:67%;" />

我们用  <img src="https://www.zhihu.com/equation?tex=−\delta ^k _{u,i}" alt="−\delta ^k _{u,i}" class="ee_img tr_noresize" eeimg="1">  表示这个表达式。 此外，我们计算  <img src="https://www.zhihu.com/equation?tex=θ" alt="θ" class="ee_img tr_noresize" eeimg="1">  中其余参数的  <img src="https://www.zhihu.com/equation?tex=Q" alt="Q" class="ee_img tr_noresize" eeimg="1">  的偏导数，并执行如下梯度下降步骤：

<img src="https://gitee.com/Wales-Z/image_bed/raw/master/img/image-20211020202212659.png" alt="image-20211020202212659" style="zoom: 50%;" />

与矩阵分解 [14] 的情况一样，迭代进行：首先通过固定  <img src="https://www.zhihu.com/equation?tex=θ" alt="θ" class="ee_img tr_noresize" eeimg="1">  中的其余参数来优化  <img src="https://www.zhihu.com/equation?tex=θ_s" alt="θ_s" class="ee_img tr_noresize" eeimg="1">  中与用户相关的参数，然后通过固定  <img src="https://www.zhihu.com/equation?tex=θ" alt="θ" class="ee_img tr_noresize" eeimg="1">  中的其余参数对  <img src="https://www.zhihu.com/equation?tex=θ_s" alt="θ_s" class="ee_img tr_noresize" eeimg="1">  中与物品相关的的参数进行优化，最后，通过调整参数  <img src="https://www.zhihu.com/equation?tex=θ_s" alt="θ_s" class="ee_img tr_noresize" eeimg="1">  来优化  <img src="https://www.zhihu.com/equation?tex=θ_r" alt="θ_r" class="ee_img tr_noresize" eeimg="1">  中的参数。 我们反复进行直到收敛。作为结果，我们估计了 SULM 模型的所有参数。

### 3.6 Aspect 对评分的影响

在这一步中，我们应用在第 3.5 节中训练的模型来确定在第 3 节开头讨论的用户对物品的潜在体验的最重要的 aspect 。特别是，我们通过其在回归模型 (7) 中的权重来衡量一个 aspect 的重要性 。这意味着对于用户  <img src="https://www.zhihu.com/equation?tex=u" alt="u" class="ee_img tr_noresize" eeimg="1">  对物品  <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1">  的潜在体验，我们首先预测应用程序中每个 aspect   <img src="https://www.zhihu.com/equation?tex=k \in \mathbb A" alt="k \in \mathbb A" class="ee_img tr_noresize" eeimg="1">  的情感效用值  <img src="https://www.zhihu.com/equation?tex=\hat s^k_{ui}" alt="\hat s^k_{ui}" class="ee_img tr_noresize" eeimg="1">  。 此外，我们计算潜在用户评论中每个 aspect   <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1">  对于用户  <img src="https://www.zhihu.com/equation?tex=u" alt="u" class="ee_img tr_noresize" eeimg="1">  对物品  <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1">  的总体预测满意度的影响，作为线性模型 (7)的相应加数

<img src="https://www.zhihu.com/equation?tex=impact^k_{ui}=\hat s^k_{ui} \cdot (z^k + w^k_u +v^k_i)
\tag{12}
" alt="impact^k_{ui}=\hat s^k_{ui} \cdot (z^k + w^k_u +v^k_i)
\tag{12}
" class="ee_img tr_noresize" eeimg="1">
换句话说，aspect  <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1">  对于用户  <img src="https://www.zhihu.com/equation?tex=u" alt="u" class="ee_img tr_noresize" eeimg="1">  对 item  <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1">  的体验的影响被计算为：预测的情感效用值   <img src="https://www.zhihu.com/equation?tex=\hat s^k_{ui}" alt="\hat s^k_{ui}" class="ee_img tr_noresize" eeimg="1">  与表示 item  <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1">  的 aspect  <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1">  对用户  <img src="https://www.zhihu.com/equation?tex=u" alt="u" class="ee_img tr_noresize" eeimg="1">  的重要性的相应系数的乘积 .

这些计算出的 aspect 的影响反映了用户评论的每个 aspect 对整体预测评分的重要性。 请注意，它们可以是正面的也可以是负面的，我们可以使用它们来推荐正面的体验，并在用户消费推荐商品时避免负面体验，如下一节所述。

### 3.7 推荐物品和Aspect

接下来，我们在应用场景的所有 aspect   <img src="https://www.zhihu.com/equation?tex=\mathbb A" alt="\mathbb A" class="ee_img tr_noresize" eeimg="1">  中手动识别两组 aspect ，（a）用户具有控制权，（b）机构的管理具有控制权。 我们分别称这组 aspect 为用户控制的和管理员控制的。 例如，酒店应用场景中的 aspect  “gym”在用户控制之下，因为她可以在入住酒店期间决定是否使用它。 此外，在这些组中，我们识别出了我们向用户推荐的最有价值的 aspect ，连带对应物品和管理层。  这些建议可以是积极的（建议体验某个 aspect ）或消极的（建议避免某个 aspect ）。 最后，我们向用户推荐一个物品和识别出的相应 aspect ，或者向管理层推荐最重要的 aspect 

例如，如果我们的系统识别出 “fish” 这个 aspect 对评分有很大的正面影响，我们将推荐这家餐厅并建议用户在该餐厅订购 fish。 类似地，如果“甜点” 这个aspect 对评分有很强的负面影响，如果我们预计这种情况下的餐厅评分很高，我们仍可能会建议用户访问该餐厅，并建议不要在那里订购甜点。 此外，我们可以向管理层推荐受其控制且管理层有影响力的这些 aspect 。 例如，我们可以向美容和水疗沙龙的管理层推荐为用户提供补充饮料（因为这会改善她的整体体验）并且在会话期间不要与她过多聊天

总之，我们提出了一种方法来预测用户是否喜欢某个物品，估计用户可能对物品的不同 aspect 表达的情绪，以及识别和推荐最有价值的用户控制 aspect  物品的潜在用户体验。这个方法包括对用户评论的情感分析、训练情感效用逻辑模型 (SULM)、预测情感效用值以及计算每个 aspect 对用户整体评分有贡献的个人影响因素。 在第 4 节中，我们展示了将所提出的方法应用于来自三个应用场景的真实数据的实验结果。

## 4 实验

### 4.1 数据集

为了证明我们的方法在实践中的效果如何，我们根据 6 年内在美国几个城市收集的 Yelp reviews 在餐厅、酒店和美容与水疗应用场景上对其进行了测试。在这项研究中，我们只选择那些写了至少 10 条评论的用户。 表 1 显示了三个应用场景中初始数据集中的评论数量、拥有超过 10 条评论的用户以及仅由这些用户生成的总体评分（即过滤后的评分）。

<img src="https://gitee.com/Wales-Z/image_bed/raw/master/img/image-20211020205018918.png" alt="image-20211020205018918" style="zoom:67%;" />

【表1】

尽管 Yelp 使用 5 星评级系统，但我们将其转换为二元的“高”（{4, 5}）和“低”（{1, 2, 3}），如第 3.3 节所述。 此外，我们将评分估计重新表述为一个分类问题，其中我们估计用户“喜欢”一个物品的概率（通过给它一个 4 或 5 的评级）

### 4.2 实验设定

我们将第 3 节中介绍的方法应用于餐厅、酒店和美容与水疗应用场景，并使用 Opinion Parser 分别为这些应用程序提取了 69、42 和 45 个 aspect ，如第 3.1 节所述。 表 2 显示了从餐厅应用程序的评论中提取的几个 aspect 的示例，以及与这些 aspect 对应的一些词的示例。 对于每个评论，我们还确定出现在该评论中的一组 aspect 及其相应的情绪，如第 3.1 节所述。

OP 的 aspect 提取部分在 5 个基准在线评论数据集 [21] 上进行了评估，其 F 值为 0.86。 基于 8 个在线评论数据集 [9, 17] 对 OP 的 aspect 情感分类部分的评估显示，正面和负面情绪类别的 F 分数平均为 0.90。 我们还在我们的数据集上测试了 OP 系统的性能。 特别是，我们从餐厅应用程序的评论中随机选择了 3000 个句子，并手动评估了 OP 的 aspect 提取和情感分类部分。 我们的结果与之前的研究一致，系统的两个部分的 F-score 分别为 0.89 和 0.93

所有这些评估都表明 OP 总体上表现良好，特别是在我们的应用中。 在这项研究中，我们专注于利用 OP 的输出来提供物品及其最有价值的 aspect 的建议。此外，对于每个应用场景，评论集 R 以 80% 到 20% 的比例划分为训练集和测试集。 我们还在训练集上使用交叉验证来确定 SULM 模型的最佳参数，包括  <img src="https://www.zhihu.com/equation?tex=\alpha" alt="\alpha" class="ee_img tr_noresize" eeimg="1"> 、 <img src="https://www.zhihu.com/equation?tex=\lambda_s" alt="\lambda_s" class="ee_img tr_noresize" eeimg="1">  和  <img src="https://www.zhihu.com/equation?tex=\lambda_r" alt="\lambda_r" class="ee_img tr_noresize" eeimg="1">  参数，这些参数可以最大化下一节将介绍的预测性能度量。 特别是，我们发现  <img src="https://www.zhihu.com/equation?tex=\alpha = 0.5" alt="\alpha = 0.5" class="ee_img tr_noresize" eeimg="1">  提供了三个应用场景中预测 aspect 情绪和用户评分的性能之间的最佳平衡

我们在 MacBook Air 1.4 GHz Intel Core i5 上训练了 SULM。 为酒店和美容和水疗应用场景训练模型花费了不到一分钟的时间（~4, 000 条评论），为餐厅应用场景训练 SULM 花费了大约一个小时（~ 480, 000 条评论）。

在餐厅、酒店和美容水疗应用场景上训练模型，我们预测测试数据（评论）上的未知评分和情绪，并确定 aspects 对预测评分的影响。 此外，如第 3.7 节所述，在餐厅、酒店和美容 & 水疗应用的所有 aspect 中，我们确定用户分别拥有 49、14 和 17 个 aspect 的控制权，机构管理层分别拥有超过 54、29 和 31 个 aspect 的控制权。 此外，SULM 提供了从这些已识别集合中体验物品的积极 aspect 或避免体验物品消极 aspect 的建议。

这些实验的结果将在下一节中介绍。

