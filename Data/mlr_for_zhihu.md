# Metric Learning to Rank

用度量学习排序

## 摘要

我们将度量学习作为信息检索问题来研究。我们提出了一种基于结构化 SVM 框架的通用度量学习算法来，来学习一个度量，以便可以针对各种排序度量（例如 AUC、Precision-at-  <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1"> 、MRR、MAP 或 NDCG）来优化由与查询 (query) 的距离引起的数据的排序。 我们展示了标准分类数据集和大规模在线约会推荐问题的实验结果。

## 1 介绍

在许多机器学习任务中，良好的性能取决于对象之间相似性的定义。尽管原始特征的欧几里得距离提供了一个简单且数学上方便的度量，但通常没有理由假设它对于手头的任务是最佳的。 因此，许多研究人员开发了算法来自动学习监督设置中的距离度量。

除了少数例外，这些度量学习算法都遵循同样的指导原则：一个点的好 neighbors 应该比坏 neighbors 更接近自己。 好的和坏的确切定义因问题的设置有所不同，但他们通常从 proximity 和 label agreement 的一些组合中获得。 在这一原则下，度量学习算法一般通过在测试集上 (held out data) 进行 k-NN 算法测试，用label的accuracy作为评估。

在高层次上，如果给定测试点  <img src="https://www.zhihu.com/equation?tex=q" alt="q" class="ee_img tr_noresize" eeimg="1">  ，按与  <img src="https://www.zhihu.com/equation?tex=q" alt="q" class="ee_img tr_noresize" eeimg="1">  的距离升序对训练集进行排序，导致好neighbor  <img src="https://www.zhihu.com/equation?tex=a" alt="a" class="ee_img tr_noresize" eeimg="1">  在列表的前面，而坏neighbor在末尾，我们认为度量是好的。从这个角度来看，我们可以将最近邻预测作为排序问题，并将预测的标签错误率作为排序的损失函数。因此，从本质上讲，度量学习问题是 *query by example* 范式中信息检索的特例。

近年来，排序学习算法的开发取得了许多进展（Joachims, 2005; Burges et al., 2005; Xu & Li, 2007; Volkovs & Zemel, 2009）。与通常由度量学习解决的分类问题不同，排序问题通常缺乏单一的评估标准。 相反，已经提出了几种评估方法，每一种都包含不同的“正确性 (correctness) ”概念。 由于排序本质上是组合对象，因此这些评估指标在模型参数方面通常不可微分，因此难以通过学习算法进行优化。 尽管排序问题存在组合困难，但现在有几种算法技术可以优化各种排序评估方法(Joachims, 2005; Chakrabarti et al., 2008; Volkovs & Zemel, 2009)

在目前的工作中，我们试图减小度量学习和排序的差距。 通过采用来自信息检索的技术，我们推导出了一种通用度量学习算法，该算法针对真正感兴趣的数量 (true quantity of interest) 进行了优化：由学习到的度量中的距离引起的数据排列。

相反，对于许多信息检索应用，包括多媒体推荐，我们用距离度量来参数化排序函数是很自然的。

本方法基于结构化SVM（Tsochantaridis et al.，2005），在一个统一的算法框架下很容易支持各种排序评估方法。将度量学习解释为一个信息检索问题，使我们能够将损失应用于排序级别，而不是pair-wise的距离，并且能够使用比以前的度量学习算法更 general 的相似性概念，包括非对称和不可传递的相关性定义。

### 1.1 相关工作

有大量的研究致力于设计在有监督的环境下学习最优度量的算法。通常，这些算法遵循一个通用方案：学习数据的线性投影（优选低秩），使得到预先决定的“好 neighbors ”集合的距离最小化，而“坏 neighbors ”距离最大化。

Xing等人（2003）将好邻域定义为所有 similarly-labeled 的点，并用半定规划求解度量。相似点对的距离由一个常数规定了上界，并最大化不相似点对的距离。该算法试图将每个类映射成一个固定半径的球，但不强制类之间的分离。

Weinberger等人（2006）将目标 neighbors 定义为原始特征空间中  <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1">  个最近的 similarly-labeled 的点，并强制目标 neighbors 和所有其他（不相似）点之间的距离为正。这放松了Xing等人（2003）的限制，即给定类的所有点必须彼此靠近，并且该算法在许多实际场景中表现良好。然而，如图1所示，依赖于原始特征空间来确定目标 neighbors 可能会使算法不适用于包含噪声或异构特征的问题域：一个有损 (corrupted) 特征可能主导初始距离的计算，并阻止算法找到最佳投影。

<img src="https://raw.githubusercontent.com/wales-z/Markdown4Zhihu/master/Data/mlr_for_zhihu/figure1.png" alt="figure1" style="zoom:80%;" />

邻域成分分析（NCA）（Goldberger et al.，2005）通过在随机邻域选择规则下最大化正确检索点的期望数量来缓解问题。虽然这种松弛有直观的意义，但得到的优化是非凸的，它不能识别和优化学习空间中的前  <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1">  个最近邻。Globerson&Roweis（2006）优化了一个类似的随机邻域选择规则，同时试图将每个类collapse到一个点。这种思想在输出空间上比NCA具有更多的正则性，并导致了一个凸优化问题，但在实际中，整个类可以被collapse到不同的点的假设很少成立。

我们的方法的核心是基于结构化SVM框架（Tsochantaridis等人，2005）。我们在第2节中提供了一个简短的概述，并在第4节中讨论了与排序相关的拓展。

### 1.2 准备工作

令  <img src="https://www.zhihu.com/equation?tex=X \subset {\mathbb R}^d，" alt="X \subset {\mathbb R}^d，" class="ee_img tr_noresize" eeimg="1">  表示训练集（语料库），且  <img src="https://www.zhihu.com/equation?tex=|X|=n" alt="|X|=n" class="ee_img tr_noresize" eeimg="1"> ， <img src="https://www.zhihu.com/equation?tex=Y" alt="Y" class="ee_img tr_noresize" eeimg="1">  表示  <img src="https://www.zhihu.com/equation?tex=X" alt="X" class="ee_img tr_noresize" eeimg="1">  的排列的集合。对于一个 query  <img src="https://www.zhihu.com/equation?tex=q" alt="q" class="ee_img tr_noresize" eeimg="1">  ，让  <img src="https://www.zhihu.com/equation?tex=X_q^+，X_q^-" alt="X_q^+，X_q^-" class="ee_img tr_noresize" eeimg="1"> 分别表示训练集中相关的点和不相关的点的子集。对于排序  <img src="https://www.zhihu.com/equation?tex=y \in Y" alt="y \in Y" class="ee_img tr_noresize" eeimg="1">  和两个点  <img src="https://www.zhihu.com/equation?tex=x,y \in X" alt="x,y \in X" class="ee_img tr_noresize" eeimg="1">  ，用  <img src="https://www.zhihu.com/equation?tex=i \prec_y j(i \succ_y j)" alt="i \prec_y j(i \succ_y j)" class="ee_img tr_noresize" eeimg="1">  表示在  <img src="https://www.zhihu.com/equation?tex=y" alt="y" class="ee_img tr_noresize" eeimg="1">   中， <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1">  排在  <img src="https://www.zhihu.com/equation?tex=j" alt="j" class="ee_img tr_noresize" eeimg="1">  前面 。

 <img src="https://www.zhihu.com/equation?tex=W \succeq 0，W \in {\mathbb R}^{d \times d}" alt="W \succeq 0，W \in {\mathbb R}^{d \times d}" class="ee_img tr_noresize" eeimg="1">  表示一个对称的、半正定矩阵。对于  <img src="https://www.zhihu.com/equation?tex=i,j \in {\mathbb R}" alt="i,j \in {\mathbb R}" class="ee_img tr_noresize" eeimg="1">  ，由  <img src="https://www.zhihu.com/equation?tex=W" alt="W" class="ee_img tr_noresize" eeimg="1">  定义的度量下的距离表示为：  <img src="https://www.zhihu.com/equation?tex=||i-j||_W=\sqrt{(i-j)^T W(i-j)}" alt="||i-j||_W=\sqrt{(i-j)^T W(i-j)}" class="ee_img tr_noresize" eeimg="1">  。对于矩阵  <img src="https://www.zhihu.com/equation?tex=A,B \in {\mathbb R}^{d \times d}" alt="A,B \in {\mathbb R}^{d \times d}" class="ee_img tr_noresize" eeimg="1">  ，将他们的Frobenius内积表示为  <img src="https://www.zhihu.com/equation?tex=\big < A,B\big >_F = {\rm tr}(A^TB)" alt="\big < A,B\big >_F = {\rm tr}(A^TB)" class="ee_img tr_noresize" eeimg="1">  。最后，  <img src="https://www.zhihu.com/equation?tex=\mathbb 1 X" alt="\mathbb 1 X" class="ee_img tr_noresize" eeimg="1">  表示在  <img src="https://www.zhihu.com/equation?tex=X" alt="X" class="ee_img tr_noresize" eeimg="1">  上的 0-1 指示函数。



## 2 结构化SVM

结构化 SVM 可以被视为多分类 SVM（Crammer & Singer，2002）的泛化，其中可能的预测结果集合从label泛化到结构，例如 parse tree、排列、序列对齐等。（Tsochantaridis 等人，2005 ）。  Crammer & Singer (2002) 的多分类 SVM 公式在真实label  <img src="https://www.zhihu.com/equation?tex=y^∗" alt="y^∗" class="ee_img tr_noresize" eeimg="1">  和所有其他label  <img src="https://www.zhihu.com/equation?tex=y" alt="y" class="ee_img tr_noresize" eeimg="1">  之间强制每个训练点  <img src="https://www.zhihu.com/equation?tex=q \in X" alt="q \in X" class="ee_img tr_noresize" eeimg="1">  的边距：

<img src="https://www.zhihu.com/equation?tex=\forall y \neq y^*: w^T_{y^*}q \geq w^T_y +1 -\xi,
" alt="\forall y \neq y^*: w^T_{y^*}q \geq w^T_y +1 -\xi,
" class="ee_img tr_noresize" eeimg="1">
其中  <img src="https://www.zhihu.com/equation?tex=\xi \geq 0" alt="\xi \geq 0" class="ee_img tr_noresize" eeimg="1">  是一个松弛变量，用来允许训练集中的边界违规。类似地，结构化SVM在真实结构  <img src="https://www.zhihu.com/equation?tex=y^*" alt="y^*" class="ee_img tr_noresize" eeimg="1">  和所有其他可能的结构  <img src="https://www.zhihu.com/equation?tex=y" alt="y" class="ee_img tr_noresize" eeimg="1">  之间应用边距：

<img src="https://www.zhihu.com/equation?tex=\forall y \in Y, w^T \psi(q,y^*) \geq w^T \psi(q,y) + \Delta(y^*,y)-\xi.
\tag1
" alt="\forall y \in Y, w^T \psi(q,y^*) \geq w^T \psi(q,y) + \Delta(y^*,y)-\xi.
\tag1
" class="ee_img tr_noresize" eeimg="1">
这里， <img src="https://www.zhihu.com/equation?tex=\psi(q, y)" alt="\psi(q, y)" class="ee_img tr_noresize" eeimg="1">  是向量值联合特征图，它表征输入  <img src="https://www.zhihu.com/equation?tex=q" alt="q" class="ee_img tr_noresize" eeimg="1">  和输出结构  <img src="https://www.zhihu.com/equation?tex=y" alt="y" class="ee_img tr_noresize" eeimg="1">  之间的关系。（此符号包含多分类 SVM 的class-specific的判别向量。）与类标签不同，两个不同的结构  <img src="https://www.zhihu.com/equation?tex=(y ∗ , y)" alt="(y ∗ , y)" class="ee_img tr_noresize" eeimg="1">  可能表现出相似的准确性，边际约束也应反映这一点。 为了支持更灵活的结构正确性的概念，将 margin 设置为  <img src="https://www.zhihu.com/equation?tex=\Delta(y^* , y)" alt="\Delta(y^* , y)" class="ee_img tr_noresize" eeimg="1">  ：在结构之间定义的非负损失函数，通常以 [0, 1] 为值域。

对于多分类 SVM 中的测试query  <img src="https://www.zhihu.com/equation?tex=\hat q" alt="\hat q" class="ee_img tr_noresize" eeimg="1"> ，预测label  <img src="https://www.zhihu.com/equation?tex=y" alt="y" class="ee_img tr_noresize" eeimg="1">  是最大化  <img src="https://www.zhihu.com/equation?tex=w^T_y \hat q" alt="w^T_y \hat q" class="ee_img tr_noresize" eeimg="1">  的label，即与其他label相比具有最大边距的label。 类似地，结构预测是通过找到使  <img src="https://www.zhihu.com/equation?tex=w^T \psi (\hat q, y)" alt="w^T \psi (\hat q, y)" class="ee_img tr_noresize" eeimg="1">  最大化的结构  <img src="https://www.zhihu.com/equation?tex=y" alt="y" class="ee_img tr_noresize" eeimg="1">  来进行的。 在计算输出结构  <img src="https://www.zhihu.com/equation?tex=y" alt="y" class="ee_img tr_noresize" eeimg="1">  时，预测算法必须能够有效地使用学习到的向量  <img src="https://www.zhihu.com/equation?tex=w" alt="w" class="ee_img tr_noresize" eeimg="1">  。正如我们将在第 2.2 节和第 3 节中看到的，这在一般排序中很容易实现，特别是在度量学习中。

### 2.1 优化

请注意，可能的输出结构的集合  <img src="https://www.zhihu.com/equation?tex=Y" alt="Y" class="ee_img tr_noresize" eeimg="1">  通常非常大（例如，训练集的所有可能排列），因此在实践中强制执行 (1) 中的所有边际约束可能不可行。 然而，可以应用切割平面来有效地找到一个小的 active constraints 工作集，这些工作集足以在某些规定的 tolerance 内优化  <img src="https://www.zhihu.com/equation?tex=w" alt="w" class="ee_img tr_noresize" eeimg="1"> （Tsochan taridis 等，2005）。

切割平面方法的核心组件是分离预言机 (separation oracle) ，它给定一个固定的  <img src="https://www.zhihu.com/equation?tex=w" alt="w" class="ee_img tr_noresize" eeimg="1">  和输入点  <img src="https://www.zhihu.com/equation?tex=q" alt="q" class="ee_img tr_noresize" eeimg="1"> ，输出 the structure  <img src="https://www.zhihu.com/equation?tex=y" alt="y" class="ee_img tr_noresize" eeimg="1">  corresponding to the margin constraint for  <img src="https://www.zhihu.com/equation?tex=q" alt="q" class="ee_img tr_noresize" eeimg="1">  which is most violated by  <img src="https://www.zhihu.com/equation?tex=w" alt="w" class="ee_img tr_noresize" eeimg="1"> ：

<img src="https://www.zhihu.com/equation?tex=y \leftarrow {\rm argmax}_{y \in Y} w^T \psi(q,y) + \Delta(y^*,y).
\tag2
" alt="y \leftarrow {\rm argmax}_{y \in Y} w^T \psi(q,y) + \Delta(y^*,y).
\tag2
" class="ee_img tr_noresize" eeimg="1">
直观地说，这计算了结构  <img src="https://www.zhihu.com/equation?tex=y" alt="y" class="ee_img tr_noresize" eeimg="1"> ，同时具有较大的损失  <img src="https://www.zhihu.com/equation?tex=\Delta(y^* , y)" alt="\Delta(y^* , y)" class="ee_img tr_noresize" eeimg="1">  和边际得分  <img src="https://www.zhihu.com/equation?tex=w^T \psi(q,y)" alt="w^T \psi(q,y)" class="ee_img tr_noresize" eeimg="1"> ：简而言之，当前模型  <img src="https://www.zhihu.com/equation?tex=w" alt="w" class="ee_img tr_noresize" eeimg="1">  的缺点。通过关注constraints which are violated the most by the current model，为这些结构添加边距约束可以有效地将优化导向全局最优。

总之，为了将结构化 SVM 应用于学习问题，需要三件事：特征图  <img src="https://www.zhihu.com/equation?tex=\psi" alt="\psi" class="ee_img tr_noresize" eeimg="1">  的定义、损失函数  <img src="https://www.zhihu.com/equation?tex=\Delta" alt="\Delta" class="ee_img tr_noresize" eeimg="1">  和用于分离预言机的有效算法。 这些过程当然都是高度相互依赖和domain-specific的。在下一节中，我们将描述在这种情况下解决排序问题的流行方法。

### 2.2 用结构化SVM排序

在排序的情况下，最常用的特征图是偏序特征（Joachims，2005）：

<img src="https://www.zhihu.com/equation?tex=\psi_{po}(q,y)=\sum_{i \in X^+_q} \sum_{j \in X^-_q} y_{ij}
(\frac{\phi(q,i)-\phi(q,j)}{|X^+_q|\cdot|X^-_q|})
\tag3
" alt="\psi_{po}(q,y)=\sum_{i \in X^+_q} \sum_{j \in X^-_q} y_{ij}
(\frac{\phi(q,i)-\phi(q,j)}{|X^+_q|\cdot|X^-_q|})
\tag3
" class="ee_img tr_noresize" eeimg="1">
其中

<img src="https://www.zhihu.com/equation?tex=y_{ij}=
\begin {cases}
+1 & i \prec_y j\\
-1 & i \succ_y j
\end {cases}
" alt="y_{ij}=
\begin {cases}
+1 & i \prec_y j\\
-1 & i \succ_y j
\end {cases}
" class="ee_img tr_noresize" eeimg="1">
​		且  <img src="https://www.zhihu.com/equation?tex=\phi(q, i)" alt="\phi(q, i)" class="ee_img tr_noresize" eeimg="1">  是表征query  <img src="https://www.zhihu.com/equation?tex=q" alt="q" class="ee_img tr_noresize" eeimg="1">  和点  <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1">  之间关系的特征图。直观地，对于每个相关-不相关对  <img src="https://www.zhihu.com/equation?tex=(i, j)" alt="(i, j)" class="ee_img tr_noresize" eeimg="1">  ，如果  <img src="https://www.zhihu.com/equation?tex=i \prec_y j" alt="i \prec_y j" class="ee_img tr_noresize" eeimg="1"> ，则添加差分向量  <img src="https://www.zhihu.com/equation?tex=\phi(q, i)−\phi(q, j)" alt="\phi(q, i)−\phi(q, j)" class="ee_img tr_noresize" eeimg="1">  否则减去。 从本质上讲， <img src="https://www.zhihu.com/equation?tex=\psi_{po}" alt="\psi_{po}" class="ee_img tr_noresize" eeimg="1">  强调特征空间中的方向，这些方向在某种意义上与正确的排序相关。 由于  <img src="https://www.zhihu.com/equation?tex=\psi" alt="\psi" class="ee_img tr_noresize" eeimg="1">  仅取决于 query 和单个点，而不是整个列表，因此它非常适合结合 domain-specific 的知识和特征。

​		已经有人为  <img src="https://www.zhihu.com/equation?tex=\psi_{po}" alt="\psi_{po}" class="ee_img tr_noresize" eeimg="1">  设计了分离预言机，并结合了各种排序评估方法（Joachims，2005；Yue 等，2007；Chakrabarti 等，2008），我们在第 4 节中给出了简要概述。

​		  <img src="https://www.zhihu.com/equation?tex=\psi_{po}" alt="\psi_{po}" class="ee_img tr_noresize" eeimg="1">  的一个诱人的特性是，对于固定的  <img src="https://www.zhihu.com/equation?tex=w" alt="w" class="ee_img tr_noresize" eeimg="1"> ，最大化  <img src="https://www.zhihu.com/equation?tex=w^T \psi_{po}(\hat q, y)" alt="w^T \psi_{po}(\hat q, y)" class="ee_img tr_noresize" eeimg="1">  的排序  <img src="https://www.zhihu.com/equation?tex=y" alt="y" class="ee_img tr_noresize" eeimg="1">  仅仅是对于  <img src="https://www.zhihu.com/equation?tex=i \in X" alt="i \in X" class="ee_img tr_noresize" eeimg="1"> ，按  <img src="https://www.zhihu.com/equation?tex=w^T\phi (\hat q, i) " alt="w^T\phi (\hat q, i) " class="ee_img tr_noresize" eeimg="1">  降序排序。正如我们将在下一节中展示的，这个简单的预测规则可以很容易地用于基于距离的排序。

## 3 用度量学习来排序

如果query  <img src="https://www.zhihu.com/equation?tex=q" alt="q" class="ee_img tr_noresize" eeimg="1">  与语料库  <img src="https://www.zhihu.com/equation?tex=X" alt="X" class="ee_img tr_noresize" eeimg="1">  位于同一空间，一种自然的排序是：按与  <img src="https://www.zhihu.com/equation?tex=q" alt="q" class="ee_img tr_noresize" eeimg="1">  的距离（的平方）升序来排序： <img src="https://www.zhihu.com/equation?tex=||q − i||^2 " alt="||q − i||^2 " class="ee_img tr_noresize" eeimg="1"> 。 由于我们的目标是学习最佳度量  <img src="https://www.zhihu.com/equation?tex=W" alt="W" class="ee_img tr_noresize" eeimg="1"> ，因此在学习空间中计算距离并相应地排序： <img src="https://www.zhihu.com/equation?tex=||q − i||^2_W" alt="||q − i||^2_W" class="ee_img tr_noresize" eeimg="1">  。 该这一计算基于 Frobenius 内积的表达式如下：

<img src="https://www.zhihu.com/equation?tex=\begin {align}
||q − i||^2_W & =(q-i)^T W(q-i)={\rm tr}(W(q-i)(q-i)^T)\\
&=\big<W, (q-i)(q-i)^T   \big>_F
\end {align}
" alt="\begin {align}
||q − i||^2_W & =(q-i)^T W(q-i)={\rm tr}(W(q-i)(q-i)^T)\\
&=\big<W, (q-i)(q-i)^T   \big>_F
\end {align}
" class="ee_img tr_noresize" eeimg="1">
其中第二个等式基于 trace 的循环性质。
这一观察提示了  <img src="https://www.zhihu.com/equation?tex=\phi" alt="\phi" class="ee_img tr_noresize" eeimg="1">  的一种自然的选择：

<img src="https://www.zhihu.com/equation?tex=\phi_M(q,i) \doteq -(q-i)(q-i)^T
\tag4
" alt="\phi_M(q,i) \doteq -(q-i)(q-i)^T
\tag4
" class="ee_img tr_noresize" eeimg="1">
（符号的变化保留了标准结构化 SVM 中使用的顺序。）因此，将语料库按  <img src="https://www.zhihu.com/equation?tex=||q − i||^2_W " alt="||q − i||^2_W " class="ee_img tr_noresize" eeimg="1"> ​​ 升序排序等效于按  <img src="https://www.zhihu.com/equation?tex=\big< W,\phi_M(q,i)\big >_F" alt="\big< W,\phi_M(q,i)\big >_F" class="ee_img tr_noresize" eeimg="1"> ​​ 降序排序。 类似地，通过将  <img src="https://www.zhihu.com/equation?tex=\phi_M" alt="\phi_M" class="ee_img tr_noresize" eeimg="1"> ​​ 与  <img src="https://www.zhihu.com/equation?tex=\psi_{po}" alt="\psi_{po}" class="ee_img tr_noresize" eeimg="1"> ​​ 一起使用，使泛化的内积  <img src="https://www.zhihu.com/equation?tex=\big< W,\psi_{po}(q,y)\big>_F" alt="\big< W,\psi_{po}(q,y)\big>_F" class="ee_img tr_noresize" eeimg="1"> ​​ 最大化的排序  <img src="https://www.zhihu.com/equation?tex=y" alt="y" class="ee_img tr_noresize" eeimg="1"> ​​ 恰好是  <img src="https://www.zhihu.com/equation?tex=X" alt="X" class="ee_img tr_noresize" eeimg="1"> ​​ 在  <img src="https://www.zhihu.com/equation?tex=W" alt="W" class="ee_img tr_noresize" eeimg="1"> ​​ 定义的度量下按与  <img src="https://www.zhihu.com/equation?tex=q" alt="q" class="ee_img tr_noresize" eeimg="1"> ​​ 的距离升序的排序。

因此，通过将公式 1 和 2 中的向量积推广到 Frobenius 内积，我们可以推导出一种算法来学习一个度量，此度量针对 list-wise 的排序损失测量值进行优化。

### 3.1 算法

理想情况下，我们希望求解最佳度量  <img src="https://www.zhihu.com/equation?tex=W^*" alt="W^*" class="ee_img tr_noresize" eeimg="1"> ，它可以最大化每个query的所有可能排序的边际 (margin) 。然而，由于   <img src="https://www.zhihu.com/equation?tex=|Y|" alt="|Y|" class="ee_img tr_noresize" eeimg="1">   是训练集大小的超指数 (super exponential)，使用当前技术不可能实现精确的优化过程。相反，我们通过使用切割平面算法来近似完全优化。

具体来说，我们用于学习  <img src="https://www.zhihu.com/equation?tex=W" alt="W" class="ee_img tr_noresize" eeimg="1">  的算法改编自 Joachims 等人(2009)的 1-Slack 边际再缩放切割平面 ( 1-Slack margin-rescaling cutting-plane) 算法。 在高层次上，算法交替地优化模型参数（在我们的例子中为  <img src="https://www.zhihu.com/equation?tex=W" alt="W" class="ee_img tr_noresize" eeimg="1"> ）和用新的一批排序 <img src="https://www.zhihu.com/equation?tex=（y_1,y_2,...,y_n)" alt="（y_1,y_2,...,y_n)" class="ee_img tr_noresize" eeimg="1">  更新约束（每个点一个排序）。 一旦新约束批次上的经验损失在前一组约束上的损失的规定tolerance  <img src="https://www.zhihu.com/equation?tex=\epsilon>0" alt="\epsilon>0" class="ee_img tr_noresize" eeimg="1">  范围内，算法就会终止。

1-Slack 方法与其他类似的切割平面技术之间的主要区别在于，不是为每个  <img src="https://www.zhihu.com/equation?tex=q \in X" alt="q \in X" class="ee_img tr_noresize" eeimg="1">  维护一个松弛变量  <img src="https://www.zhihu.com/equation?tex=ξ_q" alt="ξ_q" class="ee_img tr_noresize" eeimg="1"> ，而是在所有约束批次之间共享一个单一的松弛变量  <img src="https://www.zhihu.com/equation?tex=ξ" alt="ξ" class="ee_img tr_noresize" eeimg="1"> ，所有约束批次依次通过平均训练集中的每个点来aggregate。

我们引入了两个修改来使原始算法适应度量学习。 首先， <img src="https://www.zhihu.com/equation?tex=W" alt="W" class="ee_img tr_noresize" eeimg="1">  必须被限制为半正定以定义有效的度量。其次，我们将标准的二次 (quadratic) 正则化  <img src="https://www.zhihu.com/equation?tex=\frac{1}{2}w^Tw" alt="\frac{1}{2}w^Tw" class="ee_img tr_noresize" eeimg="1">  （或   <img src="https://www.zhihu.com/equation?tex=\frac{1}{2}{\rm tr}(W^T W)" alt="\frac{1}{2}{\rm tr}(W^T W)" class="ee_img tr_noresize" eeimg="1">  ）替换为  <img src="https://www.zhihu.com/equation?tex={\rm tr}(W)" alt="{\rm tr}(W)" class="ee_img tr_noresize" eeimg="1"> 。直观地说，这将  <img src="https://www.zhihu.com/equation?tex=W" alt="W" class="ee_img tr_noresize" eeimg="1">  的特征值的  <img src="https://www.zhihu.com/equation?tex=l_2" alt="l_2" class="ee_img tr_noresize" eeimg="1">  penalty交换为  <img src="https://www.zhihu.com/equation?tex=l_1" alt="l_1" class="ee_img tr_noresize" eeimg="1">  penalty，从而促进了对稀疏性和低秩的解决。

一般的优化过程被列为算法 1。 为了紧凑，我们定义

<img src="https://www.zhihu.com/equation?tex=\delta \psi_{po}(q,y^*,y)=\psi_{po}(q,y^*)-\psi_{po}(q,y)
" alt="\delta \psi_{po}(q,y^*,y)=\psi_{po}(q,y^*)-\psi_{po}(q,y)
" class="ee_img tr_noresize" eeimg="1">
<img src="https://raw.githubusercontent.com/wales-z/Markdown4Zhihu/master/Data/mlr_for_zhihu/algorithm1.png" alt="algorithm1" style="zoom:67%;" />

### 3.2 实现

为了解决算法 1 中的优化问题，我们在 MATLAB 中实现了梯度下降求解器。在每个梯度步骤之后，更新后的  <img src="https://www.zhihu.com/equation?tex=W" alt="W" class="ee_img tr_noresize" eeimg="1">  通过谱分解（特征分解）被投影回可行的 PSD 矩阵集

尽管算法中使用了很多特征向量（ <img src="https://www.zhihu.com/equation?tex=\delta \psi_{po}" alt="\delta \psi_{po}" class="ee_img tr_noresize" eeimg="1"> ），但有效的 book-keeping 使我们能够减少梯度计算的开销。请注意，可以将  <img src="https://www.zhihu.com/equation?tex=\xi" alt="\xi" class="ee_img tr_noresize" eeimg="1">  解释为集合  <img src="https://www.zhihu.com/equation?tex=\{ξ1,ξ2,...\}" alt="\{ξ1,ξ2,...\}" class="ee_img tr_noresize" eeimg="1"> 的 point-wise 最大值。其中  <img src="https://www.zhihu.com/equation?tex=ξ_i" alt="ξ_i" class="ee_img tr_noresize" eeimg="1">  对应于第  <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1">  个批次的边际约束。
因此，在  <img src="https://www.zhihu.com/equation?tex=\xi > 0" alt="\xi > 0" class="ee_img tr_noresize" eeimg="1">  的任何时候，目标  <img src="https://www.zhihu.com/equation?tex=f(W, \xi)" alt="f(W, \xi)" class="ee_img tr_noresize" eeimg="1">  的梯度可以用实现当前最大边界违规的单个批次  <img src="https://www.zhihu.com/equation?tex=(\hat y1, ..., \hat yn)" alt="(\hat y1, ..., \hat yn)" class="ee_img tr_noresize" eeimg="1">  表示：

<img src="https://www.zhihu.com/equation?tex=\frac{\partial f}{\partial W}= I-
\frac{C}{n}  \sum_{i=1}^n
\delta \psi_{po}(q_i,y_i^*,\hat y_i)
" alt="\frac{\partial f}{\partial W}= I-
\frac{C}{n}  \sum_{i=1}^n
\delta \psi_{po}(q_i,y_i^*,\hat y_i)
" class="ee_img tr_noresize" eeimg="1">
请注意，  <img src="https://www.zhihu.com/equation?tex=\psi_{po}" alt="\psi_{po}" class="ee_img tr_noresize" eeimg="1">  仅以约束批次的平均值的形式出现在算法 1 中。 这表明对于每个批次，而不是每个点的单独矩阵，只维护一个  <img src="https://www.zhihu.com/equation?tex=d\times d" alt="d\times d" class="ee_img tr_noresize" eeimg="1">  矩阵就足够了

<img src="https://www.zhihu.com/equation?tex=\Psi = \frac{1}{n} \sum_{i=1}^n
\delta \psi_{po}(q_i, y^*_i, y_i)
" alt="\Psi = \frac{1}{n} \sum_{i=1}^n
\delta \psi_{po}(q_i, y^*_i, y_i)
" class="ee_img tr_noresize" eeimg="1">
因为  <img src="https://www.zhihu.com/equation?tex=\phi_M" alt="\phi_M" class="ee_img tr_noresize" eeimg="1">  来自数据的外积，所以每个   <img src="https://www.zhihu.com/equation?tex=\psi_{po}(q,y)" alt="\psi_{po}(q,y)" class="ee_img tr_noresize" eeimg="1">   可以被分解为

<img src="https://www.zhihu.com/equation?tex=\psi_{po}(q,y)=XS(q,y)X^T
" alt="\psi_{po}(q,y)=XS(q,y)X^T
" class="ee_img tr_noresize" eeimg="1">
其中  <img src="https://www.zhihu.com/equation?tex=X" alt="X" class="ee_img tr_noresize" eeimg="1">  的列包含数据， <img src="https://www.zhihu.com/equation?tex=S(q,y)" alt="S(q,y)" class="ee_img tr_noresize" eeimg="1">  是一个对称的  <img src="https://www.zhihu.com/equation?tex=n × n" alt="n × n" class="ee_img tr_noresize" eeimg="1">  矩阵，其中：

<img src="https://www.zhihu.com/equation?tex=S(q,y)=\sum_{i \in X^+_q} \sum_{j \in X^-_q}y_{ij}
\frac{(A_{qi}-A_{qj})}{|X^+_q| \cdot |X^-_q|}
\tag5 \\
A_{qx}=-(e_q-e_x)(e_q-e_x)^T
" alt="S(q,y)=\sum_{i \in X^+_q} \sum_{j \in X^-_q}y_{ij}
\frac{(A_{qi}-A_{qj})}{|X^+_q| \cdot |X^-_q|}
\tag5 \\
A_{qx}=-(e_q-e_x)(e_q-e_x)^T
" class="ee_img tr_noresize" eeimg="1">
 <img src="https://www.zhihu.com/equation?tex=e_i" alt="e_i" class="ee_img tr_noresize" eeimg="1">  是  <img src="https://www.zhihu.com/equation?tex={\mathbb R}^n" alt="{\mathbb R}^n" class="ee_img tr_noresize" eeimg="1">  中的第  <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1">  个标准基向量。通过线性，这种因式分解也可以用于  <img src="https://www.zhihu.com/equation?tex=δ\psi_{po}(q,y^*,y)" alt="δ\psi_{po}(q,y^*,y)" class="ee_img tr_noresize" eeimg="1">  和  <img src="https://www.zhihu.com/equation?tex=\Psi" alt="\Psi" class="ee_img tr_noresize" eeimg="1"> 。

通过计算具有正号和负号的  <img src="https://www.zhihu.com/equation?tex=A_{qx}" alt="A_{qx}" class="ee_img tr_noresize" eeimg="1">  的出现次数并收集这些项，可以更直接地计算公式 5 中的和。 这可以通过单次遍历  <img src="https://www.zhihu.com/equation?tex=y" alt="y" class="ee_img tr_noresize" eeimg="1">  在线性时间内完成。

通过以因式分解形式表示  <img src="https://www.zhihu.com/equation?tex=\Psi" alt="\Psi" class="ee_img tr_noresize" eeimg="1">  ，我们可以将所有矩阵乘法延迟到最终的  <img src="https://www.zhihu.com/equation?tex=\Psi" alt="\Psi" class="ee_img tr_noresize" eeimg="1">  计算。 由于可以直接构造  <img src="https://www.zhihu.com/equation?tex=S(q, y)" alt="S(q, y)" class="ee_img tr_noresize" eeimg="1">  而无需显式构建外积矩阵  <img src="https://www.zhihu.com/equation?tex=A_{qi}" alt="A_{qi}" class="ee_img tr_noresize" eeimg="1"> ，我们将每次梯度计算的矩阵乘法次数从 O(n) 减少到 2。

## 4 排序方法

在这里，我们简要概述了流行的信息检索评估标准，以及如何将它们纳入学习算法。

回顾一下，分离预言机（公式 2）寻求排序  <img src="https://www.zhihu.com/equation?tex=y" alt="y" class="ee_img tr_noresize" eeimg="1"> ，它使判别分数  <img src="https://www.zhihu.com/equation?tex=\big< W,\psi_{po}(q,y)\big>_F" alt="\big< W,\psi_{po}(q,y)\big>_F" class="ee_img tr_noresize" eeimg="1">  和排序损失  <img src="https://www.zhihu.com/equation?tex=\Delta(y^*,y)" alt="\Delta(y^*,y)" class="ee_img tr_noresize" eeimg="1">  之和最大化。
我们考虑的评估标准所共有的一个属性是相关（或不相关）集合内排列的不变性 (invariance)。正如之前所观察到的，对  <img src="https://www.zhihu.com/equation?tex=y" alt="y" class="ee_img tr_noresize" eeimg="1">  的优化简化为找到相关和不相关集合的最佳交织，每个集合都已通过 point-wise 判别分数  <img src="https://www.zhihu.com/equation?tex=\big< W,\phi_{M}(q,i)\big>_F" alt="\big< W,\phi_{M}(q,i)\big>_F" class="ee_img tr_noresize" eeimg="1">  预先排序 （Yue 等人，2007 年）。

由于这里讨论的所有方法都取 [0, 1] 中的值（1 是完美排名的分数），我们考虑以下形式的损失函数

<img src="https://www.zhihu.com/equation?tex=\Delta(y^*,y)={\rm Score}(y^*)-{\rm Score}(y)=1-{\rm Score}(y)
" alt="\Delta(y^*,y)={\rm Score}(y^*)-{\rm Score}(y)=1-{\rm Score}(y)
" class="ee_img tr_noresize" eeimg="1">

### AUC

ROC 曲线下面积 (AUC) 是一种常用的度量，它表征了真阳性 (true positives) 和假阳性 (false positives) 之间随阈值参数变化的权衡。在我们的例子中，此参数对应于返回的物品数（或预测为相关）。 AUC 可以通过从 1 中减去不正确排序对所占比例（即， <img src="https://www.zhihu.com/equation?tex=j\prec_y i" alt="j\prec_y i" class="ee_img tr_noresize" eeimg="1"> ， <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1">  相关而  <img src="https://www.zhihu.com/equation?tex=j" alt="j" class="ee_img tr_noresize" eeimg="1">  不相关）来计算。这个公式导致了一个简单而有效的分离预言机，由 Joachims (2005) 描述 。

请注意，AUC 与位置无关 (position-independent)：列表底部不正确的成对排序对分数的影响与列表顶部的错误一样。 实际上，AUC 是对list-wise cohesion的全局度量。

### Precision-at- <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1"> 

Precision-at- <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1">  (Prec@ <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1"> ) 是相关的结果在返回的前  <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1">  个结果中的分数/所占比例 (fraction)。 因此，Prec@ <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1">  是一个高度本地化的评估标准，它捕获仅前几个结果重要的应用程序的排名质量，例如，网络搜索。
Prec@ <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1">  的分离预言机利用了两个事实：Prec@ <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1">  只有  <img src="https://www.zhihu.com/equation?tex=k + 1" alt="k + 1" class="ee_img tr_noresize" eeimg="1">  个可能的值 <img src="https://www.zhihu.com/equation?tex=(0, 1/k, 2/k, . . , 1)" alt="(0, 1/k, 2/k, . . , 1)" class="ee_img tr_noresize" eeimg="1"> ，并且对于任何固定值，最佳的  <img src="https://www.zhihu.com/equation?tex=y" alt="y" class="ee_img tr_noresize" eeimg="1">  完全由判别分数引起的排序决定。 然后，我们可以评估数据的所有  <img src="https://www.zhihu.com/equation?tex=k + 1" alt="k + 1" class="ee_img tr_noresize" eeimg="1">  次交织 (interleaving)，以找到达到最大值的  <img src="https://www.zhihu.com/equation?tex=y" alt="y" class="ee_img tr_noresize" eeimg="1"> 。 有关详细信息，请参阅 Joachims (2005)。
与 Prec@ <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1">  密切相关的是  <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1"> -最近邻预测分数。 在二分类设置中，两者通过下式联系起来：

<img src="https://raw.githubusercontent.com/wales-z/Markdown4Zhihu/master/Data/mlr_for_zhihu/knn.png" alt="knn" style="zoom: 67%;" />

而且 Prec@ <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1">  分离预言机可以很容易地适应  <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1"> -最近邻。 然而，在多分类设置下，交织技术失败了，因为正确分类所需的相关点分数不仅取决于每个点的相关性或不相关性，还取决于标签本身。

在非正式实验中，我们注意到为（二进制）KNN 和 Prec@ <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1">  训练的指标之间的性能没有量化的差异，我们在第 5 节的实验中省略了 KNN。

### Average Precision

Average Precision（或Mean Average Precision，MAP）（Baeza-Yates 和 Ribeiro-Neto，1999）是排序  <img src="https://www.zhihu.com/equation?tex=y" alt="y" class="ee_img tr_noresize" eeimg="1">  的Precision-at- <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1">  分数，在相关文档的所有位置  <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1">  上取平均值：

<img src="https://raw.githubusercontent.com/wales-z/Markdown4Zhihu/master/Data/mlr_for_zhihu/AP.png" alt="AP" style="zoom:67%;" />

Yue等人 (2007) 提供了一个Average Precision的贪心 (greedy) 分离预言机，运行时间为  <img src="https://www.zhihu.com/equation?tex=O(|X^+ _q|·|X^−_q|)" alt="O(|X^+ _q|·|X^−_q|)" class="ee_img tr_noresize" eeimg="1"> 。
我们的实现使用了一种相对简单的动态编程方法，它具有等效的渐近运行时间。（为简洁起见，此处省略了详细信息。）

### Mean Reciprocal Rank

Mean Reciprocal Rank (MRR) 是  <img src="https://www.zhihu.com/equation?tex=y" alt="y" class="ee_img tr_noresize" eeimg="1">  中第一个相关文档的倒数，因此非常适合仅第一个结果重要的应用程序。
与 Prec@ <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1">  一样，对于 MRR ，有一组有限的可能得分值 <img src="https://www.zhihu.com/equation?tex=(1,1/2,1/3,...,1/(1+|X^−_q|))" alt="(1,1/2,1/3,...,1/(1+|X^−_q|))" class="ee_img tr_noresize" eeimg="1"> ，并且对于固定的 MRR 得分，最优  <img src="https://www.zhihu.com/equation?tex=y" alt="y" class="ee_img tr_noresize" eeimg="1">  完全确定。 搜索maximizer的得分值也同样简单直接。 对优化 MRR 的更完整处理见 Chakrabarti 等人(2008) 

### Normalized Discounted Cumulative Gain

Normalized Discounted Cumulative Gain (NDCG) (Jarvelin & Kekalainen, 2000) 类似于 MRR，但不是只奖励第一个相关文档，而是所有前  <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1">  个文档都以递减的折扣因子 (discount factor) 计分。在当前具有二元相关性级别的设置中，我们采用的公式表示为：

<img src="https://raw.githubusercontent.com/wales-z/Markdown4Zhihu/master/Data/mlr_for_zhihu/ndcg.png" alt="ndcg" style="zoom:67%;" />

Chakrabarti等人 (2008) 提出了一种用于 NDCG 分离预言机的动态规划算法，我们在这里采用了该算法。

## 5 实验

为了评估 MLR 算法，我们在小规模和大规模数据集上进行了实验，如下两节所述。 在所有实验中，我们将accuracy 阈值固定为  <img src="https://www.zhihu.com/equation?tex=\epsilon = 0.01" alt="\epsilon = 0.01" class="ee_img tr_noresize" eeimg="1">  。

### 5.1 在UCI数据上的分类

我们首先在来自 UCI 存储库 (Asuncion & Newman, 2007) 的五个数据集 (Balance、Ionosphere、WDBC、Wine 和 IsoLet) 上测试了我们算法的 accuracy 和降维性能。 对于前四组，我们生成了 50 个随机的 80/20 训练和测试分割。 数据的每个维度都由训练集的统计数据进行 z-score。

对于 IsoLet，我们重复了 Weinberger 等人(2006)的实验，通过生成训练集的 10 个随机 80/20 分割用于测试和验证，然后在提供的测试集上进行测试。 我们通过 PCA（根据训练集计算得出）投影到 170 个维度，足以捕获 95% 的方差。

表 1 包含使用的数据集的摘要。

<img src="https://raw.githubusercontent.com/wales-z/Markdown4Zhihu/master/Data/mlr_for_zhihu/table1.png" alt="table1" style="zoom:67%;" />

我们使用 MLR 的五种变体在每个数据集上训练指标：MLR-AUC、MLR-Prec@ <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1"> 、MLR-MAP、MLR-MRR 和 MLR-NDCG。为了比较，我们还使用大边距最近邻 (LMNN) (Weinberger et al., 2006)、邻域组件分析 (NCA) 和 Collapsing Classes 度量学习 (MLCC）。

为了评估每种算法的性能，我们在学习的度量中测试了  <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1">  最近邻分类的准确性。 分类结果见表2 。 除了 Balance 集上的 NCA 和 MLCC 之外，Balance、Ionosphere、WDBC 和 Wine 的所有结果都在误差范围内。 一般来说，MLR 的准确度与比较中的最佳算法相当，而不依赖于选择目标neighbors的输入特征。

<img src="https://raw.githubusercontent.com/wales-z/Markdown4Zhihu/master/Data/mlr_for_zhihu/table2.png" alt="table2" style="zoom:67%;" />

图 2 说明了 MLR 算法的降维特性。 在所有情况下，MLR 都实现了输入空间维度的显着降低，可与最佳竞争算法相媲美。

<img src="https://raw.githubusercontent.com/wales-z/Markdown4Zhihu/master/Data/mlr_for_zhihu/figure2.png" alt="figure2" style="zoom:67%;" />

### 5.2 eHarmony 数据

为了在信息检索上下文中的大型数据集上评估 MLR，我们对 eHarmony 提供的匹配数据的训练了指标：(eHarmony 是一种通过个性特征匹配用户的在线约会服务)。

对于我们的实验，我们专注于数据和问题的以下简化：每个匹配都以一对用户呈现，匹配成功时带有正标签（即用户表达了共同兴趣），否则带有负标签。 每个用户由  <img src="https://www.zhihu.com/equation?tex={\mathbb R}^{56}" alt="{\mathbb R}^{56}" class="ee_img tr_noresize" eeimg="1">  中的一个向量表示，该向量描述了用户的个性、兴趣等。如果两个用户被呈现为成功匹配，我们认为两个用户是相互相关的，如果匹配不成功，我们认为两个用户是不相关的。对于不匹配的对，不假设不相关。

在两个等长的连续时间间隔内收集匹配信息，并分为训练（间隔 1）和测试（间隔 2）。 训练集包含大约 295000 个唯一用户，并非所有用户都定义了有用的queries：一些只出现在正匹配中，而其他只出现在负匹配中。由于这些用户不提供判别数据，我们从 query 用户集中省略了他们。 请注意，此类用户仍然是信息丰富的，并且作为要排序的结果包含在训练集中。

我们进一步减少了训练查询的数量，只包括至少有 2 个成功匹配和 5 个不成功匹配的用户，留下大约 22000 个训练 query。 数据汇总见表 3。

<img src="https://raw.githubusercontent.com/wales-z/Markdown4Zhihu/master/Data/mlr_for_zhihu/table3.png" alt="table3" style="zoom:67%;" />

我们使用 MLR-AUC、MLR-MAP 和 MLR-MRR 训练指标。由于每个query的最小正 (positive) 结果数量很少，我们在这个实验中省略了 MLR-P@ <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1">  和 MLR-NDCG。 请注意，由于我们做的是信息检索，而不是分类，所以上一节中比较的其他度量学习算法不适用。为了比较，我们使用 SVM-MAP (Yue et al., 2007) 和特征图  <img src="https://www.zhihu.com/equation?tex=\phi(q,i)=(q−i)" alt="\phi(q,i)=(q−i)" class="ee_img tr_noresize" eeimg="1">  训练模型。 在训练 SVM-MAP 时，我们使用  <img src="https://www.zhihu.com/equation?tex=C \in \{10^{−2}, 10^{−1},...,10^5\}" alt="C \in \{10^{−2}, 10^{−1},...,10^5\}" class="ee_img tr_noresize" eeimg="1"> 。

表 4 显示了 MLR 和 SVM-MAP 的精度和时间结果。  MLR-MAP 和 MLR MRR 模型比 SVM-MAP 模型显示出轻微但统计意义上显着的改进。 请注意，MLR 算法的训练时间比 SVM-MAP 少得多，并且对分离预言机的调用更少。

<img src="https://raw.githubusercontent.com/wales-z/Markdown4Zhihu/master/Data/mlr_for_zhihu/table4.png" alt="table4" style="zoom:67%;" />

尽管在此检索任务中，MLR 比 baseline 欧几里得距离有所改进，但似乎线性模型可能不足以捕获数据中的复杂结构。将 MLR 推广到产生非线性变换将是未来研究的重点。

## 6 结论

我们提出了一种度量学习算法，该算法针对基于排序的损失函数进行了优化。通过将问题作为信息检索任务，我们将注意力集中在我们认为感兴趣的关键数量上：由距离引起的数据排列。

