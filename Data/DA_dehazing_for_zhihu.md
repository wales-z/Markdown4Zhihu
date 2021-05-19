# 【论文翻译】DA_dahazing: Domain Adaptation for Image Dehazing

## 摘要

- 近年来，使用基于学习的方法进行图像去雾已经达到了最先进的性能。

- 然而，大多数现有方法在合成模糊图像上训练去雾模型，由于域迁移（domain shift)，这些模型面对真实的模糊图像泛化（generalize)能力不强。为了解决这个问题，我们提出了一种领域适应范式(范例？paradigm)，它由一个图像翻译模块和两个图像去雾模块组成。


- 具体来说，我们首先应用双向翻译网络，通过将图像从一个域转换到另一个域来缩小合成域和真实域之间的差距。然后，我们使用**翻译前后**的图像来训练这两个具有一致性约束的图像去雾网络。在这一阶段，我们利用清晰图像的属性（例如，暗通道先验(prior)和图像梯度平滑）将真实的模糊图像纳入到去雾训练中，以进一步提高域适应性。


- 通过以端到端的方式训练图像翻译和去雾网络，我们可以同时获得更好的图像翻译和除雾效果。在合成图像和真实世界图像上的实验结果表明，我们的模型性能优于目前最先进的除雾算法。

## 1 介绍

- 单个图像去雾的目的是从有雾的输入中恢复干净的图像，这对于后续的高级任务（例如对象识别和场景理解）必不可少。因此，在过去的几年中，它已经在视觉界引起了极大的关注。

- 根据物理散射模型[21，23，18]，通常将雾化过程公式为：

<img src="https://www.zhihu.com/equation?tex=I(x)=J(x)t(x)+A(1-t(x))\tag1
  " alt="I(x)=J(x)t(x)+A(1-t(x))\tag1
  " class="ee_img tr_noresize" eeimg="1">
  
  其中  <img src="https://www.zhihu.com/equation?tex=I(x)" alt="I(x)" class="ee_img tr_noresize" eeimg="1">  和  <img src="https://www.zhihu.com/equation?tex=J(x)" alt="J(x)" class="ee_img tr_noresize" eeimg="1">  代表有雾图像和干净图像，  <img src="https://www.zhihu.com/equation?tex=A" alt="A" class="ee_img tr_noresize" eeimg="1">  是全局大气光（global atmospheric ligt)， <img src="https://www.zhihu.com/equation?tex=t(x)" alt="t(x)" class="ee_img tr_noresize" eeimg="1">  是介质传输图（transmission map).   <img src="https://www.zhihu.com/equation?tex=t(x)" alt="t(x)" class="ee_img tr_noresize" eeimg="1">  可以表示为  <img src="https://www.zhihu.com/equation?tex=t(x)=e^{\beta d(x)}" alt="t(x)=e^{\beta d(x)}" class="ee_img tr_noresize" eeimg="1"> ，其中 <img src="https://www.zhihu.com/equation?tex=d(x)" alt="d(x)" class="ee_img tr_noresize" eeimg="1">  和  <img src="https://www.zhihu.com/equation?tex=\beta" alt="\beta" class="ee_img tr_noresize" eeimg="1">  分别表示景深（scene depth)和大气散射参数（atmosphere scattering parameter）。给定有雾图像  <img src="https://www.zhihu.com/equation?tex=I(x)" alt="I(x)" class="ee_img tr_noresize" eeimg="1">  ，大多数去雾算法都是尽量去估算  <img src="https://www.zhihu.com/equation?tex=t(x)" alt="t(x)" class="ee_img tr_noresize" eeimg="1">  和  <img src="https://www.zhihu.com/equation?tex=A" alt="A" class="ee_img tr_noresize" eeimg="1"> 。

- 然而，从模糊图像估计透射图通常是**不适定问题**（ill-posed problem)。早期的基于先验的方法尝试通过利用清晰图像的统计特性（例如，暗通道先验[9]和色线(color line)先验[8]）来估计透射图。不幸的是，这些图像先验很容易与实践不一致，这可能导致不正确的传输近似值（transmission approximations）。因此，恢复的图像的质量是不太理想的。

- 为了解决这个问题，卷积神经网络（CNN)开始被用于估计传输[4、26、35]或直接预测清晰的图像[12、27、16、25]。这些方法有效且优于基于先验的算法，并具有显着的性能提升。但是，基于深度学习的方法需要依赖大量真实的有雾图像及其无雾的对应图像进行训练。而通常，在现实世界中获取大量这样的真实图像是不切实际的。因此，大多数除雾模型都依靠合成去雾图像的数据集进行训练。但是，由于域迁移问题，从合成数据中学到的模型通常无法很好地泛化到实际数据。

- 为了解决上面这个问题，我们提出了一种用于单图像去雾的域自适应框架。框架包括两部分，称为图像翻译模块和两个与域相关的去雾模块（一个用于合成域，另一个用于真实域）。

  - 为了减少域之间的差异，我们的方法首先使用双向图像翻译网络将图像从一个域翻译到另一个域。由于图像雾度是一种噪声并且高度不均匀，取决于景深，因此我们将深度信息纳入转换网络，以指导合成图像翻译为真实有雾图像的翻译过程。

  - 然后，域相关的去雾网络将包括原始图像和翻译后图像在内的该域的图像作为输入以执行图像去雾。此外，我们使用一致性损失（consistency loss）来确保两个去雾网络生成一致的结果。

  - 在此训练阶段，为了进一步提高网络在真实域中的泛化能力，我们将真实的有雾图像纳入到训练中。我们希望真实的有雾图像的去雾结果能够具有清晰图像的一些特性，例如暗通道先验和图像梯度平滑。

  - 我们以端到端的方式训练图像翻译网络和除雾网络，以便它们可以互相改进。
    如图1所示，与最近EPDN的去雾效果相比，我们的模型产生的图像更清晰[25]。

    ![figure1](.\figure1.png)

<center>图1 真实有雾图像的去雾结果</center>

- 我们将本文的贡献总结如下：
  - 我们提出了一种用于图像去雾的端到端域自适应框架，该框架有效地减小了合成和真实世界有雾图像之间的差异
  - 我们的结果表明将真实的有雾图像合并到训练过程中可以提高去雾性能。
  - 我们在合成数据集和真实世界的有雾图像上进行了大量实验，证明了所提出的方法的去雾性能优于目前最先进的去雾方法。

## 2 相关工作

本节简要讨论与我们的工作有关的单个图像去雾方法和域自适应方法。

### 2.1 单图像去雾

- **基于先验的方法。**基于先验的方法借助清晰图像的统计信息来估计介质传输图和大气光强度。在这方面的代表性作品包括[29、9、39、8、2]。具体来说，Tan [29]提出了一种用于图像去雾的对比度最大化方法，因为它观察到清晰的图像往往比其有雾的对应图像具有更高的对比度。He等人[9]利用暗通道先验（DCP）来估计介质传输图，该图基于以下假设：在至少一个颜色通道中，无雾斑块（haze-free patches)中的像素值接近零。后续工作提高了DCP方法的效率和性能[30，22，17，24，34]。此外，在[39]中采用衰减（attenuation）先验来恢复有雾图像的深度信息。 Fattal [8]使用色线假设来恢复场景传输（scene transmission)，他断言小图像块的像素呈现一维分布。同样，Berman等人[2]假设数百种不同的颜色可以很好地逼近清晰图像的颜色，然后基于此先验进行图像去雾。尽管已经证明这些方法对于图像去雾是有效的，但是由于假定的先验并不适合于所有真实图像，因此它们的性能固有地受到限制。

- **基于学习的方法。**随着深度卷积神经网络（CNN）的进步以及大规模合成数据集的可用性（availability），近年来，数据驱动的图像去雾方法受到了广泛的关注。许多方法[4，26，35，12]直接利用深层CNN来估计介质传输图和大气光，然后根据**退化模型**（1）恢复清晰图像。蔡等人[4]提出了端到端的除雾模型DehazeNet，从有雾的图像中估计传输图。任等人 [26]利用一种从粗到精（coarse-to-fine)的策略来学习有雾的输入和介质传输图的映射。 Zhang 和 Patel [35]提出了一个密集连接的金字塔网络来估计介质传输图。 Li等人[12]提出了一个AOD-Net来估计重新构造的物理散射模型的参数，该模型综合了传输和大气光。而且，已经提出了一些端到端方法[27、16、25]，以直接恢复干净的图像，而不是估计介质传输图和大气光。任等人[27]采用门控融合（gate fusion）网络直接从有雾的输入中恢复干净的图像。 Qu等人[25]将图像去雾问题转化为图像到图像翻译问题，并提出了一种增强的pix2pix去雾网络。

- 但是，由于合成数据与真实数据之间的域差距，在合成图像上训练的基于CNN的模型在应用于真实域时往往会出现明显的性能下降。为此，李等人[14]提出了一种半监督除雾模型，该模型在合成和真实有雾图像上都经过训练，因此享有合成和真实有雾图像之间的域适应性。然而，仅将真实的雾度图像应用于训练并不能真正解决域偏移的问题。
  与上述方法不同，我们的模型首先应用图像翻译网络将图像从一个域翻译到另一域，然后使用翻译后的图像及其原始图像（合成图像或真实图像）在合成域和真实域上执行图像去雾处理。我们提出的方法可以有效地解决域移位问题。

### 2.2 域自适应性

- 域自适应旨在减少不同域之间的差异[1、6、20]。现有的工作要么做特征级别要么像素级的自适应工作。
  - 特征级适应方法旨在通过最小化最大平均差异[19]（maximum mean discrepancy）或在特征空间上应用对抗性学习策略[32，31]来调整源域和目标域之间的特征分布。
  - 另一研究重点是像素级自适应[3，28，7]。这些方法通过应用图像到图像的翻译[3，28]学习或样式迁移[7]（style transfer）方法来增加目标域中的数据，从而解决了域偏移问题。

- 最近，很多方法在许多视觉任务中共同执行特征级和像素级自适应，例如图像分类[10]，语义分割[5]和深度预测[37]。这些方法[5、37]通过图像到图像转换网络，以像素级自适应将图像从一个域转换到另一域，然后将翻译后的图像以特征级别的对齐方式输入到任务网络，例如CycleGAN [38]。
- 在我们的工作中，我们利用CycleGAN来使真实的有雾图像适应我们在合成数据上训练的去雾模型。此外，由于**深度信息**与图像雾度的形成密切相关，因此我们将深度信息纳入到翻译网络中，以更好地指导真实有雾图像翻译。

## 3 提出的方法

本节介绍了我们的域适应框架的详细信息。首先，我们概述了我们的方法，然后描述图像翻译模块和图像去雾模块的细节，最后，给出了用于训练网络的损失函数。

### 3.1 方法总览

- 给定合成数据集 <img src="https://www.zhihu.com/equation?tex=X_s=\{x_s,y_s\}^{N_l}_{s=1}" alt="X_s=\{x_s,y_s\}^{N_l}_{s=1}" class="ee_img tr_noresize" eeimg="1"> 和真实有雾图像集 <img src="https://www.zhihu.com/equation?tex=X_R=\{x_r\}^{N_u}_{r=1}" alt="X_R=\{x_r\}^{N_u}_{r=1}" class="ee_img tr_noresize" eeimg="1"> ，其中 <img src="https://www.zhihu.com/equation?tex=N_l" alt="N_l" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=N_u" alt="N_u" class="ee_img tr_noresize" eeimg="1"> 分别表示合成图像和真实模糊图像的数量。我们的目标是学习一个单一图像去雾模型，该模型可以从真实的有雾图像中准确预测出清晰的图像。由于域移位，仅在合成数据上训练的除雾模型无法很好地推广到真实的模糊图像。
- 为了解决这个问题，我们提出了一种域自适应框架，该框架包括两个主要部分：图像翻译网络 <img src="https://www.zhihu.com/equation?tex=G_{S→R}" alt="G_{S→R}" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=G_{R→S}" alt="G_{R→S}" class="ee_img tr_noresize" eeimg="1"> ，以及两个除雾网络 <img src="https://www.zhihu.com/equation?tex=\mathcal G_S" alt="\mathcal G_S" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=\mathcal G_R" alt="\mathcal G_R" class="ee_img tr_noresize" eeimg="1"> 。图像翻译网络将图像从一个域翻译到另一个域，以减小它们之间的差异。然后，去雾网络使用翻译后的图像和源图像（例如，合成的或真实的）执行图像去雾。
- 如图2所示，所提出的模型将真实的模糊图像 <img src="https://www.zhihu.com/equation?tex=x_r" alt="x_r" class="ee_img tr_noresize" eeimg="1"> 和合成图像 <img src="https://www.zhihu.com/equation?tex=x_s" alt="x_s" class="ee_img tr_noresize" eeimg="1"> 及其对应的深度图像 <img src="https://www.zhihu.com/equation?tex=d_s" alt="d_s" class="ee_img tr_noresize" eeimg="1"> 用作输入。我们首先使用两个图像翻译器获得相应的转换图像 <img src="https://www.zhihu.com/equation?tex=x_{s→r}=G_{S→R}(x_s,d_s)" alt="x_{s→r}=G_{S→R}(x_s,d_s)" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=x_{r→s}=G_{R→S}(x_r)" alt="x_{r→s}=G_{R→S}(x_r)" class="ee_img tr_noresize" eeimg="1"> 。然后，将 <img src="https://www.zhihu.com/equation?tex=x_s" alt="x_s" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=x_{r→s}" alt="x_{r→s}" class="ee_img tr_noresize" eeimg="1"> 传递给 <img src="https://www.zhihu.com/equation?tex=\mathcal G_S" alt="\mathcal G_S" class="ee_img tr_noresize" eeimg="1"> ，将 <img src="https://www.zhihu.com/equation?tex=x_r" alt="x_r" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=x_{r→s}" alt="x_{r→s}" class="ee_img tr_noresize" eeimg="1"> 传递给 <img src="https://www.zhihu.com/equation?tex=\mathcal G_R" alt="\mathcal G_R" class="ee_img tr_noresize" eeimg="1"> 进行图像去雾。

![figure2](.\figure2.png)

<center>图2 本文提出的用于图像去雾的域自适应框架的架构。</center>

### 3.2 图像翻译模块

- 图像翻译模块包括两个翻译器：从合成到真实的网络 <img src="https://www.zhihu.com/equation?tex=G_{S→R}" alt="G_{S→R}" class="ee_img tr_noresize" eeimg="1"> 和从真实到合成的网络 <img src="https://www.zhihu.com/equation?tex=G_{R→S}" alt="G_{R→S}" class="ee_img tr_noresize" eeimg="1"> 。  <img src="https://www.zhihu.com/equation?tex=G_{S→R}" alt="G_{S→R}" class="ee_img tr_noresize" eeimg="1"> 网络以 <img src="https://www.zhihu.com/equation?tex=（X_s，D_s）" alt="（X_s，D_s）" class="ee_img tr_noresize" eeimg="1"> 作为输入，并生成样式与真实有雾图像相似的翻译图像 <img src="https://www.zhihu.com/equation?tex=G_{S→R}（Xs，Ds）" alt="G_{S→R}（Xs，Ds）" class="ee_img tr_noresize" eeimg="1"> 。另一个翻译器 <img src="https://www.zhihu.com/equation?tex=G_{R→S}" alt="G_{R→S}" class="ee_img tr_noresize" eeimg="1"> 执行的是逆向的图像翻译。由于深度信息与雾度公式高度相关，因此我们将其纳入到生成器 <img src="https://www.zhihu.com/equation?tex=G_{S→R}" alt="G_{S→R}" class="ee_img tr_noresize" eeimg="1"> 中，以在实际情况下生成具有相似雾度分布的图像。

- 我们采用空间特征变换（SFT）层[33，15]将深度信息纳入到翻译网络中，这可以有效地融合深度图和合成图像中的特征。

- 如图3所示，SFT层首先应用三个卷积层以从深度图提取条件图 <img src="https://www.zhihu.com/equation?tex=\phi" alt="\phi" class="ee_img tr_noresize" eeimg="1"> 。然后将条件图(conditional maps）馈送到其他两个卷积层以分别预测调制参数 <img src="https://www.zhihu.com/equation?tex=\gamma" alt="\gamma" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=\beta" alt="\beta" class="ee_img tr_noresize" eeimg="1"> 。最后，我们可以通过以下方式获得输出移位特征（output shifted features）：

<img src="https://www.zhihu.com/equation?tex=SFT(F|\gamma,\beta)=\gamma⊙F+\beta\tag2
  " alt="SFT(F|\gamma,\beta)=\gamma⊙F+\beta\tag2
  " class="ee_img tr_noresize" eeimg="1">
  其中 <img src="https://www.zhihu.com/equation?tex=⊙" alt="⊙" class="ee_img tr_noresize" eeimg="1"> 是按元素的乘法。在翻译器 <img src="https://www.zhihu.com/equation?tex=G_{S→R}" alt="G_{S→R}" class="ee_img tr_noresize" eeimg="1"> 中，我们将深度图作为指导，并使用SFT层来变换倒数第二层卷积层的特征。

  ![figure3](.\figure3.png)

  <center>图3 STL层的结构</center>

  如图4所示，翻译后，合成图像相对更接近真实世界的有雾图像。

![figure4](.\figure4.png)

<center>图4 在两个合成有雾图像上的翻译结果。</center>

- 我们在去雾中展示了翻译器 <img src="https://www.zhihu.com/equation?tex=G_{S→R}" alt="G_{S→R}" class="ee_img tr_noresize" eeimg="1"> 的详细配置。我们还采用了CycleGAN [38]提供的架构，用于生成器 <img src="https://www.zhihu.com/equation?tex=G_{R→S}" alt="G_{R→S}" class="ee_img tr_noresize" eeimg="1"> 和鉴别器(discriminators)（ <img src="https://www.zhihu.com/equation?tex=D^{img}_R" alt="D^{img}_R" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=D^{img}_S" alt="D^{img}_S" class="ee_img tr_noresize" eeimg="1"> ）。

<center>去雾 图像翻译模块的配置。“ Conv”表示卷积层，“ Res”表示残差块，“ Upconv”表示通过转置卷积算子的上采样层，“ Tanh”表示非线性Tanh层。</center>

![table1](.\table1.png)



### 3.3 去雾模块

我们的方法包括两个去雾模块 <img src="https://www.zhihu.com/equation?tex=\mathcal G_S" alt="\mathcal G_S" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=\mathcal G_R" alt="\mathcal G_R" class="ee_img tr_noresize" eeimg="1"> ，分别在合成域和真实域上执行图像去雾。  <img src="https://www.zhihu.com/equation?tex=\mathcal G_S" alt="\mathcal G_S" class="ee_img tr_noresize" eeimg="1"> 将合成图像 <img src="https://www.zhihu.com/equation?tex=x_s" alt="x_s" class="ee_img tr_noresize" eeimg="1"> 和转换后的图像 <img src="https://www.zhihu.com/equation?tex=x_{r→s}" alt="x_{r→s}" class="ee_img tr_noresize" eeimg="1"> 作为输入来执行图像去雾。  <img src="https://www.zhihu.com/equation?tex=\mathcal G_R" alt="\mathcal G_R" class="ee_img tr_noresize" eeimg="1"> 则用 <img src="https://www.zhihu.com/equation?tex=x_r" alt="x_r" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=x_{s→r}" alt="x_{s→r}" class="ee_img tr_noresize" eeimg="1"> 训练。对于这两个图像去雾网络，我们都使用标准的编码器-解码器体系结构，其跳过连接（skip connections）和侧输出（side outputs）同[37]。每个域中的除雾网络共享相同的网络架构，但具有不同的学习参数。

### 3.4 训练误差

在域适应框架中，我们采用以下损失来训练网络。

#### 图像翻译损失

- 我们的翻译模块的目的是学习翻译器 <img src="https://www.zhihu.com/equation?tex=G_{S→R}" alt="G_{S→R}" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=G_{R→S}" alt="G_{R→S}" class="ee_img tr_noresize" eeimg="1"> ，以减少合成域 <img src="https://www.zhihu.com/equation?tex=X_S" alt="X_S" class="ee_img tr_noresize" eeimg="1"> 和真实域 <img src="https://www.zhihu.com/equation?tex=X_R" alt="X_R" class="ee_img tr_noresize" eeimg="1"> 之间的差异。对于翻译器 <img src="https://www.zhihu.com/equation?tex=G_{S→R}" alt="G_{S→R}" class="ee_img tr_noresize" eeimg="1"> ，我们希望的效果是 <img src="https://www.zhihu.com/equation?tex=G_{S→R}（x_s,d_s)" alt="G_{S→R}（x_s,d_s)" class="ee_img tr_noresize" eeimg="1"> 与真实的模糊图像 <img src="https://www.zhihu.com/equation?tex=x_r" alt="x_r" class="ee_img tr_noresize" eeimg="1"> 不能区分。因此，我们采用图像级鉴别器 <img src="https://www.zhihu.com/equation?tex=D^{img}_R" alt="D^{img}_R" class="ee_img tr_noresize" eeimg="1"> 和特征级鉴别器 <img src="https://www.zhihu.com/equation?tex=D^{feat}_R" alt="D^{feat}_R" class="ee_img tr_noresize" eeimg="1"> ，通过对抗性学习方式进行**minmax game**。  <img src="https://www.zhihu.com/equation?tex=D^{img}_R" alt="D^{img}_R" class="ee_img tr_noresize" eeimg="1"> 的目的是校准真实图像 <img src="https://www.zhihu.com/equation?tex=x_r" alt="x_r" class="ee_img tr_noresize" eeimg="1"> 和转换后的图像 <img src="https://www.zhihu.com/equation?tex=G_{S→R}（x_s,d_s)" alt="G_{S→R}（x_s,d_s)" class="ee_img tr_noresize" eeimg="1"> 之间的分布。
  鉴别器 <img src="https://www.zhihu.com/equation?tex=D^{feat}_R" alt="D^{feat}_R" class="ee_img tr_noresize" eeimg="1"> 帮助校准 <img src="https://www.zhihu.com/equation?tex=x_r" alt="x_r" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=G_{S→R}（x_s,d_s)" alt="G_{S→R}（x_s,d_s)" class="ee_img tr_noresize" eeimg="1"> 的特征图（feature map）之间的分布。
  定义对抗性损失（adversarial losses）为：


<img src="https://www.zhihu.com/equation?tex=\begin{align}
L&^{img}_{Gan}(X_R,(X_S,D_S),D^{img}_R,G_{S→R})\\
&=\mathbb{E}_{x_s\sim X_S,d_s\sim D_S}[D^{img}_R(G_{S→R}(x_s,d_s))]\\
&+\mathbb{E}_{x_r\sim X_R}[D^{img}_R(x_r)-1]\tag3
\end{align}
" alt="\begin{align}
L&^{img}_{Gan}(X_R,(X_S,D_S),D^{img}_R,G_{S→R})\\
&=\mathbb{E}_{x_s\sim X_S,d_s\sim D_S}[D^{img}_R(G_{S→R}(x_s,d_s))]\\
&+\mathbb{E}_{x_r\sim X_R}[D^{img}_R(x_r)-1]\tag3
\end{align}
" class="ee_img tr_noresize" eeimg="1">


<img src="https://www.zhihu.com/equation?tex=\begin{align}
L&^{feat}_{Gan}(X_R,(X_S,D_S),D^{feat}_R,G_{S→R},\mathcal G_R)\\
&=\mathbb{E}_{x_s\sim X_S,d_s\sim D_S}[D^{feat}_R(\mathcal G_R(G_{S→R}(x_s,d_s)))]\\
&+\mathbb{E}_{x_r\sim X_R}[D^{feat}_R(\mathcal G_R(x_r))-1]\tag4
\end{align}
" alt="\begin{align}
L&^{feat}_{Gan}(X_R,(X_S,D_S),D^{feat}_R,G_{S→R},\mathcal G_R)\\
&=\mathbb{E}_{x_s\sim X_S,d_s\sim D_S}[D^{feat}_R(\mathcal G_R(G_{S→R}(x_s,d_s)))]\\
&+\mathbb{E}_{x_r\sim X_R}[D^{feat}_R(\mathcal G_R(x_r))-1]\tag4
\end{align}
" class="ee_img tr_noresize" eeimg="1">

- 与 <img src="https://www.zhihu.com/equation?tex=G_{S→R}" alt="G_{S→R}" class="ee_img tr_noresize" eeimg="1"> 类似，翻译器 <img src="https://www.zhihu.com/equation?tex=G_{R→S}" alt="G_{R→S}" class="ee_img tr_noresize" eeimg="1"> 具有另一种图像级对抗损失和特征级对抗损失，分别表示为 <img src="https://www.zhihu.com/equation?tex=L^{img}_{Gan}(X_S,X_R,D_S^{img},G_{R→S})" alt="L^{img}_{Gan}(X_S,X_R,D_S^{img},G_{R→S})" class="ee_img tr_noresize" eeimg="1"> ， <img src="https://www.zhihu.com/equation?tex=L^{feat}_{Gan}(X_S,X_R,D_S^{feat},G_{R→S},G_S)" alt="L^{feat}_{Gan}(X_S,X_R,D_S^{feat},G_{R→S},G_S)" class="ee_img tr_noresize" eeimg="1"> 

- 另外，我们利用循环一致性损失（cycle consistency loss）[38]来正则化（regularize）翻译网络的训练。具体来说，当将图像 <img src="https://www.zhihu.com/equation?tex=x_s" alt="x_s" class="ee_img tr_noresize" eeimg="1"> 依次传递到 <img src="https://www.zhihu.com/equation?tex=G_{S→R}" alt="G_{S→R}" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=G_{R→S}" alt="G_{R→S}" class="ee_img tr_noresize" eeimg="1"> 时，我们期望输出应该是相同的图像，类似地 <img src="https://www.zhihu.com/equation?tex=x_r" alt="x_r" class="ee_img tr_noresize" eeimg="1"> 就是反过来。即， <img src="https://www.zhihu.com/equation?tex=G_{R→S}(G_{S→R}(x_s,ds))=x_s" alt="G_{R→S}(G_{S→R}(x_s,ds))=x_s" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=G_{S→R}(G_{R→S}(x_r),dr)=x_r" alt="G_{S→R}(G_{R→S}(x_r),dr)=x_r" class="ee_img tr_noresize" eeimg="1"> 
  循环一致性损失可以表示为：


<img src="https://www.zhihu.com/equation?tex=\begin{align}
L_c&=\mathbb{E}_{x_s\sim X_S,d_s\sim D_S}[||G_{R→S}(G_{S→R}(x_s,d_s))-x_s||_1]\\
&+\mathbb{E}_{x_r\sim X_R,d_r\sim D_R}[||G_{S→R}(G_{R→S}(x_r),d_r)-x_r||_1]\tag5
\end{align}
" alt="\begin{align}
L_c&=\mathbb{E}_{x_s\sim X_S,d_s\sim D_S}[||G_{R→S}(G_{S→R}(x_s,d_s))-x_s||_1]\\
&+\mathbb{E}_{x_r\sim X_R,d_r\sim D_R}[||G_{S→R}(G_{R→S}(x_r),d_r)-x_r||_1]\tag5
\end{align}
" class="ee_img tr_noresize" eeimg="1">

- 最后，为了激励生成器在输入和输出之间保留内容信息，我们还利用了一个身份映射损失（identity mapping loss）[38]，它表示为：

<img src="https://www.zhihu.com/equation?tex=\begin{align}
  L_{idt} & =\mathbb{E}_{x_s\sim X_S}[||G_{R→S}(x_s)-x_s||_1]\\
  &+\mathbb{E}_{x_s\sim X_S,d_r\sim D_R}[||G_{S→R}(X_r,d_r)-x_r||_1]\tag6
  \end{align}
  " alt="\begin{align}
  L_{idt} & =\mathbb{E}_{x_s\sim X_S}[||G_{R→S}(x_s)-x_s||_1]\\
  &+\mathbb{E}_{x_s\sim X_S,d_r\sim D_R}[||G_{S→R}(X_r,d_r)-x_r||_1]\tag6
  \end{align}
  " class="ee_img tr_noresize" eeimg="1">
  
- 翻译模块的完整损失函数如下所示：


<img src="https://www.zhihu.com/equation?tex=\begin{align}
L_{tran} & =L^{img}_{Gan}(X_R,(X_S,D_S),D^{img}_R,G_{S→R})\\
&+L^{feat}_{Gan}(X_R,(X_S,D_S),D^{feat}_R,G_{S→R},\mathcal G_R)\\
&+L^{img}_{Gan}(X_S,X_R,D^{img}_S,G_{R→S})\\
&+L^{feat}_{Gan}(X_S,X_R,D^{feat}_S,G_{R→S},\mathcal G_S)\\
&+\lambda_1 L_c+\lambda_2 L_{idt}\tag7
&\end{align}
" alt="\begin{align}
L_{tran} & =L^{img}_{Gan}(X_R,(X_S,D_S),D^{img}_R,G_{S→R})\\
&+L^{feat}_{Gan}(X_R,(X_S,D_S),D^{feat}_R,G_{S→R},\mathcal G_R)\\
&+L^{img}_{Gan}(X_S,X_R,D^{img}_S,G_{R→S})\\
&+L^{feat}_{Gan}(X_S,X_R,D^{feat}_S,G_{R→S},\mathcal G_S)\\
&+\lambda_1 L_c+\lambda_2 L_{idt}\tag7
&\end{align}
" class="ee_img tr_noresize" eeimg="1">

#### 图像去雾损失

- 现在，我们可以将合成图像 <img src="https://www.zhihu.com/equation?tex=X_S" alt="X_S" class="ee_img tr_noresize" eeimg="1"> 和相应的深度图像 <img src="https://www.zhihu.com/equation?tex=D_S" alt="D_S" class="ee_img tr_noresize" eeimg="1"> 传输到生成器 <img src="https://www.zhihu.com/equation?tex=G_{S→R}" alt="G_{S→R}" class="ee_img tr_noresize" eeimg="1"> ，并获得一个新的数据集 <img src="https://www.zhihu.com/equation?tex=X_{S→R}=G_{S→R}(X_S,D_S)" alt="X_{S→R}=G_{S→R}(X_S,D_S)" class="ee_img tr_noresize" eeimg="1"> ，它与真实的有雾图像具有相似的样式。然后，我们以半监督的方式在 <img src="https://www.zhihu.com/equation?tex=X_{S→R}" alt="X_{S→R}" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=X_R" alt="X_R" class="ee_img tr_noresize" eeimg="1"> 上训练图像去雾网络 <img src="https://www.zhihu.com/equation?tex=\mathcal G_R" alt="\mathcal G_R" class="ee_img tr_noresize" eeimg="1"> 。对于监督分支，我们用均方损失以确保预测的图像 <img src="https://www.zhihu.com/equation?tex=J_{S→R}" alt="J_{S→R}" class="ee_img tr_noresize" eeimg="1"> 接近干净图像 <img src="https://www.zhihu.com/equation?tex=Y_S" alt="Y_S" class="ee_img tr_noresize" eeimg="1"> ，将其定义为：


<img src="https://www.zhihu.com/equation?tex=L_{rm}=||J_{S→R}-Y_S||^2_2\tag8
" alt="L_{rm}=||J_{S→R}-Y_S||^2_2\tag8
" class="ee_img tr_noresize" eeimg="1">

- 在无监督的分支中，我们引入了总变化（total variation）和暗通道损失，它们将去雾网络正则化，以产生具有与清晰图像相似的统计特征的图像。总变化损失在预测图像 <img src="https://www.zhihu.com/equation?tex=J_R" alt="J_R" class="ee_img tr_noresize" eeimg="1"> 上是 <img src="https://www.zhihu.com/equation?tex=\ell_1" alt="\ell_1" class="ee_img tr_noresize" eeimg="1"> 正则化梯度先验（gradient prior）：


<img src="https://www.zhihu.com/equation?tex=L_{rt}=||\partial_h J_R||_1+||\partial_v J_R||_1\tag9
" alt="L_{rt}=||\partial_h J_R||_1+||\partial_v J_R||_1\tag9
" class="ee_img tr_noresize" eeimg="1">

​		其中 <img src="https://www.zhihu.com/equation?tex=\partial_h" alt="\partial_h" class="ee_img tr_noresize" eeimg="1"> 表示水平梯度算子， <img src="https://www.zhihu.com/equation?tex=\partial_v" alt="\partial_v" class="ee_img tr_noresize" eeimg="1"> 表示垂直梯度算子。

- 此外，[9]提出了暗通道的概念，表示为：


<img src="https://www.zhihu.com/equation?tex=D(I)={\underset {y \in N(x)}{\operatorname {min}}}\Bigg[{\underset {c \in \{r,g,b\}}{\operatorname {min}}}I^c(y)\Bigg]\tag{10}
" alt="D(I)={\underset {y \in N(x)}{\operatorname {min}}}\Bigg[{\underset {c \in \{r,g,b\}}{\operatorname {min}}}I^c(y)\Bigg]\tag{10}
" class="ee_img tr_noresize" eeimg="1">

​		其中 <img src="https://www.zhihu.com/equation?tex=x" alt="x" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=y" alt="y" class="ee_img tr_noresize" eeimg="1"> 是图像 <img src="https://www.zhihu.com/equation?tex=I" alt="I" class="ee_img tr_noresize" eeimg="1"> 的像素坐标， <img src="https://www.zhihu.com/equation?tex=I^c" alt="I^c" class="ee_img tr_noresize" eeimg="1"> 表示 <img src="https://www.zhihu.com/equation?tex=I" alt="I" class="ee_img tr_noresize" eeimg="1"> 的第 <img src="https://www.zhihu.com/equation?tex=c" alt="c" class="ee_img tr_noresize" eeimg="1"> 个颜色通道， <img src="https://www.zhihu.com/equation?tex=N(x)" alt="N(x)" class="ee_img tr_noresize" eeimg="1"> 表示以 <img src="https://www.zhihu.com/equation?tex=x" alt="x" class="ee_img tr_noresize" eeimg="1"> 为中心的局部邻域。He等人[9]还表明，暗通道图像的大多数强度为零或接近于零。因此，我们应用以下暗通道（DC）损失以确保预测图像的暗通道与清晰图像的暗通道一致：

<img src="https://www.zhihu.com/equation?tex=L_{rd}=||D(J_R)||_1\tag{11}
" alt="L_{rd}=||D(J_R)||_1\tag{11}
" class="ee_img tr_noresize" eeimg="1">

- 此外，我们还在 <img src="https://www.zhihu.com/equation?tex=X_S" alt="X_S" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=X_{R→S}" alt="X_{R→S}" class="ee_img tr_noresize" eeimg="1"> 上训练了互补的图像去雾网络 <img src="https://www.zhihu.com/equation?tex=\mathcal G_S" alt="\mathcal G_S" class="ee_img tr_noresize" eeimg="1"> 。类似地，我们用相同的监督损失和无监督损失来训练除雾网络 <img src="https://www.zhihu.com/equation?tex=\mathcal G_S" alt="\mathcal G_S" class="ee_img tr_noresize" eeimg="1"> ，如下所示：


<img src="https://www.zhihu.com/equation?tex=L_{sm}=||J_S-Y_s||_2^2\tag{12}\\
" alt="L_{sm}=||J_S-Y_s||_2^2\tag{12}\\
" class="ee_img tr_noresize" eeimg="1">


<img src="https://www.zhihu.com/equation?tex=L_{st}=||\partial_h J_{R→S}||_1+||\partial_v J_{R→S}||_1\tag{13}\\
" alt="L_{st}=||\partial_h J_{R→S}||_1+||\partial_v J_{R→S}||_1\tag{13}\\
" class="ee_img tr_noresize" eeimg="1">


<img src="https://www.zhihu.com/equation?tex=L_{sd}=||D(J_{R→S})||_1\tag{14}
" alt="L_{sd}=||D(J_{R→S})||_1\tag{14}
" class="ee_img tr_noresize" eeimg="1">

- 最后，考虑到两个去雾网络的输出对于真实的有雾图像应该是一致的，即 <img src="https://www.zhihu.com/equation?tex=\mathcal G_R(X_R)≈\mathcal G_S(G_{R→S}(X_R))" alt="\mathcal G_R(X_R)≈\mathcal G_S(G_{R→S}(X_R))" class="ee_img tr_noresize" eeimg="1"> ，我们引入以下一致性损失：


<img src="https://www.zhihu.com/equation?tex=L_c=||\mathcal G_R(X_R)-\mathcal G_S(G_{R→S}(X_R))||_1\tag{15}
" alt="L_c=||\mathcal G_R(X_R)-\mathcal G_S(G_{R→S}(X_R))||_1\tag{15}
" class="ee_img tr_noresize" eeimg="1">

#### 总损失

总损失函数定义如下：

<img src="https://www.zhihu.com/equation?tex=\begin{align}
L&=L_{tran}+\lambda_m(L_{rm}+L_{sm})+\lambda_d(L_{rd}+L_{sd})\\
&+\lambda_t(L_{rt}+L_{st})+\lambda_c L_c\tag{16}
\end{align}
" alt="\begin{align}
L&=L_{tran}+\lambda_m(L_{rm}+L_{sm})+\lambda_d(L_{rd}+L_{sd})\\
&+\lambda_t(L_{rt}+L_{st})+\lambda_c L_c\tag{16}
\end{align}
" class="ee_img tr_noresize" eeimg="1">
其中 <img src="https://www.zhihu.com/equation?tex=\lambda_m,\lambda_d,\lambda_t" alt="\lambda_m,\lambda_d,\lambda_t" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=\lambda_c" alt="\lambda_c" class="ee_img tr_noresize" eeimg="1"> 是权衡权重（trade-off weights）。

## 4 实验结果

在这一部分中，我们首先介绍框架的实现细节。然后，我们分别针对合成数据集和真实图像评估我们的域自适应方法。最后，进行消融研究（ablation study）以分析我们提出的方法。

### 4.1 实现细节

#### 数据集

我们从RESIDE数据集[13]中随机选择合成图像和真实有雾图像进行训练。数据集分为五个子集，即ITS（室内训练集），OTS（室外训练集），SOTS（合成对象测试集），URHI（无注释的真实有雾图像）和RTTS（真实的任务驱动测试集）。对于合成数据集，我们选择6000张合成有雾图像进行训练，从ITS中选择3000张，从OTS中选择3000张。对于真实的有雾图像，我们通过从URHI中随机选择1000个真实的有雾图像来训练网络。在训练阶段，我们将所有图像随机裁剪为256×256，并将像素值归一化为[-1，1]。

#### 训练细节

我们用PyTorch [24]实现我们的框架，并使用batch size为2的ADAM [11]优化器来训练网络。
首先，我们训练了90个epoch的图像转换网络 <img src="https://www.zhihu.com/equation?tex=G_{S→R}" alt="G_{S→R}" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=G_{R→S}" alt="G_{R→S}" class="ee_img tr_noresize" eeimg="1"> ，动量为 <img src="https://www.zhihu.com/equation?tex=β_1" alt="β_1" class="ee_img tr_noresize" eeimg="1"> = 0.5， <img src="https://www.zhihu.com/equation?tex=β_2" alt="β_2" class="ee_img tr_noresize" eeimg="1"> = 0.999，学习率设为 <img src="https://www.zhihu.com/equation?tex=5×10^{-5}" alt="5×10^{-5}" class="ee_img tr_noresize" eeimg="1"> 。然后，我们使用预先训练的 <img src="https://www.zhihu.com/equation?tex=G_{S→R}" alt="G_{S→R}" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=G_{R→S}" alt="G_{R→S}" class="ee_img tr_noresize" eeimg="1"> 在 <img src="https://www.zhihu.com/equation?tex=\{X_R,G_{S→R}(X_S，D_S)\}" alt="\{X_R,G_{S→R}(X_S，D_S)\}" class="ee_img tr_noresize" eeimg="1"> 上训练 <img src="https://www.zhihu.com/equation?tex=\mathcal G_R" alt="\mathcal G_R" class="ee_img tr_noresize" eeimg="1"> ，和在 <img src="https://www.zhihu.com/equation?tex=\{X_S,G_{R→S}(X_R)\}" alt="\{X_S,G_{R→S}(X_R)\}" class="ee_img tr_noresize" eeimg="1"> 上训练 <img src="https://www.zhihu.com/equation?tex=\mathcal G_S" alt="\mathcal G_S" class="ee_img tr_noresize" eeimg="1"> 90个epochs，动量和学习率设置为： <img src="https://www.zhihu.com/equation?tex=β_1= 0.95，β_2= 0.999，lr = 10^{-4}" alt="β_1= 0.95，β_2= 0.999，lr = 10^{-4}" class="ee_img tr_noresize" eeimg="1"> 。最后，我们使用上述预先训练的模型对整个网络进行微调（fine tune）。在计算暗通道（DC）损失时，我们将patch 大小设置为35×35。权衡权重设置为： <img src="https://www.zhihu.com/equation?tex=λ_{tran}= 1，λ_m= 10，λ_d= 10^{-2}，λ_t= 10^{-3}，λ_c= 10^{-1}" alt="λ_{tran}= 1，λ_m= 10，λ_d= 10^{-2}，λ_t= 10^{-3}，λ_c= 10^{-1}" class="ee_img tr_noresize" eeimg="1"> 。

#### 比较方法

我们对自己提出的方法同以下方法进行了比较：DCP [9]，MSCNN [26]，DehazeNet [4]，NLD [2]，AOD-Net [12]，GFN [27]，DCPDN [35]和EPDN [25]。补充材料中包含更多图像去雾结果以及与其他去雾方法的比较。

### 4.2 在合成数据集上的实验

我们使用两个综合数据集，SOTS [13]和HazeRD [36]，来评估我们提出的方法的性能。
这两个数据集上不同方法的去雾图像如图5和6所示。从图5(b)中，我们可以观察到NLD [2]和GFN [27]都有一些颜色失真，结果看起来不接近真实。在某些情况下，EPDN [25]的除雾结果也比ground  truth更暗，如图5(g)所示。此外，DehazeNet [4]，AOD-Net [12]和DCPDN [35]在去雾后的图像中仍然存在一些残留的雾。与这些方法相比，我们的算法还原的图像具有更清晰（sharp）的结构和细节，更接近ground  truth。在图6的HazeRD数据集的除雾结果中可以找到相似的结果，我们的算法生成的结果具有更好的视觉效果。

![figure5](.\figure5.png)

<center>图5 在SOTS[13]数据集上的可视化对比</center>

![figure6](.\figure6.png)

<center>图6 在HazeRD[36]数据集上的可视化对比</center>

我们还在表2中给出了去雾结果的定量比较。如图所示，该方法在两个数据集上均获得了最高的PSNR和SSIM值。与STOA的EPDN相比[25]，我们的方法在STOS数据集上的PSNR和SSIM分别获得了3.94 dB和0.04的提升。对于HazeRD数据集，我们的方法产生的PSNR和SSIM分别比EPDN [25]高出0.7dB和0.07。

<center>表2 两个合成数据集上的除雾结果的定量比较（平均PSNR / SSIM）</center>

![table2](.\table2.png)

### 4.3 在真实图像上的实验

为了评估我们的方法在真实图像上的泛化能力（generalization），我们比较了从URHI数据集获得的真实有雾图像上不同方法的视觉结果。如图7所示，NLD [2]遭受了严重的颜色失真（例如，参见图7(b))中的天空）。从图7(f)中我们可以看出GFN [27]在某些情况下也遭受了颜色失真，并且去雾后的结果看起来比我们的方法更暗。另外，由DehazeNet [4]，AOD-Net [12]和DCPDN [35]进行的去雾结果具有一些残留的雾的伪影，如图7(c-e)的第五行所示。尽管EPDN [25]比上述方法具有更好的视觉效果，但除雾效果的亮度普遍比我们的方法低。总体而言，我们提出的方法可还原更多细节并获得视觉上pleasing的图像。

![figure7](.\figure7.png)

<center>图7 在真实有雾图像上的可视化对比</center>

### 4.4 消融研究

为了验证图像翻译网络的有效性和无监督的损失，我们进行了一系列消融（ablationos）以分析我们的方法。我们构建以下除雾模型进行比较：

1. **SYN**： <img src="https://www.zhihu.com/equation?tex=\mathcal G_S" alt="\mathcal G_S" class="ee_img tr_noresize" eeimg="1"> 仅在 <img src="https://www.zhihu.com/equation?tex=X_S" alt="X_S" class="ee_img tr_noresize" eeimg="1"> 上训练； 
2. **SYN + U**： <img src="https://www.zhihu.com/equation?tex=\mathcal G_S" alt="\mathcal G_S" class="ee_img tr_noresize" eeimg="1"> 在 <img src="https://www.zhihu.com/equation?tex=X_S" alt="X_S" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=X_R" alt="X_R" class="ee_img tr_noresize" eeimg="1"> 上训练； 
3. **R2S + U**： <img src="https://www.zhihu.com/equation?tex=\mathcal G_S" alt="\mathcal G_S" class="ee_img tr_noresize" eeimg="1"> 仅在 <img src="https://www.zhihu.com/equation?tex=X_S" alt="X_S" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=G_{R→S}(X_R)" alt="G_{R→S}(X_R)" class="ee_img tr_noresize" eeimg="1"> 上训练； 
4. **S2R**： <img src="https://www.zhihu.com/equation?tex=\mathcal G_R" alt="\mathcal G_R" class="ee_img tr_noresize" eeimg="1"> 在 <img src="https://www.zhihu.com/equation?tex=G_{S→R}(X_S,D_S)" alt="G_{S→R}(X_S,D_S)" class="ee_img tr_noresize" eeimg="1"> 上训练。

我们在合成和真实有雾图像上，针对这四种除雾模型，对提出的域自适应方法进行了比较。
视觉和定量结果显示在表3和图8中，结果表明我们的方法在PSNR和SSIM以及视觉效果方面达到了图像去雾的最佳性能。如图8(b)所示，由于域迁移，**SYN**方法会导致颜色失真或较暗的伪影（例如，参见天空部分和红色矩形）。相比之下，如图8(c)所示，在翻译后的图像**（S2R）**上训练的去雾模型 <img src="https://www.zhihu.com/equation?tex=\mathcal G_R" alt="\mathcal G_R" class="ee_img tr_noresize" eeimg="1"> 获得了更好的图像质量，这证明翻译器有效地减少了合成数据与真实图像之间的差异。此外，图8(b)和(d)显示具有无监督损失**（SYN + U）**的去雾模型比**SYN**可以产生更好的结果，这证明了无监督损失的有效性。最后，我们可以观察到，图8(e)中提出的同时具有转换器和无监督损耗的方法产生了更清晰和视觉上更pleasing的结果（例如，天空更明亮）。表3中通过应用图像平移和无监督损失得出的定量结果也与图8中的定性结果一致。

![figure8](figure8.png)

<center>图8 真实有雾图像上几种除雾模型除雾结果的比较</center>

<center>表3 合成域上不同除雾模型的定量结果</center>

![table3](.\table3.png)

总之，这些消融研究表明，图像翻译模型和无监督损失对于减小合成数据与真实世界图像之间的域差异以及提高图像在合成域和真实域上的除雾性能很有用。

## 5 结论

在本次工作中，我们提出了一种用于单图像去雾的新颖的域自适应框架，该框架包含一个图像翻译模块和两个图像去雾模块。我们首先使用图像翻译网络将图像从一个域翻译到另一个域，以减少域差异。然后，图像去雾网络将翻译后的图像及其原始图像作为输入以执行图像去雾。
为了进一步提高泛化能力，我们通过利用清晰图像的属性将真实的有雾图像合并到除雾训练中。在合成数据集和真实世界图像上的大量实验结果表明，我们的算法性能优于STOA。
