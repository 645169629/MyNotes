## Context Encoding for Semantic Segmentation

> CVPR 2018 作者 与PSPNet 是一个组的

### Abstract

​	近期在提高空间分辨率的研究上取得了重大进展，Dialted/Atrous 卷积、多尺度特征、refining 边缘。本文研究全局上下文信息的影响，引入Context Encoding Module，来捕获场景语义上下文并有选择的强调类别依赖（class-dependent）的特征图。PASCAL-Context mIoU 51.7%，PASCAL VOC 2012 mIoU 85.9%，ADE20K 测试集上得分0.5567。此外，我们也研究了Context Encoding Module如何提高相对浅层网络（CIFAR-10 上的图像分类网络）的特征表示。

### Introduction

![f1](images\f1.png)

​	Dilated/Atrous 卷积策略孤立了像素与全局场景上下文，导致误分类问题。如图4第3行，baseline 方法把窗户的一些像素分类成了门。最近的state-of-the-art方法使用多分辨率金字塔表示来扩大感知野。如，PSPNet采用了空间金字塔池化，将特征图池化到不同大小，并concat它们接着上采样；Deeplab提出Atrous 空间金字塔池化，利用了large rate 的dilated/atrous卷积。尽管这些方法提升了性能，上下文表示却没有明确，导致了如下问题：`是否捕获上下文信息和扩大感知野是一样的？`考虑对一个大数据集（如ADE20K，包含150类）进行标注（如图1）。假设我们有一个工具，可以让标注器首先选择图像的语义上下文（如，床）。接着工具可以提供一个相关类更小的sublist（如，床，椅子等） ，这样可以显著减少类别的搜索空间。类似的，如果我们可以设计一种方法来充分利用场景上下文和类别概率之间的关系，语义分割将会变得更简单。

​	经典的计算机视觉方法可以捕获场景语义上下文信息。给定一幅图片，使用SIFT或过滤器组响应（filter bank responses）来提取手工特征，接着学习一个视觉词汇（字典），全局特征统计会使用经典的编码器来表示，如Bag-of-Words，VLAD或Fisher Vector。经典表示通过捕获特征统计编码了全局上下文信息。尽管CNN大大提升了手工特征，但是传统方法整体的编码过程还是有效的。最近的研究在传统编码器运用于CNN框架的方法上取得了很大的进展。Zhang等提出Encoding Layer来集成整个词典学习以及残差编码pipeline到一个CNN层中，来捕获无条理的表示。该方法在纹理分类上取得了state-of-the-art。本文拓展了Encoding Layer来捕获全局特征统计，用于理解语义上下文。

贡献：

  1. 提出Context Encoding Module结合Semantic Encoding Loss(SE-loss)，利用全局场景上下文信息。该模块集成了一个Encoding Layer来捕获全局上下文，并有选择性的强调类别相关的特征图。（如，我们想降低室内场景中机动车出现的概率）。标准训练过程只利用了逐像素分割损失，没有充分利用场景全局上下文。我们提出SE-loss来对训练规范化，使网络预测场景中出现的目标类别，以此强制网络学习语义上下文。不像逐像素损失，SE-loss对于大小目标同等重视，小目标的性能会提高。
  2. 设计实现了一个新语义分割框架 Context Encoding Network（EncNet）。在ResNet上增加了Context Encoding Module（如图2所示）。使用了dilation策略。

### Context Encoding Module

![f2](images\f2.png)

**Context Encoding**  在多种图像集上预训练的网络，特征图编码了物体丰富的信息。我们利用Encoding Layer来捕获特征统计作为全局语义信息。我们称Encoding Layer的输出为 *encoded semantics*。为了利用上下文，预测一系列缩放因子(scaling factors)来有选择性的强调类别依赖的特征图。Encoding Layer 学习一个固有的词典，携带数据集中的语义上下文，并输出具有丰富上下文信息的残差编码器。

​	Encoding Layer 将$C\times H\times W$的输入特征图视为一系列C维输入特征$\ X = \{x_1,...,x_N\}\ $，其中N为$\ H\times W\ $，学习一个固有的codebook $\ D = \{d_1,...d_K\}\ $包含K个codewords（visual centers）以及一系列视觉中心的平滑factor$\ S=\{s_ 1,...,s_K\}\ $。Encoding Layer 输出残差编码器，通过soft-assignment weights$\ e_k = \sum^{N}_{i=1}e_{ik}\ $来集成残差，其中
$$
e_{ik} = \frac{exp(-s_k\parallel r_{ik}\parallel ^2)}{\sum^{K}_{j=1}exp(-s_j\parallel r_{ij}\parallel^2)}r_{ik}
$$
残差为$\ r_{ik} = x_i - d_k\ $。我们集成各个编码器，而不是concat。即：$\ e=\sum^{K}_{k=1}\phi(e_k)\ $，其中$\ \phi\ $表示Batch Normalization和ReLU，避免了K个独立的编码器有顺序，并减少了特征表示的维度。

**Featuremap Attention** 为了利用Encoding Layer捕获的encoded semantics，我们预测一系列特征图的缩放尺度，作为加强或不强调类别依赖特征图的反馈环路。我们在Encoding Layer上加了一个全连接层以及一个sigmoid激活，输出预测的特征图缩放尺度$\ \gamma = \delta(We)\ $，其中$\ W\ $表示层权重，$\ \delta\ $为sigmoid函数。接着模块的输出为$\ Y = X\otimes \gamma\ $，在输入特征图$\ X\ $和缩放尺度$\ \gamma\ $之间的逐通道相乘。该反馈策略是收到先前风更转换研究以及SE-Net的启发。

**Semantic Encoding Loss** 为了规则化Context Encoding Module的训练，我们提出Semantic Encoding Loss(SE-Loss)，强制网络理解全局上下文信息，同时计算开销很小。我们在Encoding Layer上附加了一个全连接层以及sigmoid激活，以对类别存在与否进行单独预测，使用二分类cross entropy loss学习。不同于逐像素损失，SE-loss对大物体和小物体同等重视。在实践中，我们发现对小物体的分割有所提升。

![f3](images\f3.png)

#### Context Encoding Network（EncNet）

​	我们在ResNet stage 3和stage 4上运用了dilated 策略，如图3。我们在最后预测巻积层上加入了Context Encoding Module，如图2。我们社了另一个分支来学习SE-loss（接收 encoded semantics作为输入，物体类别是否存在作为输出）。我们在stage 3上加了另一个Context Encoding Module来最小化SE-loss。

#### Relation to Other Approaches

**Segmentation Approaches** 从降采样特征图中恢复细节，一种方法是学习上采样filter，即, fractionally-strided convolution 或解码器。另一种方法是利用Dilated 卷积策略。之前的研究采用密集CRF来修正分割边界，CRF-RNN做到了端到端。最近基于FCN的方法的提升都源自于larger rate atrous convolution或global/pyramid pooling，但这些模型都牺牲了效率。

**Featuremap Attention and Scaling** 逐通道特征图注意力策略是受到先前研究的启发。Spatial Transformer Network 学习一种网络内部变换条件，对特征图提供了空间注意力。Batch Normalization使得对mini-batch的数据归一化集成进了网络，使得可以使用更大的学习率，并且网络对于初始化没有那么敏感。最近在风格转换上的研究对特征图平均和方差或2阶统计量进行操作，已实现网络内风格转换。SE-Net利用了通道间信息医学到逐通道的注意力。因此，我们使用encoded semantics来预测特征图通道的缩放尺度。

![f4](images\f4.png)