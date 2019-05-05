## UNet++: A Nested U-Net Architecture for Medical Image Segmentation

### Abstract

​	我们提出一个新的医学图像分割结构，UNet++。我们的网络结构本质上是一个encoder-decoder网络，其中encoder和decoder子网络通过一系列内嵌的，密集连接路径连接起来。新设计的skip pathways可以减少encoder和decoder特征图之间的语义差距。平均上，UNet++比UNet、wide U-Net提高了3.9个点和3.4个点的IoU。

### Introduction

skip connections，结合了深层，语义，粗粒度特征图与浅层，low-level，精细的特征图。我们认为当decoder和encoder网络语义相似时，网络将处理更简单的学习任务。

### Related Work

​	FCN、UNet。收到DenseNet结构的启发，Li等提出了H-denseunet用于liver和liver tumor 分割。类似的，Drozdzal等系统的研究了skip connections的重要性，并引入了encoder的short skip connections。尽管上述架构之间存在细微差异，但他们都景象与融合encoder和decoder特征图之间的语义不一致性，根据我们的实验，这可能会降低分割性能。

​	另外两个最近的工作：GridNet和Mask-RCNN。

### Proposed Network Architecture: UNet++

![f11](images\f11.png)

![f12](images\f12.png)

​	图1展示了网络的结构。UNet++从encoder子网络或backbone开始，然后是decoder子网络。重设计的skip pathways（绿色和蓝色）连接了两个子网络并使用了深度监督（红色）。

#### Re-designed skip pathways

​	在UNet++中，encoder的特征图经过密集卷积block（巻积层书取决于pyramid level）。如结点$\ X^{0,0}\ $和$\ X^{1,3}\ $之间的skip pathway包含了一个三卷积层的dense convolution block，每个卷积层之前都有一个concatenation 层，concatenation层融合了之前巻积层的输出与对应的上采样输出。本质上，dense convolution block使得encoder特征图和decoder特征图之间的语义更接近。

​	skip pathway可以表示为：设$\ x^{i,j}\ $表示结点$\ X^{i,j}\ $的输出，$\ i\ $表示down-sampling层的索引，$\ j\ $表示skip pathway上第$\ j\ $个巻积层。
$$
x^{i.j}=\begin{cases}
\mathcal{H}(x^{i-1,j})&  \text{j=0}\\
\mathcal{H}([[x^{i,k}]^{j-1}_{k=0},\mathcal{u}(x^{i+1,j-1})])&  \text{j>0}
\end{cases}
$$
其中$\ \mathcal{H}(\cdot)\ $表示卷积+激活操作，$\ u(\cdot)\ $表示上采样层，$\ [\ ]\ $表示concatenation层。图1b展示了等式1.

#### Deep supervision

​	深度监督使得模型有两种操作模式：1）accurate mode，所有分割分支的输出都被利用；2）fast mode，最终的分割map只选择一个分割分支。图1c展示了分割分支的选择产生不同复杂度的结构。

​	由于skip pathways，UNet++会产生多个语义level的输出，$\ \{x^{0,j},j\in\{1,2,3,4\}\}\ $。每个语义level上增加了binary cross-entropy和dice 结合的损失。
$$
\mathcal{L}（Y,\hat{Y}）=-\frac{1}{N}\sum^N_{b=1}(\frac{1}{2}\cdot Y_b\cdot\log\hat{Y}_b+\frac{2\cdot Y_b\cdot\hat{Y}_b}{Y_b+\hat{Y}_b})
$$
其中$\ \hat{Y}_b\ $和$\ Y_b\ $表示第b幅图像flatten的预测概率和flatten的ground truth，N表示batch size。

### Experiments

![t1t2](images\t1t2.png)

#### Datasets

如表1所示，我们使用四个医学图像数据集来评估模型，包括来自不同医学成像方式的病变/器官。

#### Baseline models

我们使用了原始UNet可以自定义的wide UNet结构。表2详细展示了UNet和wide UNet结构。

#### Implementation details

我们使用Dice和IoU，并在验证集上使用early-stop机制。使用Adam优化器，学习率为3e-4。为了使用深度监督，每个目标结点后加一个1x1卷积以及sigmoid激活。最后四个分割结构平局来产生最终分割图。

#### Results

![f2](images\f2.png)

![t3](images\t3.png)

![f3](images\f3.png)

