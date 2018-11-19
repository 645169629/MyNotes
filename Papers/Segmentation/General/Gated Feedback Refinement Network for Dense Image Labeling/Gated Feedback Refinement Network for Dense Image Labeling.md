## Gated Feedback Refinement Network for Dense Image Labeling

> CVPR 2017

### Abstract

​	大部分现有的编解码结构简单的concate低层的特征，以获得在refinement阶段高频的细节。但是，如果模糊的信息进行传递，可能会导致refinement的限制。我们提出了Gated Feedback Refinement Netwrok（G-FRNet），来解决现有方法的限制。最开始，G-FRNet产生一个粗预测，然后通过在refinement阶段有效的集成局部和全局上下文信息来逐渐的修正细节。我们引入gate units 来控制传入的信息。

### Introduction

​	有趣的是，这（编码、解码层级结构）反映了人类视觉观察到的计算结构，其中空间被抽象为丰富的特征，模式的识别先于它们的精准定位。

​	在编码最深的阶段，具有最丰富的特征表示，而相对较少的空间分辨率（从每个神经元的角度），但从此恢复准确的空间信息也是可能的。例如，一个粗编码策略可以考虑到高度的空间定位准确性，却损失了特征的多样性。对于该方法一个重要的暗示是，如果最高层不需要准确定位模式，就可以用一个更丰富的特征级表示。

​	前面层携带的信息具有更好的空间定位性，但不够辨别性。考虑到每一层都有图像特点的表示，很自然的认为在解码阶段利用浅层编码表示是有价值的。这种方式，在深层损失的空间准确度可以从浅层表示中逐渐恢复。这一直观理解在我们提出的模型中有体现，在编码层和解码层间的连接中，隐含了责任（分类、定位）的转移。

​	如果依赖卷积和unpooling来恢复信息并最终赋值表情，这意味着任何磨合的表示必然会参与进解码中，这样会降低预测的质量。如，网络较深的卷积层可能提供对于牛和马较强的辨别能力，浅层的可能只能辨别出动物，对于牛和马都表现出confidence。如果这种confidence传到了解码阶段，使用一个固定的结合表示的模式，会导致标注错误。这一观察是我们模型的主要动机，如图1所示。浅层信息对于定位有重要价值，需要过滤该信息以减少类别的不确定性。并且，使用深层来过滤浅层信息是很自然的。

![f1](images\f1.png)

### Background

**Encoder-Decoder Architecture:** 如图2所示，我们的模型基于深度编解码结构。

**Skip Connections**

![f2](images\f2.png)

### Gated Feedback Refinement Network

#### Network Overview

​	编码网络基于VGG-16网络，移除了softmax以及全连接层。增加了两个巻积层$conv6$和$conv7$。对于输入图像$\ I\ $，编码网络产生7张特征图$f_1,f_2,...,f_7$。在$\ f_7\ $上采用3x3卷积来获得粗预测图$\ Pm^G\ $。因为$\ Pm^G\ $具有较小的空间维度，因此只是输入图像的粗预测。尽管可以直接上采样$\ Pm^G\ $，但不准确。

​	Feedback Refinement Network（FRN）作为解码网络。采用了skip connection。

​	使用门机制来调节skip connection传递的信息。如，我们要从编码层$\ f_5\ $传递信息到解码层$\ Pm^{RU_1}\ $，我们首先基于$\ f_5\ $和$\ f_6\ $（上面层）来计算门特征图$\ G_1\ $。直观理解是$\ f_6\ $包含了信息，可以解决$\ f_5\ $中的不确定性。通过从$\ f_5\ $和$\ f_6\ $计算门特征，类别不确定性在到达解码阶段前可以被筛选出来。图1展示了类别不确定性。

​	将$\ G_1\ $和粗预测图$\ Pm^G\ $结合来产生扩张的预测图$\ Pm^{RU_1}\ $。

#### Gate Unit

​	门单元接受两个连续的特征图作为输入。$\ f^i_g\ $的特征具有高分辨率，小感受野（即小上下文），$\ f^{i+1}_g\ $具有低分辨率，大感受野。门单元结合两个特征图来产生丰富的上下文信息。

​	首先对两个特征图进行3x3卷积，batch normalization 以及 ReLU，使得$\ f^i_g\ $和$\ f^{i+1}_g\ $的通道数$\ c^i_g\ $和$\ c^{i+1}_g\ $相等。$\ f^{i+1}_g\ $接着以因子2上采样，产生新特征图$\ f^{i+1}_{g'}\ $，其空间维度与$\ f^i_g\ $相同。我们再通过$\ f^i_g\ $和$\ f^{i+1}_g\ $的逐像素相乘获得第$\ i\ $阶段门特征图$\ M_f\ $。然后$\ M_f\ $被送入门refinement单元中。
$$
v_i = T_f(f^{i+1}_g),u_i=T_f(f^i_g),Mf = v_i\otimes u_i
$$

#### Gated Refinement Unit

![f3](images\f3.png)

​	图3展示了gated refinement unit的详细结构。每个refinement unit $RU^i$接收粗标注图$\ R_f\ $（通道为$\ k^i_r\ $，在FRN第$\ i-1\ $阶段生成）以及门特征图$\ M_f\ $作为输入。$\ RU\ $学习集成信息并产生新的标注图$\ R'_f\ $（更大的空间维度），其过程如下：首先，对$\ M_f\ $做3x3卷积，batch normalization，获得特征图$\ m_f\ $，通道数为$\ k^i_m\ $。其中$\ k^i_m=k^i_r=C\ $，C为类别数。接着，$\ m_f\ $与前一阶段标注图$\ R_f\ $concat，产生特征图$\ (R+m)_f\ $，通道数为$\ k^i_m+k^i_r\ $。使得$\ k^i_m+k^i_r\ $主要有两个原因：1）编码器获得的特征图通常由很多通道，直接将$\ R_f\ $与特征图concat计算量很大。2）concat两个通道数差异很大的特征图会导致一些层丢失信号。最后在$\ (R+m)_f\ $特征图上做3x3卷积，产生修正表注图$\ R'_f\ $。接着$\ R'_f\ $以因子2上采样，送入下一阶段gated refinement unit。
$$
m_f = \mathbb{C}_{3\times3}(M_f),\gamma=m_f\oplus R_f,R'_f = \mathbb{C}_{3\times3}(\gamma)
$$

#### Stage-wise Supervision

​	解码前面阶段产生的标注图可能提供有用的信息。我们采用深度监督的思想来提供每阶段的监督。设$\ I\in\mathbb{R}^{h\times w\times d}\ $为训练样本，其gt mask为$\ \eta\in\mathbb{R}^{h\times w}\ $。我们获得$\ k\ $个resized的gt图（$\ R_1,R_2,...,R_k\ $）。使用逐像素交叉熵损失$\ l_i\ ​$来评估resized的gt和预测图的不同。

![f4](images\f4.png)

​	图4展示了gated refinement模式的有效性。