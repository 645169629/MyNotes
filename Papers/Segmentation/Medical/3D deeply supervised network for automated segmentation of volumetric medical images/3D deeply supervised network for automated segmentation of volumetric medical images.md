## 3D deeply supervised network for automated segmentation of volumetric medical images

> Medical Image Analysis  SCI 1区

### Abstract

​	虽然深度卷积神经网络（CNNs）已经在2D医学图像分割上取得了杰出的成功，但由于一些相互影响的挑战（包括体积图像复杂的解剖环境，3D网络的优化困难以及训练样本的不足），从3D医学图像中分割重要器官或结构依然是困难的任务。本文中，我们提出一种新颖有效的3D全卷积网络及3D深度监督机制来全面解决这些挑战；我们称其为3D DSN。我们的网络可以执行端到端学习，消除了冗余计算，减轻了过拟合的风险。此外，3D深度监督机制可以有效的解决梯度消失或爆炸问题，加速收敛同时提高辨别能力。该机制通过一个目标函数，直接指导网络upper和lower层的训练，从而抵消不稳定梯度变化的影响。我们也用了CRF作为后处理。我们在两个分割任务上进行了实验：1）肝脏分割；2）心脏和大血管分割。

### Introduction

​	目标器官或结构和其附近组织的边界总是模糊的，具有低对比度，这是由相似的成像相关的物理属性造成的，如CT成像中的衰减系数（attenuation coefficients）和MR成像中的松弛时间（relaxation times）。

本文的贡献如下：

​	1）提出一个带有3D反卷积层的3D网络结构，来连接粗特征到密集概率图预测。该结构消除了patch-based方法的计算冗余。并且，逐voxel的误差回传拓展了训练数据，避免了过拟合问题。

​	2）提出一个3D深度监督机制，通过一个目标函数，直接指导upper和lower层的训练，增强梯度流，因此可以学到更好的特征。该机制可以加速优化同时提高辨别能力。

​	3）3D DSN在两个挑战上取得了很好的结果。

### Related Work

​	目前的3D网络包括U-Net，V-Net，I2I-3D和VoxResNet。VoxResNet借鉴了2D深度残差网络，构建了一个很深的3D网络。并利用了多模态输入和多层上下文信息来达到sota结果。I2I-3D用于血管边界检测。为了定位小血管结构，将upper和lower层的特征concatenated到一起，紧接着1x1x1卷积。I2I-3D还利用了边路辅助监督（整体和密集的方式）。我们的方法以稀疏的方式利用了监督，如此可以大大减少网络参数以及计算负担。

​	肝脏分割的挑战来自于患者之间形状的巨大差异，肝脏和邻近器官（如胃、胰腺和心肝）之间低强度对比，以及各种病理（如肿瘤、肝硬化和囊肿）的存在。

​	心脏图像分割的挑战来自于心肌形状变化大，与周围组织的对比度低，血管分支结构复杂。

### Methods

![f1](images\f1.png)

​	如图1所示。我们利用3D 反卷积层将粗特征volumes连接到密集概率预测。特别的，我们迭代的执行一系列3x3x3卷积（反向strided output，如2为两倍上采样）。

#### 3D deep supervision mechanism

​	我们首先使用附加的反卷积层上采样一些lower-level和middle-level特征图。接着我们在其上利用softmax函数来生成额外的密集预测。我们计算这些分支的损失与最后的损失集成起来，共同回传梯度。
$$
L = L(X;W)+\sum_{d\in D}n_dL_d(X;W_d;\hat{w}_d)+\lambda(\parallel W\parallel^2 + \sum_{d\in D}\parallel \hat{w}_d\parallel^2)
$$
其中$\ n_d\ $是$\ L_d\ $的平衡权重，随着训练衰减。第一项是原损失，第二项是深度监督，第三项是权重衰减正则化。

​	Lee等提出了提高收敛率和CNN辨别能力，通过监督隐藏层的训练，深度监督可以直接从low和mid-level的隐藏层导出来帮助提高特征辨别能力。	

#### Contour refinement with conditional random field

​	我们利用概率阈值来生成最后的区域。为了提高分割结果，我们使用CRF模型来精修segmentation mask。该模型解决能量函数$\ E(y)=\sum_i-\log\hat{p}(y_i|x_i)+\sum_{i,j}f(y_i,y_j)\phi(x_i,x_j)\ $，其中第一项为一元可能性，表示label $\ y_i\ $在voxel $x_i$的分布。为了集成多尺度信息，$\ \hat{p}(y_i|x_i)\ $初始化为最后层预测和分支网络预测的线性组合：
$$
\hat{p}(y_i|x_i) = (1-\sum_{d\in D}\tau_d)\ p(y_i|x_i;W)+\sum_{d\in D}\tau_d\ p(y_i|x_i;W_d,\hat{w}_d)
$$
$\ E(y)\ $的第二项是成对的可能性，其中$\ f(y_i,y_j)=1\ $，如果$\ y_i\ne y_j\ $，否则为0；$\ \phi(x_i,x_j)\ $包含了局部外观和smoothness，通过灰度值value$\ I_i\ $和$\ I_j\ $以及voxel$\ x_i\ $和$\ x_j\ $的双边位置$\ s_i\ $和$\ s_j\ $：
$$
\begin{align*}
\phi(x_i,x_j) = &\mu_1\exp(-\frac{\parallel s_i-s_j\parallel^2}{2\theta^2_{\alpha}}-\frac{\parallel I_i-I_j\parallel^2}{2\theta^2_{\beta}})\\
&+\mu_2\exp(-\frac{\parallel s_i-s_j\parallel^2}{2\theta^2_{\gamma}})
\end{align*}
$$

### Experiment

肝脏分割：

​	输入层->两个巻积层（8@9x9x7），一个max-pooling层->两个巻积层（16@7x7x5，32@7x7x5），一个max-pooling层->两个巻积层（32@5x5x3，32@1x1x1）->两个反卷积->softmax。网络输入随机裁剪（160x160x72），随机旋转[90,180,270]度。

心脏分割：

​	input-conv1a(32)-pool1-conv2a(64)-conv2b(64)-pool2-conv3a(128)-conv3b(128)-pool3-conv4a(256)-conv4b(256)。

![f2](images\f2.png)

![f3](images\f3.png)

![f4](images\f4.png)

![f5](images\f5.png)

![f6](images\f6.png)

![f7](images\f7.png)

![t1](images\t1.png)

![t2](images\t2.png)

![t3](images\t3.png)