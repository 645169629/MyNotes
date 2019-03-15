## Automatic 3D Cardiovascular MR Segmentation with Densely-Connected Volumetric ConvNets

> MICCAI 2017

### Abstract

​	心脏MR图像分割挑战：心脏边界模糊，不同研究对象之间存在较大的解剖差异。问题提出一个新的densely-connected volumetric CNN，称为DenseVoxNet，来自动分割心脏和血管结构。DenseVoxNet有三个好处：1）通过密集连接机制最大限度保留了信息流，因此减轻了网络训练。2）通过特征重用避免了学习冗余的特征，因此需要更少的参数，这对于具有有限数据的医学应用很重要。3）我们增加了辅助路径来增强梯度传播并稳固学习过程。我们的网络在HVSMR 2016挑战中达到最好的dice 系数。

### Introduction

​	心脏一些部分边界不清晰，与周围组织的对比度较低；心脏结构内部主体间的变化较大，为分割带来更大的困难。3D ConvNets通常在每一层产生大量的特征通道，有大量的参数需要训练。尽管这些网络引入了不同的跳跃连接来减轻训练，但使用有限的数据来训练有效的心脏分割模型还是具有挑战的。

​	DenseVoxNet结合了密集连接的概念，有三个好处：1）实现了一层到后面所有层的直接连接。每一层都能收到损失函数额外的监督信息，因此网络更容易训练；2）DenseVoxNet相比于其他3D网络具有更少的参数。因为每一层都可以获得其所有前面层的信息，可以避免学习冗余特征图。因此DenseVoxNet每层具有更少的特征图，在数据有限时不容易过拟合；3）我们通过辅助路径提高了梯度流并稳定了学习过程。

### Method

#### 2.1 Dense Connection

​	记$\ \mathrm{x}_{\ell}\ $为第$\ \ell^{th}\ $层的输出，$\ \mathrm{x}_{\ell}\ $可以经过$\ H_l(x)\ $变换得到：
$$
\mathrm{x}_{\ell} = H_{\ell}(\mathrm{x}_{\ell -1})
$$
其中$\ H_{\ell}(\mathrm{x})\ $可以看作是卷积、池化、BN、ReLU等的组合。为了解决梯度消失问题，ResNet引入了一种跳跃连接，将

$\ H_{\ell}(\mathrm{x})\ $的响应与前一层的恒等变换相结合来增强信息传播：
$$
\mathrm{x}_{\ell} = H_{\ell}(\mathrm{x}_{\ell -1})+\mathrm{x}_{\ell -1}
$$
但是，恒等函数和$\ H_{\ell}\ $的输出是通过相加来结合的，这可能阻碍信息流。

​	为了进一步增强网络中的信息流，密集连接实现了从一层到后面所有层的连接。即：
$$
\mathrm{x}_{\ell} = H_{\ell}([\mathrm{x}_0,\mathrm{x}_1,...,\mathrm{x}_{\ell -1}])
$$
其中$\ [...]\ $表示concatenation操作。如图1所示，密集连接使得所有层获得直接的监督信息。更重要的是，这种机制可以对特征重用。

#### 2.2 The Architecture of DenseVoxNet

![f1](images\f1.png)

​	我们将降采样部分分成两个densely-connected blocks，称为DenseBlock，每个DenseBlock包含12个变换层。每个变换层都是由BN，ReLU，3x3x3卷积顺序组成。growth rate k 为12。第一个DenseBlock以16输出通道，stride 为2的Conv为前缀，以学习原始特征。中间两个DenseBlocks 是transition block，包含BN，ReLU，1x1x1卷积核2x2x2 max pooling。

​	上采样部分包含BN，ReLU，1x1x1卷积核2x2x2反卷积。接着通过1x1x1卷积核softmax层来生成最后的label map。每个Conv层后都跟0.2的dropout。

#### 2.3 Training Procedure

​	随机剪裁64x64x64，最后在重叠区域的分割结果通过投票策略获得。

### Experiments and Results

#### Dataset and Pre-processing

​	10 3D心脏MR图像训练，10个测试。HVSMR 2016数据集包含心肌和大血管的标注。所有的MR图像normalized到0均值和统一方差。我们没有空间重采样。为了利用有限的训练数据，我们通过旋转90，180，270度以及平面翻转来数据增广。

#### Qualitative Results

![f2](images\f2.png)

​	如图2所示，蓝色和紫色分别表示血池和心肌。

​	评估标准，Dice、ABD、Hausdorff。Higher Dice 表示分割结果和gt更相似，lower ABD 和 Hausdorff表示更高的边界相似性。

![t1](images\t1.png)

![t2](images\t2.png)