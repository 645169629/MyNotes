## 3D Segmentation with Exponential Logarithmic Loss for Highly Unbalanced Object Sizes*

### Abstract

​	很多2D和3D分割方法取得了不错的结果，然而，大部分网络只解决相对少的标注（<10），在物体大小高度不平衡的分割研究很有限。本文提出一个网络结构和对应的损失函数，用于提高对小结构的分割。针对3D分割的计算可行性，将跳跃连接和深度监督相结合，提出了一种快速收敛高效计算的网络结构。此外，受到focal loss的启发，我们提出一个指数对数损失，不仅可以对标注的相对大小进行平衡，还能对分割困难程度进行平衡。我们在脑部分割（20个label）上达到了82%的平均dice。只需要100epoch就可以达到该精度，分割128x128x128的volume只需要0.4s。

### Introduction

​	当需要细致分割更多解剖结构时，就需要解决之前未被发现的问题，如计算可行性和高度不平衡的物体大小。只有很少的框架用于解决高度不平衡的标注。[8]中提出了一个2D网络用于分割3D脑部volume的所有切片。引入了误差矫正增强来计算label权重，它强调更新在验证集上准确率较低的类别的参数。尽管结果不错，label权重只应用于了加权交叉熵损失上，而没有用于Dice loss，并且将2D结果堆叠用于3D分割可能会造成连续切片的不一致性。

​	[9]中运用了一般化的Dice loss。他们不是对每个label计算Dice loss，而是对于一般Dice loss，计算乘积的加权除以GT和预测概率之和的加权和，权重与label频率成反比。事实上，Dice对于小结构的不利的，几个像素的误分类就会导致Dice系数大幅下降，而这种敏感性与结构之间的相对大小无关。因此，通过label频率来平衡对于Dice loss不是最优的。

​	为了解决3D分割中高度不平衡的物体尺寸和计算有效性，本文提出两个贡献：1）我们提出指数对数loss函数。[4]中，为了解决2类图像分类高度不平衡问题，仅从网络输出的softmax概率计算的调制因子乘以加权交叉熵，以关注于较不准确的类别。受到该不平衡分类的启发，我们提出一个包含对数Dice loss的损失函数，更多的关注于分割不准确的结构。对数Dice loss和加权交叉熵的非线性可以进一步由提出的指数参数来控制。以这种方法，网络可以对小结构和大结构都分割的很好。2）我们提出一个快速收敛，计算高效的网络结构，通过结合跳跃连接和深度监督，只有VNet 1/14的参数量，并比其快两倍。

### Methodology

#### 2.1 Proposed Network Architecture

![f1](images\f1.png)

​	3D分割网络需要更多的计算资源，我们提出的网络结构如图1所示。与大部分的分割网络相似，我们的网络包括编码path和解码path。网络包含多个conv block，每个由k个3x3x3卷积层（n通道）和BN层以及ReLU组成。为了更好的收敛，每个block还是用了一个带有1x1x1卷积层的跳跃连接。

​	我们将两个分支相加而不是concatenation，以减少内存消耗，因此该模块可以实现多尺度的预处理并可以训练更深的网络。通道数n在每次max pooling后加倍，在每次上采样后减少一半。更小的张量尺寸使用更多的层（k），以学得更多的抽象信息。编码path和解码path对应的特征通道concat，以更好的收敛。我们同时也加入了高斯噪声层和dropout层以避免过拟合。

​	与[5]相似的，我们利用了深度监督，以实现到hidden layer直接的回传。尽管深度监督提高了收敛性，但很占用内存。因此，我们忽略有最多channels的block。最后的1x1x1卷积+softmax提供了分割概率图。

#### 2.2 Exponential Logarithmic Loss

$$
L_{Exp} = w_{Dice}L_{Dice}+w_{Cross}L_{Cross}
$$

其中$\ w_{Dice}\ $和$\ w_{Cross}\ $分别是指数对数Dice损失（$\ L_{Dice}\ $）和加权指数交叉熵损失（$\ L_{Cross}\ $）的权重。
$$
L_{Dice} = E[(-\ln(Dice_{i}))^{\gamma_{Dice}}]\\
Dice_i = \frac{2(\sum_x\delta_{il}(x)p_i(x))+\epsilon}{(\sum_x\delta_{il}(x)+p_i(x))+\epsilon}
$$

$$
L_{Cross} = E[w_l(-\ln(p_l(x)))^{\gamma_{Cross}}]
$$

其中x是像素位置，i 为label。l是x处的gt label。E[]为 i 和x 分别在$\ L_{Dice}\ $和$\ L_{Cross}\ $的平均值。$\delta_{il}(x)$是Kronecker delta，当i=l为1，否则为0。$\ p_i(x)\ $是softmax概率。$\ \epsilon=1\ $为处理训练样本中缺失标签的加法平滑伪计数。$\ w_l=((\sum_kf_k)/f_l)^{0.5}\ $，$\ f_k\ $是label k的频率，$\ w_l\ $是label加权，以减少频率高的label的影响。$\ \gamma_{Dice}\ $和$\ \gamma_{Cross}\ $进一步控制损失函数的非线性。这里我们使用$\ \gamma_{Dice}=\gamma_{Cross}=\gamma\ $。

![f2](images\f2.png)

​	图2展示了$\ E[1-Dice_i]\ $和对数Dice 损失的比较。

​	[4]中使用调制因子，$\ (1-p_l)^{\gamma}\ $，与加权交叉熵损失相乘用于两类图像分类。除了使用label加权来平衡label频率，该focal loss也平衡了简单和困难样本。我们的指数损失也达到了类似的目标。$\ \gamma >1\ $，损失更关注于分割不准确的label。尽管focal loss对于2类图像分类很有用，但对于20类分割问题表现不好。我们设置$\ \gamma\in(0,1)\ $结果更好。图2展示出当$\ \gamma=0.3\ $，在$\ x=0.5\ $处出现了inflection 点。对于$\ x< 0.5\ $，loss的表现与$\ \gamma \ge 1\ $的loss表现类似。$\ x >0.5\ $时则相反。因此，该损失对低准确率和高准确率的预测都有提高。

#### 2.3 Training Strategy

​	运用了图像增强。如rotation，shifting，scaling。每个图像都有80%的概率被变换。使用Adam优化器，Nesterov momentum，学习率0.001，batch size为1，100epoch。

### Experiments

#### 3.1 Data and Experimental Setups

​	43个3D 脑部 MR图像。每个分割都有19个语义类别，以及背景。图像尺寸（128到337），spacing（0.9到1.5mm），每个图像都resample到相同的spacing（最小的spacing），以0填充短边，并resize到128x128x128。

​	表1展示了label高度不平衡。背景占据了93.5%。除了背景，最小和最大的结构分部为0.07%和50.24%。

​	我们研究了六个损失函数，并将最好的应用于VNet结构上。$\ w_{Dice}=0.8\ $以及$\ w_{Cross}=0.2\ $。

#### 3.2 Results and Discussion

![t1a](images\t1a.png)

![t1b](images\t1b.png)

​	表1（b）展示了五个实验的平均Dice。线性Dice 损失$\ E[1-Dice_i)]\ $表现最差。在大结构上表现很好，但小结构都丢失了。对数Dice loss（$\ L_{Dice} (\gamma =1)\ $）表现更好。加权交叉熵损失比线性 Dice loss好，但不如对数Dice loss。两个损失的加权比单个损失好。$\ L_{Exp}(\gamma =0.3)\ $给出了最好的结果。

​	当把最好的损失函数应用于VNet，它只比线性Dice loss和$\ L_{Exp}(\gamma=2)\ ​$表现好一点。这表明我们提出的网络结构在该问题上比VNet要好。

​	图3展示了验证集 Dice vs epoch。

![f3](images\f3.png)

​	图4展示了可视化结果。

![f4](images\f4.png)

