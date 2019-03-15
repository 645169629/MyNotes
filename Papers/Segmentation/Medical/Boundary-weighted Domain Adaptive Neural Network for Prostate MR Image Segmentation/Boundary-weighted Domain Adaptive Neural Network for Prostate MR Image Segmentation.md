## Boundary-weighted Domain Adaptive Neural Network for Prostate MR Image Segmentation

### Abstract

​	自动分割前列腺面临的挑战：前列腺与其他解剖结构之间缺乏清晰的边缘；前列腺本身的背景纹理复杂，大小、形状、强度分布变化大；缺少足够的训练数据。本文提出边界加权域自适应网络。为了使网络在分割时对边界更敏感，我们提出了边界加权分割损失。另外，我们还提出一个边界加权迁移学习方法来解决小数据集问题。我们的模型在PROMISE12上比其他state-of-the-art方法要好。

### Introduction

​	Yu[20]等设计了一个volumetric CNN 利用了固定的长短残差连接，以在有限的数据上提高性能。Nie[21]等提出了一个region-attention的半监督学习策略，利用未标注数据来克服缺少训练数据的困难。为了减少噪声的影响并抑制前列腺周围具有相似亮度的组织，Wang[22]等提出了一个新的深度神经网络，利用注意力机制，有选择性的利用多层特征用于前列腺分割。

​	为了准确分割弱边界的地方，本文提出边界加权分割损失，使得网络对边界更加敏感并对分割结果规则化。同时，受到对抗学习和迁移学习的影响，我们也采用了迁移学习以利用其他数据集的有用信息，来克服训练数据不足的影响。与一般迁移学习不同的是，边界加权知识迁移学习被设计使得迁移学习关注于边界信息。

### Related Works

#### A. Medical Image Segmentation

​	Ronneberger等[35]提出U-net，可以减少信息损失并加速收敛速度。Zhu等[33]提出双向卷积循环层，可以捕获intra-slice和inter-slice的信息。Li等[36]提出混合网络用于分割，包含 2D Dense-U-Net和3D网络。Yu等[37]提出densely-connected volumetric 网络。Chen等[28]受到深度残差网络的启发，提出一个3D volumetric网络，此外，该模型还集成了低层图像外观特征、隐式形状信息和高层上下文信息来提高分割性能。

​	一般来说，3D网络在医学图像分析中有一定优势，但其包含了大量参数，很难优化。并且医学数据较少，很容易过拟合。因此，还有很多工作需要做，以在有限训练数据下提高图像分割性能。

#### B. Deep Domain Adaptation

​	如图1所示。现有的深度领域自适应方法可以分为三类：无监督，有监督和半监督自适应。

​	无监督自适应指target领域数据标注不可用。如，Zhang等[26]提出全卷积自适应网络（FCAN）。Hoffman等[41]提出一个无监督领域自适应框架。他们的模型使用了一种全新的语义分割网络，通过全卷积领域对抗学习实现全局对齐。Sun等[42]提出一个新的CORAL损失，通过最小化source和target领域的差异来达到无监督领域自适应。

​	当target领域数据可用时，是有监督领域自适应。Tzeng等提出一个自适应层以及领域混淆损失，以学习语义有效并且领域不变的表示。Tzeng等[44]还提出一个网络结构，通过同时迁移学到的source语义结构到target领域，以有效的适应新领域。

​	半监督自适应指target领域一部分有label，一部分没有。

### Materials

![f1](images\f1.png)

​	target domain是 PROMISE12，source domain是单独的数据集。为了可视化两个数据集，我们从每个域随机取了280个slice，使用VGG-16网络将每个slice映射到长度为4096的特征向量。接着使用t-SNE来可视化，如图1。我们在这项工作中的假设是迁移学习可以帮助达到更好的图像分割结果（相比于简单的扩充数据集）。

### Boundary-Weighted Domain Adaptation

![f2](images\f2.png)

​	图2展示了BOWDA-Net的概览，包含了三个主要部分，source domain image segmentation network（SNet-s），target domain image segmentation network（SNet-t）以及domain feature discriminator（D）。SNet和D被设计为对抗的方式，用于克服领域偏移问题，利用source领域的数据信息，解决数据不足和弱边界问题。

#### A. Boundary-weighted Knowledge Transfer

​	数据分布差异导致的域偏移适应性迁移学习效率和性能的常见问题。近年来，对抗适应方法被提出来解决该问题，这种方法通过最小化对抗损失来减少领域距离。训练时，表示提取器分别学习source 和 target 领域的表示，领域判别器试图区别source和target领域的特征。当领域判别器无法区分两个领域的数据时，领域适应完成。尽管现有方法对解决领域偏移和提高迁移学习性能很有效，迁移学习过程没有关注目标领域需要的信息，使得现有方法无法有效处理弱边界问题。

​	为了解决上述挑战，本文提出一个有监督边界加权对抗领域适应策略。在我们的方法中，为了提取source领域的特征信息，我们先使用source 领域数据训练SNet-s，然后固定权重。训练时，SNet-s，SNet-t分别从source和target领域学习特征表示，然后提取的特征送到D中。为了提高D识别边界的能力，我们提出边界加权损失。设$\ \{x_s,\ y_s\}=\{(x^i_s,\ y^i_s)|i=1,...,m\}\ $表示source域的训练图像和gt，$\ \{x_t,\ y_t\}=\{(x^i_t,\ y^i_t)|i=1,...,n\}\ $为target域的训练图像和gt。$\mathrm{W}$ 表示高斯分布的边界图，由3x3的高斯函数构成。D的BWL为：
$$
\begin{align*}
L_D = &-\mathbb{E}_{x_s}[(1+\alpha\mathrm{W}_s)\log(D(SNet-s(x_s)))]\\
&-\mathbb{E}_{x_t}[(1+\alpha\mathrm{W}_t)\log(1-D(SNet-t(x_t)))]
\end{align*}
$$
其中$\ \mathrm{W}_s\ $表示source域的边界图，$\ \mathrm{W}_t\ $表示target域的边界图，$\ \alpha\ $表示加权系数。

#### B. Boundary-weighted Segmentation Loss

​	一般来说，对于图像分割，cross entropy（$L_{ce}$）是有效的损失函数。设$\ y\ $表示gt，$\hat{y}$表示分割结果，$L_{ce}$表示为：
$$
L_{ce}=-\sum_yy\log(\hat{y})+(1-y)\log(1-\hat{y})
$$
使用cross entropy损失的问题是其依赖于区域信息，使得网络无法准确分辨出边界。在训练时，BWL利用一个距离损失（$L_{dis}$）来规则化位置，形状以及分割一致性，使分割结果更接近物体边界。BWL可表示为：


$$
\begin{align*}
L_{seg}&=L_{dis}+L_{ce}\\
&=\beta\sum_{\hat{y}\in B}\hat{y}M_{dist}(y)-\sum_{y\in R}y\log(\hat{y})+(1-y)\log(1-\hat{y})
\end{align*}
$$
其中$\ R\ $表示整个分割区域，$\ B\ $表示分割结果中的边界点，$\beta$表示加权系数。$M_{dist}(y)$为距离图，由分割边界点的距离偏移构造。

​	总的来说，当训练SNet-s时，我们使用$\ L_{ce}\ $作为损失函数。在训练SNet-t时，使用包含$L_{seg}$和对抗损失的总损失$L_{total}$
$$
L_{total} = L_{seg}-\mathbb{E}_{x_t}[(1+\alpha\mathrm{W}_t)\log(1-D(SNet-t(x_t)))]
$$

#### C. Network Design and Configurations

​	如图2所示，SNet-s和SNet-t包含两条路径：降采样和上采样路径。降采样路径包含一个卷积块，三个densely-connected 残差块（DRBs）和三个平均池化层。pooling层的stride为2。上采样层包含三个反卷积层和三个DRBs。加上长连接，好处是1：可以在降采样和上采样之间传播上下文和梯度信息，环节梯度消失问题。2：可以帮助解决信息损失的问题。

​	DRB是我们提出的一个新结构，结合了densely connected 层，transition 层和残差连接，共同解决小数据集过拟合问题并提高信息传播，加快收敛。在DRB内部，密集连接提供了所有后续层之间的直接连接，所有之前层生成的特征图作为后续层的输入连接在一起。为了减少特征数量并融合密集连接层的特征，在密集连接层后增加了transition层。transition层包含一个1x1卷积层，减少了特征图数量，融合了特征图因此提高了模型的紧凑性。为了进一步提升信息传播并使网络易于后话，DRBs使用了残差连接。形式上的，考虑输入图像$\ x_0\ $通过DRB，设$\ x_l\ $为第$\ l^{th}\ $巻积层的输出，$\ H_l\ $为第$\ l^{th}\ $层的非线性变换（巻积层后跟BN层后跟ReLU）。对于DRBs，其输出为：
$$
x = H_t(H_l([x_0,x_1,...,x_{l-1}]))+x_0
$$
其中$\ [x_0,x_1,...x_{l-1}]\ $表示前面特征图的concatenation。$\ H_t\ $是transition层的非线性变换。DRBs包含不同数量（4,8,16,8,2）的BN-ReLU-Conv(1x1x1)-BN-ReLU-Conv(3x3x3)，growth rate为32。每个卷积层后面，加了dropout(0.3)。此外，为了使D获得更多有用信息，我们利用了多层表示。上采样路径中每个DRB提取的特征表示都作为D的输入。为了消除SNet-t和D之间的权重不平衡，以及关注边界信息，我们特别设计了domain discriminator的输出与输入大小相同，每个输出空间单元代表对应的图像像素输入target 域的概率。在domain discriminator中，我们使用了三个ConvBlock，两个反卷积层和一个输出层。

#### Experiments

![t1](images\t1.png)

![t2](images\t2.png)

![t3](images\t3.png)

![t4](images\t4.png)

![f3](images\f3.png)

![f4](images\f4.png)