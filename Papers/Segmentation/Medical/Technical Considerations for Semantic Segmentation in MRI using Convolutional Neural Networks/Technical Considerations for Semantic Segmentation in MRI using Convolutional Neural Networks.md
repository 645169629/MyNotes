## Technical Considerations for Semantic Segmentation in MRI using Convolutional Neural Networks

### Abstract

​	MR的准确语义分割是组织形态测量和松弛参数估计的关键。人工分割被作为金标准，最近深度学习的一些研究可以有效的自动分割软组织。但是由于深度学习的随机性以及大量的超参，预测网络表现具有挑战。本文中，我们量化了三种因素对于分割的影响：网络结构，loss和训练数据特征。我们评估了这些变化对股骨软骨分割的影响，并提出潜在的网络结构和训练方案的修改。

### Introduction

​	MRI具有高分辨率和细致的软组织对比，使其普遍用于组织结构的可视化。使用MR图像定量分析组织特异性信息对于诊断和预后方案至关重要。人工标注作为金标准，耗时并且易出现观察者间差异。因此需要开发鲁棒的全自动组织分割技术。

​	分割的一个常见应用是关节软骨的分割，以研究与骨关节炎（OA）的相关变化。近年来MRI的研究进展主要集中在无创形态学和复合生物标志物的开发，已跟踪OA的发病和进展。有很好的证据证明，软骨形态和组成的变化可能作为早期的生物标志物。准确的测量阮国形态需要繁琐的手工分割。由于股骨软骨组织形态较薄，与周围组织结构的对比度较低，使得自动分割具有挑战性。

​	传统方法对形状建模，对膝关节的形变非常敏感。深度学习又具有随机性，需要针对问题调参。

​	`weighted cross-entropy 和soft Dice loss用来最小化类别不平衡的影响`

​	本文中，我们研究了三个因素对分割网络性能和泛化能力的影响：网络结构，损失函数和数据拓展。我们还利用训练的网络在不同的FOV下分割MR图像的灵敏度来量化网络的泛化性。

### Methods

#### 1. Dataset

​	数据来自OAI，DESS数据集。（FOV=14cm，Matrix=384x307，0填充到384x384，TE/TR=5/16ms，160切片，thickness 0.7mm）。88个病人，60训练，14验证，14测试。每个病人两个时间点的数据。（120 volumes training，28 volumes validation，28 volumes testing）

#### 2. Data Pre-processing

​	网络训练之前，所有数据在slice dimension降采样2倍（384x384x80），以增加SNR并减少计算复杂度。先前的研究证明，约1.5mm的切片足以进行软骨形态测量。使用平方和组合（sum-of-squares）对图像进行下采样，取mask的并集对部分体积伪影进行补偿，对相应的mask进行下采样。图像裁剪到288x288（通过计算质心）。接着从中间和外侧移除4slice，volume维度为288x288x72。所有扫描0均值化。

#### 3. Network Architecture

![f1](images\f1.png)

​	使用了U-Net，SegNet和DeeplabV3+，这些网络都是encoder-decoder模型的变体。U-Net使用max-pooling和反卷积来降采样和上采样，还有skip connections以共享空间信息。SegNet通过传递pooling指数到upsampling层以避免复制编码器权重的开销。DeeplabV3+实现了“Xception”块以及使用空间金字塔池化和空洞卷积，以此不增加参数量的情况下获得更大的感受野。DeepliabV3+ decoder采用了双线性上采样。

​	所有结构都训练100epoch，接着fine-tuned 100epoch。

#### 4. Volumetric Architectures

​	我们训练了2.5D和3D U-Net结构。2.5D网络使用t个连续切片堆叠以生成中心slice的mask。三个2.5D网络，输入thickness t分别为3，5，7。

​	3D网络需要slice数是$2^{N_p}$，其中$N_P$是pooling的步数。为了保持和2D，2.5D网络统一的pooling步数，3D U-Net使用32个切片作为输入。所有网络保持相当的权重数量。

#### 5. Loss Function

​	通常，binary cross-entropy用于二分类任务。但是类别不平衡会限制最优的性能。我们使用三个其他loss作为比较：soft Dice loss，weighted cross-entropy，以及focal loss（$\gamma = 3$）。

#### 6. Data Augmentation

![f2](images\f2.png)

​	每个2D slice和对应mask都随机都增加异质性。增广过程包含一系列变换：1. 缩放 2. 剪切 3. gamma变换 4. 运动模糊。所有3D slice都被增广了4倍，生成包含5x的数据。使用增广数据的网络只训练1/5的epoch。

#### 7. Generalizability to Multiple Fields of View

​	比较了不同FOV（相同分辨率）的影响。V0（288x288x72），V1（320x320x80），V2（352x352x80），V3（384x384x80）。

#### 8. Training Data Extent

​	使用不同大小的训练数据子集（60，5，15，30）来比较性能，分部训练100，1200，400，200个epoch。

#### 9. Network Training Hyperparameters

![t1](images\t1.png)

​	巻积层使用“He”初始化。使用Adam optimizer（$\beta_1 = 0.9，\beta_2 = 0.999，\varepsilon = 1e-8​$）。

#### 10. Quantitative Comparisons

​	使用DSC，VOE，CV来度量。

### Results

![t2](images\t2.png)

![f3](images\f3.png)

![f4](images\f4.png)

![f5](images\f5.png)

![f6](images\f6.png)

![f7](images\f7.png)

#### 1. Network Architecture Comparison

​	在含有全层软骨丢失区域和软骨下骨剥落区域、边缘切片和中外侧过渡区域的切片中，所有网络的表现均较差。除了这些区域，这些网络准确的分割了病理引起的信号不均匀的切片，并且与具有相似信号的解剖接近。图4A展示了在边缘区域和中外侧过渡区域的性能下降。U-Net，SegNet，DeeplabV3+性能没有特别的差异。

#### 2. Volumetric Architectures Comparison

2D，2.5D和3D U-Net结构性能没有重大差异。2D U-Net表现比3D U-Net要好。图4B展示了性能下降的区域。3D U-Net模型中，膝关节外侧间室的DSC较内侧间室大。

#### 3. Loss Function Comparison

​	BCE，soft Dice和focal losses性能差不多，都比WCE要好（图4C）。使用WCE loss时，false-positives（背景分成前景）比false-negatives（前景分成背景）要多很多。WCE 99%以上的误差来源于虚警（图5C）。对于BCE，soft Dice和focal loss，false-positive和false-negative差别不大。

#### 4. Data Augmentation Comparison

The use of augmented training data significantly decreased network performance (p<0.001) compared
to the augmented training data set (Figure 4B).   `???`

#### 5. FOV Generalizability Comparison

​	baseline U-Net 在四个数据集上差别较大（图6）。V1，V2上的性能比V0差很多。V0和V3差别不大。augmented U-Net差别不大。

#### 6. Data Extent Trend

​	三个网络性能都随训练数据增加而提升（图7）。

### Discussion

​	U-Net，SegNet和DeeplabV3+之间没有太大差别，2D和2.5D网络也没有太大差别。BCE，soft Dice和focal losses有相似的false-positive和false-negative率，而BCE容易产生false-positive误差。此外，虽然数据增广降低了U-Net性能，但它提高了不同FOV之间性能的泛化能力。此外，本研究要证明了随着数据量增加，分割性能会按照幂律关系改变。

#### 1. Base Architecture Variations

​	Raghu等和Bengio等建议表达能力受到网络结构的影响较大（如深度、正则化）。网络结构对整体表现力的有限影响的趋势表明，改进体系结构在训练性能更好的网络方面没有那么有效。

#### 2. Practical Design for Volumetric Architecures

​	2.5D网络接受输入为（M x N x t），2D网络接受输入为（M x N x 1），输出是同样大小，两个网络只有第一个卷积层不同。

#### 3. Selecting Loss Functions

​	WCE 目的是对不平衡类之间的损失进行归一化，人工加权会使网络对不平衡类进行过度分类。

​	在由不同损失函数引起的四种误差分布中，focal loss 产生的误差以$\ p_T\ $为中心最密集，这可能使$\ p_T\ $最适合未来的优化。

#### 4. Achieving Multi-FOV Robustness through Data Augmentation

​	在未增广数据上训练的U-Net对不同的FOV会表现不同的性能。而在增广数据上训练的U-Net在不同FOV上表现出相同的性能。

​	尽管数据增广是广泛使用的方法，其训练的2D U-Net只有次优的性能。我们建议，扩增应该精心策划，以增加对预期图像变化的网络表达率，特别是对潜在测试图像中具有可变大小的感兴趣的组织。

#### 5. Navigating Training with Limited Data

​	Hestness等最近的一项研究中，证明了无论在何种架构下，图像分类的错误率都是随着$\beta < 1​$的幂律缩放而降低的。语义分割表现出这种趋势。这种缓慢增长的幂律性能拓展可以让我们根据数据量的增加来经验估计这些网络的性能。

#### 6. Limitations

​	超参是根据经验确定的；3D U-Net需要调节slice以增加batch size和filter数量；理解多类分割对每个组织的性能影响是很有用的。

